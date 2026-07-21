"""NSE website client. Implements §8.2 scraping etiquette.

NSE's public JSON endpoints reject naive HTTP clients: no cookies, no
browser-ish headers, or too many requests, and you get a 401/403 forever. The
rules this module encodes, all from §8.2:

* **One shared session per run.** Cookies earned by the warm-up are reused.
* **Warm up on the homepage** to collect cookies before any API call, and
  re-warm on a 401/403 rather than hammering the endpoint.
* **Realistic headers** (User-Agent, Accept, Accept-Language, Referer), all
  from config.
* **Hard rate cap:** >= 2s between calls *per endpoint*, except the 30s
  announcements poll which is its own budget.
* **Exponential backoff** on failure.
* **Fail LOUDLY** after 5 consecutive failures -- Telegram alert, journalled
  error. Never silently degrade.
* **Every endpoint URL lives in config**, because NSE changes them and the
  owner maintains them.

The HTTP client is injectable, so tests exercise all of the above without a
network.
"""

from __future__ import annotations

import datetime as _dt
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from core import clock
from core.config import Settings, get_settings
from core.journal import Journal, get_journal

log = logging.getLogger(__name__)


class NSEError(RuntimeError):
    """A request to NSE failed after all retries."""


class NSEBlocked(NSEError):
    """NSE returned 401/403 even after re-warming -- we are being refused."""


@dataclass
class NSEResponse:
    """A successful fetch plus the metadata callers need for freshness checks."""

    data: Any
    endpoint: str
    fetched_at: _dt.datetime
    from_cache: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


class RateLimiter:
    """Per-endpoint minimum spacing. §8.2's 'hard cap request rates'.

    Keyed by endpoint name, not by host: the announcements poll gets its own
    30s budget while every other endpoint shares the 2s floor.
    """

    def __init__(self, default_interval: float, sleeper: Callable[[float], None] | None = None):
        self.default_interval = default_interval
        self._overrides: dict[str, float] = {}
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()
        self._sleep = sleeper or time.sleep

    def set_interval(self, endpoint: str, seconds: float) -> None:
        self._overrides[endpoint] = seconds

    def wait(self, endpoint: str) -> float:
        """Block until this endpoint may be called again. Returns seconds slept."""
        interval = self._overrides.get(endpoint, self.default_interval)
        with self._lock:
            last = self._last.get(endpoint)
            now = time.monotonic()
            delay = 0.0
            if last is not None:
                elapsed = now - last
                if elapsed < interval:
                    delay = interval - elapsed
            self._last[endpoint] = now + delay
        if delay > 0:
            self._sleep(delay)
        return delay


class NSEClient:
    """Shared-session NSE client with warm-up, backoff and loud failure."""

    def __init__(
        self,
        settings: Settings | None = None,
        http: Any = None,
        journal: Journal | None = None,
        alerts: Any = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self.settings = (settings or get_settings()).section("nse")
        self.journal = journal or get_journal()
        self._alerts = alerts
        self._sleep = sleeper or time.sleep
        self._http = http
        self._owns_http = http is None

        self.base_url = str(self.settings.require("base_url"))
        self.timeout = float(self.settings.get("timeout_seconds", 15))
        self.max_retries = int(self.settings.get("max_retries", 4))
        self.backoff_base = float(self.settings.get("backoff_base_seconds", 2.0))
        self.cookie_ttl = float(self.settings.get("cookie_ttl_seconds", 900))
        self.failure_threshold = int(self.settings.get("consecutive_failures_before_alert", 5))

        self.limiter = RateLimiter(
            float(self.settings.get("min_seconds_between_calls", 2.0)), sleeper=self._sleep
        )
        self.limiter.set_interval(
            "announcements", float(self.settings.get("announcements_poll_seconds", 30))
        )

        self._warmed_at: Optional[float] = None
        self._consecutive_failures = 0
        self._alerted = False
        self._lock = threading.RLock()

    # -- plumbing ---------------------------------------------------------

    @property
    def alerts(self) -> Any:
        if self._alerts is None:
            from live.alerts import get_alerts

            self._alerts = get_alerts()
        return self._alerts

    @property
    def http(self) -> Any:
        """The shared session. One per run (§8.2)."""
        if self._http is None:
            import httpx

            self._http = httpx.Client(
                headers=dict(self.settings.get("headers", {}) or {}),
                timeout=self.timeout,
                follow_redirects=True,
            )
        return self._http

    def close(self) -> None:
        if self._owns_http and self._http is not None:
            try:
                self._http.close()
            except Exception:  # pragma: no cover - best effort
                pass
            self._http = None

    def __enter__(self) -> "NSEClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- warm-up ----------------------------------------------------------

    def warm_up(self, force: bool = False) -> bool:
        """Hit the NSE homepage to collect cookies (§8.2). Returns success."""
        with self._lock:
            fresh = (
                self._warmed_at is not None
                and (time.monotonic() - self._warmed_at) < self.cookie_ttl
            )
            if fresh and not force:
                return True
            url = str(self.settings.get("warmup_url", self.base_url))
            try:
                response = self.http.get(url, headers=dict(self.settings.get("headers", {}) or {}))
                status = getattr(response, "status_code", 200)
                if status >= 400:
                    log.warning("NSE warm-up returned %s", status)
                    return False
                self._warmed_at = time.monotonic()
                log.debug("NSE cookies warmed")
                return True
            except Exception as exc:
                log.warning("NSE warm-up failed: %s", exc)
                return False

    # -- fetching ---------------------------------------------------------

    def fetch(
        self, endpoint: str, params: dict[str, Any] | None = None, **path_args: Any
    ) -> NSEResponse:
        """Fetch a configured endpoint by name, with etiquette and retries.

        ``endpoint`` keys into ``nse.endpoints`` in settings. Path templates
        like ``"/api/quote-equity?symbol={symbol}"`` are filled from
        ``path_args``.
        """
        endpoints = self.settings.get("endpoints", {}) or {}
        if endpoint not in endpoints:
            raise NSEError(
                f"Endpoint {endpoint!r} is not configured. Add it to "
                f"config/settings.yaml `nse.endpoints` -- URLs live in config "
                f"because NSE changes them (§8.2)."
            )
        path = str(endpoints[endpoint])
        if path_args:
            path = path.format(**path_args)
        url = path if path.startswith("http") else f"{self.base_url}{path}"

        self.limiter.wait(endpoint)
        self.warm_up()

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self.http.get(
                    url,
                    params=params,
                    headers=dict(self.settings.get("headers", {}) or {}),
                )
                status = getattr(response, "status_code", 200)

                if status in (401, 403):
                    # §8.2: re-warm rather than hammer the endpoint.
                    log.warning("NSE %s on %s (attempt %d); re-warming cookies",
                                status, endpoint, attempt + 1)
                    self.warm_up(force=True)
                    last_error = NSEBlocked(f"{status} on {endpoint}")
                    self._backoff(attempt)
                    continue

                if status >= 500:
                    last_error = NSEError(f"{status} on {endpoint}")
                    self._backoff(attempt)
                    continue

                if status >= 400:
                    raise NSEError(f"{status} on {endpoint}: {getattr(response, 'text', '')[:200]}")

                data = response.json()
                self._on_success()
                return NSEResponse(data=data, endpoint=endpoint, fetched_at=clock.now_ist())

            except NSEError:
                raise
            except Exception as exc:
                last_error = exc
                log.warning("NSE fetch %s failed (attempt %d): %s", endpoint, attempt + 1, exc)
                self._backoff(attempt)

        self._on_failure(endpoint, last_error)
        raise NSEError(f"NSE {endpoint} failed after {self.max_retries} attempts: {last_error}")

    def _backoff(self, attempt: int) -> None:
        """Exponential backoff with jitter, so retries do not synchronise."""
        delay = self.backoff_base * (2 ** attempt)
        self._sleep(delay + random.uniform(0, self.backoff_base / 2))

    def _on_success(self) -> None:
        with self._lock:
            if self._consecutive_failures and self._alerted:
                self.alerts.send(
                    f"✅ NSE client recovered after {self._consecutive_failures} failures"
                )
            self._consecutive_failures = 0
            self._alerted = False

    def _on_failure(self, endpoint: str, error: Exception | None) -> None:
        """§8.2: fail LOUDLY after 5 consecutive failures. Never degrade quietly."""
        with self._lock:
            self._consecutive_failures += 1
            count = self._consecutive_failures
        message = f"NSE {endpoint} failed ({count} consecutive): {error}"
        self.journal.record_error("nse", message, severity="ERROR", endpoint=endpoint)
        if count >= self.failure_threshold and not self._alerted:
            self._alerted = True
            self.alerts.error(
                "nse",
                f"{count} consecutive NSE failures (last endpoint: {endpoint}).\n"
                f"The scraper is down -- see docs/RUNBOOK.md. Endpoints may have "
                f"changed; they live in config/settings.yaml `nse.endpoints`.",
                severity="CRITICAL",
            )

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    # -- typed endpoint wrappers -----------------------------------------
    #
    # Each returns plain Python structures, normalised into the shapes the
    # engines expect. NSE's field names change; normalisation happens here so
    # only this file needs editing when it does.

    def announcements(self, since: _dt.datetime | None = None) -> list[dict[str, Any]]:
        """§6.1 corporate announcements feed, newest first."""
        raw = self.fetch("announcements").data
        rows = raw if isinstance(raw, list) else raw.get("data", [])
        out: list[dict[str, Any]] = []
        for row in rows:
            record = {
                "symbol": (row.get("symbol") or row.get("sym") or "").strip().upper(),
                "headline": (row.get("desc") or row.get("subject") or "").strip(),
                "body": (row.get("attchmntText") or row.get("smIndustry") or "").strip(),
                "attachment_url": row.get("attchmntFile") or None,
                "announcement_id": str(
                    row.get("seqId") or row.get("id") or f"{row.get('symbol')}-{row.get('an_dt')}"
                ),
                "timestamp": _parse_nse_datetime(row.get("an_dt") or row.get("exchdisstime")),
                "raw": row,
            }
            if since and record["timestamp"] and record["timestamp"] < since:
                continue
            out.append(record)
        return out

    def preopen_snapshot(self, key: str = "NIFTY") -> list[dict[str, Any]]:
        """§6.5 pre-open call-auction feed with matched/unmatched quantities."""
        raw = self.fetch("preopen").data
        rows = raw.get("data", []) if isinstance(raw, dict) else raw
        out: list[dict[str, Any]] = []
        for row in rows:
            meta = row.get("metadata", row)
            detail = row.get("detail", {}).get("preOpenMarket", {}) if isinstance(row, dict) else {}
            symbol = (meta.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            out.append({
                "symbol": symbol,
                "indicative_price": _as_float(meta.get("lastPrice") or detail.get("IEP")),
                "prev_close": _as_float(meta.get("previousClose")),
                "change_pct": _as_float(meta.get("pChange")),
                "final_quantity": _as_int(detail.get("finalQuantity") or meta.get("finalQuantity")),
                "total_buy_quantity": _as_int(detail.get("totalBuyQuantity")),
                "total_sell_quantity": _as_int(detail.get("totalSellQuantity")),
                "ato_buy_qty": _as_int(detail.get("atoBuyQty")),
                "ato_sell_qty": _as_int(detail.get("atoSellQty")),
                "raw": row,
            })
        return out

    def fno_ban_list(self) -> set[str]:
        """§3 veto input: symbols at MWPL >= 95%."""
        raw = self.fetch("ban_list").data
        if isinstance(raw, dict):
            rows = raw.get("data") or raw.get("BanList") or []
        else:
            rows = raw
        symbols: set[str] = set()
        for row in rows:
            if isinstance(row, str):
                symbols.add(row.strip().upper())
            elif isinstance(row, dict):
                value = row.get("symbol") or row.get("Symbol") or row.get("secName")
                if value:
                    symbols.add(str(value).strip().upper())
        return symbols

    def surveillance_lists(self) -> dict[str, list[dict[str, Any]]]:
        """§6.10 ASM and GSM stage lists, as ``{list_name: [{symbol, stage}]}``."""
        out: dict[str, list[dict[str, Any]]] = {}
        for name, endpoint in (("asm", "asm_list"), ("gsm", "gsm_list")):
            try:
                raw = self.fetch(endpoint).data
            except NSEError as exc:
                log.error("Could not fetch %s list: %s", name, exc)
                out[name] = []
                continue
            rows = raw.get("data", []) if isinstance(raw, dict) else raw
            parsed: list[dict[str, Any]] = []
            for row in rows:
                symbol = row.get("symbol") or row.get("Symbol")
                if not symbol:
                    continue
                parsed.append({
                    "symbol": str(symbol).strip().upper(),
                    "stage": str(
                        row.get("asmSurvIndicator")
                        or row.get("gsmSurvIndicator")
                        or row.get("stage")
                        or ""
                    ).strip() or None,
                })
            out[name] = parsed
        return out

    def fii_dii_cash(self) -> list[dict[str, Any]]:
        """§6.9 FII/DII cash flows (INR crore)."""
        raw = self.fetch("fii_dii").data
        rows = raw if isinstance(raw, list) else raw.get("data", [])
        return [
            {
                "category": (row.get("category") or "").strip(),
                "date": row.get("date"),
                "buy_value_cr": _as_float(row.get("buyValue")),
                "sell_value_cr": _as_float(row.get("sellValue")),
                "net_value_cr": _as_float(row.get("netValue")),
            }
            for row in rows
        ]

    def fii_derivatives(self) -> list[dict[str, Any]]:
        """§6.9 FII participant-wise open interest in index futures."""
        raw = self.fetch("fii_derivatives").data
        rows = raw if isinstance(raw, list) else raw.get("data", [])
        return [
            {
                "client_type": (row.get("client_type") or row.get("clientType") or "").strip(),
                "future_index_long": _as_float(row.get("future_index_long")),
                "future_index_short": _as_float(row.get("future_index_short")),
                "raw": row,
            }
            for row in rows
        ]

    def india_vix(self) -> Optional[float]:
        """India VIX spot -- the §6.8 IV gate and the §7 PANIC input."""
        raw = self.fetch("india_vix").data
        rows = raw.get("data", []) if isinstance(raw, dict) else raw
        for row in rows:
            name = (row.get("index") or row.get("indexSymbol") or "").upper()
            if "VIX" in name:
                return _as_float(row.get("last") or row.get("lastPrice"))
        return None

    def bulk_deals(self) -> list[dict[str, Any]]:
        raw = self.fetch("bulk_deals").data
        return raw.get("data", []) if isinstance(raw, dict) else raw

    def block_deals(self) -> list[dict[str, Any]]:
        raw = self.fetch("block_deals").data
        return raw.get("data", []) if isinstance(raw, dict) else raw

    def equity_quote(self, symbol: str) -> dict[str, Any]:
        """Full quote including the §8.4 price band, for the kernel's veto."""
        raw = self.fetch("equity_quote", symbol=symbol.upper()).data
        price_info = raw.get("priceInfo", {}) if isinstance(raw, dict) else {}
        band = price_info.get("pPriceBand") or {}
        return {
            "symbol": symbol.upper(),
            "last_price": _as_float(price_info.get("lastPrice")),
            "prev_close": _as_float(price_info.get("previousClose")),
            "upper_band": _as_float(band.get("upper") or price_info.get("upperCP")),
            "lower_band": _as_float(band.get("lower") or price_info.get("lowerCP")),
            "raw": raw,
        }

    def trading_holidays(self) -> list[dict[str, Any]]:
        """NSE holiday master -- diffed against config/holidays.yaml."""
        raw = self.fetch("holidays").data
        if isinstance(raw, dict):
            for key in ("CM", "cm", "EQ"):
                if key in raw:
                    return raw[key]
            return next((v for v in raw.values() if isinstance(v, list)), [])
        return raw


# ---------------------------------------------------------------------------
# parsing helpers -- NSE is inconsistent about types and formats
# ---------------------------------------------------------------------------


def _as_float(value: Any) -> Optional[float]:
    if value is None or value == "" or value == "-":
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> Optional[int]:
    parsed = _as_float(value)
    return int(parsed) if parsed is not None else None


_NSE_DATETIME_FORMATS = (
    "%d-%b-%Y %H:%M:%S",
    "%d-%b-%Y %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%d-%m-%Y %H:%M:%S",
)


def _parse_nse_datetime(value: Any) -> Optional[_dt.datetime]:
    """Parse NSE's several datetime formats into tz-aware IST."""
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return clock.to_ist(value)
    text = str(value).strip()
    if not text:
        return None
    for fmt in _NSE_DATETIME_FORMATS:
        try:
            return clock.to_ist(_dt.datetime.strptime(text, fmt))
        except ValueError:
            continue
    log.debug("Unparseable NSE datetime: %r", text)
    return None
