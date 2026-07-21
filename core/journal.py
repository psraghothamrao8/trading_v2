"""SQLite journal. Implements §0.6: *everything* is journalled.

Signals, orders, fills, trades, rejections, errors, regime decisions, daily
summaries, nightly data snapshots and backtest runs all land here, with
timestamps in Asia/Kolkata.

Design notes
------------
* One connection per :class:`Journal` instance, ``check_same_thread=False``
  because APScheduler runs jobs on a worker thread pool. Writes are wrapped in
  a lock so a websocket thread and a scheduler thread cannot interleave.
* WAL mode: the digest job reads while the tick handler writes.
* Schema is created idempotently at open, so a restart mid-day is safe (§9.5).
* Nothing here raises on a duplicate insert -- journalling must never be the
  reason a trade fails. Failures are logged, not propagated.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

from core import clock
from core.config import data_path, get_settings
from core.types import Fill, Order, RiskDecision, Signal

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Every Signal an engine emitted, whether or not it became an order.
CREATE TABLE IF NOT EXISTS signals (
    signal_id   TEXT PRIMARY KEY,
    ts          TEXT NOT NULL,
    trade_date  TEXT NOT NULL,
    engine      TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    side        TEXT NOT NULL,
    entry_type  TEXT NOT NULL,
    ttl         TEXT NOT NULL,
    stop        REAL,
    targets     TEXT,
    reference_price REAL,
    reason      TEXT,
    meta        TEXT
);
CREATE INDEX IF NOT EXISTS idx_signals_date   ON signals(trade_date);
CREATE INDEX IF NOT EXISTS idx_signals_engine ON signals(engine, trade_date);

-- Orders that passed the risk kernel and were routed (paper or live).
CREATE TABLE IF NOT EXISTS orders (
    order_id        TEXT PRIMARY KEY,
    signal_id       TEXT,
    ts              TEXT NOT NULL,
    trade_date      TEXT NOT NULL,
    engine          TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    quantity        INTEGER NOT NULL,
    entry_type      TEXT NOT NULL,
    product         TEXT NOT NULL,
    segment         TEXT NOT NULL,
    price           REAL,
    trigger_price   REAL,
    stop            REAL,
    is_entry        INTEGER NOT NULL DEFAULT 1,
    mode            TEXT NOT NULL,          -- paper | live
    status          TEXT NOT NULL,          -- SUBMITTED | FILLED | CANCELLED | FAILED
    broker_order_id TEXT,
    reason          TEXT,
    meta            TEXT
    -- Deliberately NO foreign key to signals: the journal is an append-only
    -- log, and a flatten/kill order has no originating signal. Referential
    -- integrity here would let journalling reject a real order (§0.6).
);
CREATE INDEX IF NOT EXISTS idx_orders_date   ON orders(trade_date);
CREATE INDEX IF NOT EXISTS idx_orders_engine ON orders(engine, trade_date);

-- Fills, real or simulated. `costs` is the §4 model output, never zero in prod.
CREATE TABLE IF NOT EXISTS fills (
    fill_id         TEXT PRIMARY KEY,
    order_id        TEXT NOT NULL,
    ts              TEXT NOT NULL,
    trade_date      TEXT NOT NULL,
    engine          TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    quantity        INTEGER NOT NULL,
    price           REAL NOT NULL,
    costs           REAL NOT NULL DEFAULT 0,
    is_paper        INTEGER NOT NULL DEFAULT 1,
    broker_order_id TEXT,
    meta            TEXT
);
CREATE INDEX IF NOT EXISTS idx_fills_date ON fills(trade_date);

-- Round-trip trades: one row per completed position, net of costs.
CREATE TABLE IF NOT EXISTS trades (
    trade_id     TEXT PRIMARY KEY,
    engine       TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    side         TEXT NOT NULL,
    quantity     INTEGER NOT NULL,
    entry_ts     TEXT NOT NULL,
    entry_price  REAL NOT NULL,
    exit_ts      TEXT,
    exit_price   REAL,
    gross_pnl    REAL,
    costs        REAL NOT NULL DEFAULT 0,
    net_pnl      REAL,
    r_multiple   REAL,
    exit_reason  TEXT,
    trade_date   TEXT NOT NULL,
    mode         TEXT NOT NULL,
    meta         TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_date   ON trades(trade_date);
CREATE INDEX IF NOT EXISTS idx_trades_engine ON trades(engine);

-- §3: every rejection, with the veto that fired.
CREATE TABLE IF NOT EXISTS rejections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    trade_date  TEXT NOT NULL,
    engine      TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    side        TEXT,
    quantity    INTEGER,
    signal_id   TEXT,
    order_id    TEXT,
    reason_code TEXT NOT NULL,
    reason      TEXT NOT NULL,
    meta        TEXT
);
CREATE INDEX IF NOT EXISTS idx_rejections_date ON rejections(trade_date);
CREATE INDEX IF NOT EXISTS idx_rejections_code ON rejections(reason_code);

-- Errors and loud degradations (scraper failures, websocket gaps, auth loss).
CREATE TABLE IF NOT EXISTS errors (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    source     TEXT NOT NULL,
    severity   TEXT NOT NULL,       -- WARNING | ERROR | CRITICAL
    message    TEXT NOT NULL,
    meta       TEXT
);
CREATE INDEX IF NOT EXISTS idx_errors_date ON errors(trade_date);

-- §7 regime decisions, one row per classification, with the inputs.
CREATE TABLE IF NOT EXISTS regime_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    trade_date  TEXT NOT NULL,
    regime      TEXT NOT NULL,
    inputs      TEXT,
    enabled_engines TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_regime_day ON regime_log(trade_date, ts);

-- §7 15:45 digest source-of-truth, one row per session.
CREATE TABLE IF NOT EXISTS daily_summary (
    trade_date       TEXT PRIMARY KEY,
    mode             TEXT NOT NULL,
    regime           TEXT,
    starting_capital REAL,
    ending_capital   REAL,
    gross_pnl        REAL,
    costs            REAL,
    net_pnl          REAL,
    trades           INTEGER,
    wins             INTEGER,
    losses           INTEGER,
    signals          INTEGER,
    rejections       INTEGER,
    kill_switch_fired INTEGER DEFAULT 0,
    notes            TEXT
);

-- §6.1 announcements + LLM verdicts. Deduped by (announcement_id, content_hash)
-- so a filing is never classified twice, and 6.3/6.7 can query "was there a
-- MATERIAL filing on this symbol today?".
CREATE TABLE IF NOT EXISTS announcements (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    announcement_id TEXT NOT NULL,
    content_hash    TEXT NOT NULL,
    ts              TEXT NOT NULL,
    trade_date      TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    headline        TEXT,
    body            TEXT,
    attachment_url  TEXT,
    label           TEXT,             -- MATERIAL_POSITIVE | MATERIAL_NEGATIVE | NOISE
    confidence      REAL,
    llm_reason      TEXT,
    est_revenue_impact_pct REAL,
    classify_latency_sec   REAL,
    traded          INTEGER DEFAULT 0,
    meta            TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_ann_dedupe ON announcements(announcement_id, content_hash);
CREATE INDEX IF NOT EXISTS idx_ann_symbol_date  ON announcements(symbol, trade_date);
CREATE INDEX IF NOT EXISTS idx_ann_label_date   ON announcements(label, trade_date);

-- §6.3 cointegrated pairs, refreshed monthly by scripts/refresh_pairs.py.
CREATE TABLE IF NOT EXISTS pairs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    refreshed_on  TEXT NOT NULL,
    sector        TEXT NOT NULL,
    symbol_a      TEXT NOT NULL,
    symbol_b      TEXT NOT NULL,
    hedge_ratio   REAL NOT NULL,
    pvalue        REAL NOT NULL,
    lookback_days INTEGER NOT NULL,
    active        INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_pairs_active ON pairs(active, refreshed_on);

-- §6.9 nightly FII/DII cash + FII index-futures positioning.
CREATE TABLE IF NOT EXISTS flows (
    trade_date          TEXT PRIMARY KEY,
    fii_cash_cr         REAL,
    dii_cash_cr         REAL,
    fii_idx_fut_long    REAL,
    fii_idx_fut_short   REAL,
    long_ratio          REAL,
    ratio_percentile_3y REAL,
    meta                TEXT
);

-- §6.10 nightly ASM/GSM/ban-list snapshots; the diff drives the alert and the
-- §3 veto. Stored as full snapshots so a missed night is recoverable.
CREATE TABLE IF NOT EXISTS surveillance_snapshots (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date TEXT NOT NULL,
    list_name  TEXT NOT NULL,        -- asm | gsm | fno_ban
    symbol     TEXT NOT NULL,
    stage      TEXT,
    meta       TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_surv_unique ON surveillance_snapshots(trade_date, list_name, symbol);
CREATE INDEX IF NOT EXISTS idx_surv_symbol ON surveillance_snapshots(symbol, trade_date);

-- §4 backtest bookkeeping. `test` window consumption is recorded here so the
-- runner can WARN on a re-run of the untouched window.
CREATE TABLE IF NOT EXISTS backtest_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    engine      TEXT NOT NULL,
    window_name TEXT NOT NULL,       -- tune | validate | test
    start_date  TEXT NOT NULL,
    end_date    TEXT,
    verdict     TEXT,                -- PROMOTED | FAILED | NA
    metrics     TEXT
);
CREATE INDEX IF NOT EXISTS idx_bt_engine ON backtest_runs(engine, window_name);

-- Kill-switch activations (§3 kill(), Telegram /kill).
CREATE TABLE IF NOT EXISTS kill_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    trade_date  TEXT NOT NULL,
    source      TEXT NOT NULL,       -- cli | telegram | risk_kernel
    reason      TEXT,
    orders_cancelled INTEGER DEFAULT 0,
    positions_flattened INTEGER DEFAULT 0
);
"""


def _dumps(value: Any) -> Optional[str]:
    """JSON-encode a meta blob, tolerating non-serialisable values."""
    if value is None:
        return None
    try:
        return json.dumps(value, default=str, separators=(",", ":"))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return json.dumps({"unserialisable": str(value)})


class Journal:
    """SQLite-backed event log. Thread-safe for the writes this system makes."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            settings = get_settings()
            configured = str(settings.get("storage.journal_db", "data/journal.sqlite"))
            db_path = data_path(Path(configured).name)
        self.db_path = Path(db_path)
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # -- lifecycle --------------------------------------------------------

    def _init_schema(self) -> None:
        with self._lock:
            if str(self.db_path) != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES('version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "Journal":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """Serialise writes and never let a journal failure kill a trade."""
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except sqlite3.Error as exc:
                self._conn.rollback()
                log.error("journal write failed: %s", exc, exc_info=True)

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        """Run a read query. Used by engines (§6.3, §6.7) and the digest."""
        with self._lock:
            return list(self._conn.execute(sql, params).fetchall())

    # -- writers ----------------------------------------------------------

    def record_signal(self, signal: Signal) -> None:
        """Persist an emitted Signal (§0.6)."""
        ts = signal.created_at or clock.now_ist()
        with self._write() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO signals
                   (signal_id, ts, trade_date, engine, symbol, side, entry_type, ttl,
                    stop, targets, reference_price, reason, meta)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    signal.signal_id,
                    clock.isoformat(ts),
                    clock.to_ist(ts).date().isoformat(),
                    signal.engine,
                    signal.symbol,
                    signal.side.value,
                    signal.entry_type.value,
                    signal.ttl.value,
                    signal.stop,
                    _dumps(list(signal.targets)),
                    signal.reference_price,
                    signal.reason,
                    _dumps(signal.meta),
                ),
            )

    def record_order(self, order: Order, mode: str, status: str = "SUBMITTED") -> None:
        """Persist a routed order."""
        ts = order.created_at or clock.now_ist()
        with self._write() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO orders
                   (order_id, signal_id, ts, trade_date, engine, symbol, side, quantity,
                    entry_type, product, segment, price, trigger_price, stop, is_entry,
                    mode, status, broker_order_id, reason, meta)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    order.order_id,
                    order.signal_id,
                    clock.isoformat(ts),
                    clock.to_ist(ts).date().isoformat(),
                    order.engine,
                    order.symbol,
                    order.side.value,
                    order.quantity,
                    order.entry_type.value,
                    order.product.value,
                    order.segment.value,
                    order.price,
                    order.trigger_price,
                    order.stop,
                    1 if order.is_entry else 0,
                    mode,
                    status,
                    order.broker_order_id,
                    order.reason,
                    _dumps(order.meta),
                ),
            )

    def update_order_status(
        self, order_id: str, status: str, broker_order_id: str | None = None
    ) -> None:
        with self._write() as conn:
            if broker_order_id is None:
                conn.execute("UPDATE orders SET status=? WHERE order_id=?", (status, order_id))
            else:
                conn.execute(
                    "UPDATE orders SET status=?, broker_order_id=? WHERE order_id=?",
                    (status, broker_order_id, order_id),
                )

    def record_fill(self, fill: Fill) -> None:
        with self._write() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO fills
                   (fill_id, order_id, ts, trade_date, engine, symbol, side, quantity,
                    price, costs, is_paper, broker_order_id, meta)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    fill.fill_id,
                    fill.order_id,
                    clock.isoformat(fill.timestamp),
                    clock.to_ist(fill.timestamp).date().isoformat(),
                    fill.engine,
                    fill.symbol,
                    fill.side.value,
                    fill.quantity,
                    fill.price,
                    fill.costs,
                    1 if fill.is_paper else 0,
                    fill.broker_order_id,
                    _dumps(fill.meta),
                ),
            )

    def record_trade(self, **kwargs: Any) -> None:
        """Persist a completed round trip. Keys mirror the ``trades`` columns."""
        kwargs.setdefault("trade_date", clock.today_ist().isoformat())
        kwargs["meta"] = _dumps(kwargs.get("meta"))
        columns = [
            "trade_id", "engine", "symbol", "side", "quantity", "entry_ts", "entry_price",
            "exit_ts", "exit_price", "gross_pnl", "costs", "net_pnl", "r_multiple",
            "exit_reason", "trade_date", "mode", "meta",
        ]
        values = [kwargs.get(c) for c in columns]
        with self._write() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO trades ({','.join(columns)}) "
                f"VALUES ({','.join('?' * len(columns))})",
                values,
            )

    def record_rejection(
        self,
        decision: RiskDecision,
        *,
        engine: str,
        symbol: str,
        side: str | None = None,
        quantity: int | None = None,
        signal_id: str | None = None,
        order_id: str | None = None,
    ) -> None:
        """§3: *every* rejection is journalled with its reason."""
        ts = clock.now_ist()
        with self._write() as conn:
            conn.execute(
                """INSERT INTO rejections
                   (ts, trade_date, engine, symbol, side, quantity, signal_id, order_id,
                    reason_code, reason, meta)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    clock.isoformat(ts),
                    ts.date().isoformat(),
                    engine,
                    symbol,
                    side,
                    quantity,
                    signal_id,
                    order_id,
                    decision.reason_code,
                    decision.reason,
                    _dumps(decision.meta),
                ),
            )

    def record_error(
        self, source: str, message: str, severity: str = "ERROR", **meta: Any
    ) -> None:
        ts = clock.now_ist()
        with self._write() as conn:
            conn.execute(
                "INSERT INTO errors (ts, trade_date, source, severity, message, meta) "
                "VALUES (?,?,?,?,?,?)",
                (clock.isoformat(ts), ts.date().isoformat(), source, severity, message, _dumps(meta)),
            )

    def record_regime(
        self, regime: str, inputs: dict[str, Any], enabled_engines: Iterable[str]
    ) -> None:
        ts = clock.now_ist()
        with self._write() as conn:
            conn.execute(
                "INSERT INTO regime_log (ts, trade_date, regime, inputs, enabled_engines) "
                "VALUES (?,?,?,?,?)",
                (
                    clock.isoformat(ts),
                    ts.date().isoformat(),
                    regime,
                    _dumps(inputs),
                    _dumps(list(enabled_engines)),
                ),
            )

    def record_announcement(self, **kwargs: Any) -> bool:
        """Insert an announcement. Returns False when it was already seen.

        §6.1 dedupes by announcement id + content hash; the UNIQUE index does
        the work so two poller threads cannot race a duplicate through.
        """
        kwargs.setdefault("ts", clock.isoformat(clock.now_ist()))
        kwargs.setdefault("trade_date", clock.today_ist().isoformat())
        kwargs["meta"] = _dumps(kwargs.get("meta"))
        columns = [
            "announcement_id", "content_hash", "ts", "trade_date", "symbol", "headline",
            "body", "attachment_url", "label", "confidence", "llm_reason",
            "est_revenue_impact_pct", "classify_latency_sec", "traded", "meta",
        ]
        values = [kwargs.get(c) for c in columns]
        with self._lock:
            try:
                self._conn.execute(
                    f"INSERT INTO announcements ({','.join(columns)}) "
                    f"VALUES ({','.join('?' * len(columns))})",
                    values,
                )
                self._conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False  # already seen -- the dedupe working as designed
            except sqlite3.Error as exc:  # pragma: no cover - defensive
                self._conn.rollback()
                log.error("announcement insert failed: %s", exc)
                return False

    def record_kill(
        self, source: str, reason: str, orders_cancelled: int = 0, positions_flattened: int = 0
    ) -> None:
        ts = clock.now_ist()
        with self._write() as conn:
            conn.execute(
                """INSERT INTO kill_events
                   (ts, trade_date, source, reason, orders_cancelled, positions_flattened)
                   VALUES (?,?,?,?,?,?)""",
                (
                    clock.isoformat(ts),
                    ts.date().isoformat(),
                    source,
                    reason,
                    orders_cancelled,
                    positions_flattened,
                ),
            )

    def upsert_daily_summary(self, trade_date: str, **fields: Any) -> None:
        allowed = {
            "mode", "regime", "starting_capital", "ending_capital", "gross_pnl", "costs",
            "net_pnl", "trades", "wins", "losses", "signals", "rejections",
            "kill_switch_fired", "notes",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"Unknown daily_summary fields: {sorted(unknown)}")
        with self._write() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO daily_summary (trade_date, mode) VALUES (?, ?)",
                (trade_date, fields.get("mode", "paper")),
            )
            if fields:
                assignments = ", ".join(f"{k}=?" for k in fields)
                conn.execute(
                    f"UPDATE daily_summary SET {assignments} WHERE trade_date=?",
                    [*fields.values(), trade_date],
                )

    def record_surveillance_snapshot(
        self, trade_date: str, list_name: str, rows: Iterable[dict[str, Any]]
    ) -> int:
        """Store a full nightly list snapshot (§6.10). Returns rows written."""
        written = 0
        with self._write() as conn:
            for row in rows:
                try:
                    conn.execute(
                        """INSERT OR REPLACE INTO surveillance_snapshots
                           (trade_date, list_name, symbol, stage, meta) VALUES (?,?,?,?,?)""",
                        (
                            trade_date,
                            list_name,
                            row["symbol"],
                            row.get("stage"),
                            _dumps({k: v for k, v in row.items() if k not in {"symbol", "stage"}}),
                        ),
                    )
                    written += 1
                except (KeyError, sqlite3.Error) as exc:  # pragma: no cover
                    log.warning("bad surveillance row %r: %s", row, exc)
        return written

    def record_flows(self, trade_date: str, **fields: Any) -> None:
        """§6.9 nightly FII/DII row."""
        fields["meta"] = _dumps(fields.get("meta"))
        columns = [
            "fii_cash_cr", "dii_cash_cr", "fii_idx_fut_long", "fii_idx_fut_short",
            "long_ratio", "ratio_percentile_3y", "meta",
        ]
        values = [fields.get(c) for c in columns]
        with self._write() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO flows (trade_date, {','.join(columns)}) "
                f"VALUES (?, {','.join('?' * len(columns))})",
                [trade_date, *values],
            )

    def record_backtest_run(
        self,
        engine: str,
        window_name: str,
        start_date: str,
        end_date: str | None,
        verdict: str,
        metrics: dict[str, Any],
    ) -> None:
        with self._write() as conn:
            conn.execute(
                """INSERT INTO backtest_runs
                   (ts, engine, window_name, start_date, end_date, verdict, metrics)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    clock.isoformat(clock.now_ist()),
                    engine,
                    window_name,
                    start_date,
                    end_date,
                    verdict,
                    _dumps(metrics),
                ),
            )

    def save_pairs(self, sector_pairs: Iterable[dict[str, Any]], refreshed_on: str) -> int:
        """Replace the active pair set (§6.3 monthly refresh)."""
        count = 0
        with self._write() as conn:
            conn.execute("UPDATE pairs SET active=0")
            for pair in sector_pairs:
                conn.execute(
                    """INSERT INTO pairs
                       (refreshed_on, sector, symbol_a, symbol_b, hedge_ratio, pvalue,
                        lookback_days, active)
                       VALUES (?,?,?,?,?,?,?,1)""",
                    (
                        refreshed_on,
                        pair["sector"],
                        pair["symbol_a"],
                        pair["symbol_b"],
                        pair["hedge_ratio"],
                        pair["pvalue"],
                        pair["lookback_days"],
                    ),
                )
                count += 1
        return count

    # -- readers used by engines and the kernel ---------------------------

    def active_pairs(self) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM pairs WHERE active=1")

    def material_filing_symbols(self, trade_date: str) -> set[str]:
        """Symbols with a MATERIAL filing today -- §6.3 skips those legs."""
        rows = self.query(
            "SELECT DISTINCT symbol FROM announcements "
            "WHERE trade_date=? AND label IN ('MATERIAL_POSITIVE','MATERIAL_NEGATIVE')",
            (trade_date,),
        )
        return {r["symbol"] for r in rows}

    def has_negative_filing(self, symbol: str, trade_date: str) -> bool:
        """§6.7 mandatory cross-check before any panic-reversion entry."""
        rows = self.query(
            "SELECT 1 FROM announcements WHERE symbol=? AND trade_date=? "
            "AND label='MATERIAL_NEGATIVE' LIMIT 1",
            (symbol, trade_date),
        )
        return bool(rows)

    def orders_today(self, trade_date: str, engine: str | None = None) -> list[sqlite3.Row]:
        if engine:
            return self.query(
                "SELECT * FROM orders WHERE trade_date=? AND engine=?", (trade_date, engine)
            )
        return self.query("SELECT * FROM orders WHERE trade_date=?", (trade_date,))

    def new_entry_count_today(self, trade_date: str, engine: str) -> int:
        """§3 ``max_new_trades_per_day_per_engine`` counter."""
        rows = self.query(
            "SELECT COUNT(*) AS n FROM orders "
            "WHERE trade_date=? AND engine=? AND is_entry=1 AND status != 'FAILED'",
            (trade_date, engine),
        )
        return int(rows[0]["n"]) if rows else 0

    def realised_pnl_between(self, start_date: str, end_date: str) -> float:
        """Net realised P&L over an inclusive date range -- §3 loss limits."""
        rows = self.query(
            "SELECT COALESCE(SUM(net_pnl), 0.0) AS pnl FROM trades "
            "WHERE trade_date >= ? AND trade_date <= ?",
            (start_date, end_date),
        )
        return float(rows[0]["pnl"]) if rows else 0.0

    def surveillance_symbols(self, trade_date: str, list_names: Sequence[str]) -> set[str]:
        placeholders = ",".join("?" * len(list_names))
        rows = self.query(
            f"SELECT DISTINCT symbol FROM surveillance_snapshots "
            f"WHERE trade_date=? AND list_name IN ({placeholders})",
            (trade_date, *list_names),
        )
        return {r["symbol"] for r in rows}

    def latest_surveillance_date(self, list_name: str) -> Optional[str]:
        rows = self.query(
            "SELECT MAX(trade_date) AS d FROM surveillance_snapshots WHERE list_name=?",
            (list_name,),
        )
        return rows[0]["d"] if rows and rows[0]["d"] else None

    def test_window_runs(self, engine: str) -> int:
        """How many times the untouched §4 test window has been consumed."""
        rows = self.query(
            "SELECT COUNT(*) AS n FROM backtest_runs WHERE engine=? AND window_name='test'",
            (engine,),
        )
        return int(rows[0]["n"]) if rows else 0

    def counts_for_date(self, trade_date: str) -> dict[str, int]:
        """Row counts used by the 15:45 digest."""
        out: dict[str, int] = {}
        for table in ("signals", "orders", "fills", "trades", "rejections", "errors"):
            rows = self.query(f"SELECT COUNT(*) AS n FROM {table} WHERE trade_date=?", (trade_date,))
            out[table] = int(rows[0]["n"]) if rows else 0
        return out


_default: Optional[Journal] = None


def get_journal() -> Journal:
    """Process-wide journal singleton."""
    global _default
    if _default is None:
        _default = Journal()
    return _default


def set_journal(journal: Optional[Journal]) -> None:
    """Swap the singleton. Tests inject an in-memory journal through here."""
    global _default
    _default = journal
