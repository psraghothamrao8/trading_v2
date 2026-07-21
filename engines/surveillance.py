"""surveillance.py — ASM/GSM list-change engine. Implements §6.10.

WHY (from the spec): stocks entering ASM/GSM surveillance get 100% margin and
position caps -- forced selling follows, then relief rallies on exit from the
lists. The lists are published nightly; almost nobody diffs them systematically.

**ALERT-ONLY, structurally.** Shorting these names is operationally restricted,
so the value is the veto plus the heads-up, not a trade. This engine inherits
:class:`~engines.base.AlertOnlyEngine`, whose ``on_schedule`` always returns
``[]``, and the §3 kernel rejects its orders anyway. Two independent barriers,
because "alert-only" is a safety property and not a preference.
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from engines.base import AlertOnlyEngine, Context

log = logging.getLogger(__name__)

LIST_NAMES = ("asm", "gsm", "fno_ban")


@dataclass
class ListDiff:
    """Entries and exits for one surveillance list."""

    list_name: str
    added: list[dict[str, Any]] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    previous_date: Optional[str] = None
    current_date: Optional[str] = None

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed)


class SurveillanceEngine(AlertOnlyEngine):
    """§6.10. Nightly diff -> Telegram digest + the §3 kernel veto set."""

    name = "surveillance"

    def __init__(self, alerts: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._alerts = alerts

    @property
    def alerts(self) -> Any:
        if self._alerts is None:
            from live.alerts import get_alerts

            self._alerts = get_alerts()
        return self._alerts

    def universe(self) -> list[str]:
        """This engine watches the lists, not a fixed symbol set."""
        return []

    # -- the diff ---------------------------------------------------------

    def diff(self, ctx: Context, trade_date: Optional[str] = None) -> dict[str, ListDiff]:
        """Diff tonight's snapshots against the previous ones (§6.10)."""
        journal = ctx.the_journal()
        today = trade_date or ctx.today.isoformat()
        out: dict[str, ListDiff] = {}

        for list_name in LIST_NAMES:
            dates = journal.query(
                "SELECT DISTINCT trade_date FROM surveillance_snapshots "
                "WHERE list_name=? ORDER BY trade_date DESC LIMIT 2",
                (list_name,),
            )
            if not dates:
                out[list_name] = ListDiff(list_name=list_name)
                continue

            current_date = dates[0]["trade_date"]
            previous_date = dates[1]["trade_date"] if len(dates) > 1 else None

            current_rows = journal.query(
                "SELECT symbol, stage FROM surveillance_snapshots "
                "WHERE list_name=? AND trade_date=?",
                (list_name, current_date),
            )
            current = {row["symbol"]: row["stage"] for row in current_rows}
            previous = (
                journal.surveillance_symbols(previous_date, [list_name])
                if previous_date else set()
            )

            out[list_name] = ListDiff(
                list_name=list_name,
                added=[
                    {"symbol": symbol, "stage": stage}
                    for symbol, stage in sorted(current.items())
                    if symbol not in previous
                ],
                removed=sorted(previous - set(current)),
                previous_date=previous_date,
                current_date=current_date,
            )
        return out

    # -- the veto set -----------------------------------------------------

    def veto_symbols(self, ctx: Context) -> set[str]:
        """Symbols the §3 kernel must refuse new entries on.

        The system-wide effect the spec describes: additions feed the kernel
        veto so **no engine** touches them.
        """
        journal = ctx.the_journal()
        symbols: set[str] = set()
        for list_name in ("asm", "gsm"):
            latest = journal.latest_surveillance_date(list_name)
            if latest:
                symbols |= journal.surveillance_symbols(latest, [list_name])
        return symbols

    def ban_list(self, ctx: Context) -> set[str]:
        """The F&O ban list, for the §3 derivatives veto."""
        journal = ctx.the_journal()
        latest = journal.latest_surveillance_date("fno_ban")
        return journal.surveillance_symbols(latest, ["fno_ban"]) if latest else set()

    def staleness_days(self, ctx: Context) -> Optional[int]:
        """How old the newest snapshot is. Stale lists are a silent-degrade risk."""
        journal = ctx.the_journal()
        newest: Optional[_dt.date] = None
        for list_name in LIST_NAMES:
            latest = journal.latest_surveillance_date(list_name)
            if latest:
                try:
                    day = _dt.date.fromisoformat(str(latest)[:10])
                except ValueError:
                    continue
                newest = day if newest is None else max(newest, day)
        if newest is None:
            return None
        return (ctx.today - newest).days

    # -- alerting ---------------------------------------------------------

    def alerts_for(self, ctx: Context) -> list[str]:
        """§6.10: a Telegram digest of entries and exits with stage."""
        lines: list[str] = []
        for list_name, change in self.diff(ctx).items():
            if not change.changed:
                continue
            label = list_name.upper().replace("FNO_BAN", "F&O BAN")
            for entry in change.added:
                stage = f" [{entry['stage']}]" if entry.get("stage") else ""
                lines.append(f"IN  {label}: {entry['symbol']}{stage}")
            for symbol in change.removed:
                lines.append(f"OUT {label}: {symbol}")
        return lines

    def send_digest(self, ctx: Context) -> bool:
        """Send the nightly §6.10 digest. Returns False when nothing changed."""
        lines = self.alerts_for(ctx)
        stale = self.staleness_days(ctx)

        if stale is not None and stale > 3:
            # §8.2: never silently degrade. A stale list means the §3 veto is
            # operating on old data, which is worse than knowing it is absent.
            self.alerts.error(
                "surveillance",
                f"Surveillance lists are {stale} days old. The §3 ASM/GSM and "
                f"F&O-ban vetoes are running on stale data. Check the nightly job.",
                severity="WARNING",
            )

        if not lines:
            return False

        self.alerts.send(
            "🛡️ <b>SURVEILLANCE DIFF</b>\n<code>"
            + "\n".join(lines[:40])
            + "</code>\n<i>Additions feed the §3 kernel veto — no engine may open a "
              "new position in them. Exits are a watchlist note only (§6.10).</i>"
        )
        return True
