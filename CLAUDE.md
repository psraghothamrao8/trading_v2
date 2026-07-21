# CLAUDE.md — Indian Markets Multi-Engine Trading System (v2)

Distilled operating directives for any agent or human working in this repo.
The authoritative specification is [MASTER_BUILD_PROMPT.md](MASTER_BUILD_PROMPT.md).
Section references below (`§3`, `§6.1`, …) point into that document.

---

## PRIME DIRECTIVES (never violate)

1. **Paper mode is the default.** `EXECUTION_MODE` is `paper` | `live`.
   `live` requires BOTH `execution.mode: live` in `config/settings.yaml` AND an
   interactive `y/N` confirmation at startup. Paper mode simulates fills locally
   and never sends a broker order.
2. **The strategy owner owns the trading logic.** Implement every rule EXACTLY as
   specified. Never add, remove, "improve", or re-tune a trading rule. If a spec
   is ambiguous or two rules conflict, STOP and ask. Asking is success; assuming
   is failure.
3. **No engine calls the broker directly.** Every order passes through
   `core/risk.py::check()`. This is architectural law, enforced by
   `tests/test_architecture.py`.
4. **Secrets live in `.env` only.** Kite key/secret/access token, Anthropic key,
   Telegram token. `.env.example` is committed; `.env` never is.
5. **Every module ships pytest tests** on mocked data. `core/risk.py` and
   `core/costs.py` carry the heaviest coverage.
6. **Everything is journaled** — signals, orders, fills, rejections, errors — to
   SQLite, with timestamps in `Asia/Kolkata`.
7. **One phase at a time** (§5). After each phase: run the full test suite,
   summarize what was built plus test results, commit.
8. **No placeholder code.** No `TODO: implement later`, no stub that silently
   returns fake success. If something cannot be built yet, raise
   `NotImplementedError` loudly and say so.

---

## CONVENTIONS

### Language and style
- Python 3.11+. Type hints everywhere.
- Every module, class and public function carries a docstring, and any function
  implementing a spec rule cites its section: `"""Implements §6.1."""`
- Logging via the stdlib `logging` module, configured once in
  `core/logging_config.py` with a rotating file handler. Never `print()` from
  library code — `print` belongs to CLI entry points only.
- Money is `float` INR at the boundary but always rounded through
  `core/costs.py` helpers before journalling. Quantities are `int`.
- Timezone: **all** datetimes are timezone-aware `Asia/Kolkata`. Use
  `core.clock.now_ist()` — never `datetime.now()` bare.

### Numbers and configuration
- **No magic numbers in code.** Every threshold, rate, cap and percentage lives
  in `config/settings.yaml` and is read through `core.config.get_settings()`.
  If you find yourself typing a number into a `.py` file, it belongs in YAML.
- Rate tables that the government or exchange can change (STT, GST, stamp duty,
  transaction charges, lot sizes, expiry weekday) carry an `as_of` date in YAML.

### Testing
- `pytest`, mocked data only — tests never hit the network or the broker.
- Network clients are injected, so tests substitute a fake transport.
- Run: `python -m pytest -q`.

### Git
- Small commits, one logical unit each, descriptive messages.
- Never commit `.env`, `data/`, `*.sqlite`, `*.parquet`, or logs.

---

## ARCHITECTURE IN ONE PAGE

```
                    ┌──────────────────────────────┐
                    │   live/orchestrator.py  §7   │
                    │  regime router + scheduler   │
                    └──────────┬───────────────────┘
             Signals           │            enable/disable
        ┌───────────────┬──────┴───────┬─────────────────┐
        │               │              │                 │
   engines/*.py    core/nse.py    core/datafeed.py   core/llm.py
     (§6.1-6.11)     (§8.2)        candles/ticks      classify()
        │
        │ list[Signal]                  ┌───────────────────────┐
        └──────────────────────────────►│   core/risk.py  §3    │
                                        │  check() ALLOW/REJECT │
                                        └──────────┬────────────┘
                                                   │ ALLOW only
                                        ┌──────────▼────────────┐
                                        │   core/broker.py      │
                                        │  paper sim │ Kite     │
                                        └──────────┬────────────┘
                                                   │
                            core/journal.py ◄──────┴──────► live/alerts.py
                            (SQLite, all events)            (Telegram, /kill)
```

**The one-way rule:** engines produce `Signal`s and nothing else. The
orchestrator converts a `Signal` into an `Order`, hands it to `risk.check()`,
and only an `ALLOW` verdict reaches `broker.py`. An engine that imports
`core.broker` is a bug.

---

## THE RISK KERNEL (§3) — what `check()` enforces

Position sizing, daily/weekly loss limits, per-engine trade counts and capital
caps, plus these India-specific vetoes:

| Veto | Rule |
|---|---|
| F&O ban list | MWPL ≥ 95% ⇒ reject derivatives orders |
| ASM/GSM surveillance | reject ALL new entries, every engine |
| Circuit band | within 1% of band ⇒ reject new entries |
| Event days | `events.yaml` blocked days ⇒ no new entries, no overnight holds |
| Time | no new intraday entries after 14:45; MIS force-flat by 15:10 |
| Physical settlement | short stock option ITM or within 2% of spot must be closed/rolled by expiry−2 sessions unless `allow_delivery: true` |
| STT trap | never carry long ITM options into expiry close — exit by 15:00 on expiry day |

Every rejection is journalled with its reason. Every veto has an explicit test.

---

## INDIA-SPECIFIC FACTS THE CODE MUST RESPECT (§8)

- **Kite access tokens expire daily (~07:30 IST).** `scripts/morning_auth.py`
  prints a login URL; the human completes login + TOTP and pastes the
  `request_token`. Credential entry is never automated.
- **NSE rejects naive HTTP clients.** One shared session per run, homepage
  warm-up for cookies, realistic headers, ≥2s between calls per endpoint
  (except the 30s announcements poll), exponential backoff with re-warm on
  401/403, and a loud Telegram failure after 5 consecutive errors. Endpoint
  URLs live in config because NSE changes them.
- **Session structure:** pre-open 09:00–09:08 (entry to 09:07:59, matching
  09:08–09:12), continuous 09:15–15:30, closing auction 15:40–16:00.
- **Expiry weekday and lot sizes are NOT hardcoded** — SEBI and the exchanges
  have changed both repeatedly. They live in `config/settings.yaml` with
  `as_of` dates.
- **Price bands:** 2/5/10/20% per stock; index circuit breakers at ±10/15/20%.
- **Settlement:** equity T+1. Stock derivatives settle *physically*. Long ITM
  options held into expiry incur the higher delivery STT.
- **Product types:** intraday equity/short = `MIS`; overnight equity = `CNC`;
  overnight F&O = `NRML`. Shorting equity overnight is impossible in cash — any
  such Signal is rejected.

---

## COSTS ARE NOT OPTIONAL (§4)

Every backtest fill and every paper fill runs through `core/costs.py`. Gross
backtests are lies. The model covers brokerage, STT/CTT, exchange transaction
charges, SEBI fees, stamp duty, GST and slippage (0.03%/side liquid equity,
0.05%/side options, 1 tick minimum), and is validated in tests against
Zerodha's public brokerage calculator for three sample trades.

## PROMOTION GATES (§4)

Per engine, on the validation window: profit factor ≥ 1.3; trades ≥ 150
(≥ 40 for event/quarterly engines); max drawdown ≤ 12% of engine capital;
net-of-costs expectancy > 0.

**A FAILED verdict is valuable output, not a bug.** Never soften a gate to make
an engine pass. A failed engine stays alert-only.

---

## WORKING PROTOCOL

1. Start each phase with a short PLAN: files to create or change, interfaces,
   test list.
2. Small commits, descriptive messages.
3. After coding: run pytest, show the summary, list skipped/failing tests with
   reasons. **Never claim green without running.**
4. When the spec conflicts with reality (a field is missing, an endpoint is
   dead): STOP, show the evidence, propose ≤3 options. Do not improvise data.
5. Websockets auto-reconnect with backoff, resubscribe, and journal the gap.
   Scheduler jobs are idempotent so a mid-day restart is safe.
6. Every embedded LLM prompt lives in `config/prompts/` as an editable text
   file with a strict JSON schema and two few-shot examples.
7. Backtest verdicts are sacred.
