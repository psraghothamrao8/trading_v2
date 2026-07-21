# MASTER BUILD INSTRUCTION — Indian Markets Multi-Engine Trading System (v2)

You are Claude Code (Sonnet), acting as the sole engineer for my personal
algorithmic trading system for Indian markets (NSE), built on Zerodha Kite
Connect. I am the strategy owner; you are the implementer. Build the ENTIRE
system described below, phase by phase, in this repository.

**How to read this document:** Section 0 is law. Section 1–2 is stack and
layout. Section 3 is the risk kernel every order must pass through. Section 4
is the backtester and promotion gates. Section 5 is your build order. Section 6
specifies every engine — each has a WHY (the market logic, so you understand
intent), RULES (implement exactly), and NOTES (edge cases). Section 7 is the
orchestrator. Section 8 is India-specific market rules you must encode.
Section 9 is your working protocol. Section 10 is deliverables.

---

## 0. PRIME DIRECTIVES (never violate)

1. **Paper mode is default.** Global `EXECUTION_MODE` (`paper` | `live`).
   `live` requires BOTH the config flag AND an interactive y/N confirmation at
   startup. Paper mode simulates fills locally and never sends broker orders.
2. **You write code; I own the trading logic.** Implement every rule EXACTLY.
   Never add, remove, "improve," or re-tune any trading rule. If a spec is
   ambiguous or two rules conflict, STOP and ask me. Asking is success;
   assuming is failure.
3. **No engine calls the broker directly.** Every order passes through
   `core/risk.py::check()`. Architectural law.
4. **Secrets in `.env` only** (Kite key/secret/access token, Anthropic key,
   Telegram token). Ship `.env.example`. Never commit secrets.
5. **Every module ships pytest tests** on mocked data. `core/risk.py` and
   `core/costs.py` get the heaviest coverage.
6. **Everything is journaled** — signals, orders, fills, rejections, errors —
   to SQLite, timestamps in Asia/Kolkata.
7. **One phase at a time** (Section 5). After each phase: run the full test
   suite, summarize what was built + test results, wait for my go-ahead,
   commit to git.
8. **No placeholder code.** No `TODO: implement later`, no stubbed logic
   silently returning fake success. If something can't be built yet, raise
   `NotImplementedError` loudly and tell me.

## 1. TECH STACK

Python 3.11+, `kiteconnect`, `pandas`, `numpy`, `pyarrow`, SQLite,
`APScheduler`, `pytest`, `python-telegram-bot`, `httpx`, `statsmodels`,
`anthropic`. Custom event-driven backtester (Section 4) — no heavy framework.
Type hints everywhere; docstrings cite the spec section they implement
(e.g., "Implements §6.1"). Logging via `logging` with rotating file handler.

`core/llm.py`: single entry `classify(task: str, payload: dict,
schema: dict) -> dict`. Model name from config. Force strict-JSON output,
validate against schema, retry ×3 with backoff, and cache by content hash in
SQLite so no filing/transcript is classified twice.

## 2. REPO STRUCTURE

```
├── CLAUDE.md                # distilled directives + conventions (write first)
├── .env.example
├── config/
│   ├── settings.yaml        # capital, risk, engine caps, regime map
│   ├── universe.yaml        # NIFTY50/100/200/500 lists, approved wheel stocks, pair sectors
│   └── events.yaml          # blocked event dates I maintain (RBI MPC, Budget, elections, US Fed)
├── core/
│   ├── broker.py            # Kite wrapper: auth, orders, positions, margins, option chain
│   ├── datafeed.py          # historical candles (chunked for Kite limits) + websocket ticks
│   ├── risk.py              # THE KERNEL — §3
│   ├── costs.py             # full Indian cost model — §4
│   ├── calendar.py          # NSE holidays, expiry schedule, Muhurat, event blocks
│   ├── nse.py               # NSE website client: cookie warm-up, announcements, pre-open,
│   │                        #   surveillance lists, ban list, bulk/block deals, FII/DII stats
│   ├── journal.py           # SQLite: signals, orders, trades, rejections, daily_summary
│   └── llm.py
├── engines/
│   ├── base.py              # Strategy ABC — §6.0
│   ├── filings.py, sympathy.py, pairs.py, overnight.py, pead.py,
│   ├── panic_reversion.py, wheel.py, special_situations.py,
│   ├── preopen.py, surveillance.py, flows.py
├── live/
│   ├── orchestrator.py      # regime router + scheduler — §7
│   └── alerts.py            # Telegram real-time + nightly digest + /kill command
├── backtest/
│   ├── runner.py
│   └── metrics.py
├── scripts/                 # download_history.py, refresh_pairs.py, morning_auth.py
└── tests/
```

## 3. RISK KERNEL — `core/risk.py`

All numbers live in `settings.yaml`, never hardcoded. Defaults:

- `capital: 800000` (INR)
- `risk_per_trade_pct: 0.5` → qty = floor((capital × 0.005) / |entry − stop|)
- `daily_loss_limit_pct: 1.5` → breach ⇒ flatten all intraday, block new
  orders until next session, Telegram alert
- `weekly_loss_limit_pct: 3.0` → breach ⇒ block new orders until Monday
- `max_new_trades_per_day_per_engine: 3`; `max_concurrent_positions_total: 8`
- `per_engine_capital_cap_pct`: filings 25, sympathy 10, pairs 25,
  overnight 25, pead 25, panic_reversion 15, wheel 40 (margin), preopen 10,
  flows 15; surveillance & special_situations alert-only
- `kill()`: cancel all open orders + flatten everything; callable from CLI and
  the Telegram `/kill` command.

`check(order) -> ALLOW | REJECT(reason)` runs before EVERY order (paper and
live) and enforces, beyond the limits above, these **India-specific vetoes**:

- Symbol on the F&O **ban list** (MWPL ≥ 95%) → reject derivatives orders.
- Symbol in **ASM/GSM surveillance** stages → reject ALL new entries, any
  engine (these carry 100% margin and trap intraday traders).
- Symbol within 1% of its **circuit band** → reject new entries (you may not
  be able to exit a locked stock).
- New entries on `events.yaml` blocked days; no overnight holds into them.
- Time vetoes: no new intraday entries after 14:45; MIS positions force-flat
  by 15:10 (brokers auto-square ~15:20 with penalties — never rely on that).
- **Physical-settlement guard**: any short stock-option position that is ITM
  or within 2% of spot must be closed/rolled by expiry-minus-2 sessions
  unless I set `allow_delivery: true` for that trade (stock F&O settle
  physically; surprise delivery obligations can exceed my capital).
- **STT trap guard**: never carry long ITM options into expiry close — exit
  by 15:00 on expiry day.

Every rejection journaled with reason. Test all vetoes explicitly.

## 4. BACKTESTER, COSTS, PROMOTION GATES

**Runner:** event-driven over daily + 5-minute Parquet. Walk-forward: tune on
2019–2022, validate on 2023–2024, final untouched test 2025→present (test
window consumed exactly once; print a warning if re-run).

**Costs — `core/costs.py`:** implement Zerodha's full charge structure as a
config-driven table: flat ₹20/executed order (₹0 equity delivery), STT/CTT by
segment, exchange transaction charges, SEBI fees, stamp duty, GST — plus
slippage: 0.03%/side liquid equity, 0.05%/side options, 1 tick minimum.
Because these rates change, put every rate in `settings.yaml` with an
`as_of` date, and validate the model against Zerodha's public brokerage
calculator for 3 sample trades (document the comparison in tests). Every
backtest and paper fill runs through this model — gross backtests are lies.

**Promotion gates (per engine, on validation window):** profit factor ≥ 1.3;
trades ≥ 150 (≥ 40 for event/quarterly engines); max drawdown ≤ 12% of engine
capital; and net-of-costs expectancy > 0. Print PROMOTED / FAILED with the
numbers. An engine that fails stays alert-only. Do not soften a gate to make
an engine pass — a FAILED verdict is valuable output, not a bug.

**Metrics:** trades, win rate, avg win/loss, expectancy, profit factor, max
DD, monthly returns table, and equity curve CSV.

## 5. BUILD PHASES (strict order; acceptance criteria included)

1. **Skeleton** — repo, CLAUDE.md, configs, `.env.example`, broker auth
   (§8.1 morning token flow), journal schema, Telegram alerts incl. `/kill`,
   calendar with NSE holidays + expiry schedule from config.
   ✓ `python -m live.orchestrator --status` prints account, config, regime=NA.
2. **NSE client + data** — `core/nse.py` (§8.2 scraping etiquette),
   `download_history.py`: NIFTY-500 daily since 2015, 5-min as far back as
   Kite's historical API allows (respect per-request range + rate limits —
   chunk and sleep), instruments cache, announcements poller, pre-open
   snapshot, surveillance/ban lists, bulk/block deals, FII/DII stats, India
   VIX. ✓ Data in Parquet/SQLite; each fetcher survives a live dry run.
3. **Costs + backtester** — §4. ✓ Buy-and-hold NIFTY sanity backtest produces
   believable net numbers; cost model matches brokerage calculator samples.
4. **Engines** in order: filings → pairs → overnight → preopen → pead →
   panic_reversion → wheel → sympathy → flows → surveillance →
   special_situations. Each engine: implement → unit tests → backtest →
   verdict → my go-ahead → next.
5. **Orchestrator** — regime router, scheduler, engine enablement, digest.
6. **Paper runtime** — full session loop on live data. ✓ One complete
   simulated session end-to-end with journal + 15:45 digest.

## 6. ENGINE SPECIFICATIONS

### 6.0 Common interface — `engines/base.py`

```python
class Engine(ABC):
    name: str
    def universe(self) -> list[str]: ...
    def on_schedule(self, ctx: Context) -> list[Signal]: ...   # timed engines
    def on_tick(self, tick: Tick, ctx: Context) -> list[Signal]: ...  # streaming engines
    def on_fill(self, fill: Fill, ctx: Context) -> None: ...
    def manage(self, ctx: Context) -> list[Signal]: ...        # stops/trails/exits
```

`Signal` = dataclass(symbol, side, entry_type, stop, targets, ttl, reason,
engine, meta). The orchestrator converts Signals to orders via risk kernel.
Engines never place orders themselves.

### 6.1 filings.py — Disclosure-latency engine
**WHY:** Every price-moving fact must legally hit the exchange announcements
feed first. Most humans read it hours later; HFT trades order flow, not text.
Reacting to *clearly material* filings within a minute captures the drift
before the crowd. This engine is also the shared news-sensor for 6.2, 6.6.

**RULES:**
- Poll NSE corporate announcements every 30s, 08:00–15:35 IST; dedupe by
  announcement ID + content hash; download attached PDF when present and
  extract text (first 4 pages max) for the classifier.
- LLM classify → `{label: MATERIAL_POSITIVE|MATERIAL_NEGATIVE|NOISE,
  confidence: 0-1, reason: str, est_revenue_impact_pct: float|null}`.
  Embed this materiality guidance in the prompt: POSITIVE = order wins
  plausibly >5% of annual revenue, promoter stake purchases, key regulatory
  approvals (USFDA etc.), rating upgrades, buyback/open-offer announcements.
  NEGATIVE = defaults, fraud/raids/investigations, auditor resignation,
  rating downgrades, key plant shutdown, pledge invocation. NOISE = board
  meetings, ESOPs, routine compliance, investor-call schedules.
- Every MATERIAL item ⇒ instant Telegram alert (label, confidence, reason).
- Auto-trade (paper until promoted): stock ∈ NIFTY-200, confidence ≥ 0.8,
  time 09:20–14:30, stock not gapped >5% already since the filing.
  POSITIVE ⇒ long at market; stop = entry − 1.5×ATR(5m,14), tightened to VWAP
  if VWAP is nearer. NEGATIVE ⇒ MIS short, mirrored stop above VWAP.
  Book 50% at +1R; trail rest by VWAP; force-flat 15:10. POSITIVE trades may
  carry `swing_hold` (config, default false) → hold ≤5 sessions, stop at entry.

**NOTES:** If classification latency >20s, alert anyway and skip auto-trade.
Never trade a filing older than 10 minutes (edge is gone).

### 6.2 sympathy.py — Second-order reaction
**WHY:** A material event moves the subject stock in seconds, but its listed
suppliers/customers/peers reprice over hours. The LLM knows business
relationships no price feed contains — this is the rare edge where my AI
stack beats faster money.

**RULES:** Trigger only from a 6.1 MATERIAL event. LLM returns ≤3 listed NSE
names with `{symbol, relation, direction, confidence}`. Trade one name only
if: liquid (NIFTY-500 + avg daily turnover > ₹25 cr), confidence ≥ 0.7, and
its move since the filing < 1/3 of the primary stock's move. Mechanics
mirror 6.1. Max 1 sympathy trade per filing, 2 per day.

### 6.3 pairs.py — Intraday pair mean reversion
**WHY:** Same-sector large caps are chained together by flows; intraday
divergences snap back. Market-neutral, so it earns on days the directional
engines sit out — its job in the ensemble is smoothing.

**RULES:** Universe from `universe.yaml` sector groups (NIFTY-50 only).
Monthly `refresh_pairs.py`: Engle-Granger on 1y daily closes, keep p<0.05,
store hedge ratio. Live: 5-min spread z-score vs rolling 20-day mean/σ.
Enter |z|≥2 (short rich / long cheap, rupee-neutral); exit |z|≤0.25; stop
|z|≥3.5; force-flat EOD; max 2 concurrent pairs; skip if either leg has a
MATERIAL filing today (ask 6.1's DB — a real event breaks mean reversion).

### 6.4 overnight.py — Overnight drift
**WHY:** A large share of long-run index return accrues overnight — news
lands while the market is shut and gaps carry it. Harvest the drift with
filters that skip the toxic nights.

**RULES:** 15:20 — long 1 lot NIFTY futures (paper: index level) IF index
close > 200-DMA AND next session not in `events.yaml` AND today ≥ −1.5%.
Exit 09:16 next session at market. If GIFT Nifty (§8.3) shows ≤ −1% before
open, exit in pre-open instead of waiting. Size so a 2% adverse gap ≈ daily
loss limit; this should normally be the book's only overnight index position.

### 6.5 preopen.py — Pre-open auction imbalance  *(new)*
**WHY:** NSE's 09:00–09:08 call auction publishes indicative price and
matched/unmatched order quantities. Persistent one-sided imbalance in the
auction routinely continues into the first minutes of trade — public data
almost no retail trader reads programmatically.

**RULES:** Snapshot the NSE pre-open feed for NIFTY-50 names at 09:06:30 and
09:07:45. Compute indicative gap vs prev close and imbalance ratio =
unmatched buy qty / unmatched sell qty. Candidates: |gap| ≥ 1% AND
(ratio ≥ 3 for longs, ≤ 1/3 for shorts) AND direction agrees with any 6.1
overnight filing (if a filing exists and disagrees, veto). At 09:15–09:20:
enter on continuation only (price trades beyond indicative price in gap
direction); stop = indicative price ∓ 0.5×ATR(5m); book 50% at +1R; flat by
10:30 — this edge is minutes-scale. Max 2 trades/day. Alert-first for the
first two paper weeks.

### 6.6 pead.py — Post-earnings drift + concall tone
**WHY:** Institutions can't buy a surprise in one day; big beats drift for
weeks (documented anomaly, persistent in India). The LLM reads what the
price gap can't: whether management's guidance backs the number.

**RULES:** Results season: detect gap ≥ +3% with volume ≥ 2× 20-day avg in
NIFTY-500 on result day. Fetch results PDF/concall transcript; LLM scores
`tone` 0–10 (guidance confidence, demand commentary, margin trajectory);
require tone ≥ 7. Entry: first pullback to 20-EMA (daily) within 5 sessions.
Stop: below pre-earnings close. Exit: 3×ATR(14,d) trail or 30-day time exit.
Max 4 concurrent.

### 6.7 panic_reversion.py
**WHY:** India's market is structurally dip-bought (SIP flows arrive
monthly regardless of mood). Sentiment crashes without a fundamental filing
snap back with high frequency — but only sentiment crashes, hence the
mandatory no-negative-filing cross-check.

**RULES:** Trigger A: NIFTY day ≤ −3%. Trigger B: NIFTY-100 stock ≤ −6% with
NO negative filing that day (query 6.1 DB — mandatory). Entry next session
09:30–10:30 on reclaim of first-15-min high. Stop below session low. Book
50% at +1R, trail rest by 10-EMA(15m). Max 2 concurrent. Enabled only in
PANIC regime. Never enter a stock sitting at lower circuit (§3 veto covers
this — do not remove).

### 6.8 wheel.py — Cash-secured puts / covered calls
**WHY:** Selling puts on stocks I'd happily own converts waiting into
premium. The IV gate ensures I only sell insurance when it's expensive.

**RULES:** Universe: approved list in `universe.yaml` (2–3 liquid large-cap
F&O names). Run only when India VIX 1-year percentile ≥ 50. Sell 1 lot
cash-secured put ~0.25 delta, 30–45 DTE. Manage: buy back at 50% of credit;
roll down-and-out if spot breaches strike with >10 DTE; else accept
assignment ONLY with my Telegram confirmation (physical settlement, §3
guard applies). Post-assignment: 0.25-delta covered calls monthly until
called away. Alert-first: every wheel order proposed via Telegram and
requires my confirmation even in paper.

### 6.9 flows.py — FII positioning tilt  *(new)*
**WHY:** NSE publishes FII/DII cash flows and FII index-futures positioning
daily. Extremes are contrarian gold: when FII index-futures long share drops
to historic lows, forward index returns have skewed sharply positive — the
sellers are exhausted. Pros watch this; almost no retail codes it.

**RULES:** Nightly: pull FII/DII cash figures + FII index futures long/short
ratio; store; compute 3-year percentile. Signals: ratio percentile ≤ 10 ⇒
swing long NIFTY (futures or ETF proxy) next open, stop −2%, exit at
percentile > 40 or +4% or 20 sessions. Percentile ≥ 90 ⇒ no shorting —
instead veto new overnight longs (6.4) and halve premium-selling size.
Also feed the daily reading to the regime router as context. Max 1 position.

### 6.10 surveillance.py — ASM/GSM list-change engine  *(new)*
**WHY:** Stocks entering ASM/GSM surveillance get 100% margin and position
caps — forced selling follows, then relief rallies on exit from the lists.
The lists are published nightly; almost nobody diffs them systematically.

**RULES:** Nightly diff of ASM/GSM stage lists and F&O ban list. ALERT-ONLY:
Telegram digest of entries/exits with stage. System-wide effect: additions
feed the §3 kernel veto (no engine touches them); exits create a watchlist
note for me. No automated trading in this engine — shorting these names is
operationally restricted; the value is the veto + the heads-up.

### 6.11 special_situations.py — Alert-only
**RULES:** Daily scan of announcements for tender buybacks, open offers,
delistings, index-inclusion news. Alert with: offer price, market price,
retail entitlement where determinable, acceptance-ratio estimate, indicative
expected value. No automated execution.

## 7. ORCHESTRATOR + REGIME ROUTER — `live/orchestrator.py`

**WHY (understand this to build it right):** No engine wins every day; each
wins on its *kind* of day. The router's job is deciding which engines are
allowed to play today, so each engine avoids the days it loses on. This
meta-layer is worth more than any single signal.

- 10:00 IST classification:
  - `PANIC`: India VIX +8% intraday OR index ≤ −1.5% intraday.
  - `TREND`: |gap| ≥ 0.4% AND index one side of VWAP ≥ 80% of 09:15–10:00
    AND advance/decline ≥ 2:1 (or ≤ 1:2 for down-trend).
  - else `CHOP`.
- Pre-open context inputs (log with the day's record): GIFT Nifty gap at
  08:45 (§8.3), prior US session S&P move, FII positioning percentile (6.9).
- Enablement map (config-driven, editable): TREND → filings, sympathy,
  preopen, overnight; CHOP → pairs, wheel management; PANIC →
  panic_reversion only, all new premium selling disabled. filings/surveillance
  ALERTS run in every regime. Expiry days: halve all new sizes.
- Scheduler (APScheduler, idempotent jobs, misfire grace): 08:30 auth check;
  08:45 pre-open context; 09:06 preopen snapshots; 30s announcement polls;
  10:00 regime; 14:45 entry cutoff; 15:10 force-flat; 15:20 overnight check;
  15:45 digest; 20:30 nightly downloads (bhavcopy, surveillance, flows,
  bulk/block); Sunday pairs refresh reminder.

## 8. INDIA-SPECIFIC MARKET RULES (encode all of these)

### 8.1 Kite auth reality
Kite access tokens **expire daily** (~07:30 IST). Build
`scripts/morning_auth.py`: prints login URL, I complete login + TOTP in
browser, paste the `request_token`, script exchanges and stores the access
token in `.env`/keyring. The 08:30 scheduler job verifies token validity and
alerts me on Telegram if the system is unauthenticated. Do NOT fully
automate credential entry.

### 8.2 NSE website etiquette (`core/nse.py`)
NSE endpoints reject naive clients. One shared `httpx` session per run:
warm up by hitting the NSE homepage for cookies, send realistic
User-Agent/Accept headers, reuse cookies, exponential backoff on 401/403 by
re-warming, hard cap request rates (≥2s between calls per endpoint except
the 30s announcements poll), and fail LOUDLY (Telegram) after 5 consecutive
failures — never silently degrade. All endpoint URLs in config; they change
periodically and I will maintain them.

### 8.3 Session structure & instruments
Pre-open 09:00–09:08 (order entry to 09:07:59, matching 09:08–09:12),
continuous 09:15–15:30, closing auction 15:40–16:00. GIFT Nifty trades
nearly round-the-clock — use its level vs prior close as the overnight gap
proxy (config URL; if unavailable, degrade gracefully and log). Expiry
schedule (weekly index day, monthly expiries) lives in `calendar.py` config
— **do not hardcode weekdays**; SEBI/exchanges have changed expiry days and
lot sizes multiple times recently. Verify current NIFTY lot size, expiry
weekday, and intraday-data lookback limits from Kite/NSE docs during
Phase 2 and record them in config with `as_of` dates.

### 8.4 Price bands & circuits
Stocks carry 2/5/10/20% bands (no band for F&O stocks, but dynamic
price-freeze applies); index circuit breakers halt the market at ±10/15/20%.
`datafeed` must expose band status per symbol; kernel veto per §3.

### 8.5 Settlement traps
Equity is T+1. Stock derivatives settle **physically** — §3 guard is
mandatory. Long ITM options held to expiry incur the higher delivery STT —
the §3 STT-trap guard is mandatory. MIS auto square-off ≈ 15:20 with broker
charges — our own 15:10 flat rule always fires first.

### 8.6 Product types
Intraday equity/short = MIS. Overnight equity = CNC. Overnight F&O = NRML.
The broker wrapper picks product type from the Signal's ttl; shorting
equity overnight is impossible in cash — reject any such Signal.

## 9. HOW YOU (SONNET) SHOULD WORK

1. Start each phase with a short PLAN: files to create/change, interfaces,
   test list. Wait for my OK on the plan only if it deviates from this spec;
   otherwise proceed.
2. Small commits, descriptive messages, one logical unit each.
3. After coding: run pytest, show the summary table, list any skipped/failing
   tests with reasons. Never claim green without running.
4. When a spec conflicts with reality (API missing a field, endpoint dead),
   STOP, show me the evidence, propose ≤3 options. Do not improvise data.
5. Websocket handling: auto-reconnect with backoff, resubscribe, journal the
   gap; scheduler jobs idempotent so a restart mid-day is safe.
6. Every LLM prompt you embed (filings, sympathy, PEAD tone) goes in
   `config/prompts/` as editable text files, each with a strict JSON schema
   and 2 few-shot examples. I will tune these — make them easy to edit.
7. Backtest verdicts are sacred: report FAILED engines as FAILED.

## 10. FINAL DELIVERABLES

- Paper-mode system runnable via `python -m live.orchestrator`.
- README: setup, morning auth walkthrough, running backtests, reading the
  journal, promoting an engine, updating universe/events configs, and the
  go-live checklist: per-engine PROMOTED verdict + ≥4 weeks clean paper
  trading + my explicit sign-off + first two live weeks at minimum size.
- `docs/RUNBOOK.md`: what to do when — scraper breaks, token expires
  mid-day, websocket drops, kill-switch fires, NSE changes an endpoint.

Begin with Phase 1 now. Show me CLAUDE.md and settings.yaml first.
