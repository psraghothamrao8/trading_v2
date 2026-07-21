# Indian Markets Multi-Engine Trading System (v2)

A personal algorithmic trading system for NSE, built on Zerodha Kite Connect.
Eleven engines, one risk kernel, an event-driven backtester with a full Indian
cost model, and a regime router that decides which engines are allowed to play
on any given day.

**Paper mode is the default and live mode requires a typed confirmation.**
Nothing here trades real money until you deliberately make it.

- Specification: [MASTER_BUILD_PROMPT.md](MASTER_BUILD_PROMPT.md) — the authority
- Conventions and directives: [CLAUDE.md](CLAUDE.md)
- Operational playbook: [docs/RUNBOOK.md](docs/RUNBOOK.md)

---

## Contents

1. [Setup](#1-setup)
2. [Morning auth walkthrough](#2-morning-auth-walkthrough)
3. [Running the system](#3-running-the-system)
4. [Running backtests](#4-running-backtests)
5. [Reading the journal](#5-reading-the-journal)
6. [Promoting an engine](#6-promoting-an-engine)
7. [Updating universe and events configs](#7-updating-universe-and-events-configs)
8. [The go-live checklist](#8-the-go-live-checklist)
9. [What each engine does](#9-what-each-engine-does)
10. [Architecture](#10-architecture)

---

## 1. Setup

Python 3.11 or newer.

```bash
git clone https://github.com/psraghothamrao8/trading_v2.git
cd trading_v2
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

Copy the environment template and fill it in:

```bash
cp .env.example .env
```

| Variable | Where it comes from |
|---|---|
| `KITE_API_KEY`, `KITE_API_SECRET` | [developers.kite.trade/apps](https://developers.kite.trade/apps) |
| `KITE_ACCESS_TOKEN` | written for you by `scripts/morning_auth.py` |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) |
| `TELEGRAM_BOT_TOKEN` | message `@BotFather` → `/newbot` |
| `TELEGRAM_CHAT_ID` | message your bot once, then open `https://api.telegram.org/bot<TOKEN>/getUpdates` |

`TELEGRAM_CHAT_ID` is an allowlist — only that chat can issue `/kill`.

Verify the install:

```bash
python -m pytest -q
python -m live.orchestrator --status
```

`--status` prints the account, the risk configuration, every engine's state,
the calendar and today's journal counts. It never places an order.

---

## 2. Morning auth walkthrough

**Kite access tokens expire every day at about 07:30 IST.** There is no way
around this, and credential entry is deliberately not automated (§8.1).

```bash
python scripts/morning_auth.py
```

1. The script prints a login URL. Open it in a browser.
2. Log in with your Zerodha credentials and TOTP.
3. Kite redirects to your app's redirect URL. Copy either the whole URL or
   just the `request_token=` value.
4. Paste it back at the prompt.

The access token is written into `.env`. Check it any time with:

```bash
python scripts/morning_auth.py --check
```

The 08:30 scheduler job runs this check automatically and sends a Telegram
alert if the system is unauthenticated. If you see that alert, the system is
blind until you log in — it will not trade on stale data.

---

## 3. Running the system

### First-time data load

```bash
python scripts/download_history.py --instruments
python scripts/download_history.py --universe NIFTY500 --interval day
python scripts/download_history.py --universe NIFTY200 --interval 5minute
python scripts/refresh_pairs.py            # §6.3 cointegration
python scripts/nightly_downloads.py        # surveillance, flows, VIX
```

The NIFTY-500 daily pull takes a while — Kite rate-limits, and the downloader
respects it. It is resumable: `--update` fetches only what is missing.

### A trading day

```bash
python -m live.orchestrator --run
```

This starts the scheduler and blocks. The day then runs itself:

| Time (IST) | What happens |
|---|---|
| 08:30 | Kite token check; Telegram alert if unauthenticated |
| 08:45 | Pre-open context: GIFT Nifty gap, FII positioning, India VIX |
| 09:06:30, 09:07:45 | §6.5 pre-open auction snapshots |
| 08:00–15:35, every 30s | §6.1 announcement poll → classify → alert |
| 09:15–09:20 | §6.5 continuation entries |
| 10:00 | §7 regime classification → engine enablement |
| 14:45 | Entry cutoff; management only from here |
| 15:10 | §3 force-flat of all MIS positions |
| 15:20 | §6.4 overnight check |
| 15:45 | Daily digest to Telegram |
| 20:30 | Nightly downloads, surveillance diff, special situations scan |
| Sunday | Pairs-refresh reminder |

### Other commands

```bash
python -m live.orchestrator --status          # account, config, regime
python -m live.orchestrator --status --json   # the same, machine-readable
python -m live.orchestrator --kill --reason "stepping away"
```

### Telegram commands

| Command | Effect |
|---|---|
| `/kill` | Cancel every open order, flatten every position, block new orders |
| `/status` | Mode, regime, open positions, available capital |
| `/confirm <id>` | Approve a §6.8 wheel proposal |
| `/reject <id>` | Decline one |

---

## 4. Running backtests

```bash
# Plumbing sanity check first — if this is not believable, nothing else is
python scripts/run_backtest.py --sanity --window validate

python scripts/run_backtest.py --engine filings --window validate
python scripts/run_backtest.py --engine pairs --window tune --interval 5minute
python scripts/run_backtest.py --all --window validate
```

Walk-forward windows (§4), configured in `settings.yaml`:

| Window | Range | Purpose |
|---|---|---|
| `tune` | 2019–2022 | Parameter selection |
| `validate` | 2023–2024 | Promotion decisions |
| `test` | 2025–present | **Consumed exactly once** |

Re-running `test` prints a loud warning and the result should be treated as
tainted. That warning is not a nuisance — it is the difference between an
out-of-sample measurement and a number you fitted to.

Every backtest fill runs through the §4 cost model and the §3 risk kernel. The
output includes the metric set, a monthly returns table, a PROMOTED/FAILED
verdict with every gate's number, and an equity-curve CSV in
`backtest_output/`.

---

## 5. Reading the journal

Everything lands in `data/journal.sqlite`, timestamped in Asia/Kolkata.

```bash
sqlite3 data/journal.sqlite
```

| Table | Contents |
|---|---|
| `signals` | Every Signal an engine emitted, traded or not |
| `orders` | Orders that passed the kernel and were routed |
| `fills` | Executions, with the cost breakdown |
| `trades` | Completed round trips, net of costs |
| `rejections` | Every kernel veto, with its reason code |
| `errors` | Scraper failures, websocket gaps, auth loss |
| `announcements` | Filings with their LLM verdicts (the §6.1 store) |
| `regime_log` | Daily §7 classification with its inputs |
| `daily_summary` | One row per session — the digest source |
| `flows` | Nightly FII/DII readings and percentiles |
| `surveillance_snapshots` | ASM/GSM/ban-list snapshots |
| `backtest_runs` | Every backtest, incl. test-window consumption |
| `pairs` | Cointegrated pairs with hedge ratios |
| `kill_events` | Kill-switch activations |

Useful queries:

```sql
-- Why did nothing trade today?
SELECT reason_code, COUNT(*) FROM rejections
WHERE trade_date = date('now') GROUP BY reason_code ORDER BY 2 DESC;

-- What has each engine actually made, net of costs?
SELECT engine, COUNT(*) AS trades, ROUND(SUM(net_pnl), 2) AS net,
       ROUND(SUM(costs), 2) AS costs
FROM trades GROUP BY engine ORDER BY net DESC;

-- Did the cost model eat the edge?
SELECT engine, ROUND(SUM(gross_pnl), 2) AS gross, ROUND(SUM(costs), 2) AS costs,
       ROUND(SUM(net_pnl), 2) AS net FROM trades GROUP BY engine;

-- Today's material filings and whether they were traded
SELECT symbol, label, confidence, traded FROM announcements
WHERE trade_date = date('now') AND label != 'NOISE';
```

---

## 6. Promoting an engine

An engine ships with `auto_trade: false`. That means it emits signals, they are
journalled, and alerts fire — but no order is placed. Promotion is the only way
that changes, and it is a manual decision.

1. **Backtest on the validation window.**
   ```bash
   python scripts/run_backtest.py --engine <name> --window validate
   ```

2. **Read the verdict.** All four gates must pass:

   | Gate | Threshold |
   |---|---|
   | Profit factor | ≥ 1.3 |
   | Trades | ≥ 150 (≥ 40 for event engines) |
   | Max drawdown | ≤ 12% of that engine's capital |
   | Net expectancy | > 0, after costs |

3. **If FAILED, stop.** A FAILED verdict is a result, not a bug. The engine
   stays alert-only. **Do not soften a gate to make an engine pass** — the
   gates exist precisely for the moment you want to move them.

4. **If PROMOTED**, run the untouched test window once:
   ```bash
   python scripts/run_backtest.py --engine <name> --window test
   ```

5. **Then paper trade it for at least four weeks** with `auto_trade: false`
   still set, and compare the paper signals against the backtest expectation.

6. **Only then** flip the flag in `config/settings.yaml`:
   ```yaml
   engines:
     <name>:
       auto_trade: true
   ```

`surveillance` and `special_situations` can never be promoted. They are
alert-only structurally — their `on_schedule` returns nothing regardless of
configuration, and the risk kernel rejects their orders anyway.

---

## 7. Updating universe and events configs

### `config/universe.yaml`

Index constituents change at every semi-annual rebalance. A backtest run
against a stale list has survivorship bias.

- `nifty50`, `nifty_next_50`, `nifty_midcap_100` — index members. Update the
  `meta.as_of` date whenever you edit them.
- `pair_sectors` — §6.3 same-sector groups, NIFTY-50 only. Re-run
  `scripts/refresh_pairs.py` after any change.
- `wheel_approved` — §6.8. **Only add a name you would genuinely be happy to
  take delivery of**, because stock options settle physically. Every symbol
  here also needs a lot size in `settings.yaml → market.lot_sizes`.
- `blacklist.symbols` — never traded by any engine, whatever the signal.

### `config/events.yaml`

You maintain this. The kernel refuses new entries on these dates and refuses to
carry positions overnight into them.

```yaml
rbi_mpc:
  - date: "2026-08-06"
    note: "RBI MPC decision"
```

Categories: `rbi_mpc`, `union_budget`, `election_result`, `us_fed`, `other`.
For FOMC, list the **Indian session that follows** the decision, not the US
date. Past dates are kept for backtest fidelity.

### `config/holidays.yaml`

```bash
python scripts/refresh_holidays.py --check
```

This diffs your file against NSE's holiday master and prints what to change. It
never edits the file itself — a scraper silently rewriting the trading calendar
is exactly how a backtest gets quietly corrupted.

---

## 8. The go-live checklist

Every item. No exceptions.

- [ ] **Per-engine PROMOTED verdict** on the validation window, with the numbers
      recorded. Any engine without one stays `auto_trade: false`.
- [ ] **≥ 4 weeks of clean paper trading** — the full session loop running daily
      with journal and digest, no unexplained errors, and paper results in the
      same territory as the backtest.
- [ ] **Your explicit sign-off** on each engine individually.
- [ ] **First two live weeks at minimum size** — reduce `risk.capital` in
      `settings.yaml` to a fraction of the real account, and raise it only after
      those two weeks are clean.

Before flipping the switch, confirm:

- [ ] `costs.verified_against_calculator: true` — you have run the three sample
      trades in `tests/test_costs.py` through
      [Zerodha's calculator](https://zerodha.com/brokerage-calculator/) and the
      totals match.
- [ ] `market.expiry.verified: true` and `market.lot_sizes.verified: true` —
      confirmed against current NSE/Kite data.
- [ ] `holidays.yaml → meta.verified_against_nse: true`.
- [ ] Telegram works: you have received a test alert and `/kill` has been
      exercised at least once against a paper position.
- [ ] `.env` is not in git (`git check-ignore .env` should print `.env`).
- [ ] You have read [docs/RUNBOOK.md](docs/RUNBOOK.md) and know what to do when
      the scraper breaks at 09:20 on a Monday.

Then, and only then:

```yaml
# config/settings.yaml
execution:
  mode: live
```

```bash
python -m live.orchestrator --run
```

You will be asked to type `y`. A non-interactive process can never answer that
question, which is the point.

---

## 9. What each engine does

| § | Engine | Edge | Default |
|---|---|---|---|
| 6.1 | `filings` | Reacts to clearly material announcements before the crowd reads them. Also the shared news sensor for 6.2 and 6.6. | alert-only |
| 6.2 | `sympathy` | Trades the listed supplier/customer/peer that has not repriced yet. | alert-only |
| 6.3 | `pairs` | Intraday same-sector mean reversion. Market-neutral; smooths the ensemble. | alert-only |
| 6.4 | `overnight` | Harvests overnight index drift, with filters that skip the toxic nights. | alert-only |
| 6.5 | `preopen` | Pre-open auction imbalance that continues into the first minutes. | alert-only |
| 6.6 | `pead` | Post-earnings drift, gated on whether management's guidance backs the beat. | alert-only |
| 6.7 | `panic_reversion` | Buys sentiment crashes, never news-driven ones. PANIC regime only. | alert-only |
| 6.8 | `wheel` | Cash-secured puts on stocks you would own, only when IV is expensive. Every order needs your confirmation. | alert-only |
| 6.9 | `flows` | FII positioning extremes. Low percentile = long; high percentile = a veto, never a short. | alert-only |
| 6.10 | `surveillance` | Nightly ASM/GSM/ban diff. Feeds the kernel veto. | **alert-only always** |
| 6.11 | `special_situations` | Buybacks, open offers, delistings, index changes, with the economics computed. | **alert-only always** |

---

## 10. Architecture

```
                    ┌──────────────────────────────┐
                    │  live/session.py  +  §7      │
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

**The one-way rule:** engines produce `Signal`s and nothing else. Only
`live/session.py::route()` turns a Signal into an Order, and only a kernel
`ALLOW` reaches the broker. An engine that imports `core.broker` fails
`tests/test_architecture.py`.

### The risk kernel (§3)

Every order, paper or live, passes `core/risk.py::check()`. Beyond sizing and
the loss limits, it enforces:

| Veto | Rule |
|---|---|
| F&O ban list | MWPL ≥ 95% ⇒ reject derivatives entries |
| ASM/GSM surveillance | Reject **all** new entries, every engine |
| Circuit band | Within 1% of a band ⇒ reject (a locked stock cannot be exited) |
| Event days | No entries, no overnight holds into them |
| Time | No new intraday entries after 14:45; MIS flat by 15:10 |
| Physical settlement | Short stock options ITM/near-money must close by expiry−2 |
| STT trap | No long ITM options into the expiry close |

Exits are deliberately exempt from the instrument vetoes. Blocking an exit
because a stock is near its circuit band would trap you in exactly the position
you are trying to leave.

### Running the tests

```bash
python -m pytest -q                       # everything
python -m pytest tests/test_risk.py -v    # the kernel, veto by veto
python -m pytest tests/test_costs.py -v   # the cost model + calculator samples
```

`tests/test_architecture.py` is not a unit test — it is a guard rail. It fails
the build if an engine imports the broker, if a secret appears in a tracked
file, if a stub sneaks in, or if a bare `datetime.now()` appears anywhere.

---

## Licence and disclaimer

Personal software, provided as-is. Trading carries risk of loss. Nothing here
is investment advice. You are responsible for every order this system places on
your behalf — which is why it will not place one until you have said yes, in
writing, in a terminal.
