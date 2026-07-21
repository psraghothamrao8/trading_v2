# RUNBOOK — what to do when

Operational playbook for the Indian Markets Multi-Engine Trading System.
Written for the person reading it at 09:20 on a Monday with something broken.

**First rule: nothing here is urgent enough to skip the kill switch.** If you
are unsure what state the system is in, kill it and diagnose calmly.

```bash
python -m live.orchestrator --kill --reason "diagnosing"
```
or send `/kill` on Telegram.

---

## Contents

- [Fast triage](#fast-triage)
- [The kill switch fires](#the-kill-switch-fires)
- [The Kite token expires mid-day](#the-kite-token-expires-mid-day)
- [The NSE scraper breaks](#the-nse-scraper-breaks)
- [NSE changes an endpoint](#nse-changes-an-endpoint)
- [The websocket drops](#the-websocket-drops)
- [A daily or weekly loss limit trips](#a-daily-or-weekly-loss-limit-trips)
- [The physical-settlement guard fires](#the-physical-settlement-guard-fires)
- [An engine is trading when it should not be](#an-engine-is-trading-when-it-should-not-be)
- [An engine is not trading when it should be](#an-engine-is-not-trading-when-it-should-be)
- [The LLM is failing or slow](#the-llm-is-failing-or-slow)
- [Telegram goes quiet](#telegram-goes-quiet)
- [A scheduler job did not run](#a-scheduler-job-did-not-run)
- [Restarting mid-session](#restarting-mid-session)
- [Data problems](#data-problems)
- [Rates, lot sizes or expiry days change](#rates-lot-sizes-or-expiry-days-change)
- [Escalation: when to stop trading entirely](#escalation-when-to-stop-trading-entirely)

---

## Fast triage

```bash
python -m live.orchestrator --status          # is it authenticated? what regime?
tail -n 100 logs/trading.log                  # what did it just do?
```

```sql
-- Recent errors, worst first
SELECT ts, source, severity, message FROM errors
WHERE trade_date >= date('now','-1 day') ORDER BY id DESC LIMIT 30;

-- Why is nothing trading?
SELECT reason_code, COUNT(*) FROM rejections
WHERE trade_date = date('now') GROUP BY reason_code ORDER BY 2 DESC;

-- What is actually open?
SELECT * FROM orders WHERE trade_date = date('now') ORDER BY ts DESC;
```

Three questions, in order:

1. **Is money at risk right now?** Open positions with a broken data feed is the
   only true emergency. Kill first.
2. **Is it going to place a wrong order?** If yes, kill. If no, you have time.
3. **Is it just silent?** Silence is safe. Diagnose properly.

---

## The kill switch fires

**Symptom:** Telegram shows `🚨 KILL SWITCH FIRED`.

**What already happened:** every open order was cancelled, every position was
flattened, and the kernel is latched — no new order passes until the process
restarts.

**What to do:**

1. Find out who fired it and why:
   ```sql
   SELECT * FROM kill_events ORDER BY id DESC LIMIT 5;
   ```
   `source` is `cli`, `telegram`, or `risk_kernel`.
2. Confirm the book is actually flat. In live mode, check the Kite web console
   directly — do not trust the system's own view of a system you just killed.
3. Check for partial failures:
   ```sql
   SELECT * FROM errors WHERE source IN ('flatten_all','kill')
   ORDER BY id DESC LIMIT 20;
   ```
   Anything here means a position may still be open. Close it manually in Kite.
4. Do not restart until you know why it fired.

**Restarting** clears the latch. That is deliberate — the latch is a
within-session stop, not a permanent one.

---

## The Kite token expires mid-day

**Symptom:** broker calls start failing; `--status` shows `authenticated: NO`;
Telegram shows `❗ ERROR [broker]`.

Tokens are meant to last the session, but a Kite-side invalidation, a second
login elsewhere, or an app-key change can end one early.

**What to do:**

1. Confirm:
   ```bash
   python scripts/morning_auth.py --check
   ```
2. **If you have open live positions**, manage them in the Kite web console
   yourself. The system cannot place an exit without a session.
3. Re-authenticate:
   ```bash
   python scripts/morning_auth.py
   ```
4. Restart the session. It reads open positions from the broker and continues.

**Do not** try to trade through a dead session. Every order will fail, each
failure is journalled, and you will be reading a wall of noise instead of the
one line that mattered.

---

## The NSE scraper breaks

**Symptom:** `🚨 CRITICAL [nse] 5 consecutive NSE failures`.

The client already tried: cookie re-warm, exponential backoff, four retries per
call. Five consecutive failures means it is genuinely refused or the endpoint is
gone.

**What breaks downstream:**

| Broken fetcher | Consequence |
|---|---|
| `announcements` | §6.1 blind; §6.2 has no trigger; §6.3 and §6.7 lose their filing cross-check |
| `preopen` | §6.5 takes no snapshots and does not trade |
| `ban_list`, `asm`/`gsm` | **The §3 vetoes run on stale data** — the serious one |
| `fii_dii`, `fii_derivatives` | §6.9 has no reading and stays flat |
| `india_vix` | §6.8's IV gate cannot open; §7 loses a PANIC input |

**What to do:**

1. Is it NSE or is it you? Open `https://www.nseindia.com` in a browser.
   - Site down → wait. The client retries. Consider disabling the affected
     engines for the day.
   - Site fine → the endpoint or the anti-bot rules changed. See the next
     section.
2. **Check how stale the veto lists are.** This is the item that actually risks
   money:
   ```sql
   SELECT list_name, MAX(trade_date) FROM surveillance_snapshots GROUP BY list_name;
   ```
   More than about three days old and the §6.10 engine will warn you on its own.
   Older than a week: **stop trading**, or manually add the current ASM/GSM
   names to `universe.yaml → blacklist.symbols` until the feed is back.
3. Test one endpoint in isolation:
   ```bash
   python -c "from core.nse import NSEClient; c=NSEClient(); print(c.fno_ban_list())"
   ```

---

## NSE changes an endpoint

NSE moves URLs and changes payload shapes without notice. Both are handled the
same way, and neither requires touching Python for the URL case.

**Symptom:** one specific fetcher fails consistently while others work, or a
fetcher returns rows that parse to empty.

**To fix a moved URL** — config only:

1. Find the new path. Open the relevant NSE page in a browser with DevTools →
   Network → Fetch/XHR, and read the request the page itself makes.
2. Edit `config/settings.yaml`:
   ```yaml
   nse:
     endpoints:
       ban_list: "/api/fno-ban-list"     # <- the new path
   ```
3. Test it:
   ```bash
   python -c "from core.nse import NSEClient; c=NSEClient(); print(c.fno_ban_list())"
   ```

**To fix a changed payload shape** — this needs code, in exactly one place:
the typed wrapper in `core/nse.py` (e.g. `preopen_snapshot`, `fii_derivatives`).
Normalisation is deliberately confined to that file so no engine ever sees a raw
NSE field name.

**If the headers stop working**, update `nse.headers.User-Agent` in config to a
current browser string.

---

## The websocket drops

**Symptom:** `⚠️ WARNING [websocket] Tick feed reconnected after a 47s gap`.

This is handled: the stream reconnects with backoff, resubscribes to everything
it had, and journals the gap. The alert exists because a silent gap looks
exactly like a quiet market, and engines would happily act on stale state.

**What to do:**

1. Judge the gap length:
   - **< 30s** — noise. Carry on.
   - **30s–5min** — check open positions. A stop may not have been evaluated
     during the gap; verify each position's price against its stop manually.
   - **> 5min** — treat as a data outage. Consider killing and going flat.
2. See the history:
   ```sql
   SELECT ts, message FROM errors WHERE source='websocket' ORDER BY id DESC LIMIT 20;
   ```
3. Repeated drops usually mean the local network, not Kite. Check for a second
   process using the same API key — Kite allows one websocket session per key,
   and two clients will knock each other off in a loop.

---

## A daily or weekly loss limit trips

**Symptom:** every new order is rejected with `DAILY_LOSS_LIMIT` or
`WEEKLY_LOSS_LIMIT`.

**This is the system working.** Daily: −1.5% of capital. Weekly: −3%.

**What happens automatically:** intraday positions are flattened, new orders are
blocked until the next session (daily) or Monday (weekly), and you are alerted.
**Exits are still allowed** — the limits exist to get you out.

**What to do:**

1. Look at the damage:
   ```sql
   SELECT engine, symbol, net_pnl, exit_reason FROM trades
   WHERE trade_date = date('now') ORDER BY net_pnl ASC;
   ```
2. Ask whether one engine did it or the whole book did. One engine having a bad
   day is normal; one engine having a bad day repeatedly is a promotion that
   should be reconsidered.
3. **Do not raise the limit to keep trading.** If you find yourself editing
   `daily_loss_limit_pct` during a losing day, stop and close the file.

---

## The physical-settlement guard fires

**Symptom:** `⚠️ WARNING [physical_settlement]` or rejections with
`PHYSICAL_SETTLEMENT_GUARD`.

**Why this matters more than any other alert:** Indian stock derivatives settle
**physically**. A short in-the-money stock option carried into expiry becomes a
real delivery obligation, and that obligation can exceed the entire account.

**What to do — same day:**

1. Identify it:
   ```sql
   SELECT * FROM orders WHERE trade_date = date('now')
   AND segment='equity_options' ORDER BY ts DESC;
   ```
2. Choose one:
   - **Close it.** Always correct if you are unsure.
   - **Roll it** down-and-out to a later expiry.
   - **Accept delivery** — only if you have the cash and genuinely want the
     shares. Set `allow_delivery: true` on that trade explicitly.
3. Never leave it to expiry to "see what happens".

The same guard has an exit side: `RiskKernel.stt_trap_exits_due()` flags long
ITM options that must be closed by 15:00 on expiry day, because held into the
close they pay delivery STT on intrinsic value — often more than the option is
worth.

---

## An engine is trading when it should not be

1. **Kill first**, then diagnose.
2. Check its config:
   ```bash
   python -m live.orchestrator --status | grep -A15 ENGINES
   ```
   `auto_trade` should be `NO` for anything not deliberately promoted.
3. If an alert-only engine (`surveillance`, `special_situations`) placed an
   order, that is a serious bug, not a misconfiguration — those have two
   independent barriers. Capture the journal rows and stop trading.
4. If the regime router enabled something you did not expect, look at the map:
   ```yaml
   regime:
     enablement:
       TREND: [filings, sympathy, preopen, overnight]
   ```

---

## An engine is not trading when it should be

Almost always a veto, and the journal will tell you which:

```sql
SELECT reason_code, symbol, reason FROM rejections
WHERE trade_date = date('now') AND engine = 'filings' ORDER BY id DESC LIMIT 20;
```

| Reason code | Meaning |
|---|---|
| `ALERT_ONLY_ENGINE` | Not promoted. Working as designed. |
| `ASM_GSM_SURVEILLANCE` | The stock is under surveillance. |
| `FNO_BAN_LIST` | MWPL ≥ 95%. |
| `CIRCUIT_BAND_PROXIMITY` | Within 1% of a band. |
| `BLOCKED_EVENT_DAY` | It is in `events.yaml`. |
| `INTRADAY_ENTRY_CUTOFF` | After 14:45. |
| `MAX_NEW_TRADES_PER_DAY_PER_ENGINE` | Already opened 3 today. |
| `MAX_CONCURRENT_POSITIONS_TOTAL` | 8 positions open. |
| `ENGINE_CAPITAL_CAP` | Its capital slice is full. |
| `ZERO_QUANTITY` | The stop is too far for the risk budget. |
| `DAILY_LOSS_LIMIT` / `WEEKLY_LOSS_LIMIT` | See above. |

**No rejections at all** means no signal was generated. Check whether the engine
is enabled in today's regime, and whether its data is present:

```sql
SELECT regime, enabled_engines FROM regime_log
WHERE trade_date = date('now') ORDER BY ts DESC LIMIT 1;
```

---

## The LLM is failing or slow

**Symptom:** `❗ ERROR [filings] ... LLM task 'filings' failed after 3 attempts`,
or alerts tagged `⚠️ over budget`.

**Failures:**

1. Check the key: `ANTHROPIC_API_KEY` in `.env`.
2. Check credit and rate limits at console.anthropic.com.
3. The system degrades correctly: no classification means no alert and no trade
   from §6.1, and §6.2 and §6.6 lose their input. It does not guess.

**Slowness** (classification over the 20s budget): §6.1 alerts anyway and skips
the auto-trade, by design — a late trade on a stale filing is worse than none.
Persistent slowness usually means an oversized PDF; reduce
`engines.filings.pdf_max_pages`.

**Same filing being classified repeatedly** means the cache is not working:

```sql
SELECT task, COUNT(*), SUM(hits) FROM llm_cache GROUP BY task;
```

Zero hits with a 30-second poll running is a bug. Check `data/llm_cache.sqlite`
is writable.

---

## Telegram goes quiet

Alerts are best-effort by design — a Telegram outage must never stop trading, so
every send failure is logged and swallowed. **The journal is the source of
truth, not your phone.**

1. Verify config:
   ```bash
   python -m live.orchestrator --status | grep telegram
   ```
2. Test the token:
   ```bash
   curl "https://api.telegram.org/bot<TOKEN>/getMe"
   ```
3. Check for drops:
   ```bash
   grep -i "telegram" logs/trading.log | tail -30
   ```
   `Telegram rate limit hit` means too many alerts in a minute; raise
   `alerts.telegram.rate_limit_per_minute` or find what is generating the flood.
4. **If `/kill` is not reachable, use the CLI**:
   ```bash
   python -m live.orchestrator --kill --reason "telegram down"
   ```

---

## A scheduler job did not run

```sql
SELECT * FROM errors WHERE source='scheduler' ORDER BY id DESC LIMIT 20;
```

Jobs are wrapped so a failure is journalled and the scheduler survives. Common
causes: the machine was asleep at the fire time (misfire grace is 300s — beyond
that the job is skipped), or a dependency was unavailable.

**Running a job by hand:**

```bash
python scripts/nightly_downloads.py        # the 20:30 job
python scripts/morning_auth.py --check     # the 08:30 check
python scripts/refresh_pairs.py            # the Sunday reminder's work
```

The digest and regime jobs live inside the session process; restarting the
session re-registers them, and both are idempotent.

---

## Restarting mid-session

Safe by design (§9.5). Every job is idempotent and per-day state is keyed by
date.

```bash
# Ctrl-C, then:
python -m live.orchestrator --run
```

**What is preserved:** open positions (read from the broker), the journal,
trade counts, and the surveillance/ban veto sets.

**What is lost and why it is fine:**

- The kill latch clears. It is a within-session stop.
- The regime decision is re-read from `regime_log` at the next classification;
  before 10:00 the system is `NA` and opens nothing.
- The §6.5 pre-open snapshots are gone. If you restart after 09:08 that engine
  simply does not trade today.
- Engine watchlists (§6.6, §6.7) rebuild from daily bars on the next cycle.

**After any restart, verify the book matches reality:**

```bash
python -m live.orchestrator --status
```

---

## Data problems

**"No {interval} data for {symbol}"** — the download never ran:

```bash
python scripts/download_history.py --symbols RELIANCE --interval day
```

**Backtest reports zero trades.** Distinguish two very different cases:
missing data raises a `BacktestError` naming the fix; a genuinely empty result
prints `trades: 0` with a FAILED verdict. If you see the latter, the data was
there and the strategy found nothing.

**Suspicious backtest numbers** — check the obvious corruptions first:

```sql
SELECT engine, window_name, verdict, ts FROM backtest_runs ORDER BY id DESC LIMIT 10;
```

- Is `costs` non-zero in the trades? Zero costs means the model was bypassed and
  the result is fiction.
- Is `universe.yaml → meta.as_of` recent? A stale constituent list is
  survivorship bias.
- Has the `test` window been consumed more than once?
  ```sql
  SELECT engine, COUNT(*) FROM backtest_runs WHERE window_name='test' GROUP BY engine;
  ```

**Instruments cache stale:**

```bash
python scripts/download_history.py --instruments
```

---

## Rates, lot sizes or expiry days change

All three are config, all three carry `as_of` dates, and all three have burned
somebody before.

**Brokerage, STT, stamp duty, GST** — after any budget or SEBI circular:

1. Update `config/settings.yaml → costs`, bump every `as_of`.
2. Set `costs.verified_against_calculator: false`.
3. Run the three sample trades in `tests/test_costs.py` through
   [Zerodha's calculator](https://zerodha.com/brokerage-calculator/) by hand.
4. Update the expected values in that file, then set the flag back to `true`.

**Lot sizes** — NSE revises stock lot sizes at least twice a year:

```yaml
market:
  lot_sizes:
    as_of: "2026-07-22"
    verified: false
    NIFTY: 75
    RELIANCE: 500
```

Every symbol in `universe.yaml → wheel_approved` must have an entry.
`calendar.lot_size()` raises rather than guessing, which is why a missing entry
surfaces as a loud error and not a wrong position size.

**Expiry weekday** — SEBI and the exchanges have changed this more than once:

```yaml
market:
  expiry:
    verified: false
    weekly:
      NIFTY: {weekday: tuesday}
```

Nothing in the code hardcodes a weekday
(`tests/test_architecture.py` enforces that). After any change, re-check:

```bash
python -m live.orchestrator --status | grep -i expiry
```

**Holidays** — annually, when NSE publishes the circular:

```bash
python scripts/refresh_holidays.py --check
```

---

## Escalation: when to stop trading entirely

Set `auto_trade: false` on everything, or just do not start the session, if any
of these is true:

- **The surveillance or ban lists are more than a week stale.** The §3 vetoes
  are running blind, and those vetoes exist because trading a surveillance name
  is how you get stuck in a position you cannot exit.
- **You cannot reconcile the system's positions against the Kite console.**
- **An alert-only engine placed an order.** That requires two independent
  barriers to fail.
- **The cost model has not been verified since the last rate change.** Every
  backtest and every paper number is then unreliable in an unknown direction.
- **You are editing risk limits during a losing day.** Close the file, close the
  terminal, come back tomorrow.

Re-entry is the go-live checklist in [README](../README.md#8-the-go-live-checklist)
again, from the top. That is not bureaucracy — it is the only version of this
process that has ever worked.
