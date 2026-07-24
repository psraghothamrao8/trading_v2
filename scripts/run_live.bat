@echo off
REM Launched by the daily Windows Scheduled Task "TradingV2DailyStart".
REM Starts the paper-mode session (python -m live.orchestrator --run), which
REM then blocks and runs its own internal scheduler (08:30 auth check, 08:45
REM pre-open context, 10:00 regime, 15:10 force-flat, 15:45 digest, 20:30
REM nightly downloads, ...) for the rest of the day. Output is appended to
REM logs\daily_run.log since a scheduled task has no console to print to.
cd /d "C:\Users\Admin\Documents\trading_v2"
python -m live.orchestrator --run >> logs\daily_run.log 2>&1
