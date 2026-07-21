"""Daily Kite login. Implements §8.1.

Kite access tokens expire every morning around 07:30 IST. There is no way to
refresh one without a browser login plus TOTP, and **we do not automate
credential entry** (§8.1) -- this script prints a URL, you log in yourself, and
you paste back the ``request_token`` from the redirect.

    python scripts/morning_auth.py

The exchanged access token is written into ``.env`` as ``KITE_ACCESS_TOKEN``.
The 08:30 scheduler job checks that token and alerts on Telegram if the system
is unauthenticated.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Allow `python scripts/<name>.py` as well as `python -m scripts.<name>`.
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from core.config import REPO_ROOT, get_secrets
from core.logging_config import setup_logging

ENV_PATH = REPO_ROOT / ".env"


def extract_request_token(pasted: str) -> str:
    """Accept either a bare token or the whole redirect URL.

    The redirect looks like::

        http://127.0.0.1/?action=login&type=login&status=success&request_token=XXXX

    Pasting the full URL is what actually happens at 07:30, so handle it.
    """
    pasted = pasted.strip()
    if not pasted:
        raise ValueError("Nothing pasted.")
    if "request_token" in pasted:
        query = parse_qs(urlparse(pasted).query)
        tokens = query.get("request_token")
        if tokens and tokens[0]:
            return tokens[0]
        match = re.search(r"request_token=([A-Za-z0-9]+)", pasted)
        if match:
            return match.group(1)
        raise ValueError(f"Could not find a request_token in: {pasted[:120]}")
    if not re.fullmatch(r"[A-Za-z0-9]{6,}", pasted):
        raise ValueError(
            f"{pasted[:40]!r} does not look like a request_token or a redirect URL."
        )
    return pasted


def write_env_value(key: str, value: str, env_path: Path | None = None) -> Path:
    """Set ``key=value`` in ``.env``, preserving every other line and comment."""
    path = env_path or ENV_PATH
    if not path.exists():
        example = REPO_ROOT / ".env.example"
        if example.exists():
            path.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            path.write_text("", encoding="utf-8")

    lines = path.read_text(encoding="utf-8").splitlines()
    replaced = False
    for index, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[index] = f"{key}={value}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Daily Kite Connect login (§8.1)")
    parser.add_argument(
        "--check", action="store_true", help="only verify the existing token, do not log in"
    )
    parser.add_argument("--request-token", help="skip the prompt and use this token")
    args = parser.parse_args(argv)

    setup_logging()
    secrets = get_secrets()

    if not secrets.kite_api_key or not secrets.kite_api_secret:
        print("ERROR: KITE_API_KEY / KITE_API_SECRET are not set in .env", file=sys.stderr)
        print("       Copy .env.example to .env and fill them in.", file=sys.stderr)
        return 2

    try:
        from kiteconnect import KiteConnect
    except ImportError:
        print("ERROR: kiteconnect is not installed. pip install -r requirements.txt", file=sys.stderr)
        return 2

    kite = KiteConnect(api_key=secrets.kite_api_key)

    # --check: is the stored token still alive?
    if args.check:
        if not secrets.kite_access_token:
            print("UNAUTHENTICATED: no KITE_ACCESS_TOKEN in .env")
            return 1
        kite.set_access_token(secrets.kite_access_token)
        try:
            profile = kite.profile()
            print(f"AUTHENTICATED as {profile.get('user_id')} ({profile.get('user_name')})")
            return 0
        except Exception as exc:
            print(f"UNAUTHENTICATED: {exc}")
            print("Run `python scripts/morning_auth.py` to re-login (§8.1: tokens expire daily).")
            return 1

    print()
    print("=" * 72)
    print("  KITE MORNING AUTH  (§8.1 -- access tokens expire daily ~07:30 IST)")
    print("=" * 72)
    print()
    print("  1. Open this URL in your browser and complete login + TOTP:")
    print()
    print(f"     {kite.login_url()}")
    print()
    print("  2. You will be redirected to your app's redirect URL. Copy either the")
    print("     whole redirect URL or just the request_token value.")
    print()

    raw = args.request_token or input("  Paste request_token (or the full redirect URL): ")
    try:
        request_token = extract_request_token(raw)
    except ValueError as exc:
        print(f"\n  ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        session = kite.generate_session(request_token, api_secret=secrets.kite_api_secret)
    except Exception as exc:
        print(f"\n  ERROR: token exchange failed: {exc}", file=sys.stderr)
        print("  A request_token is single-use and expires in minutes -- redo step 1.",
              file=sys.stderr)
        return 2

    access_token = session["access_token"]
    path = write_env_value("KITE_ACCESS_TOKEN", access_token)
    if session.get("user_id"):
        write_env_value("KITE_USER_ID", session["user_id"])

    print()
    print(f"  OK: access token stored in {path}")
    print(f"      user  : {session.get('user_id')} ({session.get('user_name')})")
    print(f"      valid : until roughly 07:30 IST tomorrow")
    print()
    print("  Verify with: python -m live.orchestrator --status")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
