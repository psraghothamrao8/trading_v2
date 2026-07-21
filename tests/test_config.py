"""Config loading and the no-magic-numbers contract."""

from __future__ import annotations

import pytest

from core.config import (
    ConfigError,
    Settings,
    get_events,
    get_secrets,
    get_settings,
    get_universe,
    resolve_universe,
)


class TestSettings:
    def test_dotted_access(self):
        assert get_settings().get("risk.capital") == 800000

    def test_missing_key_names_the_path(self):
        with pytest.raises(ConfigError, match="risk.nonexistent"):
            get_settings().get("risk.nonexistent")

    def test_missing_key_with_default(self):
        assert get_settings().get("risk.nonexistent", 42) == 42

    def test_section_returns_settings(self):
        section = get_settings().section("risk.vetoes")
        assert isinstance(section, Settings)
        assert section.get("circuit_band_proximity_pct") == 1.0

    def test_section_on_a_scalar_raises(self):
        with pytest.raises(ConfigError, match="not a section"):
            get_settings().section("risk.capital")


class TestSpecDefaults:
    """§3 ships specific defaults. If these drift, the spec was violated."""

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("risk.capital", 800000),
            ("risk.risk_per_trade_pct", 0.5),
            ("risk.daily_loss_limit_pct", 1.5),
            ("risk.weekly_loss_limit_pct", 3.0),
            ("risk.max_new_trades_per_day_per_engine", 3),
            ("risk.max_concurrent_positions_total", 8),
        ],
    )
    def test_risk_defaults(self, path, expected):
        assert get_settings().require(path) == expected

    @pytest.mark.parametrize(
        "engine,cap",
        [
            ("filings", 25), ("sympathy", 10), ("pairs", 25), ("overnight", 25),
            ("pead", 25), ("panic_reversion", 15), ("wheel", 40), ("preopen", 10),
            ("flows", 15),
        ],
    )
    def test_per_engine_caps(self, engine, cap):
        assert get_settings().require(f"risk.per_engine_capital_cap_pct.{engine}") == cap

    def test_alert_only_engines(self):
        alert_only = set(get_settings().require("risk.alert_only_engines"))
        assert alert_only == {"surveillance", "special_situations"}

    def test_execution_defaults_to_paper(self):
        """§0.1: paper mode is default."""
        assert get_settings().require("execution.mode") == "paper"

    def test_promotion_gates(self):
        gates = get_settings().section("backtest.promotion_gates")
        assert gates.require("profit_factor_min") == 1.3
        assert gates.require("min_trades_default") == 150
        assert gates.require("min_trades_event_engines") == 40
        assert gates.require("max_drawdown_pct_of_engine_capital") == 12.0

    def test_slippage_model(self):
        """§4 slippage numbers."""
        slip = get_settings().section("costs.slippage")
        assert slip.require("liquid_equity_pct") == 0.03
        assert slip.require("options_pct") == 0.05
        assert slip.require("minimum_ticks") == 1

    def test_rate_tables_carry_as_of_dates(self):
        """§4: rates change, so every rate block is dated."""
        settings = get_settings()
        for path in ("costs.as_of", "costs.stt.as_of", "costs.gst.as_of",
                     "costs.stamp_duty.as_of", "costs.sebi_charges.as_of",
                     "market.expiry.as_of", "market.lot_sizes.as_of"):
            assert settings.get(path), f"{path} is missing an as_of date"


class TestUniverse:
    def test_nifty50_has_50_names(self):
        assert len(resolve_universe("NIFTY50")) == 50

    def test_nifty100_composition(self):
        n50 = set(resolve_universe("NIFTY50"))
        n100 = set(resolve_universe("NIFTY100"))
        assert n50 < n100

    def test_nifty200_includes_midcaps(self):
        assert set(resolve_universe("NIFTY100")) < set(resolve_universe("NIFTY200"))

    def test_no_duplicates(self):
        for name in ("NIFTY50", "NIFTY100", "NIFTY200"):
            symbols = resolve_universe(name)
            assert len(symbols) == len(set(symbols)), name

    def test_nifty500_directs_to_the_datafeed(self):
        """Guessing 500 tickers by hand is how survivorship bias creeps in."""
        with pytest.raises(ConfigError, match="resolve_nifty500"):
            resolve_universe("NIFTY500")

    def test_unknown_universe_raises(self):
        with pytest.raises(ConfigError, match="Unknown universe"):
            resolve_universe("NIFTY9999")

    def test_pair_sectors_are_nifty50_only(self):
        """§6.3: the pairs universe is NIFTY-50."""
        n50 = set(resolve_universe("NIFTY50"))
        for sector, symbols in get_universe().get("pair_sectors").items():
            for symbol in symbols:
                assert symbol in n50, f"{symbol} in {sector} is not in NIFTY50"

    def test_wheel_list_is_small_and_flagged_willing_to_own(self):
        """§6.8: 2-3 liquid large caps the owner would actually take delivery of."""
        approved = get_universe().get("wheel_approved")
        assert 2 <= len(approved) <= 3
        assert all(w["willing_to_own"] for w in approved)


class TestEvents:
    def test_blocking_categories_present(self):
        categories = get_events().get("blocking_categories")
        assert {"rbi_mpc", "union_budget", "us_fed"} <= set(categories)

    def test_every_entry_has_an_iso_date(self):
        import datetime as dt

        events = get_events()
        for category in events.get("blocking_categories"):
            for entry in events.get(category, []) or []:
                dt.date.fromisoformat(entry["date"])


class TestSecrets:
    def test_repr_never_leaks_values(self):
        """§0.4: a secrets object must be safe to log."""
        secrets = get_secrets()
        text = repr(secrets)
        for value in (secrets.kite_api_secret, secrets.telegram_bot_token,
                      secrets.anthropic_api_key):
            if value:
                assert value not in text

    def test_missing_reports_unset_names(self):
        from core.config import Secrets

        empty = Secrets(None, None, None, None, None, None, None, None)
        assert empty.missing(["kite_api_key", "anthropic_api_key"]) == [
            "kite_api_key", "anthropic_api_key"
        ]
