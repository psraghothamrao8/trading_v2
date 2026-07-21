"""Configuration loading. Implements the "no magic numbers" rule from CLAUDE.md.

Every threshold in this system is read from ``config/*.yaml`` through here.
Nothing else in the codebase may hardcode a trading number.

Secrets come from ``.env`` (§0.4) and are exposed only via :func:`get_secrets`;
they are never merged into the YAML settings tree, so a settings dump can be
logged safely.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"


class ConfigError(RuntimeError):
    """Raised when configuration is missing, malformed, or self-contradictory."""


# ---------------------------------------------------------------------------
# Settings tree
# ---------------------------------------------------------------------------


class Settings(Mapping[str, Any]):
    """Read-only view over the merged YAML settings with dotted-path access.

    ``settings.get("risk.vetoes.circuit_band_proximity_pct")`` beats
    ``settings["risk"]["vetoes"][...]`` because a typo raises a clear
    :class:`ConfigError` naming the path instead of a bare ``KeyError``.
    """

    def __init__(self, data: dict[str, Any], source: str = "<memory>") -> None:
        self._data = data
        self.source = source

    # -- Mapping protocol -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"Settings(source={self.source!r}, top_level={sorted(self._data)})"

    # -- Dotted access ----------------------------------------------------
    _MISSING = object()

    def get(self, path: str, default: Any = _MISSING) -> Any:
        """Fetch a dotted path. Without a default, a missing path is an error."""
        node: Any = self._data
        walked: list[str] = []
        for part in path.split("."):
            walked.append(part)
            if not isinstance(node, Mapping) or part not in node:
                if default is not Settings._MISSING:
                    return default
                raise ConfigError(
                    f"Missing config key {'.'.join(walked)!r} "
                    f"(looking up {path!r} in {self.source})"
                )
            node = node[part]
        return node

    def require(self, path: str) -> Any:
        """Alias for :meth:`get` with no default. Reads better at call sites."""
        return self.get(path)

    def section(self, path: str) -> "Settings":
        """Return a sub-tree as its own :class:`Settings`."""
        node = self.get(path)
        if not isinstance(node, Mapping):
            raise ConfigError(f"Config path {path!r} is not a section (got {type(node).__name__})")
        return Settings(dict(node), source=f"{self.source}:{path}")

    def as_dict(self) -> dict[str, Any]:
        """Deep-ish copy for callers that want to mutate (e.g. tests)."""
        import copy

        return copy.deepcopy(self._data)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} must contain a YAML mapping at the top level")
    return data


@lru_cache(maxsize=None)
def get_settings(config_dir: str | None = None) -> Settings:
    """Load ``settings.yaml``. Cached; call :func:`reset_config_cache` in tests."""
    directory = Path(config_dir) if config_dir else CONFIG_DIR
    return Settings(_load_yaml(directory / "settings.yaml"), source=str(directory / "settings.yaml"))


@lru_cache(maxsize=None)
def get_universe(config_dir: str | None = None) -> Settings:
    """Load ``universe.yaml``."""
    directory = Path(config_dir) if config_dir else CONFIG_DIR
    return Settings(_load_yaml(directory / "universe.yaml"), source=str(directory / "universe.yaml"))


@lru_cache(maxsize=None)
def get_events(config_dir: str | None = None) -> Settings:
    """Load ``events.yaml`` -- the owner-maintained blocked-date list (§3)."""
    directory = Path(config_dir) if config_dir else CONFIG_DIR
    return Settings(_load_yaml(directory / "events.yaml"), source=str(directory / "events.yaml"))


def reset_config_cache() -> None:
    """Drop all cached config. Tests call this after writing temp YAML."""
    get_settings.cache_clear()
    get_universe.cache_clear()
    get_events.cache_clear()


# ---------------------------------------------------------------------------
# Universe resolution
# ---------------------------------------------------------------------------

def resolve_universe(name: str, config_dir: str | None = None) -> list[str]:
    """Resolve a universe name (``"NIFTY50"``, ``"NIFTY200"``, ...) to symbols.

    Composition per ``universe.yaml``:
      NIFTY100 = nifty50 + nifty_next_50
      NIFTY200 = NIFTY100 + nifty_midcap_100
      NIFTY500 = explicit list, or resolved from the instruments cache at
                 runtime when ``nifty500_from_instruments`` is true.
    """
    uni = get_universe(config_dir)
    key = name.upper().replace("-", "").replace(" ", "")

    n50: list[str] = list(uni.get("nifty50", []))
    nn50: list[str] = list(uni.get("nifty_next_50", []))
    mid100: list[str] = list(uni.get("nifty_midcap_100", []))

    if key == "NIFTY50":
        return n50
    if key == "NIFTY100":
        return _dedupe(n50 + nn50)
    if key == "NIFTY200":
        return _dedupe(n50 + nn50 + mid100)
    if key == "NIFTY500":
        explicit = list(uni.get("nifty500_explicit", []))
        if explicit:
            return _dedupe(explicit)
        if uni.get("nifty500_from_instruments", False):
            # Resolved by the datafeed against the instruments cache. Until
            # Phase 2 populates that cache the honest answer is NIFTY200 --
            # but callers must know they got a narrower list, so we say so.
            raise ConfigError(
                "NIFTY500 is configured as `nifty500_from_instruments: true`; "
                "resolve it via core.datafeed.resolve_nifty500() which reads the "
                "instruments cache, not via resolve_universe()."
            )
        raise ConfigError("NIFTY500 requested but neither explicit list nor instruments flag set")
    raise ConfigError(f"Unknown universe {name!r}")


def _dedupe(items: Iterable[str]) -> list[str]:
    """Order-preserving dedupe."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# Secrets (§0.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Secrets:
    """Values from ``.env``. Never logged, never journalled, never in YAML."""

    kite_api_key: str | None
    kite_api_secret: str | None
    kite_access_token: str | None
    kite_user_id: str | None
    anthropic_api_key: str | None
    telegram_bot_token: str | None
    telegram_chat_id: str | None
    execution_mode_override: str | None

    def missing(self, required: Iterable[str]) -> list[str]:
        """Names among ``required`` that are unset or blank."""
        return [name for name in required if not getattr(self, name, None)]

    def __repr__(self) -> str:  # pragma: no cover - defensive, never show values
        present = [f.name for f in self.__dataclass_fields__.values() if getattr(self, f.name)]
        return f"Secrets(set={sorted(present)})"


def load_dotenv(path: Path | None = None) -> None:
    """Load ``.env`` into ``os.environ`` without overwriting real env vars."""
    env_path = path or (REPO_ROOT / ".env")
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Real environment wins, so `EXECUTION_MODE=live python -m ...` works.
        os.environ.setdefault(key, value)


def get_secrets(path: Path | None = None) -> Secrets:
    """Read secrets from the environment, loading ``.env`` first."""
    load_dotenv(path)

    def env(name: str) -> str | None:
        value = os.environ.get(name, "").strip()
        return value or None

    return Secrets(
        kite_api_key=env("KITE_API_KEY"),
        kite_api_secret=env("KITE_API_SECRET"),
        kite_access_token=env("KITE_ACCESS_TOKEN"),
        kite_user_id=env("KITE_USER_ID"),
        anthropic_api_key=env("ANTHROPIC_API_KEY"),
        telegram_bot_token=env("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=env("TELEGRAM_CHAT_ID"),
        execution_mode_override=env("EXECUTION_MODE"),
    )


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def data_path(*parts: str) -> Path:
    """Absolute path under the configured data dir, creating parents."""
    settings = get_settings()
    root = REPO_ROOT / str(settings.get("storage.data_dir", "data"))
    path = root.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def repo_path(*parts: str) -> Path:
    """Absolute path relative to the repository root."""
    return REPO_ROOT.joinpath(*parts)
