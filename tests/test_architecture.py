"""Architectural law tests — §0.3 and §0.8.

These are not unit tests of behaviour; they are guard rails. §0.3 says "no
engine calls the broker directly -- every order passes through
``core/risk.py::check()``". A rule that lives only in a document gets broken.
This file makes breaking it fail the build.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = REPO_ROOT / "engines"
CORE_DIR = REPO_ROOT / "core"


def _python_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*.py") if p.name != "__init__.py")


def _imported_modules(path: Path) -> set[str]:
    """Every module name imported by a file, including ``from x import y``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


class TestEngineIsolation:
    """§0.3: engines emit Signals; they never touch the broker."""

    def test_no_engine_imports_the_broker(self):
        offenders = []
        for path in _python_files(ENGINES_DIR):
            if any(m == "core.broker" or m.startswith("core.broker.")
                   for m in _imported_modules(path)):
                offenders.append(path.relative_to(REPO_ROOT))
        assert not offenders, (
            f"§0.3 violation: these engines import core.broker directly: {offenders}. "
            f"Engines return Signals; the orchestrator routes them through core.risk.check()."
        )

    def test_no_engine_imports_kiteconnect(self):
        offenders = []
        for path in _python_files(ENGINES_DIR):
            if any(m.startswith("kiteconnect") for m in _imported_modules(path)):
                offenders.append(path.relative_to(REPO_ROOT))
        assert not offenders, f"§0.3 violation: engines importing kiteconnect: {offenders}"

    def test_no_engine_calls_place_order(self):
        offenders = []
        for path in _python_files(ENGINES_DIR):
            source = path.read_text(encoding="utf-8")
            if re.search(r"\.place_order\s*\(", source):
                offenders.append(path.relative_to(REPO_ROOT))
        assert not offenders, f"§0.3 violation: engines calling place_order: {offenders}"


class TestSecretsHygiene:
    """§0.4: secrets live in .env only."""

    SECRET_PATTERNS = [
        re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"),
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
        re.compile(r"\b\d{9,10}:[A-Za-z0-9_\-]{30,}\b"),   # telegram bot token
    ]

    def test_no_hardcoded_secrets_in_source_or_config(self):
        offenders = []
        targets = list(_python_files(REPO_ROOT / "core"))
        targets += list(_python_files(REPO_ROOT / "engines"))
        targets += list(_python_files(REPO_ROOT / "live"))
        targets += list(_python_files(REPO_ROOT / "scripts"))
        targets += list((REPO_ROOT / "config").rglob("*.yaml"))
        for path in targets:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in self.SECRET_PATTERNS:
                if pattern.search(text):
                    offenders.append(f"{path.relative_to(REPO_ROOT)} ~ {pattern.pattern}")
        assert not offenders, f"§0.4 violation: possible secret in tracked files: {offenders}"

    def test_env_is_gitignored(self):
        ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert re.search(r"^\.env$", ignored, re.MULTILINE)

    def test_env_example_has_no_real_values(self):
        example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
        for line in example.splitlines():
            if line.startswith("KITE_API_KEY="):
                assert line.strip() == "KITE_API_KEY=your_kite_api_key"
            # Every other credential line must be empty or a placeholder.
            for key in ("ANTHROPIC_API_KEY=", "TELEGRAM_BOT_TOKEN=",
                        "TELEGRAM_CHAT_ID=", "KITE_ACCESS_TOKEN="):
                if line.startswith(key):
                    assert line.strip() == key, f"{key} has a value in .env.example"


class TestNoPlaceholders:
    """§0.8: no `TODO: implement later`, no stub returning fake success."""

    FORBIDDEN = re.compile(
        r"#\s*(TODO|FIXME|XXX)\b.*(implement|later|stub|placeholder)", re.IGNORECASE
    )

    def test_no_placeholder_markers(self):
        offenders = []
        for directory in ("core", "engines", "live", "backtest", "scripts"):
            for path in _python_files(REPO_ROOT / directory):
                for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if self.FORBIDDEN.search(line):
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}")
        assert not offenders, f"§0.8 violation: placeholder markers found: {offenders}"

    def test_bare_pass_bodies_are_not_used_as_stubs(self):
        """A function whose entire body is `pass` is a silent stub (§0.8).

        Abstract methods and Protocol members are exempt -- those are contracts,
        not stubs.
        """
        offenders = []
        for directory in ("core", "engines", "live", "backtest"):
            for path in _python_files(REPO_ROOT / directory):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    body = [n for n in node.body if not isinstance(n, ast.Expr)
                            or not isinstance(n.value, ast.Constant)]
                    if len(body) == 1 and isinstance(body[0], ast.Pass):
                        decorators = {
                            getattr(d, "attr", getattr(d, "id", "")) for d in node.decorator_list
                        }
                        if "abstractmethod" not in decorators:
                            offenders.append(
                                f"{path.relative_to(REPO_ROOT)}:{node.lineno} {node.name}"
                            )
        assert not offenders, f"§0.8 violation: stub functions with `pass` bodies: {offenders}"


class TestNoMagicNumbers:
    """CLAUDE.md: thresholds live in settings.yaml, not in code."""

    def test_risk_kernel_reads_limits_from_config(self):
        source = (CORE_DIR / "risk.py").read_text(encoding="utf-8")
        # The spec's defaults must not appear as literals in the kernel.
        for literal in ("800000", "800_000", "1.5", "3.0"):
            assert f"= {literal}" not in source, (
                f"risk.py contains the literal {literal}; it belongs in settings.yaml"
            )

    def test_calendar_does_not_hardcode_an_expiry_weekday(self):
        """§8.3: do not hardcode expiry weekdays."""
        source = (CORE_DIR / "calendar.py").read_text(encoding="utf-8")
        for pattern in (r"weekday\(\)\s*==\s*[0-6]", r"\.weekday\s*=\s*[0-6]"):
            assert not re.search(pattern, source), (
                f"calendar.py hardcodes a weekday ({pattern}); §8.3 forbids it"
            )


class TestTimezoneDiscipline:
    """CLAUDE.md: use core.clock.now_ist(), never a bare datetime.now()."""

    def test_no_bare_datetime_now(self):
        offenders = []
        for directory in ("core", "engines", "live", "backtest"):
            for path in _python_files(REPO_ROOT / directory):
                if path.name == "clock.py":
                    continue
                source = path.read_text(encoding="utf-8")
                for lineno, line in enumerate(source.splitlines(), 1):
                    if re.search(r"datetime\.now\(\s*\)", line):
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}")
        assert not offenders, (
            f"Naive datetime.now() found: {offenders}. Use core.clock.now_ist() "
            f"so §0.6 (IST timestamps) holds and tests can freeze time."
        )


class TestRiskKernelIsTheOnlyGate:
    """§0.3: the orchestrator must route through core.risk before the broker."""

    def test_orchestrator_imports_risk(self):
        source = (REPO_ROOT / "live" / "orchestrator.py").read_text(encoding="utf-8")
        assert "core.risk" in source, "the orchestrator must go through the risk kernel"
