"""Engine registry: name -> class.

A single place the orchestrator, the backtest CLI and ``load_engines`` all read
from, so adding an engine means editing one dict rather than three call sites.
"""

from __future__ import annotations

from typing import Type

from engines.base import Engine
from engines.filings import FilingsEngine
from engines.flows import FlowsEngine
from engines.overnight import OvernightEngine
from engines.pairs import PairsEngine
from engines.panic_reversion import PanicReversionEngine
from engines.pead import PeadEngine
from engines.preopen import PreopenEngine
from engines.special_situations import SpecialSituationsEngine
from engines.surveillance import SurveillanceEngine
from engines.sympathy import SympathyEngine
from engines.wheel import WheelEngine

# Insertion order is the §5.4 build order, which is also a dependency order:
# sympathy consumes filings' output, and pairs and panic_reversion query the
# filings database before they trade.
ENGINE_CLASSES: dict[str, Type[Engine]] = {
    "filings": FilingsEngine,
    "pairs": PairsEngine,
    "overnight": OvernightEngine,
    "preopen": PreopenEngine,
    "pead": PeadEngine,
    "panic_reversion": PanicReversionEngine,
    "wheel": WheelEngine,
    "sympathy": SympathyEngine,
    "flows": FlowsEngine,
    "surveillance": SurveillanceEngine,
    "special_situations": SpecialSituationsEngine,
}

# §6.10 and §6.11 are alert-only structurally, not by configuration.
ALERT_ONLY = {"surveillance", "special_situations"}

# Tick-driven engines. Everything else is scheduled.
TICK_DRIVEN: set[str] = set()
