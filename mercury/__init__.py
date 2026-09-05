"""Mercury Harness: embed frontier agent operations into lesser models."""

from mercury.harness import MercuryHarness
from mercury.inject import OperatingPack
from mercury.models import AgentTrace, OperationalCard, TraceEvent, TraceOutcome
from mercury.tiers import ModelTier, classify_model

__all__ = [
    "MercuryHarness",
    "OperatingPack",
    "AgentTrace",
    "OperationalCard",
    "TraceEvent",
    "TraceOutcome",
    "ModelTier",
    "classify_model",
]

__version__ = "0.1.0"
