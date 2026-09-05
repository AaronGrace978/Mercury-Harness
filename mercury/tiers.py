"""Model-tier classification: who teaches, who receives."""

from __future__ import annotations

from enum import Enum


class ModelTier(str, Enum):
    FRONTIER = "frontier"
    CAPABLE = "capable"
    LESSER = "lesser"
    UNKNOWN = "unknown"


# Explicit student overrides for flash/lite/nano/efficiency tiers of otherwise-strong families.
# Checked before frontier so e.g. gemini-3.5-flash and glm-5.3-flash receive packs.
_LESSER_OVERRIDES: tuple[str, ...] = (
    "glm-5.3-flash",
    "deepseek-v4-flash",
    "gpt-5.6-luna",
    "gpt-5-luna",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash",
    "gemini-3-flash",
    "nemotron-3-nano",
)

# Longer / more specific tokens first. Matching is case-insensitive substring.
_FRONTIER_MARKERS: tuple[str, ...] = (
    # Anthropic
    "claude-fable",
    "fable-5",
    "claude-opus",
    "opus-4",
    "opus-5",
    "claude-4-opus",
    "claude-4.1-opus",
    # OpenAI
    "gpt-5.6-sol",
    "gpt-5-sol",
    "gpt-5",
    "o3-pro",
    "o3",
    "o4",
    "o1-pro",
    "o1",
    "gpt-4.5",
    "gpt-4.1",
    # Google
    "gemini-ultra",
    "gemini-3.1-pro",
    "gemini-3-pro",
    "gemini-3.1",
    "gemini-3",
    # xAI
    "cursor-grok-4",
    "grok-4",
    "grok-3",
    # Ollama Cloud — flagship open models
    "deepseek-v4-pro",
    "kimi-k3",
    "kimi-k2.7",
    "kimi-k2.6",
    "kimi-k2.5",
    "kimi-k2",
    "glm-5.3",
    "glm-5.2",
    "glm-5.1",
    "glm-5",
    "minimax-m3",
    "minimax-m2.7",
    "minimax-m2",  # before lesser "mini" substring
    "mistral-large-3",
    "mistral-large",
    "gpt-oss:120b",
    "gpt-oss-120b",
    "120b-cloud",
    "nemotron-3-ultra",
    "qwen3.5:397b",
    "qwen3.5-397b",
    "qwen3-coder:480b",
    "qwen3-coder-480b",
)

# Mid-tier models that must beat lesser substrings like "mini" / size tags.
_CAPABLE_OVERRIDES: tuple[str, ...] = (
    "gpt-5.6-terra",
    "gpt-5-terra",
    "gpt-oss:20b",
    "gpt-oss-20b",
    ":20b-cloud",
    "gemma4:31b",
    "gemma4-31b",
    "gemma-4-31",
    "nemotron-3-super",
)

_CAPABLE_MARKERS: tuple[str, ...] = (
    "claude-sonnet",
    "sonnet-5",
    "sonnet-4",
    "sonnet-3.7",
    "sonnet-3.5",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4",
    "gemini-pro",
    "gemini-2.5-pro",
    "gemini-2.0-pro",
    "claude-3.7",
    "claude-3.5",
    "composer-2",
    "deepseek-r1",
    "deepseek-v4",
    "deepseek-v3",
    # Ollama Cloud — strong mid-tier / open coding models
    "qwen3.5",
    "qwen3-coder",
    "qwen3",
    "devstral-2",
    "devstral",
    "cogito",
)

_LESSER_MARKERS: tuple[str, ...] = (
    "haiku",
    "gpt-4o-mini",
    "gpt-4-mini",
    "gpt-3.5",
    "mini",
    "flash",
    "nano",
    "tiny",
    "lite",
    "8b",
    "7b",
    "3b",
    "1b",
    "llama-3-8",
    "llama-3.1-8",
    "mistral-7",
    "phi-3",
    "qwen-7",
    "qwen2.5-7",
    "gemma-2-9",
    "gemma4:12b",
    "gemma4:4b",
    "gemma4:e2b",
    "gemma4:e4b",
    "grok-2-mini",
    "nemotron-3-nano",
    "ministral",
)


def classify_model(name: str | None) -> ModelTier:
    """Map a model id/name onto a teaching/receiving tier."""
    if not name:
        return ModelTier.UNKNOWN
    lowered = name.strip().lower()
    for marker in _LESSER_OVERRIDES:
        if marker in lowered:
            return ModelTier.LESSER
    for marker in _CAPABLE_OVERRIDES:
        if marker in lowered:
            return ModelTier.CAPABLE
    for marker in _FRONTIER_MARKERS:
        if marker in lowered:
            return ModelTier.FRONTIER
    for marker in _LESSER_MARKERS:
        if marker in lowered:
            return ModelTier.LESSER
    for marker in _CAPABLE_MARKERS:
        if marker in lowered:
            return ModelTier.CAPABLE
    return ModelTier.UNKNOWN


def is_teacher_tier(tier: ModelTier, teacher_tiers: tuple[ModelTier, ...] | None = None) -> bool:
    allowed = teacher_tiers or (ModelTier.FRONTIER,)
    return tier in allowed


def is_student_tier(tier: ModelTier, student_tiers: tuple[ModelTier, ...] | None = None) -> bool:
    allowed = student_tiers or (ModelTier.LESSER, ModelTier.CAPABLE)
    return tier in allowed


def pack_token_budget(tier: ModelTier) -> int:
    """Smaller students get a tighter operating pack so it actually fits."""
    if tier is ModelTier.LESSER:
        return 1400
    if tier is ModelTier.CAPABLE:
        return 2800
    if tier is ModelTier.FRONTIER:
        return 0
    return 1800
