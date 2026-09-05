"""Task fingerprints used for retrieval and distillation."""

from __future__ import annotations

import re
from collections import Counter

from mercury.models import AgentTrace, TaskType, TraceEvent

_TASK_RULES: tuple[tuple[TaskType, tuple[str, ...]], ...] = (
    (TaskType.BUGFIX, ("bug", "fix", "error", "fail", "crash", "broken", "regression", "exception", "keeps", "wrong", "unexpected", "loop")),
    (TaskType.FEATURE, ("add", "implement", "feature", "support", "introduce", "create")),
    (TaskType.REFACTOR, ("refactor", "cleanup", "clean up", "rename", "restructure")),
    (TaskType.TEST, ("test", "coverage", "spec", "assert")),
    (TaskType.DOCS, ("docs", "readme", "documentation", "comment")),
    (TaskType.REVIEW, ("review", "pr ", "pull request")),
)


def classify_task(task: str) -> TaskType:
    lowered = (task or "").lower()
    scores: Counter[TaskType] = Counter()
    for task_type, needles in _TASK_RULES:
        for needle in needles:
            if needle in lowered:
                scores[task_type] += 1
    if not scores:
        return TaskType.GENERAL
    return scores.most_common(1)[0][0]


_PATH_KEYS = ("path", "file", "file_path", "target_file", "filename", "uri")
_LANG_BY_EXT = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".md": "markdown",
    ".json": "json",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".css": "css",
    ".html": "html",
}


def extract_paths(event: TraceEvent) -> list[str]:
    paths: list[str] = []
    for call in event.tool_calls:
        for key in _PATH_KEYS:
            value = call.arguments.get(key)
            if isinstance(value, str) and _looks_like_path(value):
                paths.append(value)
        query = call.arguments.get("query") or call.arguments.get("pattern")
        if isinstance(query, str):
            for match in re.findall(r"[\w./-]+\.[A-Za-z0-9]{1,8}", query):
                if _looks_like_path(match):
                    paths.append(match)
    if event.content:
        for match in re.findall(r"[\w./-]+\.[A-Za-z0-9]{1,8}", event.content):
            if _looks_like_path(match) and "/" in match:
                paths.append(match)
    return paths


def _looks_like_path(value: str) -> bool:
    if len(value) < 3 or " " in value.strip():
        return False
    return "/" in value or any(value.endswith(ext) for ext in _LANG_BY_EXT)


def languages_from_paths(paths: list[str]) -> list[str]:
    found: list[str] = []
    for path in paths:
        lower = path.lower()
        for ext, language in _LANG_BY_EXT.items():
            if lower.endswith(ext) and language not in found:
                found.append(language)
    return found


def enrich_trace(trace: AgentTrace) -> AgentTrace:
    """Fill files_touched / languages from tool arguments when missing."""
    if trace.files_touched and trace.languages:
        return trace
    paths: list[str] = []
    for event in trace.events:
        paths.extend(extract_paths(event))
    unique_paths: list[str] = []
    for path in paths:
        if path not in unique_paths:
            unique_paths.append(path)
    files = trace.files_touched or unique_paths
    langs = trace.languages or languages_from_paths(files)
    return trace.model_copy(update={"files_touched": files, "languages": langs})


def error_signature(text: str, limit: int = 180) -> str:
    """Normalize noisy tool output into a stable failure fingerprint."""
    if not text:
        return ""
    lowered = text.lower()
    # Prefer the exception / assertion line.
    lines = [line.strip() for line in lowered.splitlines() if line.strip()]
    interesting = [
        line
        for line in lines
        if any(
            token in line
            for token in (
                "error",
                "exception",
                "failed",
                "failure",
                "traceback",
                "assert",
                "fatal",
                "panic",
            )
        )
    ]
    source = " | ".join(interesting[:3] or lines[:2])
    source = re.sub(r"\b0x[0-9a-f]+\b", "<hex>", source)
    source = re.sub(r"\b\d+\b", "N", source)
    source = re.sub(r"\s+", " ", source).strip()
    return source[:limit]


def looks_like_error(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    hits = (
        "error",
        "exception",
        "traceback",
        "failed",
        "failure",
        "assert",
        "not found",
        "fatal",
        "panic",
        "exit code",
    )
    return any(token in lowered for token in hits)
