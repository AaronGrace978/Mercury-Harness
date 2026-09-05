from mercury.fingerprint import classify_task, error_signature, looks_like_error
from mercury.models import TaskType
from mercury.traceio import parse_trace


def test_canonical_and_openai_parse():
    canonical = parse_trace(
        {
            "model": "claude-opus-4.1",
            "task": "Fix the login bug",
            "outcome": "success",
            "events": [
                {"type": "user", "content": "Fix the login bug"},
                {
                    "type": "assistant",
                    "tool_calls": [{"name": "grep", "arguments": {"pattern": "login", "path": "src/auth.ts"}}],
                },
                {"type": "tool", "tool_name": "grep", "content": "src/auth.ts:1: login"},
            ],
        }
    )
    assert canonical.files_touched == ["src/auth.ts"]
    assert "typescript" in canonical.languages or canonical.files_touched

    openai = parse_trace(
        {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "Fix the login bug"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read",
                                "arguments": '{"path": "src/auth.ts"}',
                            }
                        }
                    ],
                },
                {"role": "tool", "name": "read", "content": "export function login() {}"},
            ],
            "outcome": {"status": "failure"},
        }
    )
    assert openai.task.startswith("Fix the login")
    assert openai.tool_sequence() == ["read"]
    assert openai.outcome.failed


def test_task_and_error_fingerprints():
    assert classify_task("Fix the crash in auth") is TaskType.BUGFIX
    assert classify_task("Login keeps redirecting back to /login") is TaskType.BUGFIX
    assert classify_task("Add a billing export feature") is TaskType.FEATURE
    sig = error_signature("AssertionError: expected 12 to be 13\nFAILED tests/foo.py")
    assert "assert" in sig or "fail" in sig
    assert "12" not in sig
    assert looks_like_error("Traceback (most recent call last):\nError: boom")
