"""Built-in traces that demonstrate the frontier → lesser flywheel."""

from __future__ import annotations

from mercury.models import (
    AgentTrace,
    EventType,
    OutcomeStatus,
    ToolCall,
    TraceEvent,
    TraceOutcome,
)


def frontier_auth_fix() -> AgentTrace:
    """Successful Claude Opus run: search, localize, recover from a failed test, patch cookies."""
    return AgentTrace(
        id="trace_frontier_auth_redirect",
        model="claude-opus-4.1",
        task="Login keeps redirecting back to /login after a successful password check",
        files_touched=[
            "src/middleware/auth.ts",
            "src/lib/session.ts",
            "tests/auth.redirect.test.ts",
        ],
        languages=["typescript"],
        outcome=TraceOutcome(
            status=OutcomeStatus.SUCCESS,
            summary="Cookie SameSite was Strict on a cross-site callback; switched to Lax and tests passed.",
        ),
        events=[
            TraceEvent(type=EventType.USER, content="Login keeps redirecting back to /login after a successful password check"),
            TraceEvent(
                type=EventType.ASSISTANT,
                content="I will search for the redirect and session cookie path before editing anything.",
                tool_calls=[
                    ToolCall(name="grep", arguments={"pattern": "redirect.*login|setSession|SameSite", "glob": "src/**/*.{ts,tsx}"})
                ],
            ),
            TraceEvent(
                type=EventType.TOOL,
                tool_name="grep",
                content="src/middleware/auth.ts:41: return NextResponse.redirect('/login')\nsrc/lib/session.ts:18: sameSite: 'strict'",
            ),
            TraceEvent(
                type=EventType.ASSISTANT,
                tool_calls=[
                    ToolCall(name="read", arguments={"path": "src/middleware/auth.ts"}),
                    ToolCall(name="read", arguments={"path": "src/lib/session.ts"}),
                    ToolCall(name="read", arguments={"path": "tests/auth.redirect.test.ts"}),
                ],
            ),
            TraceEvent(type=EventType.TOOL, tool_name="read", content="middleware checks session cookie then redirects to /login if missing"),
            TraceEvent(type=EventType.TOOL, tool_name="read", content="session cookie sameSite: 'strict'"),
            TraceEvent(type=EventType.TOOL, tool_name="read", content="test expects login then /dashboard without bouncing to /login"),
            TraceEvent(
                type=EventType.ASSISTANT,
                content="Reproducing with the existing test before patching.",
                tool_calls=[ToolCall(name="shell", arguments={"command": "npx vitest run tests/auth.redirect.test.ts"})],
            ),
            TraceEvent(
                type=EventType.TOOL,
                tool_name="shell",
                is_error=True,
                content="FAIL tests/auth.redirect.test.ts\nAssertionError: expected '/login' to be '/dashboard'\nError: cookie not sent on callback",
            ),
            TraceEvent(
                type=EventType.ASSISTANT,
                content="The failing assertion is the cookie not surviving the callback. I will patch session.ts, not the login page.",
                tool_calls=[
                    ToolCall(
                        name="search_replace",
                        arguments={
                            "path": "src/lib/session.ts",
                            "old_string": "sameSite: 'strict'",
                            "new_string": "sameSite: 'lax'",
                        },
                    )
                ],
            ),
            TraceEvent(type=EventType.TOOL, tool_name="search_replace", content="updated src/lib/session.ts"),
            TraceEvent(
                type=EventType.ASSISTANT,
                tool_calls=[ToolCall(name="shell", arguments={"command": "npx vitest run tests/auth.redirect.test.ts"})],
            ),
            TraceEvent(type=EventType.TOOL, tool_name="shell", content="PASS tests/auth.redirect.test.ts"),
        ],
    )


def lesser_auth_fail() -> AgentTrace:
    """GPT-4o-mini run on the same bug: edits the login page first and fails."""
    return AgentTrace(
        id="trace_lesser_auth_redirect",
        model="gpt-4o-mini",
        task="Login keeps redirecting back to /login after a successful password check",
        files_touched=["src/pages/login.tsx"],
        languages=["typescript"],
        outcome=TraceOutcome(
            status=OutcomeStatus.FAILURE,
            summary="Removed the login redirect; tests still bounce because the cookie was never fixed.",
        ),
        events=[
            TraceEvent(type=EventType.USER, content="Login keeps redirecting back to /login after a successful password check"),
            TraceEvent(
                type=EventType.ASSISTANT,
                content="I will stop the login page from redirecting.",
                tool_calls=[
                    ToolCall(
                        name="search_replace",
                        arguments={
                            "path": "src/pages/login.tsx",
                            "old_string": "router.push('/login')",
                            "new_string": "router.push('/dashboard')",
                        },
                    )
                ],
            ),
            TraceEvent(type=EventType.TOOL, tool_name="search_replace", content="updated src/pages/login.tsx"),
            TraceEvent(
                type=EventType.ASSISTANT,
                tool_calls=[ToolCall(name="shell", arguments={"command": "npx vitest run tests/auth.redirect.test.ts"})],
            ),
            TraceEvent(
                type=EventType.TOOL,
                tool_name="shell",
                is_error=True,
                content="FAIL tests/auth.redirect.test.ts\nAssertionError: expected '/login' to be '/dashboard'",
            ),
            TraceEvent(
                type=EventType.ASSISTANT,
                content="Maybe remove the redirect entirely.",
                tool_calls=[
                    ToolCall(
                        name="search_replace",
                        arguments={"path": "src/pages/login.tsx", "old_string": "redirect('/login')", "new_string": ""},
                    )
                ],
            ),
            TraceEvent(type=EventType.TOOL, tool_name="search_replace", content="updated src/pages/login.tsx"),
        ],
    )


def frontier_pytest_fix() -> AgentTrace:
    """Second frontier trace so standing orders can form."""
    return AgentTrace(
        id="trace_frontier_flaky_test",
        model="claude-opus-4.1",
        task="Fix the flaky test that fails with AssertionError in test_billing.py",
        files_touched=["billing/invoice.py", "tests/test_billing.py"],
        languages=["python"],
        outcome=TraceOutcome(status=OutcomeStatus.SUCCESS, summary="Time was not frozen; froze clock in the test."),
        events=[
            TraceEvent(type=EventType.USER, content="Fix the flaky test that fails with AssertionError in test_billing.py"),
            TraceEvent(
                type=EventType.ASSISTANT,
                tool_calls=[ToolCall(name="grep", arguments={"pattern": "AssertionError|invoice", "glob": "tests/**/*.py"})],
            ),
            TraceEvent(type=EventType.TOOL, tool_name="grep", content="tests/test_billing.py:44: assert invoice.due == today"),
            TraceEvent(
                type=EventType.ASSISTANT,
                tool_calls=[
                    ToolCall(name="read", arguments={"path": "tests/test_billing.py"}),
                    ToolCall(name="read", arguments={"path": "billing/invoice.py"}),
                ],
            ),
            TraceEvent(type=EventType.TOOL, tool_name="read", content="test uses date.today() without freeze"),
            TraceEvent(type=EventType.TOOL, tool_name="read", content="invoice.due is computed from datetime.now()"),
            TraceEvent(
                type=EventType.ASSISTANT,
                tool_calls=[ToolCall(name="shell", arguments={"command": "pytest tests/test_billing.py -q"})],
            ),
            TraceEvent(
                type=EventType.TOOL,
                tool_name="shell",
                is_error=True,
                content="FAILED tests/test_billing.py::test_due_date AssertionError: datetime mismatch",
            ),
            TraceEvent(
                type=EventType.ASSISTANT,
                content="Actually I should freeze time in the test rather than changing production rounding.",
                tool_calls=[
                    ToolCall(
                        name="search_replace",
                        arguments={"path": "tests/test_billing.py", "old_string": "today = date.today()", "new_string": "today = freeze_time('2026-01-01').date()"},
                    )
                ],
            ),
            TraceEvent(type=EventType.TOOL, tool_name="search_replace", content="updated tests/test_billing.py"),
            TraceEvent(
                type=EventType.ASSISTANT,
                tool_calls=[ToolCall(name="shell", arguments={"command": "pytest tests/test_billing.py -q"})],
            ),
            TraceEvent(type=EventType.TOOL, tool_name="shell", content="1 passed"),
        ],
    )
