from mercury.demo import frontier_auth_fix, lesser_auth_fail
from mercury.grade import blind_retry, grade_trace


def test_frontier_trace_scores_high():
    report = grade_trace(frontier_auth_fix())
    passed = set(report.passed_names)
    assert "explored_first" in passed
    assert "read_before_edit" in passed
    assert "verified_after_edit" in passed
    assert "recovered_after_error" in passed
    assert "outcome_recorded" in passed
    assert report.score >= 0.8


def test_lesser_trace_fails_policy_checks():
    report = grade_trace(lesser_auth_fail())
    failed = set(report.failed_names)
    passed = set(report.passed_names)
    assert "explored_first" in failed
    assert "read_before_edit" in failed
    # The fixture did run a test after editing and changed its second edit,
    # so the grader must NOT punish it for those.
    assert "verified_after_edit" in passed
    assert "recovered_after_error" in passed
    assert report.score <= 0.7


def test_blind_retry_failure_emits_anti_pattern_card():
    from mercury.distill import distill_trace

    cards = distill_trace(_retry_trace("y"), teacher=False)
    retry = [card for card in cards if "identical edit" in card.title]
    assert retry, "expected a blind-retry anti-pattern card"
    assert retry[0].kind.value == "anti_pattern"


def _retry_trace(new_string_two: str):
    from mercury.models import (
        AgentTrace,
        EventType,
        OutcomeStatus,
        ToolCall,
        TraceEvent,
        TraceOutcome,
    )

    return AgentTrace(
        id="t",
        model="gpt-4o-mini",
        task="retry loop",
        outcome=TraceOutcome(status=OutcomeStatus.FAILURE),
        events=[
            TraceEvent(
                type=EventType.ASSISTANT,
                tool_calls=[ToolCall(name="search_replace", arguments={"path": "a.py", "old_string": "x", "new_string": "y"})],
            ),
            TraceEvent(type=EventType.TOOL, tool_name="search_replace", is_error=True, content="error"),
            TraceEvent(
                type=EventType.ASSISTANT,
                tool_calls=[ToolCall(name="search_replace", arguments={"path": "a.py", "old_string": "x", "new_string": new_string_two})],
            ),
        ],
    )


def test_blind_retry_detects_unchanged_rerun():
    trace = _retry_trace("y")
    hit = blind_retry(trace)
    assert hit is not None
    assert "identical edit" in hit[0]
    report = grade_trace(trace)
    assert "no_blind_retry" in report.failed_names


def test_blind_retry_allows_changed_rerun():
    trace = _retry_trace("z")
    assert blind_retry(trace) is None
    report = grade_trace(trace)
    assert "no_blind_retry" in report.passed_names


def test_grade_is_deterministic():
    first = grade_trace(frontier_auth_fix())
    second = grade_trace(frontier_auth_fix())
    assert first.as_dict() == second.as_dict()


def test_grade_report_json_shape():
    report = grade_trace(lesser_auth_fail())
    data = report.as_dict()
    assert set(data) == {"model", "task", "score", "passed", "failed", "checks"}
    assert 0.0 <= data["score"] <= 1.0
