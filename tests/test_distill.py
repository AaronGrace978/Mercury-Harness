from mercury.demo import frontier_auth_fix, lesser_auth_fail
from mercury.distill import distill_standing_orders, distill_trace
from mercury.models import CardKind
from mercury.phases import first_tool, segment_phases


def test_phase_segmentation_and_first_tool():
    trace = frontier_auth_fix()
    names = [phase.name for phase in segment_phases(trace)]
    assert names[0] == "explore"
    assert "edit" in names
    assert "verify" in names
    assert first_tool(trace) == "grep"


def test_playbook_and_recovery_from_frontier_success():
    cards = distill_trace(frontier_auth_fix(), teacher=True)
    kinds = {card.kind for card in cards}
    assert CardKind.PLAYBOOK in kinds
    assert CardKind.TOOL_POLICY in kinds
    assert CardKind.RECOVERY in kinds
    recovery = next(card for card in cards if card.kind is CardKind.RECOVERY)
    assert recovery.error_signature
    assert "search_replace" in recovery.procedure or "session" in recovery.procedure.lower() or "Patch" in recovery.procedure or "patch" in recovery.procedure.lower()
    playbook = next(card for card in cards if card.kind is CardKind.PLAYBOOK)
    assert "Explore" in playbook.procedure or "explore" in playbook.procedure.lower()
    first_action = next(card for card in cards if card.title.startswith("First action"))
    assert "grep" in first_action.procedure


def test_lesser_failure_does_not_emit_playbook_when_not_teacher():
    cards = distill_trace(lesser_auth_fail(), teacher=False)
    assert all(card.kind is not CardKind.PLAYBOOK for card in cards)
    assert any(card.kind is CardKind.ANTI_PATTERN for card in cards)


def test_standing_orders_need_two_successful_traces():
    from mercury.demo import frontier_pytest_fix

    assert distill_standing_orders([frontier_auth_fix()]) == []
    orders = distill_standing_orders([frontier_auth_fix(), frontier_pytest_fix()])
    titles = {card.title for card in orders}
    assert any("Search before" in title for title in titles)
    assert any("Verify" in title for title in titles)
