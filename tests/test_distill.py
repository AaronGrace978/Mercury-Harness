from mercury.demo import frontier_auth_fix, lesser_auth_fail
from mercury.distill import distill_standing_orders, distill_trace
from mercury.models import CardKind
from mercury.phases import (
    first_tool,
    is_edit_tool,
    is_explore_tool,
    normalize_tool,
    segment_phases,
)


def test_phase_segmentation_and_first_tool():
    trace = frontier_auth_fix()
    names = [phase.name for phase in segment_phases(trace)]
    assert names[0] == "explore"
    assert "edit" in names
    assert "verify" in names
    assert first_tool(trace) == "grep"


def test_tool_normalization_cross_harness():
    assert normalize_tool("run_terminal_cmd") == "verify"
    assert normalize_tool("RunTerminalCmd") == "verify"
    assert normalize_tool("Bash") == "verify"
    assert normalize_tool("StrReplace") == "edit"
    assert normalize_tool("search_replace") == "edit"
    assert normalize_tool("apply_patch") == "edit"
    assert normalize_tool("codebase_search") == "explore"
    assert normalize_tool("Glob") == "explore"
    assert normalize_tool("read_file") == "localize"
    assert is_explore_tool("rg")
    assert is_edit_tool("Write")
    assert is_edit_tool("multi_edit")


def test_playbook_and_recovery_from_frontier_success():
    cards = distill_trace(frontier_auth_fix(), teacher=True)
    kinds = {card.kind for card in cards}
    assert CardKind.PLAYBOOK in kinds
    assert CardKind.TOOL_POLICY in kinds
    assert CardKind.RECOVERY in kinds
    assert CardKind.DECISION in kinds
    recovery = next(card for card in cards if card.kind is CardKind.RECOVERY)
    assert recovery.error_signature
    assert "search_replace" in recovery.procedure or "session" in recovery.procedure.lower() or "Patch" in recovery.procedure or "patch" in recovery.procedure.lower()
    playbook = next(card for card in cards if card.kind is CardKind.PLAYBOOK)
    assert "Explore" in playbook.procedure or "explore" in playbook.procedure.lower()
    first_action = next(card for card in cards if card.title.startswith("First action"))
    assert "grep" in first_action.procedure
    decisions = [card for card in cards if card.kind is CardKind.DECISION]
    assert any(card.chose and card.rejected for card in decisions)
    open_decision = next(card for card in decisions if "evidence-gathering" in card.title.lower() or "first" in card.title.lower() or "Chose evidence" in card.title)
    assert "edit" in " ".join(open_decision.rejected).lower() or "patch" in " ".join(open_decision.rejected).lower()


def test_lesser_failure_does_not_emit_playbook_when_not_teacher():
    cards = distill_trace(lesser_auth_fail(), teacher=False)
    assert all(card.kind is not CardKind.PLAYBOOK for card in cards)
    assert any(card.kind is CardKind.ANTI_PATTERN for card in cards)


def test_standing_orders_cold_start_and_majority():
    from mercury.demo import frontier_pytest_fix

    seeds = distill_standing_orders([])
    assert len(seeds) >= 3
    assert all(card.metadata.get("seeded") for card in seeds)
    assert all(card.confidence < 0.5 for card in seeds)

    provisional = distill_standing_orders([frontier_auth_fix()])
    assert provisional
    assert all(card.metadata.get("provisional") for card in provisional)
    assert any("Search before" in card.title for card in provisional)

    confirmed = distill_standing_orders([frontier_auth_fix(), frontier_pytest_fix()])
    titles = {card.title for card in confirmed}
    assert any("Search before" in title for title in titles)
    assert any("Verify" in title for title in titles)
    assert all(not card.metadata.get("provisional") for card in confirmed)
    assert all(card.confidence >= 0.7 for card in confirmed)

    # Same title → same id so seed/provisional/majority upsert in the store.
    seed_ids = {card.id for card in seeds}
    confirmed_ids = {card.id for card in confirmed}
    assert seed_ids & confirmed_ids
