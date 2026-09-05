from mercury import MercuryHarness
from mercury.demo import frontier_auth_fix, frontier_pytest_fix, lesser_auth_fail
from mercury.models import CardKind
from mercury.tiers import ModelTier


def test_flywheel_embeds_frontier_how_into_lesser_pack(tmp_path):
    harness = MercuryHarness.init(tmp_path / "store")
    harness.capture(frontier_auth_fix())
    harness.capture(frontier_pytest_fix())
    harness.contrast(lesser_auth_fail(), frontier_auth_fix())

    stats = harness.stats()
    assert stats["traces"] >= 2
    assert stats["cards"] >= 6
    assert stats["frontier_traces"] >= 2

    pack = harness.pack(
        "Users bounce back to /login after authenticating. Fix the redirect bug.",
        model="gpt-4o-mini",
        languages=["typescript"],
    )
    assert pack.tier is ModelTier.LESSER
    assert pack.cards
    kinds = {card.kind for card in pack.cards}
    rendered = pack.render().lower()
    assert "operating pack" in rendered
    assert "grep" in rendered
    assert CardKind.PLAYBOOK in kinds or CardKind.CONTRAST in kinds or CardKind.TOOL_POLICY in kinds
    assert "login" in rendered or "session" in rendered or "redirect" in rendered
    playbooks = [card for card in pack.cards if card.kind is CardKind.PLAYBOOK]
    if playbooks:
        top = playbooks[0].situation.lower()
        assert "login" in top or "redirect" in top
    assert "login.tsx" in rendered
    assert "session.ts" in rendered


def test_frontier_student_gets_empty_pack(tmp_path):
    harness = MercuryHarness.init(tmp_path / "store")
    harness.capture(frontier_auth_fix())
    pack = harness.pack("fix login redirect", model="claude-opus-4.1")
    assert pack.cards == []
    assert pack.token_budget == 0


def test_lesser_success_does_not_teach_by_default(tmp_path):
    from mercury.models import OutcomeStatus, TraceOutcome
    from mercury.demo import lesser_auth_fail

    harness = MercuryHarness.init(tmp_path / "store")
    student = lesser_auth_fail().model_copy(
        update={"outcome": TraceOutcome(status=OutcomeStatus.SUCCESS, summary="lucky")}
    )
    harness.capture(student)
    kinds = {card.kind.value for card in harness.store.cards()}
    assert "playbook" not in kinds


def test_cursor_rule_and_finetune_export(tmp_path):
    harness = MercuryHarness.init(tmp_path / "store")
    harness.capture(frontier_auth_fix())
    pack = harness.pack("fix the login redirect loop", model="haiku")
    rule = pack.as_cursor_rule()
    assert rule.startswith("---")
    assert "alwaysApply: true" in rule
    rows = pack.as_finetune_rows()
    assert rows
    assert "input" in rows[0] and "output" in rows[0]


def test_mid_run_error_signature_retrieves_recovery(tmp_path):
    harness = MercuryHarness.init(tmp_path / "store")
    harness.capture(frontier_auth_fix())
    pack = harness.pack(
        "still failing after an edit",
        model="gpt-4o-mini",
        error_signature="AssertionError: expected '/login' to be '/dashboard'",
    )
    blob = pack.render()
    assert pack.cards
    assert "Recover" in blob or "recover" in blob.lower() or "Assertion" in blob or "dashboard" in blob.lower()
