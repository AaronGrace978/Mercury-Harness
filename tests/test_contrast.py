from mercury.contrast import contrast_traces
from mercury.demo import frontier_auth_fix, lesser_auth_fail
from mercury.models import CardKind


def test_contrast_emits_first_action_and_file_divergence():
    cards = contrast_traces(lesser_auth_fail(), frontier_auth_fix())
    kinds = {card.kind for card in cards}
    assert CardKind.CONTRAST in kinds
    blob = "\n".join(card.procedure for card in cards).lower()
    assert "search_replace" in blob
    assert "grep" in blob
    assert "login.tsx" in blob
    assert "session.ts" in blob or "auth.ts" in blob or "middleware" in blob
