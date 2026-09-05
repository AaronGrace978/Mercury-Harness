from mercury.tiers import ModelTier, classify_model, is_student_tier, is_teacher_tier, pack_token_budget


def test_frontier_markers():
    assert classify_model("claude-opus-4.1") is ModelTier.FRONTIER
    assert classify_model("gpt-5") is ModelTier.FRONTIER
    assert classify_model("o3") is ModelTier.FRONTIER
    assert classify_model("cursor-grok-4.6-high") is ModelTier.FRONTIER


def test_lesser_wins_over_capable_substring():
    assert classify_model("gpt-4o-mini") is ModelTier.LESSER
    assert classify_model("claude-haiku-4") is ModelTier.LESSER
    assert classify_model("gemini-2.0-flash") is ModelTier.LESSER


def test_capable_and_unknown():
    assert classify_model("claude-sonnet-4") is ModelTier.CAPABLE
    assert classify_model("gpt-4o") is ModelTier.CAPABLE
    assert classify_model("") is ModelTier.UNKNOWN
    assert classify_model("mystery-model") is ModelTier.UNKNOWN


def test_teacher_student_and_budget():
    assert is_teacher_tier(ModelTier.FRONTIER)
    assert not is_teacher_tier(ModelTier.LESSER)
    assert is_student_tier(ModelTier.LESSER)
    assert pack_token_budget(ModelTier.FRONTIER) == 0
    assert pack_token_budget(ModelTier.LESSER) < pack_token_budget(ModelTier.CAPABLE)
