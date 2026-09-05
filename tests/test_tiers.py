from mercury.tiers import ModelTier, classify_model, is_student_tier, is_teacher_tier, pack_token_budget


def test_frontier_markers():
    assert classify_model("claude-opus-4.1") is ModelTier.FRONTIER
    assert classify_model("claude-fable-5") is ModelTier.FRONTIER
    assert classify_model("gpt-5") is ModelTier.FRONTIER
    assert classify_model("gpt-5.6-sol") is ModelTier.FRONTIER
    assert classify_model("o3") is ModelTier.FRONTIER
    assert classify_model("cursor-grok-4.6-high") is ModelTier.FRONTIER
    assert classify_model("grok-4.5") is ModelTier.FRONTIER


def test_ollama_cloud_frontier():
    assert classify_model("deepseek-v4-pro") is ModelTier.FRONTIER
    assert classify_model("deepseek-v4-pro:0813") is ModelTier.FRONTIER
    assert classify_model("kimi-k3:cloud") is ModelTier.FRONTIER
    assert classify_model("kimi-k2.7-code") is ModelTier.FRONTIER
    assert classify_model("kimi-k2.6") is ModelTier.FRONTIER
    assert classify_model("glm-5.3") is ModelTier.FRONTIER
    assert classify_model("glm-5.2") is ModelTier.FRONTIER
    assert classify_model("glm-5.1") is ModelTier.FRONTIER
    assert classify_model("minimax-m3:cloud") is ModelTier.FRONTIER
    assert classify_model("minimax-m2.7") is ModelTier.FRONTIER
    assert classify_model("mistral-large-3:675b") is ModelTier.FRONTIER
    assert classify_model("gpt-oss:120b-cloud") is ModelTier.FRONTIER
    assert classify_model("nemotron-3-ultra") is ModelTier.FRONTIER
    assert classify_model("qwen3.5:397b") is ModelTier.FRONTIER


def test_ollama_cloud_capable_and_lesser():
    assert classify_model("gpt-oss:20b-cloud") is ModelTier.CAPABLE
    assert classify_model("gemma4:31b") is ModelTier.CAPABLE
    assert classify_model("nemotron-3-super") is ModelTier.CAPABLE
    assert classify_model("qwen3.5:122b") is ModelTier.CAPABLE
    assert classify_model("deepseek-v4-flash:0731") is ModelTier.LESSER
    assert classify_model("glm-5.3-flash") is ModelTier.LESSER
    assert classify_model("nemotron-3-nano:30b") is ModelTier.LESSER


def test_recent_efficiency_tiers():
    assert classify_model("gpt-5.6-terra") is ModelTier.CAPABLE
    assert classify_model("gpt-5.6-luna") is ModelTier.LESSER
    assert classify_model("gemini-3.5-flash") is ModelTier.LESSER
    assert classify_model("claude-sonnet-5") is ModelTier.CAPABLE


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
