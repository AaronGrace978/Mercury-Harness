from mercury.embed import BM25Index, HashingEmbedder, cosine


def test_hashing_embedder_is_stable_and_separates_topics():
    embedder = HashingEmbedder()
    a = embedder.embed("login redirect cookie samesite session")
    b = embedder.embed("login redirect cookie samesite session")
    c = embedder.embed("flaky pytest freeze time billing invoice")
    assert a == b
    assert cosine(a, b) > 0.99
    assert cosine(a, c) < cosine(a, b)


def test_bm25_ranks_the_matching_document_first():
    index = BM25Index()
    docs = [
        "explore grep read session cookie sameSite lax",
        "freeze time in pytest billing due date",
        "rewrite the readme for the marketing site",
    ]
    index.fit(docs)
    scores = index.score("cookie sameSite login session")
    assert scores.index(max(scores)) == 0
