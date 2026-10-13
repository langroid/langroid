"""
Hermetic tests (no LLM calls, no network, no vector database) for the opt-in
retrieval score thresholds in `DocChatAgentConfig`:

- `bm25_score_threshold`: keep a BM25-retrieved chunk only if its score
  satisfies `score >= threshold`; the filter applies only when the
  threshold is > 0.0 (default 0.0 = no filtering).
- `fuzzy_score_threshold`: keep a fuzzy match only if its score satisfies
  `score > threshold` (strict), on the 0-100 scale; the default 50.0
  preserves the legacy hard-coded `score > 50` filter.

The tests drive `DocChatAgent.get_relevant_chunks()` — the retrieval path
where these thresholds are applied — using a minimal stand-in vector store
that returns no semantic-search results, so the BM25/fuzzy branches are
isolated deterministically.

Also covered: defensive handling of absent/null/odd runtime threshold
values — invalid BM25 thresholds deactivate the filter, while invalid or
non-finite fuzzy thresholds yield no fuzzy matches — and of odd runtime
BM25 *scores* (None/non-numeric/non-finite), which an active filter must
drop rather than crash on, and an inactive filter must pass through.

Semantic-query duplicates must retain their highest-scoring occurrence and
receive compact unique-document ranks before reciprocal rank fusion.
"""

from typing import Any, List, Optional, Tuple

import pytest

from langroid.agent.special.doc_chat_agent import DocChatAgent, DocChatAgentConfig
from langroid.language_models.mock_lm import MockLMConfig
from langroid.mytypes import DocMetaData, Document
from langroid.parsing.search import find_closest_matches_with_bm25
from langroid.vector_store.base import VectorStoreConfig


class _StubVecDB:
    """Minimal stand-in for a vector store.

    Provides only what `DocChatAgent.get_relevant_chunks` touches: a named
    existing collection and configurable semantic-search results (empty by
    default), without embeddings or network.
    """

    def __init__(
        self,
        collection_name: str = "test-retrieval-thresholds",
        semantic_results: Optional[dict[str, List[Tuple[Document, float]]]] = None,
    ) -> None:
        self.config = VectorStoreConfig(collection_name=collection_name)
        self.semantic_results = semantic_results or {}

    def list_collections(self, empty: bool = False) -> List[str]:
        return [str(self.config.collection_name)]

    def similar_texts_with_scores(
        self,
        text: str,
        k: int = 1,
        where: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        return self.semantic_results.get(text, [])[:k]


BM25_CONTENTS = [
    "Tigers are the largest cat species in the world.",
    "Lions are the second largest cat species in the world.",
    "The stock market fluctuated wildly this quarter.",
]

BM25_CONTENTS_CLEAN = [
    "tigers largest cat species world",
    "lions second largest cat species world",
    "stock market fluctuated wildly quarter",
]

FUZZY_CONTENTS = {
    # for the query "quantum entanglement", fuzzy partial-ratio scores are
    # ~100 ("high": contains the query verbatim), ~42 ("low"), ~15 ("noise")
    "high": "Quantum entanglement is a physical phenomenon.",
    "low": "Bananas are rich in potassium and easy to digest.",
    "noise": "zzz qqq xxx vvv www",
}


def _mk_docs(id2content: dict) -> List[Document]:
    return [
        Document(content=content, metadata=DocMetaData(id=id_))
        for id_, content in id2content.items()
    ]


def _make_agent(**config_kwargs) -> DocChatAgent:
    config = DocChatAgentConfig(
        llm=MockLMConfig(),
        vecdb=None,
        cross_encoder_reranking_model="",
        rerank_diversity=False,
        rerank_periphery=False,
        **config_kwargs,
    )
    agent = DocChatAgent(config)
    agent.vecdb = _StubVecDB()  # type: ignore[assignment]
    return agent


def _simple_preprocess_text(text: str) -> str:
    """Small deterministic cleaner for these hermetic retrieval tests."""
    return " ".join(word.strip(".,").lower() for word in text.split())


@pytest.fixture(autouse=True)
def _no_nltk_preprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep BM25 tests offline by avoiding NLTK resource downloads."""
    monkeypatch.setattr(
        "langroid.parsing.search.preprocess_text",
        _simple_preprocess_text,
    )


def _make_bm25_agent(
    bm25_score_threshold: float,
    use_reciprocal_rank_fusion: bool = False,
) -> DocChatAgent:
    agent = _make_agent(
        use_bm25_search=True,
        use_fuzzy_match=False,
        bm25_score_threshold=bm25_score_threshold,
        use_reciprocal_rank_fusion=use_reciprocal_rank_fusion,
    )
    docs = _mk_docs({str(i): c for i, c in enumerate(BM25_CONTENTS)})
    agent.chunked_docs = docs
    agent.chunked_docs_clean = [
        Document(content=content, metadata=doc.metadata)
        for doc, content in zip(docs, BM25_CONTENTS_CLEAN)
    ]
    return agent


def _make_fuzzy_agent(
    fuzzy_score_threshold: Optional[float],
    use_reciprocal_rank_fusion: bool = False,
) -> DocChatAgent:
    # None => use the config default, to verify default behavior end-to-end.
    kwargs = (
        {}
        if fuzzy_score_threshold is None
        else {"fuzzy_score_threshold": fuzzy_score_threshold}
    )
    agent = _make_agent(
        use_bm25_search=False,
        use_fuzzy_match=True,
        use_reciprocal_rank_fusion=use_reciprocal_rank_fusion,
        **kwargs,
    )
    docs = _mk_docs(FUZZY_CONTENTS)
    agent.chunked_docs = docs
    agent.chunked_docs_clean = docs
    return agent


def test_threshold_config_defaults():
    """Defaults must preserve pre-existing retrieval behavior.

    BM25: 0.0 = no filtering; fuzzy: 50.0 = the legacy `score > 50` filter.
    """
    config = DocChatAgentConfig()
    assert config.bm25_score_threshold == 0.0
    assert config.fuzzy_score_threshold == 50.0


def test_bm25_score_threshold_on_retrieval_path():
    query = "tigers"
    baseline_agent = _make_bm25_agent(0.0)
    docs = baseline_agent.chunked_docs
    docs_clean = baseline_agent.chunked_docs_clean
    # ground-truth scores, via the same BM25 scoring the agent uses
    # (k = n_similar_chunks * retrieval_multiple = 3)
    scored = find_closest_matches_with_bm25(docs, docs_clean, query, k=3)
    scores = {d.metadata.id: score for d, score in scored}
    top_id, top_score = max(scores.items(), key=lambda kv: kv[1])
    assert top_score > 0.0
    assert all(s < top_score for id_, s in scores.items() if id_ != top_id)

    # default threshold 0.0 => no filtering: BM25 top-k comes back whole,
    # even chunks with zero score
    chunks = baseline_agent.get_relevant_chunks(query)
    assert {d.metadata.id for d in chunks} == set(scores.keys())

    # threshold strictly between 0 and top score => only the top doc remains
    chunks = _make_bm25_agent(top_score / 2).get_relevant_chunks(query)
    assert {d.metadata.id for d in chunks} == {top_id}

    # boundary: comparison is inclusive (score >= threshold)
    chunks = _make_bm25_agent(top_score).get_relevant_chunks(query)
    assert {d.metadata.id for d in chunks} == {top_id}

    # threshold above the top score => all BM25 results are filtered out
    chunks = _make_bm25_agent(top_score + 1.0).get_relevant_chunks(query)
    assert chunks == []


def test_bm25_score_threshold_on_reciprocal_rank_fusion_path():
    """BM25 thresholding must happen before RRF rank collection."""
    query = "tigers"
    baseline_agent = _make_bm25_agent(
        0.0,
        use_reciprocal_rank_fusion=True,
    )
    docs = baseline_agent.chunked_docs
    docs_clean = baseline_agent.chunked_docs_clean
    scored = find_closest_matches_with_bm25(docs, docs_clean, query, k=9)
    scores = {d.metadata.id: score for d, score in scored}
    top_id, top_score = max(scores.items(), key=lambda kv: kv[1])
    assert top_score > 0.0

    chunks = baseline_agent.get_relevant_chunks(query)
    assert {d.metadata.id for d in chunks} == set(scores.keys())

    chunks = _make_bm25_agent(
        top_score / 2,
        use_reciprocal_rank_fusion=True,
    ).get_relevant_chunks(query)
    assert {d.metadata.id for d in chunks} == {top_id}

    chunks = _make_bm25_agent(
        top_score,
        use_reciprocal_rank_fusion=True,
    ).get_relevant_chunks(query)
    assert {d.metadata.id for d in chunks} == {top_id}

    chunks = _make_bm25_agent(
        top_score + 1.0,
        use_reciprocal_rank_fusion=True,
    ).get_relevant_chunks(query)
    assert chunks == []


def test_fuzzy_only_non_rrf_retrieval_path_preserves_legacy_behavior():
    """Default fuzzy-only, non-RRF behavior stays compatible with main.

    In the legacy path, fuzzy-only results are not added to `id2doc`, so when
    semantic and BM25 retrieval produce no hits, fuzzy matches are discarded
    after de-duplication. Threshold behavior is covered on the RRF path below,
    where fuzzy-only results are intentionally collected through `id2doc`.
    """
    query = "quantum entanglement"

    # default threshold (50.0) preserves the legacy hard-coded `score > 50`
    # filter, and the non-RRF fuzzy-only path still preserves legacy behavior
    # by dropping fuzzy-only matches after de-duplication.
    chunks = _make_fuzzy_agent(None).get_relevant_chunks(query)
    assert chunks == []

    # Even when threshold 0.0 admits weaker fuzzy matches, this compatibility
    # path does not start returning fuzzy-only chunks by default.
    chunks = _make_fuzzy_agent(0.0).get_relevant_chunks(query)
    assert chunks == []

    # threshold 100.0: strict comparison excludes even a perfect match
    chunks = _make_fuzzy_agent(100.0).get_relevant_chunks(query)
    assert chunks == []


def test_fuzzy_score_threshold_on_reciprocal_rank_fusion_path():
    """Fuzzy thresholding must happen before RRF rank collection."""
    query = "quantum entanglement"

    chunks = _make_fuzzy_agent(
        None,
        use_reciprocal_rank_fusion=True,
    ).get_relevant_chunks(query)
    assert {d.metadata.id for d in chunks} == {"high"}

    chunks = _make_fuzzy_agent(
        0.0,
        use_reciprocal_rank_fusion=True,
    ).get_relevant_chunks(query)
    assert {d.metadata.id for d in chunks} == {"high", "low"}

    chunks = _make_fuzzy_agent(
        100.0,
        use_reciprocal_rank_fusion=True,
    ).get_relevant_chunks(query)
    assert chunks == []


def test_fuzzy_retrieval_keeps_documents_with_overlapping_content():
    agent = _make_fuzzy_agent(None, use_reciprocal_rank_fusion=True)
    docs = _mk_docs({"long": "quantum mechanics", "short": "quantum"})
    agent.chunked_docs = docs
    agent.chunked_docs_clean = docs

    chunks = agent.get_relevant_chunks("quantum entanglement")

    assert [doc.metadata.id for doc in chunks] == ["short", "long"]


# values that are not finite numbers: invalid for BOTH thresholds
_NON_FINITE_THRESHOLDS: List[Any] = [
    None,
    float("nan"),
    float("inf"),
    float("-inf"),
    "not-a-number",
    object(),
    pytest.param(10**10000, id="huge-int"),
]
# for BM25, negative (and zero) finite values also deactivate the filter;
# for fuzzy, a negative finite threshold is a legitimate "keep all" setting
_INVALID_BM25_THRESHOLDS: List[Any] = _NON_FINITE_THRESHOLDS + [-1.0]


@pytest.mark.parametrize("bad_threshold", _INVALID_BM25_THRESHOLDS)
def test_bm25_invalid_runtime_threshold_is_inactive(bad_threshold):
    """Invalid/None/NaN/negative runtime values deactivate the BM25 filter.

    Such values must neither crash retrieval nor filter anything out:
    results must match the threshold-0.0 (filter off) baseline.
    """
    query = "tigers"
    baseline_ids = {
        d.metadata.id for d in _make_bm25_agent(0.0).get_relevant_chunks(query)
    }
    assert baseline_ids != set()  # sanity: baseline retrieves chunks

    agent = _make_bm25_agent(0.0)
    # simulate an unvalidated runtime value on the config (pydantic does
    # not validate on assignment here)
    agent.config.bm25_score_threshold = bad_threshold
    chunks = agent.get_relevant_chunks(query)
    assert {d.metadata.id for d in chunks} == baseline_ids


def test_bm25_absent_threshold_attribute_is_inactive():
    """A config lacking the attribute entirely must behave as filter-off."""
    query = "tigers"
    baseline_ids = {
        d.metadata.id for d in _make_bm25_agent(0.0).get_relevant_chunks(query)
    }
    agent = _make_bm25_agent(0.0)
    agent.config.__dict__.pop("bm25_score_threshold", None)
    assert not hasattr(agent.config, "bm25_score_threshold")  # sanity
    chunks = agent.get_relevant_chunks(query)
    assert {d.metadata.id for d in chunks} == baseline_ids


def _odd_bm25_docs_scores() -> List[Tuple[Document, Any]]:
    """Docs paired with runtime BM25 scores of assorted odd shapes."""
    docs = _mk_docs(
        {
            "kept": "Tigers are the largest cat species.",
            "kept_str": "Lions are the second largest cat species.",
            "none": "Chunk with a null score.",
            "nan": "Chunk with a NaN score.",
            "inf": "Chunk with an infinite score.",
            "text": "Chunk with a non-numeric score.",
            "obj": "Chunk with an arbitrary-object score.",
            "low": "Chunk scored below the threshold.",
        }
    )
    id2doc = {d.metadata.id: d for d in docs}
    return [
        (id2doc["kept"], 2.5),  # plain float above threshold
        (id2doc["kept_str"], "3.5"),  # coercible string above threshold
        (id2doc["none"], None),
        (id2doc["nan"], float("nan")),
        (id2doc["inf"], float("inf")),
        (id2doc["text"], "not-a-number"),
        (id2doc["obj"], object()),
        (id2doc["low"], 0.5),  # finite but below threshold
    ]


def test_bm25_odd_runtime_scores_with_active_threshold():
    """Oddly-shaped runtime BM25 scores must not crash an active filter.

    A BM25/FTS backend may hand back scores that are None, non-numeric,
    or non-finite; with the threshold active, such pairs are dropped
    (never compared raw), while coercible scores are compared after
    coercion to float.
    """
    docs_scores = _odd_bm25_docs_scores()
    agent = _make_bm25_agent(1.0)
    agent.get_similar_chunks_bm25 = (  # type: ignore[method-assign]
        lambda query, multiple: list(docs_scores)
    )
    chunks = agent.get_relevant_chunks("tigers")
    assert {d.metadata.id for d in chunks} == {"kept", "kept_str"}


def test_bm25_odd_runtime_scores_with_inactive_threshold_pass_through():
    """With the filter inactive (default 0.0), odd scores are untouched.

    No filtering and no coercion happen: even a None-scored pair flows
    through retrieval, preserving pre-threshold behavior (the final list
    is just capped at `n_relevant_chunks`, in retrieval order).
    """
    docs_scores = _odd_bm25_docs_scores()
    agent = _make_bm25_agent(0.0)
    agent.get_similar_chunks_bm25 = (  # type: ignore[method-assign]
        lambda query, multiple: list(docs_scores)
    )
    chunks = agent.get_relevant_chunks("tigers")
    # first n_relevant_chunks (=3) pairs, including the None-scored one
    assert {d.metadata.id for d in chunks} == {"kept", "kept_str", "none"}


def test_bm25_odd_runtime_scores_inactive_threshold_with_rrf():
    """RRF coerces unrankable BM25 scores when threshold is inactive."""
    docs_scores = _odd_bm25_docs_scores()
    agent = _make_bm25_agent(0.0, use_reciprocal_rank_fusion=True)
    agent.config.n_similar_chunks = len(docs_scores)
    agent.config.n_relevant_chunks = len(docs_scores)
    agent.get_similar_chunks_bm25 = (  # type: ignore[method-assign]
        lambda query, multiple: list(docs_scores)
    )
    chunks = agent.get_relevant_chunks("tigers")
    assert [d.metadata.id for d in chunks] == [
        "kept_str",
        "kept",
        "low",
        "none",
        "nan",
        "inf",
        "text",
        "obj",
    ]


@pytest.mark.parametrize("bad_threshold", _NON_FINITE_THRESHOLDS)
def test_fuzzy_invalid_runtime_threshold_yields_no_matches(bad_threshold):
    """Invalid/non-finite fuzzy thresholds yield NO fuzzy matches (no crash).

    In particular `-inf` must not admit every candidate.
    """
    query = "quantum entanglement"
    agent = _make_fuzzy_agent(None, use_reciprocal_rank_fusion=True)
    assert {d.metadata.id for d in agent.get_relevant_chunks(query)} == {"high"}

    agent.config.fuzzy_score_threshold = bad_threshold
    assert agent.get_relevant_chunks(query) == []


# Semantic and lexical retrieval share this hermetic retrieval-path fixture.
def _make_semantic_agent(
    results: dict[str, List[Tuple[Document, float]]],
    backend: str = "fuzzy",
    n: int = 1,
    rrf: bool = True,
    corpus: Optional[List[Document]] = None,
) -> DocChatAgent:
    agent = _make_agent(
        use_bm25_search=backend == "bm25",
        use_fuzzy_match=backend == "fuzzy",
        use_reciprocal_rank_fusion=rrf,
        n_similar_chunks=n,
        n_relevant_chunks=n,
        n_fuzzy_neighbor_words=0,
    )
    agent.vecdb = _StubVecDB(semantic_results=results)  # type: ignore[assignment]
    agent.chunked_docs = corpus or []
    agent.chunked_docs_clean = agent.chunked_docs
    return agent


@pytest.mark.parametrize("backend", ["fuzzy", "bm25"])
def test_rrf_lower_scoring_proxy_duplicate_keeps_best_semantic_hit(
    backend: str,
) -> None:
    a, b = _mk_docs({"a": "first", "b": "second"})
    agent = _make_semantic_agent(
        {"query": [(a, 0.9), (b, 0.8)], "proxy": [(a, 0.7)]},
        backend=backend,
    )
    assert agent.get_relevant_chunks("query") == [a]
    assert agent.get_relevant_chunks("query", ["proxy"]) == [a]


def test_rrf_higher_scoring_proxy_hit_wins() -> None:
    a, b = _mk_docs({"a": "first", "b": "second"})
    agent = _make_semantic_agent({"query": [(b, 0.8), (a, 0.5)], "proxy": [(a, 0.9)]})
    assert agent.get_relevant_chunks("query") == [b]
    assert agent.get_relevant_chunks("query", ["proxy"]) == [a]


@pytest.mark.parametrize("backend", ["fuzzy", "bm25"])
def test_rrf_semantic_duplicates_use_compact_unique_ranks(backend: str) -> None:
    a, b, c = _mk_docs(
        {
            "a": "zzz qqq xxx vvv www",
            "b": "quantum entanglement research",
            "c": "quantum entanglement",
        }
    )
    query = "quantum entanglement"
    original = [(a, 0.9), (b, 0.8), (c, 0.7)]
    # Repeat A between B and C. Keeping raw first-occurrence indices still
    # gives C rank 3 instead of 2, enough for B to displace it after RRF.
    for same_query in [False, True]:
        results = (
            {query: original[:2] + [(a, 0.75)] + original[2:]}
            if same_query
            else {query: original, "proxy": [(a, 0.75)]}
        )
        agent = _make_semantic_agent(results, backend=backend, n=2, corpus=[c, b, a])
        # Exclude the zero-scoring noise document from actual BM25 results.
        agent.config.bm25_score_threshold = 1e-9
        proxies = [] if same_query else ["proxy"]
        assert agent.get_relevant_chunks(query, proxies) == [c, b]


def test_semantic_duplicates_keep_highest_scoring_document_instance() -> None:
    best = _mk_docs({"a": "best payload"})[0]
    lower = _mk_docs({"a": "lower payload"})[0]
    agent = _make_semantic_agent({"query": [(best, 0.9)], "proxy": [(lower, 0.7)]})
    result = agent.get_relevant_chunks("query", ["proxy"])
    assert len(result) == 1
    assert result[0] is best


@pytest.mark.parametrize("backend", ["fuzzy", "bm25"])
def test_rrf_semantic_duplicate_with_real_lexical_matches(backend: str) -> None:
    a, b, c = _mk_docs(
        {
            "a": "quantum entanglement",
            "b": "quantum entanglement research",
            "c": "zzz qqq xxx vvv www",
        }
    )
    query = "quantum entanglement"
    agent = _make_semantic_agent(
        {query: [(a, 0.9), (b, 0.8), (c, 0.75)], "proxy": [(a, 0.7)]},
        backend=backend,
        corpus=[a, b, c],
    )
    lexical = (
        agent.get_fuzzy_matches(query, 3)
        if backend == "fuzzy"
        else agent.get_similar_chunks_bm25(query, 3)
    )
    assert [doc.id() for doc, _ in lexical][:2] == ["a", "b"]
    assert agent.get_relevant_chunks(query) == [a]
    assert agent.get_relevant_chunks(query, ["proxy"]) == [a]


@pytest.mark.parametrize("backend", ["fuzzy", "bm25"])
def test_rrf_unique_semantic_hits_keep_score_order(backend: str) -> None:
    a, b, c = _mk_docs({"a": "first", "b": "second", "c": "third"})
    agent = _make_semantic_agent(
        {"query": [(b, 0.8), (a, 0.9), (c, 0.7)]}, backend=backend, n=3
    )
    assert agent.get_relevant_chunks("query") == [a, b, c]


def test_semantic_deduplication_uses_ids_not_content() -> None:
    a, b = _mk_docs({"a": "same content", "b": "same content"})
    agent = _make_semantic_agent({"query": [(a, 0.9), (b, 0.8)]}, n=2)
    assert agent.get_relevant_chunks("query") == [a, b]


def test_non_rrf_semantic_duplicate_keeps_document_order() -> None:
    a, b = _mk_docs({"a": "first", "b": "second"})
    agent = _make_semantic_agent(
        {"query": [(a, 0.9), (b, 0.8)], "proxy": [(a, 0.7)]}, n=2, rrf=False
    )
    assert not agent.config.use_reciprocal_rank_fusion
    assert agent.get_relevant_chunks("query", ["proxy"]) == [a, b]


def test_rrf_without_lexical_search_keeps_document_order() -> None:
    a, b = _mk_docs({"a": "first", "b": "second"})
    agent = _make_semantic_agent(
        {"query": [(a, 0.9), (b, 0.8)], "proxy": [(a, 0.7)]},
        backend="none",
        n=2,
    )
    assert agent.get_relevant_chunks("query", ["proxy"]) == [a, b]
