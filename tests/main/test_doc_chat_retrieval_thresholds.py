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
"""

from typing import List, Optional, Tuple

from langroid.agent.special.doc_chat_agent import DocChatAgent, DocChatAgentConfig
from langroid.language_models.mock_lm import MockLMConfig
from langroid.mytypes import DocMetaData, Document
from langroid.parsing.search import find_closest_matches_with_bm25, preprocess_text
from langroid.vector_store.base import VectorStoreConfig


class _StubVecDB:
    """Minimal stand-in for a vector store.

    Provides only what `DocChatAgent.get_relevant_chunks` touches: a named
    existing collection, and (empty) semantic-search results, so tests can
    exercise the BM25/fuzzy retrieval branches without embeddings or network.
    """

    def __init__(self, collection_name: str = "test-retrieval-thresholds"):
        self.config = VectorStoreConfig(collection_name=collection_name)

    def list_collections(self, empty: bool = False) -> List[str]:
        return [str(self.config.collection_name)]

    def similar_texts_with_scores(
        self,
        text: str,
        k: int = 1,
        where: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        return []


BM25_CONTENTS = [
    "Tigers are the largest cat species in the world.",
    "Lions are the second largest cat species in the world.",
    "The stock market fluctuated wildly this quarter.",
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


def _make_bm25_agent(bm25_score_threshold: float) -> DocChatAgent:
    agent = _make_agent(
        use_bm25_search=True,
        use_fuzzy_match=False,
        bm25_score_threshold=bm25_score_threshold,
    )
    docs = _mk_docs({str(i): c for i, c in enumerate(BM25_CONTENTS)})
    agent.chunked_docs = docs
    agent.chunked_docs_clean = [
        Document(content=preprocess_text(d.content), metadata=d.metadata) for d in docs
    ]
    return agent


def _make_fuzzy_agent(fuzzy_score_threshold: Optional[float]) -> DocChatAgent:
    # None => use the config default, to verify default behavior end-to-end.
    # With RRF enabled, fuzzy-match results are ranked and returned on their
    # own, without needing semantic or bm25 results.
    kwargs = (
        {}
        if fuzzy_score_threshold is None
        else {"fuzzy_score_threshold": fuzzy_score_threshold}
    )
    agent = _make_agent(
        use_bm25_search=False,
        use_fuzzy_match=True,
        use_reciprocal_rank_fusion=True,
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


def test_fuzzy_score_threshold_on_retrieval_path():
    query = "quantum entanglement"

    # default threshold (50.0): only the verbatim-match doc survives,
    # matching the legacy hard-coded `score > 50` behavior
    chunks = _make_fuzzy_agent(None).get_relevant_chunks(query)
    assert {d.metadata.id for d in chunks} == {"high"}

    # threshold 0.0 admits weaker matches (the "noise" doc passes the score
    # filter too, but yields no usable fuzzy context window, so it's dropped
    # downstream of the threshold)
    chunks = _make_fuzzy_agent(0.0).get_relevant_chunks(query)
    assert {d.metadata.id for d in chunks} == {"high", "low"}

    # threshold 100.0: strict comparison excludes even a perfect match
    chunks = _make_fuzzy_agent(100.0).get_relevant_chunks(query)
    assert chunks == []
