"""The method filter against the real knowledge base.

The unit tests check that a question produces the right filter. They cannot
check the thing most likely to break it: whether ChromaDB accepts the clause and
returns the papers it names. A `$or` of metadata flags is only useful if the
store understands it, and a wrong clause fails at query time, in the consultant,
not in a unit test.

Skipped where the committed vector store is absent.
"""

from pathlib import Path

import pytest

from src.ingestion.corpus import METHOD_FLAG_PREFIX
from src.retrieval.method_filter import filter_for, filter_for_question

VECTORDB = Path("data/vectordb/chroma.sqlite3")

pytestmark = pytest.mark.skipif(
    not VECTORDB.exists(), reason="knowledge-base vector store not present")


@pytest.fixture(scope="module")
def retriever():
    from src.core.config import load_config
    from src.retrieval.retriever import Retriever
    return Retriever(load_config())


def _tags(hit) -> set:
    return set((hit["metadata"].get("method") or "").split(","))


class TestTheStoreUnderstandsTheClause:

    def test_a_single_method_returns_only_its_papers(self, retriever):
        hits = retriever.retrieve("What are the assumptions of this method?",
                                  top_k=5, similarity_threshold=0.0,
                                  metadata_filters=filter_for(["theis_1935"]))

        assert hits
        for hit in hits:
            assert "theis_1935" in _tags(hit)

    def test_an_or_of_several_returns_any_of_them(self, retriever):
        methods = ["neuman_1974", "neuman_1975", "neuman_1979"]
        hits = retriever.retrieve("delayed yield", top_k=5, similarity_threshold=0.0,
                                  metadata_filters=filter_for(methods))

        assert hits
        for hit in hits:
            assert _tags(hit) & set(methods)

    def test_it_combines_with_the_aquifer_filter(self, retriever):
        """What the consultant sends once a method is confirmed."""
        clause = {"$and": [{"aquifer": "leaky"},
                           filter_for(["hantush_jacob_1955", "hantush_1960"])]}
        hits = retriever.retrieve("leakage through the aquitard", top_k=5,
                                  similarity_threshold=0.0, metadata_filters=clause)

        assert hits
        for hit in hits:
            assert hit["metadata"]["aquifer"] == "leaky"
            assert _tags(hit) & {"hantush_jacob_1955", "hantush_1960"}


class TestItNarrowsWithoutSilencing:

    def test_a_question_naming_no_method_is_not_filtered(self, retriever):
        assert filter_for_question("What is transmissivity?") is None

        hits = retriever.retrieve("What is transmissivity?", top_k=5,
                                  similarity_threshold=0.0)
        assert len(hits) == 5

    def test_a_method_no_paper_carries_still_answers(self, retriever):
        """The retriever retries without filters when a filtered search finds
        nothing, so a tag that matches no paper costs a little time, not the
        answer."""
        nothing = {f"{METHOD_FLAG_PREFIX}not_a_real_method": True}

        hits = retriever.retrieve("How is drawdown measured?", top_k=5,
                                  similarity_threshold=0.0,
                                  metadata_filters=nothing)
        assert hits

    def test_the_whole_path_from_a_question(self, retriever):
        """What actually happens when someone asks about Hantush."""
        question = "What are the assumptions of the Hantush-Jacob method?"

        hits = retriever.retrieve(question, top_k=5, similarity_threshold=0.0,
                                  metadata_filters=filter_for_question(question))

        assert hits
        for hit in hits:
            assert _tags(hit) & {"hantush_jacob_1955", "hantush_1960",
                                 "hantush_1955_steady", "hantush_1955_strip",
                                 "hantush_1967"}


class TestTheCorpusIsActuallyTagged:

    def test_the_scanned_papers_are_reachable_by_method(self, retriever):
        """Agarwal contributed nothing until OCR; this is the proof it is now
        both present and findable through its tags."""
        hits = retriever.retrieve("equivalent time for pressure buildup", top_k=5,
                                  similarity_threshold=0.0,
                                  metadata_filters=filter_for(["agarwal_1980"]))

        assert hits
        assert any("Agarwal" in (h["metadata"].get("doc_id") or "") for h in hits)

    def test_no_chunk_is_left_untagged(self, retriever):
        """`method` was "multiple" for 3,837 of 3,843 chunks."""
        from src.core.config import load_config
        from src.retrieval.vectorstore import VectorStoreManager

        config = load_config()
        store = VectorStoreManager(
            persist_directory=config.vectorstore.persist_directory,
            collection_name=config.vectorstore.collection_name)
        metadatas = store.collection.get(include=["metadatas"])["metadatas"]

        untagged = [m for m in metadatas if m.get("method") in (None, "", "multiple")]
        assert not untagged, f"{len(untagged)} chunks still untagged"
