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
        # read_only, or merely opening the store writes a lock row into a 34 MB
        # tracked binary and running the suite leaves the repository dirty.
        store = VectorStoreManager(
            persist_directory=config.vectorstore.persist_directory,
            collection_name=config.vectorstore.collection_name,
            read_only=True)
        metadatas = store.collection.get(include=["metadatas"])["metadatas"]

        untagged = [m for m in metadatas if m.get("method") in (None, "", "multiple")]
        assert not untagged, f"{len(untagged)} chunks still untagged"


class TestOneDocumentCannotFillTheAnswer:
    """Tagging the AQTESOLV manual with the implemented methods made it eligible
    for every method question, and at 503 of 4,017 chunks — phrased exactly as
    people ask — it took all five slots. Hantush's own paper ranked 15th on a
    question about Hantush-Jacob, and Theis's ranked 30th on a question about
    Theis, so the consultant answered from a software manual and cited no
    literature at all.
    """

    @pytest.mark.parametrize("question,least_papers", [
        ("What are the assumptions of the Hantush-Jacob method?", 3),
        ("What are the assumptions of the Theis method?", 3),
        # Only the manual and Walton's bulletin carry the Cooper-Jacob tag
        # closely enough to rank; two is all this corpus can offer.
        ("When should I use Cooper-Jacob?", 2),
    ])
    def test_a_method_question_draws_on_several_papers(self, retriever, question,
                                                       least_papers):
        hits = retriever.retrieve(question, top_k=5, similarity_threshold=0.0,
                                  metadata_filters=filter_for_question(question))

        papers = {h["metadata"]["doc_id"].rsplit("_page_", 1)[0] for h in hits}
        assert len(papers) >= least_papers, f"answer drawn from {len(papers)} paper(s)"

    def test_results_stay_ranked_by_score(self, retriever):
        """Backfilling after the cap appends chunks that outscore what is
        already kept, so the list must be re-sorted: both the prompt and the
        sources panel present these in order."""
        for question in ["When should I use Cooper-Jacob?",
                         "What are the assumptions of the Theis method?"]:
            hits = retriever.retrieve(question, top_k=5, similarity_threshold=0.0,
                                      metadata_filters=filter_for_question(question))
            scores = [h["score"] for h in hits]

            assert scores == sorted(scores, reverse=True), question

    def test_no_paper_exceeds_the_cap(self, retriever):
        from collections import Counter

        from src.retrieval.retriever import MAX_PER_DOCUMENT

        hits = retriever.retrieve("What are the assumptions of the Theis method?",
                                  top_k=5, similarity_threshold=0.0)
        counts = Counter(h["metadata"]["doc_id"].rsplit("_page_", 1)[0] for h in hits)

        assert max(counts.values()) <= MAX_PER_DOCUMENT

    def test_the_primary_paper_is_reachable_again(self, retriever):
        """The case that exposed this: Hantush's 1955 paper, which the manual
        had pushed out of the answer entirely."""
        question = "What are the assumptions of the Hantush-Jacob method?"
        hits = retriever.retrieve(question, top_k=5, similarity_threshold=0.0,
                                  metadata_filters=filter_for_question(question))

        assert any("Hantush" in h["metadata"]["doc_id"] for h in hits)

    def test_the_cap_can_be_lifted(self, retriever):
        """0 restores the old behaviour, for a caller that wants depth from one
        source rather than breadth."""
        question = "What are the assumptions of the Theis method?"
        hits = retriever.retrieve(question, top_k=5, similarity_threshold=0.0,
                                  metadata_filters=filter_for_question(question),
                                  max_per_document=0)

        papers = {h["metadata"]["doc_id"].rsplit("_page_", 1)[0] for h in hits}
        assert len(papers) < 3

    def test_it_still_returns_a_full_answer_from_a_narrow_corner(self, retriever):
        """A question only one paper covers must still return top_k results:
        chunks displaced by the cap are appended rather than dropped."""
        hits = retriever.retrieve("equivalent time for pressure buildup", top_k=5,
                                  similarity_threshold=0.0,
                                  metadata_filters=filter_for(["agarwal_1980"]))

        assert len(hits) == 5


class TestRetrievalPrefersTheArgument:
    """Reference lists and contents pages must not be offered as evidence.

    Reported from the deployed site: the sources panel listed Walton's
    bibliography and an ordering page. Every chunk now records what kind of
    text it is, and retrieval asks for the argument by default.
    """

    def test_every_chunk_says_what_kind_of_text_it_is(self, retriever):
        """The filter excludes anything without the key, so a store that
        predates it would return nothing at all."""
        from src.core.config import load_config
        from src.retrieval.vectorstore import VectorStoreManager

        config = load_config()
        store = VectorStoreManager(
            persist_directory=config.vectorstore.persist_directory,
            collection_name=config.vectorstore.collection_name,
            read_only=True)
        metadatas = store.collection.get(include=["metadatas"])["metadatas"]

        untagged = [m for m in metadatas if not m.get("matter")]
        assert not untagged, f"{len(untagged)} chunks have no matter"

    def test_a_bibliography_is_no_longer_offered(self, retriever):
        """The exact chunk from the report: wri99-4225's reference list."""
        question = "Walton groundwater resource evaluation 1970"

        hits = retriever.retrieve(question, top_k=5, similarity_threshold=0.0)

        assert hits, "the question must still be answered from somewhere"
        assert all(hit["metadata"]["matter"] == "body" for hit in hits)

    def test_it_can_still_be_reached_deliberately(self, retriever):
        """Excluded from answers, not deleted: a caller that wants the
        apparatus can ask for it."""
        hits = retriever.retrieve("Walton groundwater resource evaluation 1970",
                                  top_k=5, similarity_threshold=0.0,
                                  body_only=False)

        assert any(hit["metadata"]["matter"] != "body" for hit in hits)

    def test_it_combines_with_a_method_filter_rather_than_replacing_it(self, retriever):
        hits = retriever.retrieve(
            "What are the assumptions of this method?", top_k=5,
            similarity_threshold=0.0, metadata_filters=filter_for(["theis_1935"]))

        assert hits
        for hit in hits:
            assert hit["metadata"]["matter"] == "body"
            assert "theis_1935" in _tags(hit)

    def test_ordinary_questions_are_unaffected(self, retriever):
        """The honest measure of this change: of twelve pinned questions, none
        had apparatus in its top five to begin with. This removes a failure
        mode rather than improving everyday answers."""
        question = "What are the assumptions of the Theis method?"

        with_filter = retriever.retrieve(question, top_k=5, similarity_threshold=0.0)
        without = retriever.retrieve(question, top_k=5, similarity_threshold=0.0,
                                     body_only=False)

        assert [h["metadata"]["chunk_id"] for h in with_filter] == \
               [h["metadata"]["chunk_id"] for h in without]
