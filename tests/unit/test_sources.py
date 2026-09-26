"""What a retrieved chunk is called.

Reported from the deployed site: an answer cited "Custom_Fractured Aquifers,
p.5". That is the folder the PDF was filed in, and it was the only name the
citation had — both in the "Sources consulted" list and inside the answers the
model wrote, because the prompt handed it the same string.

The document's identity was in the index all along, as the filename it was
chunked from. These tests cover the three ways it is turned into something a
reader can act on, and the two things that must not happen: a name invented for a
document, and a link that would not open.
"""

import pytest

from src.retrieval.sources import resolve


def chunk(doc_id="Theis-1935_page_3", page=3, **extra):
    """Metadata in the shape the index actually stores."""
    metadata = {
        "doc_id": doc_id,
        "section": f"{doc_id.rsplit('_page_', 1)[0]} - Page {page}",
        "page": page,
        "source": "Custom_Confined Aquifer",
        "type": "research_paper",
        "method": "theis_1935",
        "aquifer": "confined",
    }
    metadata.update(extra)
    return metadata


class TestNamingTheDocument:

    def test_a_folder_name_is_never_the_answer(self):
        """The whole complaint in one assertion."""
        source = resolve(chunk())

        assert "Custom_" not in source.document

    def test_a_descriptive_filename_is_used_as_it_reads(self):
        source = resolve(chunk(
            doc_id="Water Resources Research - April 1974 - Neuman - Effect of "
                   "partial penetration_page_2", page=2))

        assert source.document.startswith("Water Resources Research - April 1974")
        assert source.page == 2

    def test_the_manifest_names_a_document_whose_filename_does_not(self):
        """`spe-426-pa` is Warren and Root; nothing in the filename says so."""
        source = resolve(chunk(doc_id="spe-426-pa_page_1", page=1))

        assert "Warren" in source.document

    def test_publisher_furniture_comes_off(self):
        source = resolve(chunk(doc_id="52485_SPE-9289-Agarwal-buildup_page_4",
                               page=4))

        assert source.document == "SPE-9289-Agarwal-buildup"

    def test_a_page_travels_with_the_citation(self):
        assert resolve(chunk()).citation.endswith("p.3]")

    def test_the_page_is_recovered_from_the_chunk_id_when_the_field_is_empty(self):
        """`doc_id` carries it as well, and a page number is what makes a
        citation checkable."""
        source = resolve(chunk(doc_id="sopaqu_page_7", page=None))

        assert source.page == 7

    def test_a_document_with_no_page_anywhere_still_cites(self):
        source = resolve({"doc_id": "ISWSB-49", "section": "ISWSB-49",
                          "type": "research_paper"})

        assert source.citation.startswith("[")
        assert "p." not in source.citation

    def test_help_text_is_named_by_its_document_and_section(self):
        source = resolve({"type": "help", "document": "getting-data-in.md",
                          "section": "What HydroAssist accepts"})

        assert source.document == "getting-data-in.md — What HydroAssist accepts"
        assert source.is_help is True
        assert source.url == ""

    def test_unrecognisable_metadata_does_not_raise(self):
        """Retrieval must not fail because a chunk is odd."""
        assert resolve({}).document
        assert resolve({"doc_id": ""}).document


class TestLinks:

    def test_a_sciencedirect_filename_yields_its_doi(self):
        """The PII is in the filename by construction, so the DOI follows by a
        fixed rule rather than a guess."""
        source = resolve(chunk(doc_id="1-s2.0-002216948890025X-main_page_1",
                               page=1))

        assert source.url == "https://doi.org/10.1016/0022-1694(88)90025-X"

    def test_another_one(self):
        source = resolve(chunk(doc_id="1-s2.0-0022169477901378-main_page_1",
                               page=1))

        assert source.url == "https://doi.org/10.1016/0022-1694(77)90137-8"

    def test_no_link_is_invented_for_a_paper_that_has_none(self):
        """A link that 404s is worse than saying there is none: the reader goes
        looking, fails, and trusts the citation less than before."""
        source = resolve(chunk(doc_id="Theis-1935_page_1", page=1))

        assert source.url == ""

    @pytest.mark.parametrize("doc_id", [
        "1-s2.0-notapii-main_page_1",
        "spe-426-pa_page_1",
        "Gringarten71_page_1",
    ])
    def test_only_a_real_identifier_becomes_a_link(self, doc_id):
        assert resolve(chunk(doc_id=doc_id, page=1)).url == ""


class TestTheWholeCorpus:
    """Every document in the index, resolved. This is the test that would have
    caught the original fault: it asserts about names, not about one name."""

    @pytest.fixture(scope="class")
    def resolved(self):
        from src.retrieval.vectorstore import VectorStoreManager

        manager = VectorStoreManager(read_only=True)
        if manager.collection is None or manager.collection.count() == 0:
            pytest.skip("no knowledge base in this checkout")
        got = manager.collection.get(include=["metadatas"])
        return [resolve(metadata) for metadata in got["metadatas"]]

    def test_no_document_is_named_after_its_folder(self, resolved):
        assert not [s for s in resolved if s.document.startswith("Custom_")]

    def test_every_chunk_has_a_name_and_a_page(self, resolved):
        assert all(s.document.strip() for s in resolved)
        assert all(s.page is None or s.page >= 1 for s in resolved)

    def test_the_names_are_not_all_the_same_six(self, resolved):
        """The folder gave six values for 4,017 chunks; the corpus has 48
        documents."""
        assert len({s.document for s in resolved}) >= 40

    def test_no_name_is_a_bare_catalogue_number(self, resolved):
        """A reader cannot act on "WR024i010p01796", which is why the manifest
        names those."""
        import re

        unnamed = {s.document for s in resolved
                   if len(re.findall(r"[A-Za-z]{3,}", s.document)) < 2}
        assert not unnamed, f"still unnamed: {sorted(unnamed)}"
