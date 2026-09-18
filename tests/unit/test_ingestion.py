"""The code that builds the knowledge base.

`data/vectordb` is committed, so this code runs rarely — and that is exactly why
it needs tests. It had none, and the next two changes to the corpus (OCR for the
scanned papers, real method tags) both re-run it. A silent regression here would
not show up as a crash; it would show up months later as a consultant that
cannot find a paper.

The PDFs are built here rather than committed: reportlab writes a text PDF, and
Pillow writes an image-only one that stands in for the scanned papers.
"""

import pytest
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from src.ingestion.processors.chunker import MetadataPreservingChunker
from src.ingestion.processors.pdf_processor import PDFProcessor
from src.ingestion.validators import (
    validate_chunk, validate_chunks_batch, validate_document,
)

PAGE_ONE = "Theis (1935) describes non-steady radial flow to a well."
PAGE_TWO = "The drawdown s equals Q W(u) / (4 pi T), where u = r^2 S / (4 T t)."


def _metadata(**overrides) -> dict:
    """The metadata shape the ingestion scripts build for every chunk."""
    base = {
        "source": "Custom_Confined Aquifer",
        "type": "research_paper",
        "aquifer": "confined",
        "method": "multiple",
        "page": 1,
        "section": "theis_1935 - Page 1",
        "doc_id": "theis_1935_page_1",
        "has_equations": True,
        "confidence": 1.0,
    }
    base.update(overrides)
    return base


@pytest.fixture(scope="module")
def text_pdf(tmp_path_factory) -> str:
    """A two-page PDF with a real text layer, like most of the corpus."""
    path = tmp_path_factory.mktemp("pdfs") / "text_paper.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    for text in (PAGE_ONE, PAGE_TWO):
        pdf.drawString(72, 750, text)
        pdf.showPage()
    pdf.save()
    return str(path)


@pytest.fixture(scope="module")
def scanned_pdf(tmp_path_factory) -> str:
    """A page that is only an image — the shape of the two papers that
    contributed nothing to the index."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1240, 1754), "white")
    ImageDraw.Draw(image).text((100, 100), PAGE_ONE, fill="black")

    path = tmp_path_factory.mktemp("pdfs") / "scanned_paper.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.drawImage(ImageReader(image), 0, 0, width=A4[0], height=A4[1])
    pdf.showPage()
    pdf.save()
    return str(path)


class TestReadingAPDF:

    def test_one_entry_per_page_numbered_from_one(self, text_pdf):
        pages = PDFProcessor().extract_text_with_pages(text_pdf)
        assert [p["page"] for p in pages] == [1, 2]

    def test_the_text_of_each_page_is_recovered(self, text_pdf):
        pages = PDFProcessor().extract_text_with_pages(text_pdf)
        assert "Theis (1935)" in pages[0]["text"]
        assert "drawdown" in pages[1]["text"]

    def test_a_missing_file_is_reported_as_such(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="PDF not found"):
            PDFProcessor().extract_text_with_pages(str(tmp_path / "absent.pdf"))

    def test_cleaning_collapses_whitespace(self):
        cleaned = PDFProcessor()._clean_text("Theis\n\n\n   1935  \t was")
        assert cleaned == "Theis 1935 was"

    def test_cleaning_an_empty_page_gives_an_empty_string(self):
        assert PDFProcessor()._clean_text("") == ""
        assert PDFProcessor()._clean_text(None) == ""

    def test_an_image_only_page_yields_no_text(self, scanned_pdf):
        """Why two papers contributed nothing: pypdf reads the text layer, and a
        scan has none. Ingestion then drops the page for being under 100
        characters, without reporting the document as empty.

        This is the behaviour OCR has to change.
        """
        pages = PDFProcessor().extract_text_with_pages(scanned_pdf)

        assert len(pages) == 1
        assert len(pages[0]["text"]) < 100

    def test_the_scanned_fixture_really_is_a_pdf_of_one_page(self, scanned_pdf):
        """Guards the fixture itself: if reportlab stopped embedding the image,
        the test above would pass for the wrong reason."""
        assert len(PdfReader(scanned_pdf).pages) == 1


class TestEquationDetection:

    @pytest.mark.parametrize("text", [
        "The drawdown s = Q / (4 pi T)",
        "See equation 7 for the residual drawdown",
        "The integral ∫ of the well function",
    ])
    def test_mathematical_pages_are_flagged(self, text):
        assert PDFProcessor()._detect_equations(text) is True

    def test_plain_prose_is_not(self):
        text = ("The pumping test was carried out in June near the village, "
                "and the observation wells were read by hand.")
        assert PDFProcessor()._detect_equations(text) is False

    def test_the_flag_reaches_the_page_data(self, text_pdf):
        pages = PDFProcessor().extract_text_with_pages(text_pdf)
        assert pages[1]["has_equations"] is True


class TestChunking:

    @pytest.fixture
    def chunker(self):
        return MetadataPreservingChunker(chunk_size=200, chunk_overlap=50)

    def test_every_chunk_keeps_the_document_metadata(self, chunker):
        chunks = chunker.chunk_text("Theis. " * 200, _metadata(), "theis_1935_page_1")

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.metadata["aquifer"] == "confined"
            assert chunk.metadata["doc_id"] == "theis_1935_page_1"

    def test_chunk_ids_are_unique_and_derived_from_the_document(self, chunker):
        chunks = chunker.chunk_text("Theis. " * 200, _metadata(), "theis_1935_page_1")
        ids = [c.metadata["chunk_id"] for c in chunks]

        assert len(set(ids)) == len(ids)
        assert ids[0] == "theis_1935_page_1_chunk_0"

    def test_chunks_are_numbered_and_know_the_total(self, chunker):
        chunks = chunker.chunk_text("Theis. " * 200, _metadata(), "doc")

        assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
        assert {c.metadata["total_chunks"] for c in chunks} == {len(chunks)}

    def test_consecutive_chunks_overlap(self, chunker):
        """The overlap is what keeps a sentence split across a boundary
        retrievable from either side."""
        text = " ".join(f"word{i}" for i in range(200))
        chunks = chunker.chunk_text(text, _metadata(), "doc")
        first, second = chunks[0].page_content, chunks[1].page_content

        tail = first.split()[-3:]
        assert any(word in second for word in tail)

    def test_no_chunk_exceeds_the_size(self, chunker):
        chunks = chunker.chunk_text("Theis. " * 300, _metadata(), "doc")
        assert all(len(c.page_content) <= 200 for c in chunks)

    @pytest.mark.parametrize("text", ["", "   ", "\n\n"])
    def test_empty_text_produces_no_chunks(self, chunker, text):
        assert chunker.chunk_text(text, _metadata(), "doc") == []

    def test_documents_are_chunked_together(self, chunker):
        documents = [
            {"text": "Theis. " * 100, "metadata": _metadata(), "doc_id": "a"},
            {"text": "Hantush. " * 100, "metadata": _metadata(aquifer="leaky"),
             "doc_id": "b"},
        ]
        chunks = chunker.chunk_documents(documents)

        assert {c.metadata["doc_id"] for c in chunks} == {"a", "b"}
        assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)

    def test_a_document_with_no_text_does_not_stop_the_others(self, chunker):
        documents = [
            {"text": "", "metadata": _metadata(), "doc_id": "empty"},
            {"text": "Theis. " * 100, "metadata": _metadata(), "doc_id": "full"},
        ]
        chunks = chunker.chunk_documents(documents)

        assert chunks and {c.metadata["doc_id"] for c in chunks} == {"full"}


class TestValidation:

    def _chunk(self, **overrides) -> dict:
        chunk = {"content": PAGE_ONE, "metadata": _metadata(chunk_id="c0"),
                 "doc_id": "theis_1935_page_1"}
        chunk.update(overrides)
        return chunk

    def test_a_well_formed_chunk_passes(self):
        assert validate_chunk(self._chunk()) is True

    @pytest.mark.parametrize("field", ["source", "type", "aquifer", "method",
                                       "section", "chunk_id"])
    def test_every_required_metadata_field_is_enforced(self, field):
        metadata = _metadata(chunk_id="c0")
        del metadata[field]

        with pytest.raises(Exception, match=field):
            validate_chunk(self._chunk(metadata=metadata))

    @pytest.mark.parametrize("content", ["short", "x" * 10001])
    def test_content_outside_the_bounds_is_rejected(self, content):
        with pytest.raises(Exception):
            validate_chunk(self._chunk(content=content))

    def test_a_document_must_carry_real_text(self):
        """The 100-character floor is why a scanned page is dropped."""
        with pytest.raises(Exception):
            validate_document({"text": "too short", "metadata": _metadata(),
                               "doc_id": "d"})

        assert validate_document({"text": "Theis. " * 50, "metadata": _metadata(),
                                  "doc_id": "d"}) is True

    def test_a_batch_counts_the_good_and_names_the_bad(self):
        valid, errors = validate_chunks_batch([
            self._chunk(),
            self._chunk(content="short"),
            self._chunk(),
        ])

        assert valid == 2
        assert len(errors) == 1 and errors[0].startswith("Chunk 1:")
