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
from pathlib import Path
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from src.ingestion import corpus
from src.ingestion.processors import ocr as ocr_module
from src.ingestion.processors.chunker import MetadataPreservingChunker
from src.ingestion.processors.ocr import (
    OCRTooling, OCRUnavailable, check_tooling, ocr_pdf_pages,
)
from src.ingestion.processors.pdf_processor import (
    MIN_PAGE_CHARS, OCR_ALWAYS, OCR_AUTO, OCR_NEVER, PDFProcessor,
)
from src.ingestion.validators import (
    validate_chunk, validate_chunks_batch, validate_document,
)

needs_ocr_tooling = pytest.mark.skipif(
    not check_tooling().available,
    reason=f"OCR tooling not installed — {check_tooling().instructions}")

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
    from PIL import Image, ImageDraw, ImageFont

    # Large enough to survive being scaled into the page and rendered again for
    # OCR — a scan of a journal page is about this size relative to its text.
    image = Image.new("RGB", (1240, 1754), "white")
    ImageDraw.Draw(image).text((80, 120), PAGE_ONE, fill="black",
                               font=ImageFont.load_default(size=44))

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
        characters. Only OCR — the next class — recovers it."""
        pages = PDFProcessor().extract_text_with_pages(scanned_pdf)

        assert len(pages) == 1
        assert len(pages[0]["text"]) < MIN_PAGE_CHARS
        assert pages[0]["ocr"] is False

    def test_the_scanned_fixture_really_is_a_pdf_of_one_page(self, scanned_pdf):
        """Guards the fixture itself: if reportlab stopped embedding the image,
        the test above would pass for the wrong reason."""
        assert len(PdfReader(scanned_pdf).pages) == 1


class TestOCR:
    """Reading the pages that have no text layer.

    The tooling is optional — it is needed to rebuild the knowledge base, never
    to run HydroAssist — so the tests that need Tesseract skip without it, while
    the policy around it is tested either way.
    """

    def test_an_unknown_mode_is_rejected(self, text_pdf):
        with pytest.raises(ValueError, match="Unknown ocr mode"):
            PDFProcessor().extract_text_with_pages(text_pdf, ocr="sometimes")

    def test_missing_tooling_says_what_to_install(self, monkeypatch, scanned_pdf):
        monkeypatch.setattr(ocr_module, "check_tooling",
                            lambda: OCRTooling(False, False, None))

        with pytest.raises(OCRUnavailable) as raised:
            ocr_pdf_pages(scanned_pdf, [1])

        message = str(raised.value)
        assert "pypdfium2" in message and "pytesseract" in message
        assert "tesseract" in message.lower()

    def test_tooling_reports_what_is_present(self):
        tooling = check_tooling()
        assert isinstance(tooling.available, bool)
        assert tooling.instructions

    def test_a_missing_pdf_is_reported_before_any_rendering(self, tmp_path):
        if not check_tooling().available:
            pytest.skip("OCR tooling not installed")
        with pytest.raises(FileNotFoundError):
            ocr_pdf_pages(str(tmp_path / "absent.pdf"), [1])

    @needs_ocr_tooling
    def test_it_recovers_the_text_of_a_scanned_page(self, scanned_pdf):
        """The whole point: a page that yielded nothing now yields its words."""
        pages = PDFProcessor().extract_text_with_pages(scanned_pdf, ocr=OCR_AUTO)

        assert pages[0]["ocr"] is True
        assert "Theis" in pages[0]["text"]

    @needs_ocr_tooling
    def test_a_readable_page_is_left_alone(self, text_pdf):
        """OCR of a page that already has text is usually worse than the text
        layer, so `auto` must not touch it."""
        plain = PDFProcessor().extract_text_with_pages(text_pdf)
        auto = PDFProcessor().extract_text_with_pages(text_pdf, ocr=OCR_AUTO)

        assert [p["ocr"] for p in auto] == [False, False]
        assert [p["text"] for p in auto] == [p["text"] for p in plain]

    @needs_ocr_tooling
    def test_always_still_keeps_whichever_text_is_longer(self, text_pdf):
        """`always` re-reads every page, but a worse reading is discarded: the
        text layer of a born-digital PDF is the better source."""
        plain = PDFProcessor().extract_text_with_pages(text_pdf)
        always = PDFProcessor().extract_text_with_pages(text_pdf, ocr=OCR_ALWAYS)

        for before, after in zip(plain, always):
            assert len(after["text"]) >= len(before["text"])

    @needs_ocr_tooling
    def test_a_page_that_does_not_exist_is_skipped_not_fatal(self, scanned_pdf):
        recognised = ocr_pdf_pages(scanned_pdf, [1, 99])

        assert set(recognised) == {1}

    @needs_ocr_tooling
    def test_never_is_the_default(self, scanned_pdf):
        """Every caller that predates OCR must behave exactly as before."""
        default = PDFProcessor().extract_text_with_pages(scanned_pdf)
        explicit = PDFProcessor().extract_text_with_pages(scanned_pdf, ocr=OCR_NEVER)

        assert [p["text"] for p in default] == [p["text"] for p in explicit]
        assert all(p["ocr"] is False for p in default)


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


class TestMeasuringWhatADocumentGave:
    """The check that would have caught the two absent papers."""

    def _pages(self, *lengths, ocr=0):
        return [{"page": i, "text": "x" * n, "has_equations": False,
                 "ocr": i <= ocr}
                for i, n in enumerate(lengths, 1)]

    def test_a_scan_that_yielded_nothing_is_reported_empty(self):
        measured = corpus.measure("scan.pdf", self._pages(0, 0, 12))

        assert measured.verdict == corpus.EMPTY
        assert measured.needs_attention
        assert measured.pages_with_text == 0

    def test_a_sparse_text_layer_is_reported_thin(self):
        """Birsoy and four others yield 303-370 characters a page, against
        600-2,400 for a healthy paper."""
        measured = corpus.measure("birsoy.pdf", self._pages(*[360] * 10))

        assert measured.verdict == corpus.THIN
        assert measured.needs_attention

    def test_a_normal_paper_passes(self):
        measured = corpus.measure("theis.pdf", self._pages(*[1900] * 10))

        assert measured.verdict == corpus.OK
        assert not measured.needs_attention

    def test_pages_below_the_floor_are_not_counted_as_text(self):
        measured = corpus.measure("mixed.pdf", self._pages(2000, 50, 2000))

        assert measured.pages == 3
        assert measured.pages_with_text == 2
        assert measured.characters == 4000

    def test_ocr_pages_are_counted_and_described(self):
        measured = corpus.measure("scan.pdf", self._pages(1800, 1800, ocr=2))

        assert measured.ocr_pages == 2
        assert "by OCR" in measured.describe()

    def test_the_report_puts_the_worst_first_and_stays_quiet_when_all_is_well(self):
        good = corpus.measure("theis.pdf", self._pages(*[1900] * 5))
        thin = corpus.measure("birsoy.pdf", self._pages(*[360] * 5))
        empty = corpus.measure("scan.pdf", self._pages(0, 0))

        text = corpus.report([good, thin, empty])
        assert text.index("scan.pdf") < text.index("birsoy.pdf")
        assert "theis.pdf" not in text

        assert "Every document yielded" in corpus.report([good])


class TestBuildingDocuments:

    def _pages(self):
        return [{"page": 1, "text": "Theis. " * 50, "has_equations": True, "ocr": False},
                {"page": 2, "text": "short", "has_equations": False, "ocr": False},
                {"page": 3, "text": "Hantush. " * 50, "has_equations": False, "ocr": True}]

    def test_the_folder_decides_the_aquifer_type(self):
        assert corpus.aquifer_for(Path("knowledge source/Leaky Confined Aquifer/x.pdf")) == "leaky"
        assert corpus.aquifer_for(Path("knowledge source/Step drawdown test/x.pdf")) == "general"
        assert corpus.aquifer_for(Path("knowledge source/Unknown Folder/x.pdf")) == "general"

    def test_pages_without_usable_text_are_left_out(self):
        documents = corpus.build_documents(
            Path("knowledge source/Confined Aquifer/theis_1935.pdf"), self._pages())

        assert [d["metadata"]["page"] for d in documents] == [1, 3]

    def test_the_metadata_matches_what_the_index_already_holds(self):
        """The shape existing chunks carry; changing it would split the corpus
        into two incompatible halves."""
        documents = corpus.build_documents(
            Path("knowledge source/Confined Aquifer/theis_1935.pdf"), self._pages())
        metadata = documents[0]["metadata"]

        assert metadata["source"] == "Custom_Confined Aquifer"
        assert metadata["type"] == "research_paper"
        assert metadata["aquifer"] == "confined"
        assert metadata["doc_id"] == "theis_1935_page_1"
        assert metadata["section"] == "theis_1935 - Page 1"
        assert metadata["confidence"] == 1.0

    def test_a_paper_with_no_named_methods_keeps_the_old_tag(self):
        documents = corpus.build_documents(
            Path("knowledge source/Confined Aquifer/theis_1935.pdf"), self._pages())

        assert documents[0]["metadata"]["method"] == "multiple"

    def test_named_methods_are_recorded(self):
        documents = corpus.build_documents(
            Path("knowledge source/Confined Aquifer/theis_1935.pdf"), self._pages(),
            methods=["theis_1935", "cooper_jacob_1946"])

        assert documents[0]["metadata"]["method"] == "theis_1935,cooper_jacob_1946"

    def test_a_recognised_page_says_so(self):
        documents = corpus.build_documents(
            Path("knowledge source/Confined Aquifer/theis_1935.pdf"), self._pages())

        assert documents[0]["metadata"]["ocr"] is False
        assert documents[1]["metadata"]["ocr"] is True


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


class TestWhatKindOfTextAChunkIs:
    """A paper is not all argument.

    Its reference list and contents page were chunked and embedded exactly like
    its derivations, so a question could be answered with a bibliography beside
    it as evidence — which is what the deployed site did when the daily quota
    ran out and the sources panel listed Walton's references.

    The separation is sharper than it looks: a bibliography writes
    "Walton, W.C., 1970, Groundwater resource evaluation", while an argument
    writes "Neuman (1974) showed". Only the first form is matched.
    """

    # Verbatim from wri99-4225 page 45 — the chunk that prompted this.
    BIBLIOGRAPHY = (
        ". 197-206. Walton, W.C., 1970, Groundwater resource evaluation: New "
        "York, McGraw-Hill Book Company, 664 p. _____1987, Groundwater pumping "
        "tests-design and analysis: Chelsea, Mich., Lewis Publishers, 201 p. "
        "Weeks, E.P., 1977, Aquifer tests-the state of the art in hydrology: "
        "Proceedings, p. 14."
    )

    CONTENTS = (
        "CONTENTS\nAbstract ................... 1\nIntroduction "
        "............... 3\nMethods .................... 7\n"
    )

    # Real body text that cites several papers, which must not be caught.
    ARGUMENT = (
        "The delayed yield response described by Boulton (1963) and later "
        "reinterpreted by Neuman (1974) explains the S-shaped time-drawdown "
        "curve. Moench (1995) combined the two treatments, and the resulting "
        "model reduces to Theis (1935) at late time when gravity drainage is "
        "complete. The practical consequence is that an early-time fit "
        "returns the confined storativity rather than the specific yield."
    )

    def test_a_reference_list_is_recognised(self):
        assert corpus.classify_matter(self.BIBLIOGRAPHY) == corpus.REFERENCES

    def test_a_contents_page_is_recognised(self):
        assert corpus.classify_matter(self.CONTENTS) == corpus.CONTENTS

    def test_an_argument_that_cites_papers_is_left_alone(self):
        """The test that matters: prose naming five papers is still prose."""
        assert corpus.classify_matter(self.ARGUMENT) == corpus.BODY

    @pytest.mark.parametrize("text", [
        "", "   ",
        "Drawdown is the difference between the static water level and the "
        "level during pumping, measured in metres.",
        "See Theis (1935) for the derivation.",
    ])
    def test_anything_unrecognised_stays_body(self, text):
        """Conservative on purpose: wrongly dropping a derivation costs an
        answer, while wrongly keeping a reference list costs one line."""
        assert corpus.classify_matter(text) == corpus.BODY

    def test_one_citation_is_not_a_bibliography(self):
        """A short chunk quoting a single paper must survive."""
        single = ("The method follows Papadopulos, I.S. and Cooper, H.H., 1967, "
                  "and applies to large-diameter wells.")

        assert corpus.classify_matter(single) == corpus.BODY

    def test_the_kind_reaches_the_chunk_metadata(self):
        pages = [{"page": 1, "text": self.BIBLIOGRAPHY * 3, "has_equations": False},
                 {"page": 2, "text": self.ARGUMENT * 3, "has_equations": False}]

        documents = corpus.build_documents(
            Path("knowledge source/General/report.pdf"), pages)

        assert documents[0]["metadata"]["matter"] == corpus.REFERENCES
        assert documents[1]["metadata"]["matter"] == corpus.BODY
