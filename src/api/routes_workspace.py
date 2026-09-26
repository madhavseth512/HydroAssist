"""Session, methods, project and readings.

These are the routes behind the sidebar and Step 1 of the workspace.
"""

import io

import pandas as pd
from typing import Optional

from fastapi import (
    APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile,
)

from src.api import presenters, schemas
from src.api.deps import MAX_UPLOAD_BYTES, current_session, inspection_llm
from src.api.sessions import SessionState
from src.calculator import registry
from src.project import workspace as ws
from src.project.model import Project, ProjectFileError
from src.services import data_intake, table_ocr, table_reader

router = APIRouter(prefix="/api", tags=["workspace"])


def test_or_404(session: SessionState, test_id: str):
    try:
        return session.project.get_test(test_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No test with id {test_id!r}.")


def intake_settings(session: SessionState, test) -> dict:
    """What the validator needs to judge a dataset: the test's own setup, which
    decides both the test type and the geometry the checks use."""
    setup = test.setup
    return dict(
        test_type=setup.test_type or registry.CONSTANT_RATE,
        well_radius_m=setup.r_m,
        pumping_rate_m3d=(setup.Q_m3s or 0.0) * 86400.0,
        well_casing_radius_m=setup.r_c_m,
    )


# ── Session ───────────────────────────────────────────────────────────────────

@router.get("/session", response_model=schemas.SessionModel)
def read_session(session: SessionState = Depends(current_session)):
    """The caller's workspace, created on first contact."""
    return presenters.session_model(session)


@router.post("/session", response_model=schemas.SessionModel)
def start_session(session: SessionState = Depends(current_session)):
    return presenters.session_model(session)


# ── The method catalogue ──────────────────────────────────────────────────────

@router.get("/test-types", response_model=list[schemas.TestTypeModel])
def list_test_types():
    """Only the test types that have an implemented method."""
    return [
        schemas.TestTypeModel(id=test_type, label=ws.TEST_TYPE_LABELS[test_type])
        for test_type in ws.TEST_TYPE_LABELS
        if registry.for_test_type(test_type)
    ]


@router.get("/methods", response_model=schemas.MethodListModel)
def list_methods(
    test_type: str = Query(default=registry.CONSTANT_RATE),
    filtered: bool = Query(default=True,
                           description="Apply the validator's suitability check"),
    session: SessionState = Depends(current_session),
):
    """The methods for a test type, as the registry declares them.

    The browser builds its form from this — including which extra inputs a
    method needs — so a new method needs no frontend change.
    """
    if test_type not in registry.TEST_TYPES:
        raise HTTPException(status_code=400,
                            detail=f"Unknown test type {test_type!r}.")
    prepared = session.prepared
    # A validation computed for another test type must not gate this one: the
    # suitable-method list it carries is for its own kind of test.
    validation = (prepared.validation
                  if (filtered and prepared and prepared.test_type == test_type)
                  else None)
    return presenters.method_list(validation, test_type,
                                  preselect_for(session, test_type))


def preselect_for(session: SessionState, test_type: str) -> str:
    """Which method the form should open on.

    The workspace owns the method; the conversation only proposes one. So the
    order is: a method the user accepted from a suggestion card, then the method
    the workspace last actually ran, and only then what the conversation is
    currently talking about.

    Without that last distinction, mentioning a method in the chat quietly moved
    the form under a user who had already chosen and run a different one — which
    is how a fit in progress could end up labelled with a method nobody selected.
    """
    chat_state = session.chat.state
    if chat_state.get("suggestion_confirmed"):
        return chat_state.get("selected_method", "") or ""

    result = getattr(session, "result", None)
    ran = getattr(result, "method", "") if result is not None else ""
    if ran:
        return ran

    return chat_state.get("selected_method", "") or ""


# ── Project ───────────────────────────────────────────────────────────────────

@router.get("/project", response_model=schemas.ProjectModel)
def read_project(session: SessionState = Depends(current_session)):
    return schemas.ProjectModel.from_session(session)


@router.post("/project/new", response_model=schemas.ProjectModel)
def new_project(session: SessionState = Depends(current_session)):
    """Discard the open project. Anything not downloaded is lost, which the UI
    warns about before calling this."""
    session.open_project(ws.new_project())
    return schemas.ProjectModel.from_session(session)


@router.post("/project/open", response_model=schemas.ProjectModel)
async def open_project(session: SessionState = Depends(current_session),
                       file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="That file is too large.")
    try:
        project = Project.from_json(raw)
    except ProjectFileError as exc:
        # The model's messages say what is wrong and what to do, so they are
        # shown to the user as they are.
        raise HTTPException(status_code=400, detail=str(exc))
    session.open_project(project)
    return schemas.ProjectModel.from_session(session)


@router.get("/project/file")
def download_project(session: SessionState = Depends(current_session)):
    """The project as the file the user keeps — this is the only persistence."""
    project = session.project
    return Response(
        content=project.to_json(),
        media_type="application/json",
        headers={"Content-Disposition":
                 f'attachment; filename="{project.filename}"'},
    )


@router.patch("/project/info", response_model=schemas.ProjectModel)
def update_project_info(payload: schemas.ProjectInfoModel,
                        session: SessionState = Depends(current_session)):
    info = session.project.info
    for field, value in payload.model_dump().items():
        setattr(info, field, value)
    session.project.touch()
    return schemas.ProjectModel.from_session(session)


# ── Tests ─────────────────────────────────────────────────────────────────────

@router.post("/tests", response_model=schemas.ProjectModel)
def add_test(session: SessionState = Depends(current_session)):
    session.load_test(session.project.add_test())
    return schemas.ProjectModel.from_session(session)


@router.post("/tests/{test_id}/activate", response_model=schemas.ProjectModel)
def activate_test(test_id: str, session: SessionState = Depends(current_session)):
    session.load_test(test_or_404(session, test_id))
    return schemas.ProjectModel.from_session(session)


@router.patch("/tests/{test_id}", response_model=schemas.ProjectModel)
def update_test(test_id: str, payload: schemas.TestDetailsModel,
                session: SessionState = Depends(current_session)):
    test = test_or_404(session, test_id)
    for field, value in payload.model_dump(exclude_none=True).items():
        setattr(test, field, value)
    session.project.touch()
    return schemas.ProjectModel.from_session(session)


# ── Readings ──────────────────────────────────────────────────────────────────

@router.post("/tests/{test_id}/readings", response_model=schemas.PreparedDataModel)
async def upload_readings(test_id: str,
                          session: SessionState = Depends(current_session),
                          file: UploadFile = File(...),
                          sheet: Optional[str] = Form(None)):
    """Inspect, format, diagnose and validate uploaded readings, and record them
    against the test.

    CSV, tab- or space-delimited text, and Excel workbooks; `table_reader`
    chooses by extension.
    """
    test = test_or_404(session, test_id)
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="That file is too large.")

    try:
        raw_df, notes = table_reader.read_table(raw, file.filename or "", sheet)
    except table_reader.UnreadableTable as exc:
        raise HTTPException(status_code=400,
                            detail=f"That file could not be read: {exc}")

    try:
        prepared = data_intake.prepare_upload(
            raw_df, llm=inspection_llm(), **intake_settings(session, test))
    except Exception as exc:
        raise HTTPException(status_code=400,
                            detail=f"Failed to process the file: {exc}")

    # Which sheet was read belongs with the other conversions, where the user
    # reads what was done to their numbers.
    prepared.conversions.extend(notes)

    data_intake.store_readings(test, prepared.df,
                               source_name=file.filename or "",
                               conversions=prepared.conversions)
    # New data invalidates the last fit, and the conversation must not go on
    # discussing numbers from the previous dataset.
    session.load_test(test)
    session.prepared = prepared
    _stage_for_chat(session, prepared)
    return presenters.prepared_model(prepared)


@router.post("/tests/{test_id}/readings/text",
             response_model=schemas.PreparedDataModel)
def paste_readings(test_id: str, payload: schemas.PastedReadings,
                   session: SessionState = Depends(current_session)):
    """The same intake, for columns pasted from a spreadsheet.

    Most people have the numbers open in Excel already, and saving a file first
    to upload it is a step that earns nothing. The text goes through the same
    reader — the delimiter is sniffed — so a pasted table and an uploaded one
    cannot diverge.
    """
    test = test_or_404(session, test_id)
    raw = payload.text.encode("utf-8")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="That is too much text.")

    try:
        raw_df, notes = table_reader.read_table(raw, table_reader.PASTED)
    except table_reader.UnreadableTable as exc:
        raise HTTPException(status_code=400,
                            detail=f"That text could not be read: {exc}")

    try:
        prepared = data_intake.prepare_upload(
            raw_df, llm=inspection_llm(), **intake_settings(session, test))
    except Exception as exc:
        raise HTTPException(status_code=400,
                            detail=f"Failed to process the readings: {exc}")

    prepared.conversions.extend(notes)
    data_intake.store_readings(test, prepared.df, source_name="pasted",
                               conversions=prepared.conversions)
    session.load_test(test)
    session.prepared = prepared
    _stage_for_chat(session, prepared)
    return presenters.prepared_model(prepared)


@router.post("/tests/{test_id}/readings/image", response_model=schemas.OCRTableModel)
async def read_readings_image(test_id: str,
                              session: SessionState = Depends(current_session),
                              file: UploadFile = File(...)):
    """Propose a grid from a photograph of a field sheet. Stores nothing.

    Deliberately not an intake route. It does not touch `test.readings`, does not
    inspect or validate, and does not reach the chat: it hands back what OCR
    saw, cell by cell, with the confidence of each. The user corrects the grid
    and sends it as text, so an image reaches a calculator only through a person.

    A misread digit does not fail — it fits, and reports a wrong transmissivity
    with no warning anywhere. That is the whole reason this route is separate.
    """
    # The test need not hold the readings, but it must exist: a proposal belongs
    # to the test the user is filling in.
    test_or_404(session, test_id)
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="That image is too large.")

    try:
        read = table_ocr.read_image_table(raw)
    except table_ocr.OCRNotAvailable as exc:
        # 503 when the server cannot do it at all, 400 when this image is the
        # problem — the browser tells the two apart to decide what to offer.
        status = 400 if "could not be opened" in str(exc) else 503
        raise HTTPException(status_code=status, detail=str(exc))

    return presenters.ocr_table_model(read)


@router.get("/tests/{test_id}/readings", response_model=schemas.PreparedDataModel)
def read_readings(test_id: str, session: SessionState = Depends(current_session),
                  rows: int = Query(presenters.PREVIEW_ROWS, ge=1, le=100_000)):
    """The readings already held by a test — typically a reopened project.

    `rows` is how much of the table to return. The default is a preview; the
    workspace asks for the whole table when the user opens it to check what was
    read, which is the only way to tell a misread column from a correct one.
    """
    test = test_or_404(session, test_id)
    if test.readings is None:
        raise HTTPException(status_code=404,
                            detail="This test has no readings yet.")
    prepared = data_intake.prepare(test.readings.to_frame(),
                                   **intake_settings(session, test))
    if test.id == session.active_test().id:
        session.prepared = prepared
        _stage_for_chat(session, prepared)
    return presenters.prepared_model(prepared, rows=rows)


def _stage_for_chat(session: SessionState, prepared) -> None:
    """Publish the series to the conversation, so the method suggestion is read
    off the derivative rather than guessed from the user's words."""
    df = prepared.df
    session.chat.stage_series(df.time_s.values, df.drawdown_m.values,
                              Q=session.active_test().setup.Q_m3s)
