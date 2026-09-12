"""Session, methods, project and readings.

These are the routes behind the sidebar and Step 1 of the workspace.
"""

import io

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile

from src.api import presenters, schemas
from src.api.deps import MAX_UPLOAD_BYTES, current_session, inspection_llm
from src.api.sessions import SessionState
from src.calculator import registry
from src.project import workspace as ws
from src.project.model import Project, ProjectFileError
from src.services import data_intake

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
    suggested = session.chat.state.get("selected_method", "") or ""
    return presenters.method_list(validation, test_type, suggested)


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
                          file: UploadFile = File(...)):
    """Inspect, format, diagnose and validate an uploaded CSV, and record the
    readings against the test."""
    test = test_or_404(session, test_id)
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="That file is too large.")

    try:
        raw_df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400,
                            detail=f"That file could not be read as CSV: {exc}")

    try:
        prepared = data_intake.prepare_upload(
            raw_df, llm=inspection_llm(), **intake_settings(session, test))
    except Exception as exc:
        raise HTTPException(status_code=400,
                            detail=f"Failed to process the file: {exc}")

    data_intake.store_readings(test, prepared.df,
                               source_name=file.filename or "",
                               conversions=prepared.conversions)
    # New data invalidates the last fit, and the conversation must not go on
    # discussing numbers from the previous dataset.
    session.load_test(test)
    session.prepared = prepared
    _stage_for_chat(session, prepared)
    return presenters.prepared_model(prepared)


@router.get("/tests/{test_id}/readings", response_model=schemas.PreparedDataModel)
def read_readings(test_id: str, session: SessionState = Depends(current_session)):
    """The readings already held by a test — typically a reopened project."""
    test = test_or_404(session, test_id)
    if test.readings is None:
        raise HTTPException(status_code=404,
                            detail="This test has no readings yet.")
    prepared = data_intake.prepare(test.readings.to_frame(),
                                   **intake_settings(session, test))
    if test.id == session.active_test().id:
        session.prepared = prepared
        _stage_for_chat(session, prepared)
    return presenters.prepared_model(prepared)


def _stage_for_chat(session: SessionState, prepared) -> None:
    """Publish the series to the conversation, so the method suggestion is read
    off the derivative rather than guessed from the user's words."""
    df = prepared.df
    session.chat.stage_series(df.time_s.values, df.drawdown_m.values,
                              Q=session.active_test().setup.Q_m3s)
