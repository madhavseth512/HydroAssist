"""Running a fit, and what follows one: manual matching, saving, comparing,
reporting.

Every route works on the session's active test, which is what the browser is
showing, and on `services.analysis`, which is what the Streamlit app used too.
"""

from fastapi import APIRouter, Depends, HTTPException, Response

from src.api import presenters, schemas
from src.api.deps import current_session
from src.api.routes_workspace import test_or_404
from src.api.sessions import SessionState
from src.services import analysis as analysis_service

router = APIRouter(prefix="/api", tags=["analysis"])


@router.put("/tests/{test_id}/setup", response_model=schemas.TestModel)
def update_setup(test_id: str, payload: schemas.SetupModel,
                 session: SessionState = Depends(current_session)):
    """Store the Step 2 form against the test, so saving the project captures
    what is on screen."""
    test = test_or_404(session, test_id)
    test.setup = payload.to_setup()
    session.project.touch()
    return schemas.TestModel.from_test(test)


@router.post("/analyses/run", response_model=schemas.ResultModel)
def run_analysis(payload: schemas.RunRequest,
                 session: SessionState = Depends(current_session)):
    """Fit the chosen method to the active test's readings.

    The setup is stored on the test first, so the analysis, the project and the
    form all describe the same test afterwards.
    """
    test = session.active_test()
    if test.readings is None:
        raise HTTPException(status_code=400,
                            detail="Load readings into this test before running "
                                   "an analysis.")

    setup = payload.setup.to_setup()
    test.setup = setup
    df = test.readings.to_frame()

    try:
        result = analysis_service.run_analysis(df, payload.method, setup)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        # A setup a method cannot run with, e.g. recovery without a duration.
        raise HTTPException(status_code=400, detail=str(exc))

    record = analysis_service.run_record(setup, payload.method, result,
                                         n_points=len(df))
    session.result = result
    session.record = record
    session.saved_result = None

    # Publish the result into the conversation, which is what lets the
    # consultant discuss these numbers.
    session.chat.record_calculation(result.to_state_dict(),
                                    analysis_service.chat_record(record))
    session.chat.confirm_method(record["method"])

    return presenters.result_model(result, record)


@router.get("/analyses/current", response_model=schemas.ResultModel)
def current_result(session: SessionState = Depends(current_session)):
    """The last result, for a browser that reloaded mid-session."""
    if session.result is None:
        raise HTTPException(status_code=404, detail="No analysis has been run yet.")
    return presenters.result_model(session.result, session.record)


@router.get("/analyses/history", response_model=list[schemas.AnalysisRunModel])
def analysis_history(session: SessionState = Depends(current_session)):
    """Every fit run in this session, oldest first.

    Saving an analysis to the project is a deliberate act, so a fit that was run
    and not saved used to leave no trace at all — the next one overwrote it. This
    is what was run, which is a different question from what was kept.
    """
    history = session.chat.state.get("analysis_history") or []
    return [schemas.AnalysisRunModel(**{
        key: row.get(key) for key in
        ("method", "success", "T_day", "S", "r_squared", "rmse", "n_points")
        if row.get(key) is not None
    }) for row in history]


@router.post("/analyses/manual-match", response_model=schemas.ManualMatchModel)
def manual_match(payload: schemas.ManualMatchRequest,
                 session: SessionState = Depends(current_session)):
    """The hand-matched curve for the sliders. Every parameter other than T and
    S stays at its fitted value; a method that estimates no S ignores it."""
    if session.result is None or not session.result.success:
        raise HTTPException(status_code=400,
                            detail="Run an analysis before matching by hand.")

    result = session.result
    S = payload.S if payload.S is not None else result.S
    figure, rmse_mm = analysis_service.manual_match(result, session.record,
                                                    payload.T, S)
    return schemas.ManualMatchModel(
        figure=presenters.figure_json(figure) or "",
        rmse_mm=rmse_mm,
        auto_rmse_mm=result.rmse * 1000.0,
    )


@router.post("/analyses/save", response_model=schemas.ComparisonModel)
def save_analysis(session: SessionState = Depends(current_session)):
    """Record the last result against the active test, so methods can be
    compared rather than recomputed."""
    if session.result is None or not session.result.success:
        raise HTTPException(status_code=400, detail="There is no result to save.")
    if session.saved_result is session.result:
        raise HTTPException(status_code=409,
                            detail="This analysis is already saved.")

    test = session.active_test()
    if test.readings is None:
        raise HTTPException(status_code=400,
                            detail="Load readings into this test before saving "
                                   "an analysis.")

    analysis_service.save_analysis(test, session.result, session.record)
    session.saved_result = session.result
    session.project.touch()
    return presenters.comparison_model(test)


@router.get("/tests/{test_id}/comparison", response_model=schemas.ComparisonModel)
def comparison(test_id: str, session: SessionState = Depends(current_session)):
    """Saved analyses ranked by AICc — never by R², which cannot penalise an
    extra parameter. Analyses run on replaced readings are left out."""
    return presenters.comparison_model(test_or_404(session, test_id))


@router.delete("/tests/{test_id}/analyses/{analysis_id}",
               response_model=schemas.ComparisonModel)
def remove_analysis(test_id: str, analysis_id: str,
                    session: SessionState = Depends(current_session)):
    test = test_or_404(session, test_id)
    test.remove_analysis(analysis_id)
    session.project.touch()
    return presenters.comparison_model(test)


@router.get("/report.pdf")
def report(session: SessionState = Depends(current_session)):
    """The client deliverable: the fit, its figure, its validity notes and the
    project metadata."""
    if session.result is None or not session.result.success:
        raise HTTPException(status_code=400,
                            detail="Run an analysis before generating a report.")

    test = session.active_test()
    df = test.readings.to_frame() if test.readings is not None else None
    if df is None:
        raise HTTPException(status_code=400,
                            detail="This test no longer holds the readings the "
                                   "analysis was run on.")

    pdf, filename = analysis_service.report_pdf(
        session.result, session.record, df,
        project_info=session.project.info, test_label=test.name)
    return Response(content=pdf, media_type="application/pdf",
                    headers={"Content-Disposition":
                             f'attachment; filename="{filename}"'})
