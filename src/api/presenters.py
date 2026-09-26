"""Turning domain objects into response models.

Kept out of the routes so that each route reads as one action, and out of the
schemas so the models stay plain data. Figures are serialised here, in one
place: the browser receives exactly the Plotly figure the registry's adapters
build, which is the same one the PDF draws.
"""

from typing import Optional

import plotly.io as pio

from src.api import schemas
from src.calculator import registry
from src.project import workspace as ws
from src.services import analysis as analysis_service


def figure_json(figure) -> Optional[str]:
    """A Plotly figure as JSON, or None if it cannot be drawn.

    A missing figure must never fail a request that otherwise succeeded: the
    numbers are the result, the picture is how it is explained.
    """
    if figure is None:
        return None
    try:
        return pio.to_json(figure)
    except Exception:
        return None


# ── Prepared data ─────────────────────────────────────────────────────────────

def inspection_model(inspection) -> Optional[schemas.InspectionModel]:
    if inspection is None:
        return None
    return schemas.InspectionModel(
        time_col=inspection.time_col,
        time_unit=inspection.time_unit,
        drawdown_cols=list(inspection.drawdown_cols),
        drawdown_unit=inspection.drawdown_unit,
        confidence=inspection.confidence,
        issues=list(inspection.issues),
    )


def validation_model(validation) -> schemas.ValidationModel:
    return schemas.ValidationModel(
        is_usable=validation.is_usable,
        errors=list(validation.errors),
        warnings=list(validation.warnings),
        info=list(validation.info),
        suitable_methods=list(validation.suitable_methods),
    )


def diagnosis_model(prepared) -> Optional[schemas.DiagnosisModel]:
    if prepared.diagnosis is None:
        return None
    signature = prepared.diagnosis
    return schemas.DiagnosisModel(
        regime=signature.regime,
        evidence=signature.evidence,
        confidence=getattr(signature, "confidence", 0.0) or 0.0,
        notes=list(signature.notes),
        consistent_methods=list(prepared.consistent_methods),
    )


def prepared_model(prepared) -> schemas.PreparedDataModel:
    from src.calculator.plots import derivative_diagnostic

    df = prepared.df
    figure = None
    if prepared.diagnosis is not None:
        try:
            figure = figure_json(
                derivative_diagnostic(df.time_s.values, df.drawdown_m.values))
        except Exception:
            figure = None

    return schemas.PreparedDataModel(
        n_points=len(df),
        summary=prepared.summary(),
        preview=df.head(10).to_dict(orient="records"),
        conversions=list(prepared.conversions),
        rows_removed=list(prepared.rows_removed),
        inspection=inspection_model(prepared.inspection),
        validation=validation_model(prepared.validation),
        diagnosis=diagnosis_model(prepared),
        diagnosis_error=prepared.diagnosis_error,
        derivative_figure=figure,
    )


def ocr_table_model(read) -> schemas.OCRTableModel:
    """A photographed table as a grid to confirm.

    Every cell keeps the text Tesseract saw and its confidence, because the check
    the user performs is against the sheet in front of them, not against a
    number this has already decided to believe.
    """
    return schemas.OCRTableModel(
        rows=[[None if cell is None else schemas.OCRCellModel(
            text=cell.text, value=cell.value, confidence=cell.confidence,
            uncertain=cell.uncertain) for cell in row] for row in read.rows],
        columns=read.columns,
        notes=list(read.notes),
        total_cells=read.total_cells,
        uncertain_cells=read.uncertain_cells,
        mostly_uncertain=read.mostly_uncertain,
    )


# ── Results ───────────────────────────────────────────────────────────────────

def result_model(result, record: dict) -> schemas.ResultModel:
    if not result.success:
        return schemas.ResultModel(
            success=False,
            error_message=result.error_message,
            method=result.method,
            method_id=result.method_id,
        )

    estimates_S = analysis_service.estimates_S(result)
    return schemas.ResultModel(
        success=True,
        method=result.method,
        method_id=result.method_id,
        T=result.T,
        T_day=result.T_day,
        T_ci=list(result.T_ci) if result.T_ci else None,
        # None rather than the unset 0.0 for a method that reports no
        # storativity, such as Theis recovery.
        S=result.S if estimates_S else None,
        S_ci=list(result.S_ci) if (estimates_S and result.S_ci) else None,
        estimates_storativity=estimates_S,
        rmse=result.rmse,
        r_squared=result.r_squared,
        parameters=[schemas.ParameterModel.from_parameter(p)
                    for p in result.parameters],
        gof=result.to_state_dict().get("gof"),
        diagnostics={k: float(v) for k, v in result.diagnostics.items()
                     if isinstance(v, (int, float)) and not isinstance(v, bool)},
        validity_notes=list(result.validity_notes),
        figure=figure_json(_safe_diagnostic(result, record)),
    )


def _safe_diagnostic(result, record: dict):
    try:
        return analysis_service.diagnostic_plot(result, record)
    except Exception:
        return None


# ── Comparison ────────────────────────────────────────────────────────────────

def comparison_model(test) -> schemas.ComparisonModel:
    rows = [
        schemas.ComparisonRowModel(
            rank=rank,
            analysis_id=row["analysis_id"],
            method=row["method"],
            n_params=row["n_params"],
            T_m2_per_day=row["T_m2_per_day"],
            S=row["S"],
            r_squared=row["r_squared"],
            aicc=row["aicc"],
            delta_aicc=row["delta_aicc"],
            akaike_weight=row["akaike_weight"],
            evidence=ws.evidence(row["delta_aicc"]),
        )
        for rank, row in enumerate(test.comparison(), 1)
    ]
    return schemas.ComparisonModel(test_id=test.id, test_name=test.name,
                                   rows=rows, n_stale=ws.stale_count(test))


# ── Chat ──────────────────────────────────────────────────────────────────────

def _message_models(messages) -> list:
    out = []
    for message in messages:
        role = "user" if message.__class__.__name__ == "HumanMessage" else "assistant"
        out.append(schemas.MessageModel(role=role, content=message.content))
    return out


def _retrieved_models(docs) -> list:
    out = []
    for doc in docs:
        meta = doc.get("metadata", {})
        content = doc.get("content", "")
        # A help passage has a document and a section rather than a method and
        # an aquifer, and "HydroAssist help" alone says too little to be worth
        # showing.
        source = meta.get("source", "Unknown")
        if meta.get("type") == "help":
            source = f"{meta.get('document', source)} — {meta.get('section', '')}".strip(" —")

        out.append(schemas.RetrievedDocModel(
            source=source,
            method=meta.get("method", ""),
            aquifer=meta.get("aquifer", ""),
            score=doc.get("score", 0.0),
            preview=content[:200] + "…" if len(content) > 200 else content,
        ))
    return out


def chat_state(session) -> schemas.ChatStateModel:
    state = session.chat.state
    context = session.chat.get_context()
    raw_suggestion = state.get("method_suggestion") or {}

    suggestion = None
    # A confirmed method needs no card: the question it asks has been answered.
    if raw_suggestion and not context.get("suggestion_confirmed"):
        suggestion = schemas.SuggestionModel(
            method=raw_suggestion.get("method", ""),
            reason=raw_suggestion.get("reason", ""),
            confidence=float(raw_suggestion.get("confidence", 0.0) or 0.0),
            supported=bool(raw_suggestion.get("supported", True)),
        )

    return schemas.ChatStateModel(
        messages=_message_models(state.get("messages", [])),
        context=schemas.ChatContextModel(
            intent=context.get("intent", "unknown"),
            aquifer=context.get("aquifer", "unknown"),
            method=context.get("method", ""),
            suggestion_confirmed=bool(context.get("suggestion_confirmed")),
            has_result=bool(context.get("has_result")),
            message_count=int(context.get("message_count", 0)),
        ),
        suggestion=suggestion,
        retrieved=_retrieved_models(state.get("retrieved_docs", [])),
        actions=[schemas.ActionModel(kind=action.get("kind", "open_panel"),
                                     target=action.get("target", ""),
                                     label=action.get("label", ""))
                 for action in (state.get("proposed_actions") or [])],
    )


def session_model(session) -> schemas.SessionModel:
    return schemas.SessionModel(
        session_id=session.id,
        project=schemas.ProjectModel.from_session(session),
        chat=chat_state(session),
    )


def method_list(validation, test_type: str,
                suggested: str = "") -> schemas.MethodListModel:
    offered, gated = analysis_service.method_options(validation, test_type)
    specs = [registry.require(label) for label in offered]
    return schemas.MethodListModel(
        test_type=test_type,
        offered=[schemas.MethodModel.from_spec(spec) for spec in specs],
        gated_out=gated,
        gated_reasons=analysis_service.gating_reasons(validation, gated),
        help=analysis_service.method_help(offered),
        default_index=analysis_service.default_method_index(offered, suggested),
    )
