"""What the API accepts and returns.

These models are also the frontend's types: `openapi-typescript` generates
TypeScript from the OpenAPI schema FastAPI derives from them, so the two sides
cannot drift apart.

Two conventions:
  * Setups are exchanged in the unit the user typed (`q_value` + `q_unit`,
    `t_p_hours`), never in SI alone. A reopened form has to read back the way it
    was entered, and `TestSetup` already stores both.
  * Figures cross the wire as Plotly JSON strings, exactly as the registry's
    plot adapters build them, so the browser renders the same figure the PDF
    draws.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.calculator import registry
from src.project import workspace as ws
from src.project.model import ProjectInfo, TestSetup


# ── Test setup ────────────────────────────────────────────────────────────────

class SetupModel(BaseModel):
    """The Step 2 form: the test parameters an analysis runs with."""

    q_value: float = Field(default=500.0, description="Pumping rate, as entered")
    q_unit: str = Field(default="m³/day")
    r_m: float = Field(default=50.0, description="Observation well distance")
    r_w_m: float = Field(default=0.15, description="Well screen radius")
    r_c_m: float = Field(default=0.10, description="Casing radius")
    test_type: str = Field(default=registry.CONSTANT_RATE)
    t_p_hours: Optional[float] = Field(
        default=None, description="Pumping duration before shut-down (recovery)")
    recovery_clock: str = Field(
        default="since_stop",
        description="Whether the time column counts from pump start or stop")

    def to_setup(self) -> TestSetup:
        return TestSetup(
            Q_m3s=ws.q_to_si(self.q_value, self.q_unit),
            Q_entered=self.q_value,
            Q_unit=self.q_unit,
            r_m=self.r_m,
            r_w_m=self.r_w_m,
            r_c_m=self.r_c_m,
            test_type=self.test_type,
            t_p_s=(self.t_p_hours * 3600.0) if self.t_p_hours else None,
            recovery_clock=self.recovery_clock,
        )

    @classmethod
    def from_setup(cls, setup: TestSetup) -> "SetupModel":
        values = ws.inputs_from_setup(setup)
        return cls(
            q_value=values[ws.KEY_Q],
            q_unit=values[ws.KEY_Q_UNIT],
            r_m=values[ws.KEY_R],
            r_w_m=values[ws.KEY_RW],
            r_c_m=values[ws.KEY_RC],
            test_type=values[ws.KEY_TEST_TYPE],
            t_p_hours=(setup.t_p_s / 3600.0) if setup.t_p_s else None,
            recovery_clock=values[ws.KEY_CLOCK],
        )


# ── Methods ───────────────────────────────────────────────────────────────────

class MethodModel(BaseModel):
    """A registry entry, which is what drives the method form in the browser."""

    id: str
    label: str
    year: int
    authors: List[str]
    citation: str
    aquifer_types: List[str]
    test_types: List[str]
    parameters: List[str]
    data_source: str
    requires: List[str]
    ui_help: str
    selection_rationale: str
    # What the method takes for granted. A card that lists only what a method
    # does, and never what it assumes, is selling rather than informing.
    assumptions: List[str] = Field(default_factory=list)
    needs_well_geometry: bool
    estimates_storativity: bool

    @classmethod
    def from_spec(cls, spec) -> "MethodModel":
        return cls(
            id=spec.id, label=spec.label, year=spec.year,
            authors=list(spec.authors), citation=spec.citation,
            aquifer_types=sorted(spec.aquifer_types),
            test_types=sorted(spec.test_types),
            parameters=list(spec.parameters),
            data_source=spec.data_source,
            requires=list(spec.requires),
            ui_help=spec.ui_help,
            selection_rationale=spec.selection_rationale,
            assumptions=list(spec.assumptions),
            needs_well_geometry=spec.data_source == registry.PUMPED_WELL,
            estimates_storativity="S" in spec.parameters,
        )


class MethodListModel(BaseModel):
    test_type: str
    offered: List[MethodModel]
    gated_out: List[str] = Field(
        default_factory=list,
        description="Methods the validator does not offer for this dataset")
    gated_reasons: Dict[str, str] = Field(
        default_factory=dict,
        description="Why each excluded method was excluded. A greyed name "
                    "teaches nothing; the reason is what a practitioner learns "
                    "from.")
    help: str = ""
    default_index: int = 0


class TestTypeModel(BaseModel):
    id: str
    label: str


# ── Project ───────────────────────────────────────────────────────────────────

class ProjectInfoModel(BaseModel):
    title: str = "Untitled project"
    client: str = ""
    project_number: str = ""
    location: str = ""
    test_date: str = ""
    analyst: str = ""
    notes: str = ""

    @classmethod
    def from_info(cls, info: ProjectInfo) -> "ProjectInfoModel":
        return cls(**vars(info))


class TestDetailsModel(BaseModel):
    """The editable fields of one test."""

    name: Optional[str] = None
    pumping_well: Optional[str] = None
    observation_well: Optional[str] = None


class TestModel(BaseModel):
    id: str
    name: str
    pumping_well: str
    observation_well: str
    setup: SetupModel
    n_readings: int = 0
    readings_source: str = ""
    n_analyses: int = 0
    n_stale_analyses: int = 0

    @classmethod
    def from_test(cls, test) -> "TestModel":
        return cls(
            id=test.id, name=test.name,
            pumping_well=test.pumping_well,
            observation_well=test.observation_well,
            setup=SetupModel.from_setup(test.setup),
            n_readings=len(test.readings) if test.readings else 0,
            readings_source=test.readings.source_name if test.readings else "",
            n_analyses=len(test.analyses),
            n_stale_analyses=ws.stale_count(test),
        )


class ProjectModel(BaseModel):
    info: ProjectInfoModel
    tests: List[TestModel]
    active_test_id: str
    filename: str
    summary: str
    analysis_count: int
    modified_at: str

    @classmethod
    def from_session(cls, session) -> "ProjectModel":
        project = session.project
        return cls(
            info=ProjectInfoModel.from_info(project.info),
            tests=[TestModel.from_test(t) for t in project.tests],
            active_test_id=session.active_test().id,
            filename=project.filename,
            summary=ws.project_summary(project),
            analysis_count=project.analysis_count,
            modified_at=project.modified_at,
        )


# ── Prepared data ─────────────────────────────────────────────────────────────

class InspectionModel(BaseModel):
    time_col: str = ""
    time_unit: str = ""
    drawdown_cols: List[str] = Field(default_factory=list)
    drawdown_unit: str = ""
    confidence: float = 0.0
    issues: List[str] = Field(default_factory=list)


class ValidationModel(BaseModel):
    is_usable: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    info: List[str] = Field(default_factory=list)
    suitable_methods: List[str] = Field(default_factory=list)


class DiagnosisModel(BaseModel):
    """What the Bourdet derivative says the flow regime is."""

    regime: str
    evidence: str
    confidence: float = 0.0
    notes: List[str] = Field(default_factory=list)
    consistent_methods: List[str] = Field(default_factory=list)


class PastedReadings(BaseModel):
    """Readings pasted from a spreadsheet rather than uploaded as a file.

    The limit is generous because a long test is a long table: 20,000 readings
    of "time,drawdown" is well under it, and the byte ceiling still applies
    once encoded.
    """
    text: str = Field(min_length=1, max_length=2_000_000)


class OCRCellModel(BaseModel):
    """One cell OCR proposed, with how far it is to be trusted.

    `text` rather than a number, because what the user must check is the
    characters Tesseract saw. An empty cell is `null` in the row.
    """
    text: str
    value: Optional[float] = None
    confidence: float = 0.0
    uncertain: bool = False


class OCRTableModel(BaseModel):
    """A photographed table, proposed for confirmation and nothing more.

    Deliberately not a `PreparedDataModel`: nothing here has been inspected,
    validated or stored against the test. A misread digit yields a plausible fit
    and a wrong transmissivity, so the numbers become readings only when the
    user sends the corrected grid back through the ordinary text route.
    """
    rows: List[List[Optional[OCRCellModel]]] = Field(default_factory=list)
    columns: int = 0
    notes: List[str] = Field(default_factory=list)
    total_cells: int = 0
    uncertain_cells: int = 0
    mostly_uncertain: bool = False


class PreparedDataModel(BaseModel):
    n_points: int
    summary: str
    preview: List[dict] = Field(default_factory=list)
    conversions: List[str] = Field(default_factory=list)
    rows_removed: List[str] = Field(default_factory=list)
    inspection: Optional[InspectionModel] = None
    validation: ValidationModel
    diagnosis: Optional[DiagnosisModel] = None
    diagnosis_error: str = ""
    derivative_figure: Optional[str] = Field(
        default=None, description="Plotly JSON for the derivative diagnostic")


# ── Results ───────────────────────────────────────────────────────────────────

class ParameterModel(BaseModel):
    name: str
    symbol: str
    value: float
    unit: str
    display: float
    display_unit: str
    ci: Optional[List[float]] = None
    display_ci: Optional[List[float]] = None
    reliable: bool = True
    reliability_note: str = ""
    derived: bool = False
    fixed: bool = False
    at_bound: Optional[str] = None

    @classmethod
    def from_parameter(cls, p) -> "ParameterModel":
        return cls(
            name=p.name, symbol=p.symbol, value=p.value, unit=p.unit,
            display=p.display, display_unit=p.display_unit,
            ci=list(p.ci) if p.ci else None,
            display_ci=list(p.display_ci) if p.display_ci else None,
            reliable=p.reliable, reliability_note=p.reliability_note,
            derived=p.derived, fixed=p.fixed, at_bound=p.at_bound,
        )


class RunRequest(BaseModel):
    method: str
    setup: SetupModel


class ResultModel(BaseModel):
    success: bool
    error_message: str = ""
    method: str = ""
    method_id: str = ""
    T: float = 0.0
    T_day: float = 0.0
    T_ci: Optional[List[float]] = None
    S: Optional[float] = None
    S_ci: Optional[List[float]] = None
    estimates_storativity: bool = False
    rmse: float = 0.0
    r_squared: float = 0.0
    parameters: List[ParameterModel] = Field(default_factory=list)
    gof: Optional[dict] = None
    diagnostics: dict = Field(default_factory=dict)
    validity_notes: List[str] = Field(default_factory=list)
    figure: Optional[str] = Field(default=None,
                                  description="Plotly JSON for the fit diagnostic")


class ManualMatchRequest(BaseModel):
    T: float
    S: Optional[float] = None


class ManualMatchModel(BaseModel):
    figure: str
    rmse_mm: float
    auto_rmse_mm: float


class ComparisonRowModel(BaseModel):
    rank: int
    analysis_id: str
    method: str
    n_params: Optional[int] = None
    T_m2_per_day: Optional[float] = None
    S: Optional[float] = None
    r_squared: Optional[float] = None
    aicc: Optional[float] = None
    delta_aicc: Optional[float] = None
    akaike_weight: Optional[float] = None
    evidence: str = ""


class ComparisonModel(BaseModel):
    test_id: str
    test_name: str
    rows: List[ComparisonRowModel] = Field(default_factory=list)
    n_stale: int = 0


# ── Chat ──────────────────────────────────────────────────────────────────────

class RetrievedDocModel(BaseModel):
    """One passage an answer was built from.

    `document` is what the paper is called, not what folder it was filed in: the
    field used to carry "Custom_Confined Aquifer", which identifies nothing a
    reader could go and look up. `url` is present only where a link can be
    derived rather than guessed — most of the corpus is paywalled, and a citation
    that says so is more use than one that implies a link and 404s.
    """
    document: str = ""
    page: Optional[int] = None
    url: str = ""
    method: str = ""
    aquifer: str = ""
    score: float = 0.0
    # The opening of the passage, and the whole of it for a reader who wants to
    # check the answer against what was actually retrieved.
    preview: str = ""
    text: str = ""


class MessageModel(BaseModel):
    role: str
    content: str
    # The documents this particular answer used. One list under the whole
    # conversation belonged to the latest answer and was wrong under the rest.
    sources: List[RetrievedDocModel] = Field(default_factory=list)


class SuggestionModel(BaseModel):
    method: str = ""
    reason: str = ""
    confidence: float = 0.0
    supported: bool = True


class ChatContextModel(BaseModel):
    intent: str = "unknown"
    aquifer: str = "unknown"
    method: str = ""
    suggestion_confirmed: bool = False
    has_result: bool = False
    message_count: int = 0


class ActionModel(BaseModel):
    """Somewhere the guide offered to take the user.

    An offer, never a move: the browser renders a button and nothing happens
    until it is pressed. An assistant that navigates the workspace by itself
    while someone is mid-analysis is a worse failure than one that answers
    badly.
    """
    kind: str = "open_panel"
    target: str = ""
    label: str = ""


class ChatStateModel(BaseModel):
    messages: List[MessageModel] = Field(default_factory=list)
    context: ChatContextModel
    suggestion: Optional[SuggestionModel] = None
    retrieved: List[RetrievedDocModel] = Field(default_factory=list)
    actions: List[ActionModel] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class ConfirmMethodRequest(BaseModel):
    method: str


class SessionModel(BaseModel):
    session_id: str
    project: ProjectModel
    chat: ChatStateModel


class MessageResponse(BaseModel):
    """A plain acknowledgement, for actions whose result is the new state."""

    ok: bool = True
    detail: str = ""
