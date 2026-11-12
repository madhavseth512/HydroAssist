"""Turning an uploaded file into readings that can be analysed.

The pipeline is: inspect the columns, format to SI, read the flow regime off the
Bourdet derivative, and validate. `app.py` ran those four steps interleaved with
its own widgets, so the sequence could not be reused or tested on its own. Here
it returns plain data and renders nothing.

The derivative comes before the method choice deliberately: modern practice
reads the diagnostic first and picks the model from its shape, rather than
fitting a model chosen in advance and checking R² afterwards.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence

import pandas as pd

from src.calculator import registry
from src.project.model import PumpingTest, Readings


@dataclass
class PreparedData:
    """An uploaded or reopened dataset, ready to analyse or to explain."""

    df: pd.DataFrame
    inspection: object = None               # InspectionResult, when a file was read
    conversions: list = field(default_factory=list)
    rows_removed: list = field(default_factory=list)
    validation: object = None               # ValidationResult
    # The test type the validation was computed for. A dataset judged as a
    # recovery record says nothing about which pumping methods suit it.
    test_type: str = registry.CONSTANT_RATE
    diagnosis: object = None                # DerivativeSignature, when readable
    diagnosis_error: str = ""
    consistent_methods: list = field(default_factory=list)   # labels

    @property
    def quantity(self) -> str:
        """What the response column holds."""
        return "concentration" if self.test_type == registry.TRACER else "drawdown"

    @property
    def is_usable(self) -> bool:
        return bool(self.validation is not None and self.validation.is_usable)

    def summary(self) -> str:
        """The one-line description of the series the workspace shows."""
        if self.test_type == registry.TRACER:
            return (f"{len(self.df)} data points ready | "
                    f"t: {self.df.time_s.min():.0f}s – {self.df.time_s.max():.0f}s | "
                    f"C: {self.df.drawdown_m.min():.3g} – "
                    f"{self.df.drawdown_m.max():.3g}")
        return (f"{len(self.df)} data points ready | "
                f"t: {self.df.time_s.min():.0f}s – {self.df.time_s.max():.0f}s | "
                f"s: {self.df.drawdown_m.min():.3f}m – "
                f"{self.df.drawdown_m.max():.3f}m")


def diagnose_series(time_s, drawdown_m) -> tuple:
    """(signature, error). A dataset too short or too noisy to differentiate is
    not a failure — the workspace still analyses it, it just cannot say what the
    flow regime looks like."""
    from src.calculator.derivative import diagnose

    try:
        return diagnose(time_s, drawdown_m), ""
    except Exception as exc:                 # noqa: BLE001 - reported, not raised
        return None, str(exc)


def consistent_method_labels(signature) -> list:
    """The registry labels a derivative signature points to."""
    if signature is None:
        return []
    specs = [registry.get(method_id) for method_id in signature.suggests]
    return [spec.label for spec in specs if spec]


def inspect_and_format(raw_df: pd.DataFrame, llm=None,
                       test_type: str = registry.CONSTANT_RATE) -> tuple:
    """(formatted frame, inspection, formatting outcome).

    `llm` is the CSVInspector's layer-3 fallback for ambiguous column names; it
    degrades to regex and statistical detection when None.
    """
    from src.data.csv_inspector import CSVInspector
    from src.data.formatter import DataFormatter

    inspection = CSVInspector(llm=llm).inspect(
        raw_df, tracer=test_type == registry.TRACER)
    formatted = DataFormatter().format(raw_df, inspection)
    return formatted.df, inspection, formatted


def prepare(df: pd.DataFrame, *,
            test_type: str = registry.CONSTANT_RATE,
            well_radius_m: Optional[float] = None,
            pumping_rate_m3d: Optional[float] = None,
            well_casing_radius_m: Optional[float] = None,
            inspection=None,
            conversions: Sequence[str] = (),
            rows_removed: Sequence[str] = ()) -> PreparedData:
    """Diagnose and validate an already-formatted frame.

    Used for both a fresh upload and readings reopened from a project, so the
    two paths cannot drift apart.
    """
    from src.data.validator import DataValidator

    # The Bourdet derivative reads a flow regime off drawdown. On a breakthrough
    # curve it would name one anyway, so it is not asked.
    signature, error = (None, "") if test_type == registry.TRACER else (
        diagnose_series(df.time_s.values, df.drawdown_m.values))
    validation = DataValidator().validate(
        df,
        well_radius_m=well_radius_m,
        pumping_rate_m3d=pumping_rate_m3d,
        well_casing_radius_m=well_casing_radius_m,
        test_type=test_type,
    )
    return PreparedData(
        df=df,
        test_type=test_type,
        inspection=inspection,
        conversions=list(conversions),
        rows_removed=list(rows_removed),
        validation=validation,
        diagnosis=signature,
        diagnosis_error=error,
        consistent_methods=consistent_method_labels(signature),
    )


def prepare_upload(raw_df: pd.DataFrame, *, llm=None, **kwargs) -> PreparedData:
    """Inspect, format, diagnose and validate an uploaded file in one step.

    Readings are usually uploaded before the test type is chosen in Step 2, so a
    column named as a concentration decides the type by itself.
    """
    from src.data.csv_inspector import CONCENTRATION

    df, inspection, formatted = inspect_and_format(
        raw_df, llm=llm, test_type=kwargs.get("test_type", registry.CONSTANT_RATE))
    if inspection.drawdown_unit == CONCENTRATION:
        kwargs["test_type"] = registry.TRACER
    return prepare(df, inspection=inspection,
                   conversions=formatted.conversions_applied,
                   rows_removed=formatted.rows_removed, **kwargs)


def store_readings(test: PumpingTest, df: pd.DataFrame, source_name: str = "",
                   conversions: Sequence[str] = (),
                   quantity: str = "drawdown") -> bool:
    """Record readings against a project test. Returns whether they changed.

    Guarded by fingerprint because an upload widget re-delivers the same file on
    every interaction, and replacing readings marks existing analyses stale.
    """
    readings = Readings.from_frame(df, source_name=source_name,
                                   conversions=list(conversions),
                                   quantity=quantity)
    if test.readings is not None and test.readings.fingerprint == readings.fingerprint:
        return False
    test.set_readings(readings)
    return True
