"""CSV Inspector — detects columns, infers units, flags anomalies in pumping test data."""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Regex patterns for column name matching ──────────────────────────────────

# Column name patterns — use simple substring search (column names, not prose)
_TIME_PATTERNS = re.compile(
    r"(time|elapsed|duration|timestamp|^t$|^t_)", re.IGNORECASE
)
_DRAWDOWN_PATTERNS = re.compile(
    r"(drawdown|^dd|_dd|^s$|^s_|depth|water.?level|^wl|_wl)", re.IGNORECASE
)
_WELL_ID_PATTERNS = re.compile(
    r"(well|borehole|obs|monitoring|^mw|site|^id$|^name$)", re.IGNORECASE
)

# Use lookahead/lookbehind to match unit tokens separated by word boundaries OR underscores/parens
_UNIT_SECONDS = re.compile(r"(?<![a-zA-Z])(sec|second|secs)(?![a-zA-Z])", re.IGNORECASE)
_UNIT_MINUTES = re.compile(r"(?<![a-zA-Z])(min|minute|mins)(?![a-zA-Z])", re.IGNORECASE)
_UNIT_HOURS   = re.compile(r"(?<![a-zA-Z])(hr|hour|hrs)(?![a-zA-Z])", re.IGNORECASE)
_UNIT_FEET    = re.compile(r"(?<![a-zA-Z])(ft|feet|foot)(?![a-zA-Z])", re.IGNORECASE)
_UNIT_METERS  = re.compile(r"(?<![a-zA-Z])(meter|metre|_m\b|\(m\))", re.IGNORECASE)


@dataclass
class ColumnMapping:
    name: str
    time_unit: str = "unknown"     # seconds | minutes | hours | unknown
    drawdown_unit: str = "unknown" # meters  | feet    | unknown


@dataclass
class InspectionResult:
    time_col: str
    drawdown_cols: list
    well_id_col: Optional[str]
    time_unit: str                 # seconds | minutes | hours
    drawdown_unit: str             # meters  | feet
    issues: list = field(default_factory=list)
    confidence: float = 1.0
    needs_user_confirmation: bool = False
    llm_explanation: str = ""


class CSVInspector:
    """
    Inspects a raw pumping test DataFrame and identifies columns + units.

    Detection runs in three layers:
      1. Regex on column names  (fast, free)
      2. Statistical heuristics (fast, free)
      3. LLM call               (only if confidence < 0.8)
    """

    CONFIDENCE_THRESHOLD = 0.8

    def __init__(self, llm=None):
        """
        Args:
            llm: Optional LangChain LLM instance used only when confidence < 0.8.
                 Pass None to skip LLM fallback (useful in tests).
        """
        self.llm = llm

    # ── Public API ────────────────────────────────────────────────────────────

    def inspect(self, df: pd.DataFrame) -> InspectionResult:
        """
        Inspect df and return an InspectionResult.

        Args:
            df: Raw DataFrame from pd.read_csv

        Returns:
            InspectionResult with detected columns, units, and quality issues
        """
        issues = []
        confidence = 1.0

        time_col, time_unit, time_conf = self._detect_time_column(df)
        dd_cols,  dd_unit,   dd_conf   = self._detect_drawdown_columns(df)
        well_col                       = self._detect_well_id_column(df, exclude=[time_col] + dd_cols)

        confidence = min(time_conf, dd_conf)

        if not time_col:
            issues.append("Could not identify a time column.")
            confidence = 0.0
        if not dd_cols:
            issues.append("Could not identify a drawdown column.")
            confidence = 0.0

        # ── LLM fallback ──────────────────────────────────────────────────────
        llm_explanation = ""
        if confidence < self.CONFIDENCE_THRESHOLD and self.llm and time_col and dd_cols:
            try:
                time_col, time_unit, dd_cols, dd_unit, llm_explanation = \
                    self._llm_assisted_mapping(df, time_col, dd_cols)
                confidence = max(confidence, 0.85)
            except Exception as e:
                logger.warning(f"LLM mapping fallback failed: {e}")

        # ── Basic anomaly scan ────────────────────────────────────────────────
        if time_col and time_col in df.columns:
            issues += self._scan_time_anomalies(df[time_col])
        if dd_cols:
            for col in dd_cols:
                if col in df.columns:
                    issues += self._scan_drawdown_anomalies(df[col], col)

        needs_confirmation = confidence < self.CONFIDENCE_THRESHOLD

        return InspectionResult(
            time_col=time_col or "",
            drawdown_cols=dd_cols,
            well_id_col=well_col,
            time_unit=time_unit,
            drawdown_unit=dd_unit,
            issues=issues,
            confidence=round(confidence, 2),
            needs_user_confirmation=needs_confirmation,
            llm_explanation=llm_explanation,
        )

    # ── Layer 1: Regex detection ──────────────────────────────────────────────

    def _detect_time_column(self, df: pd.DataFrame):
        """Returns (col_name, unit, confidence)."""
        candidates = []
        for col in df.columns:
            if _TIME_PATTERNS.search(col):
                unit = self._infer_time_unit(col)
                score = 0.9 if unit != "unknown" else 0.7
                candidates.append((col, unit, score))

        if len(candidates) == 1:
            return candidates[0]

        if len(candidates) > 1:
            # Prefer the one with an explicit unit in the name
            with_unit = [c for c in candidates if c[1] != "unknown"]
            if with_unit:
                return with_unit[0][0], with_unit[0][1], 0.85
            return candidates[0][0], candidates[0][1], 0.75

        # Layer 2 fallback: find the monotonically increasing numeric column
        return self._statistical_time_detection(df)

    def _detect_drawdown_columns(self, df: pd.DataFrame):
        """Returns (col_names_list, unit, confidence)."""
        candidates = []
        for col in df.columns:
            if _DRAWDOWN_PATTERNS.search(col):
                unit = self._infer_drawdown_unit(col)
                score = 0.9 if unit != "unknown" else 0.7
                candidates.append((col, unit, score))

        if candidates:
            unit = candidates[0][1]
            conf = min(c[2] for c in candidates)
            return [c[0] for c in candidates], unit, conf

        # Layer 2 fallback: numeric column with plausible drawdown range
        return self._statistical_drawdown_detection(df)

    def _detect_well_id_column(self, df: pd.DataFrame, exclude: list) -> Optional[str]:
        """Returns column name if a well-ID column is detected, else None."""
        for col in df.columns:
            if col in exclude:
                continue
            if _WELL_ID_PATTERNS.search(col):
                # Confirm it contains string-like values
                if df[col].dtype == object or df[col].nunique() < 20:
                    return col
        return None

    # ── Layer 2: Statistical heuristics ──────────────────────────────────────

    def _statistical_time_detection(self, df: pd.DataFrame):
        """Find the most likely time column by statistical properties."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        best, best_score = None, -1

        for col in numeric_cols:
            s = df[col].dropna()
            if len(s) < 3:
                continue
            is_monotone = (s.diff().dropna() >= 0).all()
            starts_near_zero = s.iloc[0] <= s.max() * 0.05
            score = (0.6 if is_monotone else 0) + (0.3 if starts_near_zero else 0)
            if score > best_score:
                best, best_score = col, score

        if best and best_score >= 0.6:
            unit = self._infer_time_unit(best)
            return best, unit, 0.55
        return "", "unknown", 0.0

    def _statistical_drawdown_detection(self, df: pd.DataFrame):
        """Find drawdown columns by value ranges (meters: 0–150, feet: 0–500)."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        candidates = []

        for col in numeric_cols:
            s = df[col].dropna()
            if len(s) < 3:
                continue
            col_max = s.max()
            col_min = s.min()
            if col_min >= 0 and col_max <= 500:
                unit = "feet" if col_max > 150 else "meters"
                candidates.append((col, unit, 0.5))

        if candidates:
            return [c[0] for c in candidates[:1]], candidates[0][1], 0.5
        return [], "unknown", 0.0

    # ── Unit inference ────────────────────────────────────────────────────────

    def _infer_time_unit(self, col_name: str) -> str:
        if _UNIT_SECONDS.search(col_name):
            return "seconds"
        if _UNIT_MINUTES.search(col_name):
            return "minutes"
        if _UNIT_HOURS.search(col_name):
            return "hours"
        return "unknown"

    def _infer_drawdown_unit(self, col_name: str) -> str:
        if _UNIT_FEET.search(col_name):
            return "feet"
        if _UNIT_METERS.search(col_name):
            return "meters"
        return "unknown"

    # ── Layer 3: LLM fallback ─────────────────────────────────────────────────

    def _llm_assisted_mapping(self, df: pd.DataFrame, time_col: str, dd_cols: list):
        """
        Ask the LLM to resolve ambiguous column mapping.
        Returns (time_col, time_unit, dd_cols, dd_unit, explanation).
        """
        sample = df.head(5).to_string(index=False)
        col_list = list(df.columns)

        prompt = f"""You are analyzing a pumping test CSV file for hydrogeological analysis.

Column names: {col_list}
First 5 rows:
{sample}

Identify:
1. Which column contains elapsed time since pumping started?
2. What unit is the time in (seconds, minutes, or hours)?
3. Which column(s) contain drawdown (water level decline) measurements?
4. What unit is drawdown in (meters or feet)?

Respond ONLY with valid JSON, no markdown:
{{
  "time_column": "<column name>",
  "time_unit": "seconds|minutes|hours",
  "drawdown_columns": ["<col1>", "<col2>"],
  "drawdown_unit": "meters|feet",
  "explanation": "<one sentence summary>"
}}"""

        response = self.llm.invoke(prompt)
        text = response.content.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = re.sub(r"```(?:json)?", "", text).strip().strip("```").strip()

        result = json.loads(text)
        return (
            result["time_column"],
            result["time_unit"],
            result["drawdown_columns"],
            result["drawdown_unit"],
            result.get("explanation", ""),
        )

    # ── Anomaly scans ─────────────────────────────────────────────────────────

    def _scan_time_anomalies(self, series: pd.Series) -> list:
        issues = []
        if series.isnull().any():
            issues.append(f"Time column has {series.isnull().sum()} missing value(s).")
        diffs = series.diff().dropna()
        if (diffs < 0).any():
            count = (diffs < 0).sum()
            issues.append(f"Time column has {count} non-monotonic step(s) — rows may be out of order.")
        if (series == 0).any():
            issues.append("t=0 row detected — will be removed before analysis (log transform undefined at t=0).")
        return issues

    def _scan_drawdown_anomalies(self, series: pd.Series, col_name: str) -> list:
        issues = []
        if series.isnull().any():
            issues.append(f"Column '{col_name}' has {series.isnull().sum()} missing value(s).")
        if (series < 0).any():
            count = (series < 0).sum()
            issues.append(f"Column '{col_name}' has {count} negative drawdown value(s) — check for recovery data.")
        return issues
