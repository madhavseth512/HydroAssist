"""Data Validator — physical plausibility checks on formatted pumping test data."""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────

MIN_DATA_POINTS = 10          # below this the curve fit will be unreliable
MIN_LOG_CYCLES  = 1.0         # time span must cover at least one log cycle
COOPER_JACOB_U_MAX = 0.05     # u = r²S / 4Tt — Cooper-Jacob valid when u < 0.05


@dataclass
class ValidationResult:
    is_usable: bool
    errors:   list = field(default_factory=list)   # blocking — cannot compute
    warnings: list = field(default_factory=list)   # non-blocking but important
    info:     list = field(default_factory=list)   # informational
    suitable_methods: list = field(default_factory=list)  # pre-filtered method list


class DataValidator:
    """
    Runs read-only sanity checks on a formatted pumping test DataFrame.

    Does NOT modify the data — only assesses and reports.
    The suitable_methods list pre-filters which calculator methods to offer.
    """

    # All confined aquifer methods supported in Phase 2
    ALL_METHODS = ["Theis (1935)", "Cooper-Jacob (1946)", "Papadopulos-Cooper (1967)"]

    def validate(
        self,
        df: pd.DataFrame,
        well_radius_m: float = None,
        pumping_rate_m3d: float = None,
        well_casing_radius_m: float = None,
    ) -> ValidationResult:
        """
        Validate a formatted DataFrame (columns: time_s, drawdown_m[, well_id]).

        Args:
            df:                     Formatted DataFrame from DataFormatter
            well_radius_m:          Observation well distance r (optional, used for u check)
            pumping_rate_m3d:       Pumping rate Q in m³/day (optional, used for u check)
            well_casing_radius_m:   Pumping well casing radius (optional).
                                    Theis (1935) assumption (5): "infinitesimal diameter well".
                                    Values > 0.15m trigger a recommendation for
                                    Papadopulos-Cooper (1967) instead.

        Returns:
            ValidationResult with errors, warnings, info, and suitable_methods
        """
        errors, warnings, info = [], [], []

        # Work per-well if multi-well dataset
        well_ids = df["well_id"].unique().tolist() if "well_id" in df.columns else [None]

        for well_id in well_ids:
            label = f"Well '{well_id}'" if well_id else "Dataset"
            sub = df[df["well_id"] == well_id] if well_id else df

            errors  += self._check_min_points(sub, label)
            warnings += self._check_monotonic_time(sub, label)
            warnings += self._check_negative_drawdown(sub, label)
            warnings += self._check_log_cycle_span(sub, label)
            info     += self._check_drawdown_trend(sub, label)

        is_usable = len(errors) == 0
        suitable  = self._suggest_methods(
            df, well_radius_m, pumping_rate_m3d,
            well_casing_radius_m, warnings, info
        )

        return ValidationResult(
            is_usable=is_usable,
            errors=errors,
            warnings=warnings,
            info=info,
            suitable_methods=suitable,
        )

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_min_points(self, df: pd.DataFrame, label: str) -> list:
        n = len(df)
        if n < MIN_DATA_POINTS:
            return [f"{label}: only {n} data points — minimum {MIN_DATA_POINTS} required for reliable curve fitting."]
        return []

    def _check_monotonic_time(self, df: pd.DataFrame, label: str) -> list:
        diffs = df["time_s"].diff().dropna()
        bad = (diffs <= 0).sum()
        if bad:
            return [f"{label}: {bad} non-monotonic time step(s) detected — data may not be sorted correctly."]
        return []

    def _check_negative_drawdown(self, df: pd.DataFrame, label: str) -> list:
        neg = (df["drawdown_m"] < 0).sum()
        if neg:
            return [f"{label}: {neg} negative drawdown value(s) — may include recovery data or measurement errors."]
        return []

    def _check_log_cycle_span(self, df: pd.DataFrame, label: str) -> list:
        t_min = df["time_s"].min()
        t_max = df["time_s"].max()
        if t_min <= 0:
            return []
        log_cycles = np.log10(t_max / t_min)
        if log_cycles < MIN_LOG_CYCLES:
            return [
                f"{label}: time span covers only {log_cycles:.1f} log cycle(s) "
                f"({t_min:.0f}s – {t_max:.0f}s) — at least 1 recommended for stable parameter estimation."
            ]
        return []

    def _check_drawdown_trend(self, df: pd.DataFrame, label: str) -> list:
        """Check that early-time drawdown is generally increasing (expected for pumping test)."""
        early = df.iloc[:max(5, len(df) // 4)]
        if len(early) < 3:
            return []
        slope = np.polyfit(range(len(early)), early["drawdown_m"].values, 1)[0]
        if slope < 0:
            return [f"{label}: drawdown appears to decrease in early time — check that data is pumping phase, not recovery."]
        return []

    # ── Method suitability ────────────────────────────────────────────────────

    def _suggest_methods(
        self,
        df: pd.DataFrame,
        r: float,
        Q: float,
        well_casing_radius_m: float,
        warnings: list,
        info: list,
    ) -> list:
        """
        Return the list of methods that are plausibly valid given the data.
        Theis is always included if data is usable.
        Cooper-Jacob requires u < 0.05 at late time.
        Papadopulos-Cooper requires additional well geometry input — always listed as optional.
        """
        suitable = ["Theis (1935)"]

        # Well diameter check — Theis (1935) assumption (5): "infinitesimal diameter well".
        # Theis notes the error is "apparently always insignificant" for standard wells,
        # but a casing radius > 0.15m (12-inch diameter) is non-negligible.
        if well_casing_radius_m and well_casing_radius_m > 0.15:
            info.append(
                f"Well casing radius {well_casing_radius_m}m is non-negligible. "
                "Theis (1935) assumption (5) requires an infinitesimal-diameter well. "
                "Papadopulos-Cooper (1967) is recommended for large-diameter wells "
                "where wellbore storage effects are significant."
            )

        # Rough Cooper-Jacob pre-check using late-time data and rough S/T estimate
        cj_likely = self._cooper_jacob_feasible(df, r, Q)
        if cj_likely:
            suitable.append("Cooper-Jacob (1946)")
        else:
            info.append(
                "Cooper-Jacob pre-check: u may not be < 0.05 for all late-time points. "
                "The method will validate this after fitting — it may still be usable."
            )

        # Papadopulos-Cooper is always available but requires extra inputs
        suitable.append("Papadopulos-Cooper (1967)")
        info.append(
            "Papadopulos-Cooper (1967) requires well casing radius (rc) and "
            "screen radius (rw) — enter these in the calculator form."
        )

        return suitable

    def _cooper_jacob_feasible(self, df: pd.DataFrame, r: float, Q: float) -> bool:
        """
        Rough feasibility check: does late-time data plausibly satisfy u < 0.05?
        Uses a simple heuristic based on time span; exact check is done post-fitting.
        """
        if r is None or Q is None:
            # No geometry info — be optimistic, offer the method
            return True

        t_max_s = df["time_s"].max()
        # For u < 0.05 we need t > r²S / (4T × 0.05)
        # Without S and T we can only check time span heuristic:
        # if test ran for >= 30 minutes it's likely long enough
        return t_max_s >= 1800
