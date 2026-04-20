"""Abstract base for all pumping test calculator methods."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class CalculationInput:
    """
    Inputs required by every confined aquifer calculator.

    All values must be in strict SI units before being passed here.
    The calculator layer never performs unit conversion — that is the
    formatter's responsibility (src/data/formatter.py).

    Units:
        df.time_s      seconds
        df.drawdown_m  meters
        Q              m³/s   (NOT m³/day — convert before passing)
        r              meters
    """
    df: pd.DataFrame          # formatted: columns time_s, drawdown_m [, well_id]
    Q: float                  # pumping rate  [m³/s]
    r: float                  # observation distance from pumping well [m]
    well_id: Optional[str] = None   # which well to analyse (None = single-well dataset)

    def get_series(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (time_s, drawdown_m) arrays for the target well."""
        if self.well_id and "well_id" in self.df.columns:
            sub = self.df[self.df["well_id"] == self.well_id]
        else:
            sub = self.df
        return sub["time_s"].to_numpy(dtype=float), sub["drawdown_m"].to_numpy(dtype=float)


@dataclass
class CalculationResult:
    """
    Outputs produced by every confined aquifer calculator.

    T and S are always in SI units internally.
    T_day is provided as a convenience for display (hydrogeologists
    commonly use m²/day — the original Theis paper itself used two
    unit systems for exactly this reason).
    """
    method: str

    # ── Primary outputs ───────────────────────────────────────────────────────
    T: float                           # transmissivity  [m²/s]
    S: float                           # storativity     [dimensionless]
    T_day: float = 0.0                 # transmissivity  [m²/day]  = T × 86400

    # ── Uncertainty ───────────────────────────────────────────────────────────
    # None when curve_fit covariance matrix contains inf (poor convergence)
    T_ci: Optional[Tuple[float, float]] = None   # 95% CI on T  [m²/s]
    S_ci: Optional[Tuple[float, float]] = None   # 95% CI on S

    # ── Goodness-of-fit ───────────────────────────────────────────────────────
    rmse: float = 0.0          # root mean squared error  [m]
    r_squared: float = 0.0     # coefficient of determination

    # ── Plotting data ─────────────────────────────────────────────────────────
    time_s: Optional[np.ndarray] = None      # observed time array
    drawdown_obs: Optional[np.ndarray] = None    # observed drawdown
    drawdown_fitted: Optional[np.ndarray] = None # model-predicted drawdown

    # ── Interpretation ────────────────────────────────────────────────────────
    validity_notes: list = field(default_factory=list)

    # ── Status ────────────────────────────────────────────────────────────────
    success: bool = True
    error_message: str = ""

    def __post_init__(self):
        if self.T and self.T_day == 0.0:
            self.T_day = self.T * 86400.0


class BaseCalculator(ABC):
    """All pumping test calculators implement this interface."""

    @abstractmethod
    def calculate(self, inputs: CalculationInput) -> CalculationResult:
        """Run the calculation and return results."""
        ...
