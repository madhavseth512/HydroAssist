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

    def __post_init__(self):
        # Runtime Q unit guard. Real pumping wells: 0.001–0.5 m³/s.
        # Q > 1.0 m³/s almost always means m³/day was passed instead of m³/s.
        # Theis (1935): "the same units must of course be used throughout."
        if self.Q > 1.0:
            raise ValueError(
                f"Q = {self.Q} m³/s appears unrealistically large. "
                "Q must be in m³/s, not m³/day. "
                f"To convert: Q_si = {self.Q:.1f} / 86400 = {self.Q / 86400:.6f} m³/s."
            )
        if self.Q <= 0:
            raise ValueError(f"Q must be positive, got Q = {self.Q} m³/s.")
        if self.r <= 0:
            raise ValueError(f"r must be positive, got r = {self.r} m.")

    def get_series(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (time_s, drawdown_m) arrays for the target well."""
        if self.well_id:
            if "well_id" not in self.df.columns:
                raise ValueError(
                    f"well_id='{self.well_id}' was specified but the DataFrame "
                    "has no 'well_id' column. Check the formatter output."
                )
            sub = self.df[self.df["well_id"] == self.well_id]
            if len(sub) == 0:
                raise ValueError(
                    f"well_id='{self.well_id}' not found in DataFrame. "
                    f"Available IDs: {self.df['well_id'].unique().tolist()}"
                )
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
        if self.T > 0 and self.T_day == 0.0:
            self.T_day = self.T * 86400.0


class BaseCalculator(ABC):
    """All pumping test calculators implement this interface."""

    @abstractmethod
    def calculate(self, inputs: CalculationInput) -> CalculationResult:
        """Run the calculation and return results."""
        ...
