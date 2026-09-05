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

    # ── Papadopulos-Cooper (1967) additional geometry ─────────────────────────
    # Unused by Theis and Cooper-Jacob; defaults are physically valid so existing
    # code that does not pass these fields is unaffected.
    r_w: float = 0.10   # effective radius of well screen / open hole [m]
    r_c: float = 0.10   # radius of well casing where water level declines [m]

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
        if self.r_w <= 0:
            raise ValueError(f"r_w must be positive, got r_w = {self.r_w} m.")
        if self.r_c <= 0:
            raise ValueError(f"r_c must be positive, got r_c = {self.r_c} m.")

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
    # Defaulted rather than required: a method that estimates something other
    # than T and S (Warren-Root fits omega and lambda; Moench fits K and Ss)
    # populates `parameters` instead and leaves these to be projected from it.
    # Positional construction, CalculationResult("Theis (1935)", 2.5e-3, 5e-5),
    # is unaffected.
    T: float = 0.0                     # transmissivity  [m²/s]
    S: float = 0.0                     # storativity     [dimensionless]
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

    # ── Generalised results ───────────────────────────────────────────────────
    # Methods beyond the original three estimate more than two parameters:
    # Neuman (1974) fits T, S, Sy and beta; Moench (1984) fits six. `parameters`
    # is the general store; the T/S fields above remain the view of it that the
    # existing UI, plots and report read.
    method_id: str = ""                             # registry key
    parameters: list = field(default_factory=list)  # list[FittedParameter]
    gof: Optional[object] = None                    # GoodnessOfFit
    diagnostics: dict = field(default_factory=dict) # alpha, t0, n_excluded, u_max…
    curves: dict = field(default_factory=dict)      # derivative, smooth curves…
    findings: list = field(default_factory=list)    # list[ValidityFinding]

    def __post_init__(self):
        # Reconciliation runs in both directions so old and new writers coexist.
        #
        # Legacy writer — the three original calculators pass T= and S=. Build the
        # parameter list from them, so anything reading `parameters` works today.
        if not self.parameters and self.success and (self.T > 0 or self.S > 0):
            from src.calculator.parameters import storativity, transmissivity
            self.parameters = [
                transmissivity(self.T, ci=self.T_ci),
                storativity(self.S, ci=self.S_ci),
            ]
        # New writer — a method that builds `parameters` directly. Project T and S
        # back onto the scalar fields, which is what keeps every existing
        # `result.T` / `result.S` / `result.T_day` call site working unchanged.
        elif self.parameters:
            if (p := self.get("T")) is not None:
                self.T, self.T_ci = p.value, p.ci
            if (p := self.get("S")) is not None:
                self.S, self.S_ci = p.value, p.ci

        if self.T > 0 and self.T_day == 0.0:
            self.T_day = self.T * 86400.0

        # Mirror the goodness-of-fit object onto the legacy scalars.
        if self.gof is not None:
            self.rmse = self.gof.rmse
            self.r_squared = self.gof.r_squared

        # Structured findings render down to the flat note list the UI and PDF
        # already consume, so a method can emit either.
        if self.findings and not self.validity_notes:
            self.validity_notes = [f.text for f in self.findings]

    # ── Parameter access ──────────────────────────────────────────────────────

    def get(self, name: str):
        """The named FittedParameter, or None."""
        for p in self.parameters:
            if p.name == name:
                return p
        return None

    def value(self, name: str, default: float = 0.0) -> float:
        p = self.get(name)
        return p.value if p is not None else default

    @property
    def parameter_names(self) -> list:
        return [p.name for p in self.parameters]

    @property
    def unreliable(self) -> list:
        """Parameters the method itself says not to trust — e.g. storativity from
        Papadopulos-Cooper, which the paper (p. 244) puts at order-of-magnitude."""
        return [p for p in self.parameters if not p.reliable]

    @property
    def at_bounds(self) -> list:
        return [p for p in self.parameters if p.at_bound]

    def to_state_dict(self) -> dict:
        """Flatten to a plain dict for storage in the conversation AgentState.

        The numpy arrays (time_s, drawdown_obs, drawdown_fitted) are dropped:
        the chat side only needs the scalars and the notes, and keeping the
        state JSON-serialisable avoids surprises if it is ever checkpointed,
        exported or sent to an LLM.
        """
        payload = {
            "method": self.method,
            "T": float(self.T),
            "S": float(self.S),
            "T_day": float(self.T_day),
            "T_ci": list(self.T_ci) if self.T_ci else None,
            "S_ci": list(self.S_ci) if self.S_ci else None,
            "rmse": float(self.rmse),
            "r_squared": float(self.r_squared),
            "validity_notes": list(self.validity_notes),
            "success": bool(self.success),
            "error_message": self.error_message,
        }
        # Additive only — every key above is unchanged, so existing consumers
        # (the consultant prompt, summarise_result, the chat bridge) are unaffected.
        payload["method_id"] = self.method_id
        payload["parameters"] = [p.to_dict() for p in self.parameters]
        payload["diagnostics"] = {
            k: float(v) for k, v in self.diagnostics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        return payload


class BaseCalculator(ABC):
    """All pumping test calculators implement this interface."""

    @abstractmethod
    def calculate(self, inputs: CalculationInput) -> CalculationResult:
        """Run the calculation and return results."""
        ...
