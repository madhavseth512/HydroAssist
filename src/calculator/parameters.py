"""One estimated parameter, with everything needed to report it honestly.

The current CalculationResult has fixed `T` and `S` fields. That works for the
three confined methods and for nothing else: Neuman (1974) estimates T, S, Sy and
beta; Moench (1984) estimates K, Ss, K', Ss', Sw and Sf; Warren-Root estimates
omega and lambda. A result therefore has to carry a variable-length parameter set.

Carrying each parameter as an object rather than a bare float lets the report say
more than a number. Three flags exist because the source papers demand them:

  `reliable`  Papadopulos & Cooper (1967) p. 244 states plainly that S from their
              method shifts by an order of magnitude between adjacent type curves
              while T barely moves. A result that prints both to four significant
              figures, side by side and unqualified, is misleading.
  `at_bound`  A parameter pinned against its bound is a floor or a ceiling, not a
              measurement.
  `derived`   Moench fits K and Ss, not T and S; T = K*b is still worth showing,
              but it should be visibly a derivation rather than a fitted value.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class FittedParameter:
    """An estimated aquifer or well parameter in SI units."""

    name: str                       # canonical key: "T", "S", "Sy", "beta", "omega"
    value: float                    # SI
    unit: str                       # "m^2/s", "-", "m", "1/m", "m/s"

    symbol: str = ""                # display glyph: "T", "S_y", "β", "r/B"
    ci: Optional[Tuple[float, float]] = None    # 95% interval, SI units

    reliable: bool = True
    reliability_note: str = ""      # why not, when reliable is False

    fixed: bool = False             # pinned by the user rather than estimated
    derived: bool = False           # computed from fitted parameters, not fitted
    at_bound: Optional[str] = None  # None | "lower" | "upper"

    display_scale: float = 1.0      # value * scale for the UI and the report
    display_unit: str = ""

    def __post_init__(self):
        if not self.symbol:
            self.symbol = self.name
        if not self.display_unit:
            self.display_unit = self.unit

    @property
    def display(self) -> float:
        """The value in the unit a hydrogeologist expects to read."""
        return self.value * self.display_scale

    @property
    def display_ci(self) -> Optional[Tuple[float, float]]:
        if self.ci is None:
            return None
        return (self.ci[0] * self.display_scale, self.ci[1] * self.display_scale)

    @property
    def estimated(self) -> bool:
        """True only for values this fit actually estimated from the data."""
        return not (self.fixed or self.derived)

    def to_dict(self) -> dict:
        """JSON-safe form for AgentState. Tuples become lists."""
        return {
            "name": self.name,
            "value": float(self.value),
            "unit": self.unit,
            "symbol": self.symbol,
            "ci": list(self.ci) if self.ci else None,
            "reliable": bool(self.reliable),
            "reliability_note": self.reliability_note,
            "fixed": bool(self.fixed),
            "derived": bool(self.derived),
            "at_bound": self.at_bound,
            "display_scale": float(self.display_scale),
            "display_unit": self.display_unit,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FittedParameter":
        ci = data.get("ci")
        return cls(
            name=data["name"],
            value=data["value"],
            unit=data.get("unit", ""),
            symbol=data.get("symbol", ""),
            ci=tuple(ci) if ci else None,
            reliable=data.get("reliable", True),
            reliability_note=data.get("reliability_note", ""),
            fixed=data.get("fixed", False),
            derived=data.get("derived", False),
            at_bound=data.get("at_bound"),
            display_scale=data.get("display_scale", 1.0),
            display_unit=data.get("display_unit", ""),
        )


# ── Constructors for the two parameters every confined method reports ─────────
# Centralised so that T is shown in m^2/day everywhere without each method
# repeating the conversion.

def transmissivity(value: float, ci=None, **kw) -> FittedParameter:
    return FittedParameter(
        name="T", value=value, unit="m^2/s", symbol="T", ci=ci,
        display_scale=86400.0, display_unit="m^2/day", **kw
    )


def storativity(value: float, ci=None, **kw) -> FittedParameter:
    return FittedParameter(
        name="S", value=value, unit="-", symbol="S", ci=ci, **kw
    )
