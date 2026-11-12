"""Physical geometry of an aquifer test, grouped the way a hydrogeologist thinks.

`CalculationInput` currently carries `Q`, `r`, `r_w` and `r_c` — enough for the
three confined methods, where `r_w` and `r_c` were already bolted on with
defaults for Papadopulos-Cooper. The methods still to come need considerably
more: saturated thickness and vertical anisotropy for Neuman, aquitard thickness
and conductivity for Hantush, screen intervals for partial penetration, block
geometry for a double-porosity fracture model, and — for slug tests — an initial
head displacement and no pumping rate at all.

Adding those as fifteen more optional scalars would produce exactly the god-object
that made `r=r_w` necessary at the Papadopulos-Cooper call site. Grouping them
instead gives each cluster its own invariants, lets one group be reused across the
six methods that need it, and makes each group a natural section of the Step 2 form.

Every field is SI. `None` means "not supplied": a method declares what it needs
through `requires` on its registry entry, and `validation.require_geometry` turns
a missing one into a sentence the user can act on.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

from src.calculator import registry

# Re-exported from the registry so the vocabulary cannot drift between the table
# of methods and the geometry those methods consume.
CONFINED = registry.CONFINED
LEAKY = registry.LEAKY
UNCONFINED = registry.UNCONFINED
FRACTURED = registry.FRACTURED
POROUS = registry.POROUS

CONSTANT_RATE = registry.CONSTANT_RATE
RECOVERY = registry.RECOVERY
STEP_DRAWDOWN = registry.STEP_DRAWDOWN
SLUG = registry.SLUG
CONSTANT_HEAD = registry.CONSTANT_HEAD
TRACER = registry.TRACER

# How a tracer entered the aquifer.
PULSE = "pulse"             # a slug, short against the travel time
CONTINUOUS = "continuous"   # held at a constant concentration from t = 0


@dataclass
class AquiferGeometry:
    """The formation itself."""

    b: Optional[float] = None               # saturated thickness [m]
    Kz_over_Kr: Optional[float] = None      # vertical anisotropy ratio [-]
    top_m: Optional[float] = None           # depth below datum to aquifer top [m]
    bottom_m: Optional[float] = None
    aquifer_type: str = CONFINED

    def __post_init__(self):
        if self.b is not None and self.b <= 0:
            raise ValueError(
                f"Aquifer thickness b must be positive, got {self.b} m."
            )
        if self.Kz_over_Kr is not None and not (0 < self.Kz_over_Kr <= 1.0):
            # Kz > Kr would mean water moves more easily across bedding than
            # along it, which layered sediments do not do.
            raise ValueError(
                f"Kz/Kr = {self.Kz_over_Kr} is outside (0, 1]. Vertical hydraulic "
                "conductivity is not normally greater than horizontal."
            )
        if (self.top_m is not None and self.bottom_m is not None
                and self.bottom_m <= self.top_m):
            raise ValueError(
                f"Aquifer bottom ({self.bottom_m} m) must be below its top "
                f"({self.top_m} m); depths increase downward."
            )


@dataclass
class WellGeometry:
    """The pumped (or slugged) well."""

    r_w: float = 0.10                       # effective screen / open-hole radius [m]
    r_c: float = 0.10                       # casing radius where the level moves [m]
    screen_top_m: Optional[float] = None    # partial penetration
    screen_bottom_m: Optional[float] = None
    skin_factor: Optional[float] = None     # Sw, when supplied rather than fitted
    is_pumped_well_data: bool = False       # series measured at the well face

    def __post_init__(self):
        if self.r_w <= 0:
            raise ValueError(f"r_w must be positive, got r_w = {self.r_w} m.")
        if self.r_c <= 0:
            raise ValueError(f"r_c must be positive, got r_c = {self.r_c} m.")
        if (self.screen_top_m is not None and self.screen_bottom_m is not None
                and self.screen_bottom_m <= self.screen_top_m):
            raise ValueError(
                f"Screen bottom ({self.screen_bottom_m} m) must be below its top "
                f"({self.screen_top_m} m)."
            )

    @property
    def screen_length(self) -> Optional[float]:
        if self.screen_top_m is None or self.screen_bottom_m is None:
            return None
        return self.screen_bottom_m - self.screen_top_m


@dataclass
class ObservationGeometry:
    """Where the drawdown series was measured."""

    r: float                                # radial distance from the pumped well [m]
    screen_top_m: Optional[float] = None
    screen_bottom_m: Optional[float] = None
    z_m: Optional[float] = None             # piezometer depth (Neuman, partial pen.)
    is_piezometer: bool = False             # point measurement vs vertically averaged
    well_id: Optional[str] = None

    def __post_init__(self):
        if self.r <= 0:
            raise ValueError(f"r must be positive, got r = {self.r} m.")


@dataclass
class AquitardGeometry:
    """One confining layer. Neuman-Witherspoon (1969) needs upper and lower."""

    b_prime: Optional[float] = None         # thickness b' [m]
    K_prime: Optional[float] = None         # vertical conductivity K' [m/s]
    Ss_prime: Optional[float] = None        # specific storage S'_s [1/m]
    position: str = "upper"                 # "upper" | "lower"

    def __post_init__(self):
        if self.position not in ("upper", "lower"):
            raise ValueError(
                f"Aquitard position must be 'upper' or 'lower', got {self.position!r}."
            )
        for name in ("b_prime", "K_prime", "Ss_prime"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"Aquitard {name} must be positive, got {value}.")

    @property
    def leakance(self) -> Optional[float]:
        """K'/b' [1/s] — the quantity leaky solutions actually depend on."""
        if self.K_prime is None or self.b_prime is None:
            return None
        return self.K_prime / self.b_prime


@dataclass
class FractureGeometry:
    """Block geometry for a double-porosity model.

    Moench (1984) offers slab-shaped and spherical blocks; Warren-Root needs
    neither field, only the fitted omega and lambda.
    """

    block_shape: str = "slab"               # "slab" | "sphere"
    block_half_width: Optional[float] = None    # a' [m]
    n_block_terms: int = 20                 # series truncation for block diffusion

    def __post_init__(self):
        if self.block_shape not in ("slab", "sphere"):
            raise ValueError(
                f"Block shape must be 'slab' or 'sphere', got {self.block_shape!r}."
            )
        if self.block_half_width is not None and self.block_half_width <= 0:
            raise ValueError(
                f"Block half-width must be positive, got {self.block_half_width} m."
            )
        if self.n_block_terms < 1:
            raise ValueError("n_block_terms must be at least 1.")


@dataclass
class SlugTestGeometry:
    """A slug test has no pumping rate; the driving term is an instantaneous
    head displacement that then recovers."""

    H0: float                               # initial head displacement [m]
    r_casing: float                         # radius where the water level moves [m]
    r_screen: float                         # screen / gravel-pack radius [m]
    screen_length: Optional[float] = None   # L [m]
    aquifer_thickness: Optional[float] = None
    under_damped: bool = False              # oscillatory response (high-K formations)

    def __post_init__(self):
        if self.H0 == 0:
            raise ValueError("Initial head displacement H0 cannot be zero.")
        if self.r_casing <= 0:
            raise ValueError(f"r_casing must be positive, got {self.r_casing} m.")
        if self.r_screen <= 0:
            raise ValueError(f"r_screen must be positive, got {self.r_screen} m.")


@dataclass
class TracerGeometry:
    """How a tracer test was run. The travel distance is the observation
    distance `r`, and a forced-gradient test's thickness is `aquifer.b`.

    Concentrations stay in `conc_unit` throughout. The solutions scale with their
    amplitude, so no conversion is needed to fit one; the unit matters only to
    turn a fitted amplitude into a recovered mass.
    """

    injection: str = PULSE
    conc_unit: str = "mg/L"
    background: float = 0.0                 # concentration before the test
    C0: Optional[float] = None              # plateau, if known (continuous)
    M_injected: Optional[float] = None      # tracer mass [kg] (pulse)
    K: Optional[float] = None               # hydraulic conductivity [m/s]
    gradient: Optional[float] = None        # hydraulic gradient [-]

    def __post_init__(self):
        if self.injection not in (PULSE, CONTINUOUS):
            raise ValueError(
                f"Injection must be '{PULSE}' or '{CONTINUOUS}', got {self.injection!r}."
            )
        if self.background < 0:
            raise ValueError(
                f"Background concentration cannot be negative, got {self.background}."
            )
        for name in ("C0", "M_injected", "K", "gradient"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"Tracer input {name} must be positive, got {value}.")


# ── Pumping schedule ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RateStep:
    """A constant-rate period beginning at t_start_s. Q = 0 marks recovery."""

    t_start_s: float
    Q: float                                # [m^3/s]

    def __post_init__(self):
        if self.t_start_s < 0:
            raise ValueError(f"Step start time must be >= 0, got {self.t_start_s} s.")


@dataclass
class PumpingSchedule:
    """The full discharge history of a test.

    A constant-rate test is one step. A step-drawdown test is several increasing
    steps; a recovery test ends with Q = 0. Multi-rate drawdown is computed by
    superposing single-rate solutions in the TIME domain — never by inverting a
    Laplace-domain step function, which Stehfest inversion cannot handle because
    it requires a smooth transform.
    """

    steps: Tuple[RateStep, ...] = ()

    def __post_init__(self):
        if not self.steps:
            raise ValueError("A pumping schedule needs at least one rate step.")
        self.steps = tuple(sorted(self.steps, key=lambda s: s.t_start_s))
        if self.steps[0].t_start_s != 0:
            raise ValueError(
                f"The first rate step must start at t = 0, got "
                f"{self.steps[0].t_start_s} s."
            )
        starts = [s.t_start_s for s in self.steps]
        if len(set(starts)) != len(starts):
            raise ValueError("Two rate steps cannot start at the same time.")

    @property
    def is_constant(self) -> bool:
        return len(self.steps) == 1

    @property
    def initial_rate(self) -> float:
        return self.steps[0].Q

    def rate_at(self, t_s: float) -> float:
        """The discharge in force at time t."""
        rate = 0.0
        for step in self.steps:
            if step.t_start_s <= t_s:
                rate = step.Q
            else:
                break
        return rate

    def rate_changes(self) -> list:
        """[(t_i, delta_Q_i)] — the increments to superpose.

        Drawdown for a variable-rate test is the sum over these of
        delta_Q_i * s_unit(t - t_i), where s_unit is the unit-rate response of
        whichever analytical solution applies.
        """
        out, previous = [], 0.0
        for step in self.steps:
            out.append((step.t_start_s, step.Q - previous))
            previous = step.Q
        return out


@dataclass(frozen=True)
class WellFieldMember:
    """One well in a forward simulation. Never part of a fit.

    Forward prediction — what AQTESOLV calls test design — superposes the
    responses of several pumping wells in space.
    """

    well_id: str
    x_m: float
    y_m: float
    schedule: PumpingSchedule
    r_w: float = 0.10

    def distance_to(self, x_m: float, y_m: float) -> float:
        return float(((self.x_m - x_m) ** 2 + (self.y_m - y_m) ** 2) ** 0.5)
