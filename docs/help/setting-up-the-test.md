# Setting up the test

Step 2 of the workspace is where the test itself is described. These values are
what turn a curve into transmissivity.

## The fields

- **Test type** — *Constant-rate pumping*, *Recovery (after the pump stops)* or
  *Tracer test*. This decides which methods are offered.
- **Pumping rate Q**, with its unit (m³/day, m³/hour, m³/s or L/s). It must be
  the rate held through the test. The panel shows the value converted to m³/s
  beneath the field.
- **Observation well distance r** — metres from the pumping well to the well the
  readings came from.
- **Analysis method** — see below.

Two more fields appear only when the chosen method needs them:

- **Well screen radius r_w** and **casing radius r_c**, for methods fitted at the
  well face such as Papadopulos-Cooper. When drawdown was measured in the
  pumping well itself, the radius takes the place of the distance r.
- **Pumping duration before shut-down t_p** and the **recovery clock**, for a
  recovery test. The clock says whether your times count from when pumping
  started or from when it stopped.

## Tracer tests

A tracer test fits a concentration history instead of drawdown, and gives velocity
or travel time, longitudinal dispersivity and, with the right inputs, effective
porosity. It does not estimate T or S. Readings whose column is named as a
concentration, such as `concentration_mg_L`, switch the test to this type by
themselves. Otherwise choose *Tracer test* before uploading. Times must count from
the injection.

- **Ogata-Banks (1961)** is for a natural-gradient test: tracer carried by ambient
  flow to a downgradient well. No pumping rate is needed. Enter hydraulic
  conductivity and gradient to get effective porosity.
- **Thiéry (2022)** is for a forced-gradient test, also called a radially convergent
  test: tracer drawn radially to a pumped well. Enter Q. Aquifer thickness gives effective porosity, and for a pulse the
  injected mass gives the fraction recovered.

For both, the **distance** is from the injection to the sampling well, **injection**
is a short pulse or a continuous release, and the **background** is the concentration
before the test. Enter the readings as measured, because the background is removed
in the fit. For a continuous test, entering the plateau concentration C₀ holds it
fixed, which helps when the record stops before the plateau.

A pulse must be short compared with the travel time, since a long injection is not
modelled.

## Choosing a method

The dropdown offers the methods that suit the test type and the data. A method
that is not offered is listed underneath with the reason — usually that the data
cannot support it, not that it is wrong.

Under the dropdown is a line explaining what the chosen method assumes. If the
consultant has suggested a method and you accepted it, that method is already
selected here.

## Running it

**Run analysis** fits the chosen method to the readings and opens the results.
The setup is saved to the test at the same time, so the project, the form and
the result always describe the same test.
