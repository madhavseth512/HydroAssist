# Reading the results

## The four headline numbers

- **T** — transmissivity, shown in m²/day and m²/s. The unit switch in the
  header changes this to gpd/ft if you work in US customary units.
- **S** — storativity, dimensionless, usually between 1e-5 and 1e-3 in a
  confined aquifer and 0.01 to 0.3 where the aquifer behaves as unconfined.
  Some methods do not estimate S, and say so rather than showing a number.
- **R²** — how much of the variation the fit explains.
- **RMSE** — the typical distance between a measured point and the fitted curve,
  in millimetres. This is often more useful than R², which can look excellent
  for a model that is wrong.

Each carries its 95% confidence interval. An interval is not decoration: it says
how many digits of the value mean anything.

## Validity notes

Below the numbers is a list of notes from the method itself — points excluded,
assumptions that look strained, parameters at a bound, an estimate the method
does not trust. **They are part of the answer.** A fit with an excellent R² and
three validity notes is not a good fit.

Common notes:

- *early-time points excluded (u ≥ 0.02)* — Cooper-Jacob fits only the late-time
  straight line, so early readings are deliberately left out.
- *readings belong to the pumping phase* — in a recovery analysis, points from
  before the pump stopped were dropped.
- *parameter at a bound* — the fitter pushed a parameter to the edge of its
  allowed range, which usually means the model does not suit the data.

## The plots

- **Fit diagnostic** — your readings with the fitted curve, and the residuals.
- **Manual matching** — sliders for T and S so you can match the curve by eye,
  the way it is done on paper. The RMSE updates as you move them, so you can see
  what the automatic fit is worth.

## What you can do with a result

- **Save to project** keeps it for comparison against other methods.
- **Interpret in chat** sends the numbers to the consultant for comment.
- **Download report** produces a PDF with the numbers, the plots and the notes.
