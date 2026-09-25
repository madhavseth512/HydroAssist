# Comparing methods

Saving two or more analyses on the same test fills the comparison table, which
ranks them and says how much better the best one is.

## The columns

- **Rank** — best first.
- **Parameters** — how many the method estimates. More parameters fit better by
  construction, which is what the ranking corrects for.
- **T**, **S**, **R²** — as in the results panel.
- **AICc** — a score balancing fit quality against the number of parameters,
  corrected for small samples. Lower is better. Its absolute value means
  nothing; only differences do.
- **ΔAICc** — how far each method is from the best.
- **Weight** — the Akaike weight, read as the probability that this model is the
  best of those compared, given the data.
- **Evidence** — the conventional reading of ΔAICc, after Burnham and Anderson:
  under 2 is substantial support, 2 to 10 is less support, over 10 is
  essentially none.

## What the ranking does not tell you

It compares the models offered against *this* dataset. It cannot tell you that a
model is physically wrong for your aquifer — a leaky model can win on AICc for
a confined test if the data happen to curve. **Read the validity notes beside
the ranking**, not just the order.

All compared analyses must come from the same readings. If the readings are
replaced, earlier analyses are marked stale and excluded from the comparison
rather than silently compared against different data.
