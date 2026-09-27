# Projects, tests and saving

## Where your work lives

HydroAssist keeps nothing on the server. A project lives in your browser
session while you work, and on your own machine once you save it.

**Save** downloads a `.hydro` project file — readings, setup, saved analyses and
the project details. **Open** loads one back. **New project** discards
everything not saved.

If the site goes quiet for a while it may sleep, and the session ends with it.
Anything saved to a file is safe; anything only on screen is not.

## Tests

A project holds several tests — one per well, or one per pumping episode. The
sidebar switches between them, and each keeps its own readings, setup and saved
analyses. The active test is the one every panel is showing.

Each test can carry a name, a pumping well and an observation well, under
**Project and test details**.

Switching tests brings back that test's own work: its readings, the fits run
against it, and its last analysis. Nothing from another test is ever shown beside
readings it was not fitted to.

## Which fits are kept, and where

Two different things, and the difference matters:

- **The runs.** Every fit you run is noted against its test — method,
  transmissivity, R² — and listed in the project panel under the active test.
  Running a method again replaces its earlier entry, and runs against readings you
  have since replaced are not listed, because they describe data the test no longer
  holds.
- **Saved analyses.** **Save to project** keeps a fit with everything needed to
  trust it later: the setup it ran with, its full result and a fingerprint of the
  readings. Only saved analyses appear in the comparison table and the report.

Both travel in the project file. When you reopen a project, the most recent saved
analysis is fitted again so the results panel shows it whole — plots, intervals
and validity notes. A fit is deterministic, so what comes back is what you saw;
the file holds the readings rather than a copy of every curve.

## What goes into the PDF report

The report covers the current result: the numbers with their intervals, the
plots, the validity notes, the setup used, and the project details. It is
produced from the same values shown on screen — there is no second calculation
that could disagree.
