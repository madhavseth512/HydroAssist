# Getting your readings into HydroAssist

Step 1 of the workspace is **Field data**. Nothing else works until a test has
readings.

## What HydroAssist accepts

- **CSV** — the usual export from a spreadsheet or logger.
- **Text files** (`.txt`, `.tsv`, `.dat`, `.prn`) — tab, comma, semicolon or
  space separated. The separator is detected; you do not have to say which.
- **Excel workbooks** (`.xlsx`, `.xlsm`). If the workbook has several sheets,
  the first one that looks like a table of readings is used, and the panel says
  which sheet that was.
- **Pasted columns** — press **Paste data**, copy two columns straight out of a
  spreadsheet including their headers, and press **Use these readings**.

Use the **Choose file** button to upload, or **Replace file** to swap the
readings on a test that already has some.

## What the columns should hold

Two columns are enough: elapsed time, and drawdown.

- **Time** in seconds, minutes or hours. The unit is read from the column name
  where possible — `time_min`, `t (h)`, `elapsed_s` all work.
- **Drawdown** in metres or feet, and converted to metres for you.
- Extra columns are ignored. Several drawdown columns are read as several wells.

Drawdown means the *fall* in water level from the pre-pumping level, so the
numbers normally increase through a constant-rate test. Depth-to-water readings
from a logger are not drawdown and must be converted first.

## What happens after an upload

The panel reports what it found:

- the columns it detected, their units, and how confident it is;
- a **derivative diagnostic** — a plot reading the flow regime off your own
  curve, before any method is chosen;
- conversions applied, rows removed, and anything that looks wrong.

Readings at t = 0 are dropped, because drawdown at zero time is undefined in
every method here. Duplicate times are removed. Both are listed in **Rows
removed** so nothing disappears silently.

## If the upload is refused

- *"only one column was found"* — the separator was not recognised. Save as CSV,
  or paste the columns instead.
- *"that file could not be read"* — the format is not one of those listed above.
  A PDF or an image cannot be read; type or paste the numbers.
- *"the table has no rows"* — the file has a header and nothing else.
