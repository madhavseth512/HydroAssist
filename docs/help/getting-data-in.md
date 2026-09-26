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
- **Logger exports** — Solinst Levelogger (`.lev`), Van Essen Diver (`.mon`),
  In-Situ (`.xle`, or the `.csv` they write). The instrument's header block,
  clock timestamps and water levels are read as they are: the timestamps become
  elapsed time, and the level or depth to water becomes drawdown. Every
  conversion is listed under **Conversions applied**, including which level it
  measured drawdown against.
- **A photograph of a printed table** — press **Photograph of a table**. See
  below; this one you must check.

Use the **Choose file** button to upload, or **Replace file** to swap the
readings on a test that already has some.

## What the columns should hold

Two columns are enough: elapsed time, and drawdown.

- **Time** in seconds, minutes or hours. The unit is read from the column name
  where possible — `time_min`, `t (h)`, `elapsed_s` all work.
- **Drawdown** in metres or feet, and converted to metres for you.
- Extra columns are ignored. Several drawdown columns are read as several wells.

Drawdown means the *fall* in water level from the pre-pumping level, so the
numbers normally increase through a constant-rate test. A depth to water is not
drawdown — but you do not have to convert a logger export yourself, because
HydroAssist recognises one and does it, saying what it did.

Where no static water level is given, the first reading is taken as the
pre-pumping level. That is right only if recording began before the pump did; if
it did not, every drawdown is understated, and the panel says so.

## What happens after an upload

The panel reports what it found:

- the columns it detected, their units, and how confident it is;
- a **derivative diagnostic** — a plot reading the flow regime off your own
  curve, before any method is chosen;
- conversions applied, rows removed, and anything that looks wrong.

Readings at t = 0 are dropped, because drawdown at zero time is undefined in
every method here. Duplicate times are removed. Both are listed in **Rows
removed** so nothing disappears silently.

## Reading a photograph of a field sheet

Readings that exist only on paper can be photographed. Press **Photograph of a
table**, which on a phone offers the camera directly.

What comes back is **not** readings. It is a grid of what was recognised, for you
to check:

- every cell can be edited, and holds the characters that were read;
- cells the recognition was unsure of are marked in yellow, and hovering one
  shows how sure it was;
- you name each column — time in seconds, minutes or hours, drawdown in metres
  or feet — because a photograph has no header row and nothing can tell minutes
  from seconds;
- a row can be left out, and an empty cell must be filled in or its row dropped.

Only when you press **Use these readings** do the numbers enter the project, by
exactly the same route as numbers you type.

This is deliberately the one import you must confirm. A misread digit does not
fail: `3.5` recognised as `8.5` fits a type curve just as well as the right
number, reports a transmissivity that is wrong, and gives nobody a reason to look
twice. A photograph of a printed table, taken square-on and in focus, reads well;
a hand-written sheet often does not read at all, and typing it is quicker than
correcting it.

## If the upload is refused

- *"only one column was found"* — the separator was not recognised. Save as CSV,
  or paste the columns instead.
- *"that file could not be read"* — the format is not one of those listed above.
  A PDF cannot be read; type or paste the numbers. An image is not read here
  either — use **Photograph of a table**, which proposes a grid to check.
- *"a photograph could not be read"* — the server may not have the recognition
  program installed, which it will say; every other way of getting data in still
  works.
- *"the table has no rows"* — the file has a header and nothing else.
