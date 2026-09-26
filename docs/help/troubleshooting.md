# When something does not work

## "No calculator for this case yet"

The consultant recognised what you are doing but HydroAssist has no method for
it. Seven methods are implemented: Theis, Theis recovery, Cooper-Jacob,
Papadopulos-Cooper, Hantush-Jacob, Boulton and Warren-Root. Step-drawdown tests,
slug tests and Neuman's unconfined solution are not among them yet.

## A method is missing from the dropdown

Methods are offered by test type and by what the data can support. A method the
validator excluded is listed under the dropdown with the reason. Changing the
test type in Step 2 changes the whole list — recovery methods never appear for a
constant-rate test.

## The fit failed

The message says what went wrong. Common causes:

- **Q or r is wrong or missing.** A rate in the wrong unit gives a
  transmissivity out by orders of magnitude rather than an error.
- **Too few usable points.** Cooper-Jacob needs several late-time readings; if
  the u < 0.02 filter leaves fewer than four, it cannot fit a line.
- **Recovery readings mixed into a pumping test**, or the wrong recovery clock.

## Most of my points were excluded

That is usually correct. Cooper-Jacob is only valid at late time, so early
readings are excluded by design and the count is reported. If nearly everything
was excluded, the test may be too short for the method — Theis fits the whole
curve and does not have this restriction.

## Storativity looks impossible

A storativity near 0.1 for a supposedly confined aquifer usually means one of
three things: the aquifer is behaving as unconfined, the distance r is wrong, or
the drawdown is from the pumping well itself, where S cannot be estimated
reliably at all.

## The consultant stopped answering

Its daily allowance is spent. See "The consultant". Everything else still works.

## The photograph read badly, or not at all

Recognition is good at printed tables photographed square-on and in focus, and
poor at everything else — a hand-written sheet often does not read at all,
because the recognition is trained on print.

If most cells come back marked uncertain, the grid says so: correcting it cell by
cell is then slower than typing the numbers, and typing them is not a defeat.

Two specific messages:

- *"no rows of numbers were found"* — the photograph is too skewed, too dark or
  too soft. Retake it square-on, filling the frame with the table.
- *"reading a photograph needs the Tesseract program"* — the server does not have
  it installed. That is about the server, not your image, and no other way of
  getting data in is affected.

Lines that held a single number are left out and counted, because one number on a
line is as likely to be a page number as a reading. If a reading was lost that
way, type it into the grid.
