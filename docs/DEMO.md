# HydroAssist — demo plan and script

A ~13-minute screen recording that shows a professor what the platform is, what it
does that a spreadsheet and a commercial package do not, and why its numbers can
be trusted. Everything below is checked against the code as it stands: the values
in **Numbers you should see** are what the screen will show if you follow the
steps in order.

---

## 1. Before you record

### Wake the site first — this one matters

The free host stops the container after 15 minutes idle, and the first request
afterwards takes about a minute while the knowledge base loads. **Open
`https://hydroassist.onrender.com`, sign in, upload a file and run one analysis
about three minutes before you start recording.** Then start a new project, so
the recording begins on a clean workspace against a warm server.

Prove the whole path works before you talk over it:

```bash
python scripts/smoke_deploy.py https://hydroassist.onrender.com \
    --user <user> --password '<password>'
```

Eight checks. The one to watch is **read a photographed field sheet** — it needs
Tesseract inside the deployed image, which has not been verified on the live
server yet. If that check reports 503, either record §5 (the photograph) against
a local server, where Tesseract is installed, or drop that segment. Everything
else is unaffected.

### Budget the consultant's questions

The language model runs on a free allowance of about **20 answers a day**, shared
by anyone using the site. The script uses **four** questions. Write them down and
ask exactly those; do not improvise ten while recording. The allowance resets at
midnight Pacific (12:30 IST).

If you run dry mid-recording, the app says so plainly and the calculators keep
working — which is itself worth one sentence on camera — but don't plan for it.

### Files in one folder, named for the camera

Copy these out of the repository into one folder before you start, because the
file dialog will be on screen:

| Rename to | From | Used in |
|---|---|---|
| `1-pumping-test.csv` | `tests/theis_cooper_jacob_sample.csv` | §3, §4, §6 |
| `2-large-diameter-well.csv` | `tests/papadopulos_cooper_sample.csv` | §6 (optional) |
| `3-field-sheet.png` | `tests/fixtures/field_data/field_sheet.png` | §5 |
| `4-gridley-1962.csv` | `tests/fixtures/field_data/gridley_well1.csv` | §7 |

### Capture settings

- **1920×1080.** The workspace is three columns; anything narrower makes the
  consultant unreadable on a projector.
- Browser at 100% zoom, bookmarks bar hidden, one tab.
- Clear this site's data first (or use a fresh profile) so the "carry on where you
  left off?" banner does not appear until §8, where you want it.
- Record the whole screen, not just the browser — the file dialog and the
  downloaded PDF are part of the story.

---

## 2. The shape of the demo

Follow the order a hydrogeologist works in. Each segment earns the next.

| § | Segment | Minutes | The one thing it proves |
|---|---|---|---|
| 1 | What it is | 1:00 | The claim: it computes *and* justifies |
| 2 | The workspace | 0:45 | A tool, not a web form |
| 3 | Data in, as it arrives | 1:30 | CSV, Excel, logger exports, pasted columns |
| 4 | Diagnostic before method | 1:30 | Modern practice: read the regime off the data |
| 5 | A photographed field sheet | 1:15 | Paper records, into a grid you confirm |
| 6 | Fit, and refuse to mislead | 3:00 | Validity notes, bounds, AICc over R² |
| 7 | Validated against the literature | 2:00 | Walton (1962), to within 1.9% |
| 8 | The consultant | 2:30 | Grounded, cited, and honest about limits |
| 9 | Keeping the work | 1:00 | Projects, runs, report |
| 10 | How it is built and hosted | 1:00 | 1,696 tests; £0 |

**If you need to cut to 8 minutes:** drop §5 and §9, and shorten §8 to two
questions. **Do not cut §6 or §7** — they are the argument.

---

## 3. The script

Read the plain text aloud. `[ACTION]` lines are what you do on screen and are
not spoken. Pause where marked; silence while a plot draws is better than
narrating the loading.

---

### § 1 — What it is (1:00)

`[ACTION] Start on the workspace, already signed in, nothing uploaded.`

> HydroAssist is a platform for analysing pumping tests. You give it the water
> levels measured in a well while another well is being pumped, and it returns the
> two properties that matter for groundwater: transmissivity — how much water the
> aquifer transmits — and storativity, how much it releases from storage.
>
> That calculation is not new. Theis published it in 1935, and commercial software
> has done it for thirty years. What is different here is the second half of the
> job. This does not only compute a number: it argues for it. Every result comes
> with the assumptions behind the method, the ones your data violates, the pages
> of the papers those conditions come from, and a comparison against the other
> methods that could have been used instead.
>
> It runs in a browser, there is nothing to install, and it is deployed at no
> cost. I will work through one real analysis, end to end.

---

### § 2 — The workspace (0:45)

`[ACTION] Point with the cursor: left panel, centre, then the toolbar menus.`

> The layout follows the shape of the work. On the left is the project — a project
> holds several tests, one per well or per pumping episode, and each test keeps its
> own readings, its own setup and its own analyses.
>
> The centre is the analysis itself, in four steps: the field data, the test setup,
> the result, and the comparison between methods.

`[ACTION] Open the File menu, then the Analysis menu. Close them.`

> The actions are in menus and a toolbar rather than scattered through the page,
> because that is where someone who uses AQTESOLV or Excel will look for them.

`[ACTION] Drag the divider between the left panel and the centre, then drag it
back. Then click Consultant to open the right panel, and widen it a little.`

> Every divider is draggable, and each browser remembers the widths it was given —
> a long derivation in the consultant needs a wide column; a long test name needs
> a wide tree.

---

### § 3 — Data in, as it arrives (1:30)

`[ACTION] Click "Choose file" and pick 1-pumping-test.csv.`

> Field data does not arrive in one tidy format, and the single biggest barrier to
> using software like this is being made to reformat your own measurements first.
>
> This reads comma- and tab-separated text, Excel workbooks, columns pasted
> straight out of a spreadsheet, and the raw exports from the pressure transducers
> people actually leave down a well — Solinst, Van Essen Diver, In-Situ.

`[ACTION] Let the panel render. Point at the detection badges.`

> It has worked out which column is time and which is drawdown, and in what units
> — minutes here, metres here — and it says how confident it is. That matters
> because a time column read as seconds instead of minutes gives you a
> transmissivity out by a factor of sixty, with no error message anywhere.
>
> The readings are shown as they were read, so the numbers a fit is about to run
> on are in front of the person running it.

`[ACTION] Scroll to "Conversions applied" and open it.`

> And everything it did to your numbers is listed. Nothing is converted silently.

**Optional, 15 seconds, if you want to show the logger path:** a transducer export
is recognised by its own header, not by its file extension, so its clock
timestamps become elapsed time and its water levels become drawdown — and every
one of those conversions is listed here too, because getting the sign of drawdown
backwards produces a curve that still fits and a transmissivity that is wrong.

---

### § 4 — Diagnostic before method (1:30)

`[ACTION] Scroll to the derivative diagnostic plot.`

> This is the part I would point to first.
>
> The older way to use software like this is to choose a method, fit it, and look
> at R². Modern practice is the other way round: you read the flow regime off your
> own data first, and then pick the model that matches what you saw.
>
> This is the Bourdet derivative — the rate of change of drawdown against the
> logarithm of time. Its shape is diagnostic. A flat late-time derivative means
> infinite-acting radial flow, which is exactly the condition the Theis solution
> assumes. A derivative that dips means water is leaking in through a confining
> layer. A doubling means a barrier boundary.

`[ACTION] Point at the regime line: "Radial".`

> Here it reads the regime as radial, and tells you *why* — the late-time slope is
> flat — and then which methods are consistent with that reading. So the method is
> chosen by the evidence, not by habit.

---

### § 5 — A photographed field sheet (1:15)

`[ACTION] Click "Photograph of a table" and choose 3-field-sheet.png.`

> Some readings only exist on paper — a printed table, or a sheet on a clipboard.
> This reads a photograph of one.

`[ACTION] Let the grid open. Point at the yellow cell.`

> But notice what it does *not* do. It has not produced readings. It has produced a
> proposal, and it will not become data until I confirm it.
>
> That is deliberate, and it is the most important design decision in this feature.
> Recognition never fails loudly. A three-point-five read as eight-point-five fits
> a curve just as well as the right number, reports a transmissivity that is
> simply wrong, and gives nobody a reason to look twice. So every cell carries the
> confidence the recognition assigned it, the doubtful ones are marked — this one
> here was read at seventy-one per cent — and I have to name the columns myself,
> because a photograph has no header row and nothing in the software can tell
> minutes from seconds.

`[ACTION] Either confirm the grid to show it becoming readings, or press Discard
and say the next line.`

> Once I confirm, those numbers enter by exactly the same door as numbers I type.

---

### § 6 — Fit, and refuse to mislead (3:00)

`[ACTION] If you confirmed the OCR grid, re-upload 1-pumping-test.csv. Scroll to
Step 2. Point at the method cards.`

> Step two is the test setup and the method. The methods are cards rather than a
> dropdown, and each one states what it assumes.

`[ACTION] Click the "6 assumptions" line on the Theis card.`

> Theis assumes a confined aquifer, uniform and infinite, a fully penetrating well,
> a constant rate, and water released instantly from storage. Those are the
> conditions a reviewer will ask about, so they are on the card and not in a manual.

`[ACTION] Point at a greyed-out card, if one is shown.`

> And a method that cannot be used here is still shown, greyed, with the reason —
> because the reason is the part that teaches.

`[ACTION] Keep Theis selected. Q = 500 m³/day, r = 50 m. Click "Run analysis".`

> Transmissivity, two hundred and fifteen point three metres squared per day, with
> a ninety-five per cent confidence interval. Storativity, five times ten to the
> minus five — which is the right order of magnitude for a confined aquifer, and
> worth saying out loud, because a storativity of one in a hundred would mean the
> aquifer is not confined at all.

`[ACTION] Point at the validity notes.`

> Underneath are the validity notes. These are generated from the fit, not written
> in advance. This one tells me the Cooper-Jacob condition — u below nought point
> nought two — is satisfied for every late-time point in this dataset, so a second
> method applies and is worth running as a check.

`[ACTION] Select the Cooper-Jacob card. Run analysis.`

> Two hundred and fifteen point four. The two methods agree to within six
> hundredths of one per cent, on the same data, by different mathematics — one
> fitting the whole curve, one fitting a straight line to late time only.

`[ACTION] Point at the note about excluded points.`

> And it reports that twenty-three early-time points were excluded, with the
> reason: the approximation Cooper-Jacob rests on is not valid there. It is not
> hiding the points it chose not to use.

**Now the segment that matters most.**

`[ACTION] Select Papadopulos-Cooper. Run analysis on the same dataset.`

> Now let me deliberately do the wrong thing. Papadopulos-Cooper is for a
> large-diameter well, where the water stored in the borehole itself dominates the
> early response. This dataset is not that. A naive tool would fit it anyway and
> print a number.

`[ACTION] Point at the notes, one at a time.`

> It does fit it — four hundred and fifty-three — and then it refuses to let that
> stand. Storativity has hit the upper bound of the range the method allows, so it
> is reported as a limit and not a measurement. The fit is poor: R² nought point
> seven eight. And this note quotes the 1967 paper, page 244, on exactly this
> situation: the determined value of storativity changes by an order of magnitude
> when the data plot is moved from one type curve to another.
>
> It also tells me why, in terms of the physics: none of my fifty points lie in the
> wellbore-storage regime this method needs. So the method does not apply to this
> test, and the software says so, in the words of the people who derived it.

`[ACTION] Open the Compare panel.`

> Finally, the comparison. The three fits are ranked — and ranked by AICc, not by
> R².

`[ACTION] Point at the ΔAICc and weight columns.`

> That distinction is a correctness issue, not a refinement. R² always improves
> when you add parameters, so ranking on it would recommend the most complicated
> method every time. AICc charges a penalty for each extra parameter, so a simple
> model that explains the data is allowed to win. The wording of the evidence —
> strong, or not worth distinguishing — follows Burnham and Anderson's
> conventions.
>
> And the validity notes sit beside the ranking, because "best fit" and "right
> model" are different claims. A method that does not physically apply can still
> come first on a statistic.

---

### § 7 — Validated against the literature (2:00)

`[ACTION] Start a new test: "Add test" in the project panel. Upload
4-gridley-1962.csv.`

> Everything so far was computed on a dataset I generated. The obvious question is
> whether the mathematics is right on real measurements — so here is a test nobody
> in this project ran.
>
> These are readings from a pumping test at Gridley, Illinois. They were published
> by Walton in 1962, in Illinois State Water Survey Bulletin 49, which is one of
> the documents in this platform's own library.

`[ACTION] Point at the conversions list, where the file's comment lines appear.`

> The file carries its own provenance, and the software shows it rather than
> discarding it: the table, the pumping rate, and the fact that Walton analysed
> this with the non-leaky artesian type curve — the Theis method.

`[ACTION] Set Q = 1199.2 m³/day, r = 251.2 m. Run Theis.`

> Walton published a transmissivity of eleven thousand gallons per day per foot.
> Converted, that is a hundred and thirty-six point six metres squared per day. We
> get a hundred and twenty-three — about ten per cent below his.
>
> I want to be careful about how I describe that, because the honest statement is
> not "we match". Walton matched a type curve by eye, on log-log paper, in 1962.
> Least squares on the same eleven points is a different estimator, and where the
> two disagree it is not obvious that the older one is wrong.

`[ACTION] Switch the units toggle in the header to US, and point at the value.`

> The display converts to gallons per day per foot, so the comparison with the
> published number needs no arithmetic by hand.
>
> On the companion test from the same bulletin — the pumped well, analysed by
> Cooper-Jacob — we land within **one point nine per cent** of the value Walton
> printed. And on a third, a leaky aquifer at Dieterich, where our transmissivity
> differs from his by more, the test asserts something stronger instead: that our
> parameters explain his measurements better than his own do. Root mean squared
> error of three point eight centimetres against his five point five.
>
> Altogether there are forty-six tests against published well-function tables, and
> twenty-two against real field datasets with their analysts' printed answers.
> That is the basis for trusting the numbers, and it runs on every change.

---

### § 8 — The consultant (2:30)

`[ACTION] Open the Consultant panel if it is closed. Widen it by dragging.`

> The right-hand panel is the part that makes this more than a calculator. It is a
> consultant grounded in a library of forty-nine hydrogeology papers — Theis,
> Cooper and Jacob, Hantush, Neuman, Walton, USGS reports — about four thousand
> passages, indexed and searchable.
>
> It is not a chatbot talking about hydrogeology in general. Watch what comes with
> each answer.

`[ACTION] Ask question 1, exactly as written:`
`"What are the assumptions of the Theis method, and which of them does my dataset
violate?"`

> Notice three things while that is being written.

`[ACTION] Let it stream. Point at the text as it appears.`

> One: the answer arrives as it is written, rather than appearing whole after
> several seconds — the first sentence of an interpretation usually carries the
> verdict.
>
> Two: it is answering about *my* analysis. It has been given the fit I just ran —
> the method, the parameters, the validity notes — so "my dataset" means something
> specific to it.

`[ACTION] Expand "What this answer read".`

> Three, and this is the part a reviewer should care about: every answer carries the
> passages it was built from. The document, the page, how relevant the passage was,
> and the passage itself, so you can check the answer against what it actually
> read. Where a link exists it is offered; most of this library is published papers
> behind a paywall, and a citation that says "no public link" is more use than one
> that implies a link and fails to open.
>
> If an answer shows no sources, it did not read any — a message about a failure
> cites nothing, which is the behaviour you want.

`[ACTION] Ask question 2: "Why were 23 of my points excluded?"`

> That one is answered from the fit's own validity notes rather than from the
> papers — the software is asked about itself, and it reads its own result.

`[ACTION] Ask question 3: "Where do I enter the pumping rate?"`

> And this is a different kind of question — about the interface, not the
> hydrogeology. It is routed to a separate help index, so it answers from the
> application's own documentation and does not go searching Theis 1935 for a text
> field.

`[ACTION] Point at the offered button, then click it.`

> It offers to take me there. It does not move the workspace by itself — the
> assistant proposes and the user decides, which is a rule the whole platform
> follows.

`[ACTION] Point at the two badges at the top of the panel.`

> These two badges are the same rule stated plainly. "Analysing" is the method the
> workspace actually ran, and only I can change that. "Discussing" is what the
> conversation happens to be about. Asking about a method is a question, not an
> instruction to switch to it.

`[ACTION] Optional, question 4: "Compare the analyses we have run."`

> And it can compare the fits across this session, because every run is recorded
> against the test.

**One sentence to include, wherever it fits:**

> The model runs on a free allowance of about twenty answers a day. When it is
> spent the panel says so, in those words, and every calculator keeps working —
> fitting, plotting, comparing and the report never touch the model.

---

### § 9 — Keeping the work (1:00)

`[ACTION] Point at the run list in the project panel.`

> Each test keeps a log of the fits run against it, with their transmissivities.

`[ACTION] Click "Save to project" on the result, then open the project tree.`

> Saving an analysis records it with everything needed to trust it later: the setup
> it ran with, the full result, and a fingerprint of the readings — so if the
> readings are replaced, the analysis is marked stale rather than quietly compared
> against data it never saw.

`[ACTION] Click Save in the toolbar to download the project file.`

> A project is a file you keep. There are no accounts and no database — which is
> part of how this is hosted for nothing.

`[ACTION] Reload the page. The restore banner appears.`

> And the browser keeps a copy as you work. The session itself is gone after a
> reload — but it offers the project back, and restoring it brings the readings,
> the setup, the runs and the saved analysis, refitted so the plots are there too.

`[ACTION] Click "Restore it". Then click "Download report".`

> Finally the report: a PDF with the parameters, the intervals, the plots, the
> validity notes and the project details — produced from the same values on screen,
> so there is no second calculation that could disagree.

---

### § 10 — How it is built and hosted (1:00)

`[ACTION] Optionally show the repository, or a terminal with the test suite
finishing. Do not read code aloud.`

> Briefly, on the engineering, because it is part of the claim.
>
> The backend is Python — FastAPI, with the calculators built on a shared
> Laplace-domain engine, so a new method is a transfer function and a parameter
> specification rather than a new three-hundred-line module. The frontend is React
> and TypeScript, with its types generated from the API's own schema, so the two
> halves cannot drift apart silently.
>
> There are one thousand six hundred and ninety-six automated tests on the backend,
> a hundred and seventy-nine on the frontend, and eight that drive a real browser
> end to end. The fits themselves are pinned to nine decimal places, so a
> refactor that changes a number fails loudly.
>
> And the whole thing is deployed for nothing: one container, a free host, with the
> embedding model and the knowledge base baked into the image. No paid tier
> anywhere.

**Closing:**

> So: it computes what commercial software computes, in a browser, at no cost —
> and for every number it reports, it will tell you which assumptions it rests on,
> which of them your data breaks, and which page of which paper says so.

---

## 4. Numbers you should see

Confirm these on screen before you record. If one differs, stop and tell me — the
script quotes them.

| Where | Value |
|---|---|
| §4 derivative | regime **radial**, late-time log-log slope flat |
| §6 Theis on `1-pumping-test.csv` | **T = 215.29 m²/day**, S = 5.05×10⁻⁵, R² = 1.0000, RMSE 2.44 mm |
| §6 Cooper-Jacob, same data | **T = 215.42 m²/day**, S = 5.03×10⁻⁵, 23 points excluded, max u = 0.0167 |
| §6 Papadopulos-Cooper, same data | T = 453.92, R² = 0.78, **S pinned at the 0.01 bound** |
| §7 Theis on `4-gridley-1962.csv` (Q = 1199.2 m³/day, r = 251.2 m) | **T = 123.06 m²/day = 9,909 gpd/ft**; Walton published 11,000 |
| §7 quoted, not shown | Gridley pumped well, Cooper-Jacob: **+1.9%** against Walton's 12,100 gpd/ft |
| §5 `3-field-sheet.png` | **13 rows**, 1 of 26 cells marked uncertain |
| §10 | 1,696 backend tests, 179 frontend, 8 browser |

Papadopulos-Cooper on its *own* dataset (`2-large-diameter-well.csv`, Q = 300
m³/day, r_w = 0.30 m, r_c = 0.20 m) gives T = 86.39 m²/day, S = 1.00×10⁻⁴,
α = 2.2×10⁻⁴ — useful if the professor asks whether the method ever works.

---

## 5. If something goes wrong

| Problem | What to do, and what to say |
|---|---|
| First page load hangs ~1 min | The host had gone to sleep. Cut it in the edit, or say so: "the free tier stops the container when idle". |
| The consultant says the allowance is spent | Don't fight it. Say: "that is the free allowance for the day — note that every calculator still works", then continue with §9. It is an honest moment, not a failure. |
| The photograph returns a 503 | The deployed image is missing Tesseract. Skip §5 on camera and record it separately against a local server. |
| A fit reports "no result to save" | You switched test after running. Re-run, then save. |
| The restore banner appears early | You did not clear site data. Dismiss it with "Not now" and carry on; demo it properly in §9. |

---

## 6. Questions the professor is likely to ask

Have these ready; they are the places where a vague answer would undo the demo.

**"How do I know the mathematics is right?"** Two independent gates. Forty-six
tests reproduce well-function values printed in the literature — W(u) from the
AQTESOLV guide, and W(u, r/B) from Hantush's 1956 tables as reproduced in Walton's
Bulletin 49. Twenty-two more run real published field tests and compare against
the analyst's own answer. Both run on every change.

**"Is this just a wrapper around a language model?"** No. The model is used for
two things: classifying what a question is about, and writing prose answers from
retrieved passages. It touches no calculation. Every parameter on screen comes
from scipy least-squares on the equations in the papers, and the whole analysis
path — upload, fit, compare, report — works with the model switched off.

**"What does it not do yet?"** Step-drawdown and slug tests are not implemented.
Neuman's unconfined solution is not implemented — the consultant can discuss it
because the papers are in the library, but the calculator cannot run it. There is
no forward prediction for test design, no multi-well composite analysis, and no
named wells with coordinates. The comparison against an AQTESOLV demo, on the same
datasets, has not been done.

**"Why not just use AQTESOLV?"** It has roughly twenty-six methods against our
seven, and it is a mature product. Two things here are not in it: the diagnostic
drives the method choice rather than sitting beside it, and every result argues
for itself from the literature. And it runs in a browser with nothing to install.

---

*Keep this file next to the recording. If a number in §4 changes because the code
changed, the script is wrong and should be corrected before re-recording.*
