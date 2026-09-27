# The consultant

The consultant is the chat panel. It answers from a library of hydrogeology
papers, and from your own analysis once you have run one.

## What it can do

- Explain a method, its assumptions and when it applies.
- Suggest which method suits the test you describe, and hand that choice to the
  workspace so the form is already set.
- Interpret your own fitted T and S, including the validity notes.
- Explain how to use HydroAssist itself — where a field is, what a note means,
  what to do next.

## What it cannot do

- Run a calculation. It suggests a method; the workspace fits it.
- Answer from papers it does not have. Slug tests, for instance, are not in the
  library and are not implemented.
- Invent a citation. If it cannot find something it says so, which is the
  behaviour to expect and to report if you ever see otherwise.

## Sources

Each answer lists the passages it was drawn from, underneath itself, under **What
this answer read**. Every entry names the document and the page, says how
relevant the passage was, and expands to the whole passage so you can check the
answer against it.

Most of the library is published papers behind a paywall, so a source usually
says *no public link* rather than offering one that would not open. Where a link
can be derived — a journal article whose DOI is fixed — it is there.

If an answer shows no sources, it was not drawn from the library: a message about
a failure, or an answer about HydroAssist's own interface, has nothing to cite.

## What it is analysing, and what it is discussing

The panel shows both, and they are different things:

- **analysing** — the method the workspace ran. Only you change this, by choosing
  a method or accepting a suggestion.
- **discussing** — the method this conversation is about.

Asking about a method is a question, not an instruction to switch to it. If you
want the workspace to use the method you are discussing, the suggestion card's
button is what does it.

## What it remembers

Every fit you run this session is kept, not only the last one. Ask it to compare
two methods, or which ones you have tried, and it answers from the actual runs.
The list is also in the project panel, under the active test.

Two things it does *not* remember:

- **Runs on readings you have replaced.** Uploading new data clears the list,
  because fits on different readings are not comparable.
- **Anything after the session ends.** The conversation lives on the server for
  the session; a reload starts a new one. Your project is kept in the browser and
  offered back, but the discussion is not.

Saving an analysis to the project is still what makes it part of the record —
that is what the comparison table and the report are built from.

## Daily limit

The language model runs on a free allowance, which is small and shared by
everyone using the site. When it is spent the consultant says so and stops
answering until it resets.

**The calculators are unaffected.** Fitting, plots, the comparison table and the
PDF report never use the model, so they keep working normally.

Anything sent to the consultant goes to Google's free tier, which may use it to
improve their products. Do not paste anything confidential.

## Answers appear as they are written

The reply is streamed, a few words at a time, rather than appearing whole several
seconds later — the first sentence usually carries the verdict. If the connection
cannot stream, the answer arrives complete instead; nothing is lost either way.
