#!/usr/bin/env python3
"""Ask the consultant a fixed set of questions and report how it answered.

    python scripts/eval_chat.py                         # against a local server
    python scripts/eval_chat.py --base-url https://hydroassist.onrender.com \
        --user kgp --password '...'                     # against the live site
    python scripts/eval_chat.py --only honesty          # one category
    python scripts/eval_chat.py --out evals/report.md

The questions and their expectations are in `evals/chat_questions.yaml`.

Why a report rather than a pass/fail suite: half of what makes an answer good
cannot be asserted. Whether the hydrogeology is right, whether a citation
supports the sentence it follows, whether the tone suits a practitioner — those
need a person. So this checks what can be checked (routing, the suggested
method, which papers were retrieved, words that must or must not appear) and
prints the rest for reading, with the automatic verdicts already filled in.

Each question is asked in a fresh conversation, so one answer cannot prime the
next.

A run costs one Gemini call per question, against the free tier's daily
allowance.
"""

import argparse
import base64
import http.cookiejar
import json
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import yaml

QUESTIONS = Path("evals/chat_questions.yaml")

# An answer that never reached the model is not an answer, and must not be
# scored as one. The first run of this script reported nine questions as
# passing when the daily Gemini quota had run out: the honesty checks are
# "must not mention", and a failure message mentions nothing.
FAILURE_MARKERS = (
    "used up its daily allowance",
    "took too long to answer",
    "cannot reach the language model",
    "could not answer that one",
    "encountered an error generating",          # the message this replaced
    "not enough information",                    # retrieval found nothing
)


def _is_failure(reply: str) -> bool:
    lowered = (reply or "").lower()
    return not reply or any(m in lowered for m in FAILURE_MARKERS)


class Site:
    """The app over HTTP, with cookies so a conversation holds together."""

    def __init__(self, base_url, user=None, password=None, insecure=False):
        self.base_url = base_url.rstrip("/")
        handlers = [urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())]
        if insecure:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            handlers.append(urllib.request.HTTPSHandler(context=context))
        self.opener = urllib.request.build_opener(*handlers)
        self.auth = None
        if user is not None:
            token = base64.b64encode(f"{user}:{password or ''}".encode()).decode()
            self.auth = f"Basic {token}"

    def call(self, method, path, payload=None, timeout=300):
        body = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(self.base_url + path, data=body, method=method)
        if body:
            request.add_header("Content-Type", "application/json")
        if self.auth:
            request.add_header("Authorization", self.auth)
        with self.opener.open(request, timeout=timeout) as response:
            raw = response.read().decode()
        return json.loads(raw) if raw else None

    def fresh_conversation(self):
        self.call("GET", "/api/session")
        self.call("POST", "/api/chat/reset")

    def ask(self, question):
        started = time.time()
        state = self.call("POST", "/api/chat/messages", {"message": question})
        return state, time.time() - started


def _reply_of(state) -> str:
    replies = [m["content"] for m in state.get("messages", [])
               if m.get("role") != "user"]
    return replies[-1] if replies else ""


def _check(expect, state, reply):
    """The expectations a machine can settle. Returns [(ok, description)]."""
    results = []
    if not expect:
        return results

    context = state.get("context") or {}
    suggestion = state.get("suggestion") or {}
    retrieved = state.get("retrieved") or []
    lowered = reply.lower()

    if "intent" in expect:
        got = context.get("intent")
        results.append((got == expect["intent"], f"intent is {expect['intent']} (got {got})"))

    if "aquifer" in expect:
        got = context.get("aquifer")
        results.append((got == expect["aquifer"], f"aquifer is {expect['aquifer']} (got {got})"))

    if "suggests_method" in expect:
        got = suggestion.get("method")
        results.append((got == expect["suggests_method"],
                        f"suggests {expect['suggests_method']} (got {got or 'nothing'})"))

    if "least_sources" in expect:
        results.append((len(retrieved) >= expect["least_sources"],
                        f"at least {expect['least_sources']} sources (got {len(retrieved)})"))

    if "sources_include_method" in expect:
        wanted = expect["sources_include_method"]
        tags = {tag for doc in retrieved
                for tag in (doc.get("method") or "").split(",")}
        results.append((wanted in tags, f"a source tagged {wanted}"))

    for phrase in expect.get("must_mention", []):
        results.append((phrase.lower() in lowered, f"mentions {phrase!r}"))

    for phrase in expect.get("must_not_mention", []):
        results.append((phrase.lower() not in lowered, f"avoids {phrase!r}"))

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--user")
    parser.add_argument("--password")
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--only", help="run one category only")
    parser.add_argument("--out", default="evals/report.md")
    args = parser.parse_args()

    questions = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    if args.only:
        questions = [q for q in questions if q.get("category") == args.only]
    if not questions:
        print("No questions matched.")
        return 1

    site = Site(args.base_url, args.user, args.password, args.insecure)
    try:
        site.call("GET", "/api/health", timeout=180)
    except urllib.error.HTTPError as error:
        print(f"{args.base_url} answered {error.code}. "
              f"{'Credentials needed — pass --user and --password.' if error.code == 401 else ''}")
        return 1
    except Exception as error:
        print(f"Cannot reach {args.base_url}: {error}")
        return 1

    print(f"{len(questions)} questions against {args.base_url}\n")

    rows, failures, errored = [], 0, 0
    for index, item in enumerate(questions, 1):
        site.fresh_conversation()
        question = " ".join(item["question"].split())
        try:
            state, seconds = site.ask(question)
        except Exception as error:                     # a timeout, a 500, a refusal
            print(f"[{index}/{len(questions)}] {item['id']}: FAILED — {error}")
            rows.append({**item, "broken": True, "reply": f"(no answer: {error})",
                         "checks": [], "seconds": 0.0, "sources": []})
            errored += 1
            continue

        reply = _reply_of(state)
        broken = _is_failure(reply)
        # A failed call is reported as such, and its checks are not counted:
        # "must not mention X" is trivially true of an error message.
        checks = [] if broken else _check(item.get("expect"), state, reply)
        failed = [text for ok, text in checks if not ok]
        failures += len(failed)
        if broken:
            errored += 1

        verdict = "NO ANSWER" if broken else ("ok" if not failed else f"{len(failed)} unmet")
        print(f"[{index}/{len(questions)}] {item['id']:32s} {seconds:5.1f}s  {verdict}")
        for text in failed:
            print(f"       missed: {text}")

        rows.append({
            **item,
            "broken": broken,
            "reply": reply,
            "checks": checks,
            "seconds": seconds,
            "sources": [f"{d.get('source', '?')} — {(d.get('method') or '')[:40]}"
                        for d in (state.get("retrieved") or [])],
        })

    report = _render(rows, args.base_url)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    checked = sum(len(r["checks"]) for r in rows)
    print(f"\n{checked - failures}/{checked} automatic checks met.")
    print(f"Report written to {out} — the answers still need reading.")
    return 0


def _render(rows, base_url) -> str:
    """A report to read, not a score. Every answer is printed in full."""
    lines = [
        "# Consultant evaluation",
        "",
        f"{len(rows)} questions against `{base_url}`, "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}.",
        "",
        "The checks below are the ones a machine can settle. The answers are "
        "printed in full because the rest — whether the hydrogeology is right, "
        "whether a citation supports the sentence it follows — needs a reader. "
        "`Grade` is for you.",
        "",
        "| # | Question | Checks | Time | Grade |",
        "|---|---|---|---|---|",
    ]
    for index, row in enumerate(rows, 1):
        met = sum(1 for ok, _ in row["checks"] if ok)
        total = len(row["checks"])
        summary = "—" if not total else f"{met}/{total}"
        if row.get("broken"):
            summary = "**no answer**"
        elif total and met < total:
            summary = f"**{summary}**"
        lines.append(f"| {index} | {row['id']} | {summary} | {row['seconds']:.0f}s |  |")

    lines += ["", "---", ""]

    for index, row in enumerate(rows, 1):
        lines += [f"## {index}. {row['id']}", "",
                  f"*{row.get('category', 'uncategorised')}*", "",
                  f"**Asked:** {' '.join(row['question'].split())}", ""]
        if row.get("look_for"):
            lines += [f"**Read for:** {' '.join(row['look_for'].split())}", ""]
        for ok, text in row["checks"]:
            lines.append(f"- {'PASS' if ok else 'FAIL'} — {text}")
        if row["checks"]:
            lines.append("")
        if row["sources"]:
            lines += ["**Sources used:**", ""]
            lines += [f"- {s}" for s in row["sources"]]
            lines.append("")
        lines += ["**Answer:**", "", "> " + row["reply"].replace("\n", "\n> "), "",
                  "**Grade:**", "", "---", ""]

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
