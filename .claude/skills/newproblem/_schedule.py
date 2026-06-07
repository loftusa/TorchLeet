#!/usr/bin/env python3
"""Spaced-repetition bookkeeping for the /newproblem skill.

Single source of truth for `.spaced-repetition.json` mechanics. Used by:
  - schedule-review.sh  -> `due`        (list problems due for review today)
  - review.sh           -> `resurface` + `advance`  (re-open + advance a review)
  - /newproblem finalize -> `log`        (register a newly created problem)
  - one-time setup       -> `backfill`   (enroll every existing problem, staggered)

Pure stdlib. The whole point is that NO step requires a human to hand-edit JSON.

Schedule entry:
  {"problem": "llm/14-LoRA", "completed_date": "YYYY-MM-DD",
   "reviews": ["YYYY-MM-DD", ...], "schedule": [3, 7, 21, 60]}

`schedule` = remaining intervals (days). anchor = last review date, else
completed_date. next-due = anchor + schedule[0]. An entry with an empty
`schedule` has graduated and is never due again.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

DEFAULT_SCHEDULE = [3, 7, 21, 60]
SHAKY_GRADE = 3  # recall grade < this => repeat the same interval (don't advance)


# --------------------------------------------------------------------------- IO
def find_root(start: str | None = None) -> Path:
    """Walk up from `start` (or cwd) to the repo root.

    Root = first ancestor containing `.spaced-repetition.json` or `.git`.
    """
    p = Path(start or os.getcwd()).resolve()
    for d in [p, *p.parents]:
        if (d / ".spaced-repetition.json").exists() or (d / ".git").exists():
            return d
    return p


def schedule_path(root: Path) -> Path:
    return root / ".spaced-repetition.json"


def load(root: Path) -> list[dict]:
    f = schedule_path(root)
    if not f.exists():
        return []
    data = json.loads(f.read_text() or "[]")
    assert isinstance(data, list), f"{f} must hold a JSON array, got {type(data)}"
    return data


def save(root: Path, entries: list[dict]) -> None:
    """Atomic write (temp file + replace) so a crash never truncates the file."""
    f = schedule_path(root)
    tmp = f.with_suffix(f.suffix + ".tmp")
    tmp.write_text(json.dumps(entries, indent=2) + "\n")
    os.replace(tmp, f)


def today_str(override: str | None = None) -> str:
    if override:
        datetime.strptime(override, "%Y-%m-%d")  # validate
        return override
    return datetime.now().strftime("%Y-%m-%d")


def find_entry(entries: list[dict], problem: str) -> dict | None:
    return next((e for e in entries if e.get("problem") == problem), None)


# ----------------------------------------------------------------- due logic
def next_due(entry: dict) -> tuple | None:
    """Return (due_date, days_overdue) for the next pending review, or None."""
    schedule = entry.get("schedule", [])
    if not schedule:
        return None
    reviews = entry.get("reviews", [])
    anchor_s = reviews[-1] if reviews else entry.get("completed_date", "")
    if not anchor_s:
        return None
    anchor = datetime.strptime(anchor_s, "%Y-%m-%d").date()
    due = anchor + timedelta(days=schedule[0])
    return due, (datetime.now().date() - due).days


def cmd_due(root: Path, _args) -> int:
    entries = load(root)
    due = []
    for e in entries:
        nd = next_due(e)
        if nd and nd[1] >= 0:  # overdue (>0) or due today (==0)
            due.append((e["problem"], nd[0], nd[1]))
    if not due:
        return 0
    print("--- SPACED REPETITION: Problems due for review ---")
    for problem, due_date, overdue in sorted(due, key=lambda x: x[2], reverse=True):
        suffix = f"{overdue}d overdue" if overdue > 0 else "due today"
        print(f"  {problem}  (due {due_date}, {suffix})")
    print("Review one with:  bash .claude/skills/newproblem/review.sh <problem> [grade 1-5]")
    print("  (re-opens the stubbed notebook's recall prompts and advances the schedule for you)")
    print("---------------------------------------------------")
    return 0


# ----------------------------------------------------------------- mutations
def cmd_log(root: Path, args) -> int:
    """Register a new problem (idempotent). Called by /newproblem finalize."""
    entries = load(root)
    if find_entry(entries, args.problem):
        print(f"already logged: {args.problem}", file=sys.stderr)
        return 0
    entry = {
        "problem": args.problem,
        "completed_date": today_str(args.date),
        "reviews": [],
        "schedule": list(DEFAULT_SCHEDULE),
    }
    # Record a non-default modality so a later review can interleave (e.g.
    # implement now, debug later). Omitted for the default to keep entries lean.
    if getattr(args, "mode", None) and args.mode != "implement":
        entry["mode"] = args.mode
    entries.append(entry)
    save(root, entries)
    suffix = f" [{args.mode}]" if entry.get("mode") else ""
    print(f"logged: {args.problem}{suffix} (first review in {DEFAULT_SCHEDULE[0]}d)")
    return 0


def cmd_advance(root: Path, args) -> int:
    """Record a review. grade < 3 => repeat the same interval (don't pop)."""
    entries = load(root)
    e = find_entry(entries, args.problem)
    if e is None:
        print(f"not found in schedule: {args.problem}", file=sys.stderr)
        return 1
    if not e.get("schedule"):
        print(f"{args.problem} already graduated (no pending reviews)", file=sys.stderr)
        return 0
    e.setdefault("reviews", []).append(today_str(args.date))
    grade = args.grade
    if grade is not None and grade < SHAKY_GRADE:
        # Shaky recall: keep the same interval, just re-anchor to today.
        msg = f"shaky (grade {grade}) -> repeat {e['schedule'][0]}d interval"
    else:
        interval = e["schedule"].pop(0)
        nxt = e["schedule"][0] if e["schedule"] else None
        msg = (f"advanced past {interval}d interval; next in {nxt}d"
               if nxt else f"advanced past {interval}d; GRADUATED")
    save(root, entries)
    print(f"{args.problem}: {msg}")
    return 0


def cmd_backfill(root: Path, args) -> int:
    """Enroll every *-Question.ipynb problem dir not already present.

    Re-baselines as completed today, but STAGGERS the first interval so the
    enrolled wave trickles into review (~1-2/day) instead of all coming due at
    once. Existing entries are never touched.
    """
    entries = load(root)
    present = {e.get("problem") for e in entries}
    found = sorted({
        str(p.parent.relative_to(root))
        for p in root.rglob("*-Question.ipynb")
        if ".venv" not in p.parts
    })
    new = [d for d in found if d not in present]
    base = today_str(args.date)
    for k, problem in enumerate(new):
        entries.append({
            "problem": problem,
            "completed_date": base,
            "reviews": [],
            "schedule": [DEFAULT_SCHEDULE[0] + (k % 21)] + DEFAULT_SCHEDULE[1:],
        })
    save(root, entries)
    print(f"backfill: enrolled {len(new)} new problem(s), {len(present)} already present")
    for problem in new:
        print(f"  + {problem}")
    return 0


def cmd_resurface(root: Path, args) -> int:
    """Print the stubbed Question notebook's recall prompts for review.

    Surfaces the on-disk artifact (Debrief / Challenge / Anki) — O(read), no
    regeneration. Reads the STUBBED Question notebook (retrieval, not the
    student's filled-in answers).
    """
    prob_dir = root / args.problem
    qs = sorted(prob_dir.glob("*-Question.ipynb"))
    if not qs:
        print(f"no *-Question.ipynb under {prob_dir}", file=sys.stderr)
        return 1
    nb = json.loads(qs[0].read_text())
    print(f"=== REVIEW: {args.problem} ===")
    print(f"Notebook: {qs[0].relative_to(root)}")
    print("Recall from memory BEFORE re-opening the solution.\n")
    wanted = ("## Session Debrief", "## Anki", "**Challenge")
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        src = "".join(cell.get("source", []))
        if any(w in src for w in wanted):
            print(src.strip())
            print()
    return 0


# ----------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None, help="repo root (default: walk up from cwd)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("due")
    p_log = sub.add_parser("log"); p_log.add_argument("problem"); p_log.add_argument("--date")
    p_log.add_argument("--mode", default=None, help="non-default modality, e.g. debug")
    p_adv = sub.add_parser("advance"); p_adv.add_argument("problem")
    p_adv.add_argument("--grade", type=int, default=None); p_adv.add_argument("--date")
    p_bf = sub.add_parser("backfill"); p_bf.add_argument("--date")
    p_rs = sub.add_parser("resurface"); p_rs.add_argument("problem")

    args = ap.parse_args(argv)
    root = Path(args.root).resolve() if args.root else find_root()
    return {
        "due": cmd_due, "log": cmd_log, "advance": cmd_advance,
        "backfill": cmd_backfill, "resurface": cmd_resurface,
    }[args.cmd](root, args)


if __name__ == "__main__":
    raise SystemExit(main())
