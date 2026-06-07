#!/usr/bin/env bash
# Spaced-repetition REVIEW for TorchLeet — re-surface an existing problem and
# advance its schedule. This is the cheap O(read) counterpart to /newproblem:
# it re-opens the on-disk recall prompts instead of regenerating a new problem.
#
# Usage:
#   bash review.sh <problem-path> [grade 1-5]
#   e.g.  bash review.sh llm/14-LoRA 4
#
# grade (optional): your blank-notebook recall quality. <3 (shaky) repeats the
# same interval; >=3 (or omitted) advances to the next interval.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Robust root resolution: prefer the git toplevel of the cwd (works regardless
# of which skill copy this is), fall back to the conventional layout.
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$ROOT" ]] || ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

PROBLEM="${1:?usage: review.sh <problem-path> [grade 1-5]}"
GRADE="${2:-}"

# 1. Re-surface the stubbed notebook's recall prompts (Debrief / Challenge / Anki).
python3 "$SCRIPT_DIR/_schedule.py" --root "$ROOT" resurface "$PROBLEM"

echo
echo "Now do the Challenge from a blank notebook. When done, this advances your schedule:"

# 2. Advance the schedule (repeat-if-shaky when a grade is given).
if [[ -n "$GRADE" ]]; then
    python3 "$SCRIPT_DIR/_schedule.py" --root "$ROOT" advance "$PROBLEM" --grade "$GRADE"
else
    python3 "$SCRIPT_DIR/_schedule.py" --root "$ROOT" advance "$PROBLEM"
fi
