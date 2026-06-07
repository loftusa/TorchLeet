#!/usr/bin/env bash
# Spaced repetition review checker for TorchLeet (SessionStart hook).
# Prints any problems due for review today (or overdue), then how to review one.
#
# All schedule mechanics live in _schedule.py (single source of truth); this is
# a thin wrapper so the hook and review.sh can never drift apart.
#
# Schedule entry format (.spaced-repetition.json):
#   {"problem": "llm/03-Attention", "completed_date": "2025-05-20",
#    "reviews": ["2025-05-23"], "schedule": [3, 7, 21, 60]}
# "schedule" = remaining intervals in days; reviews advance it via review.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$ROOT" ]] || ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

[[ -f "$ROOT/.spaced-repetition.json" ]] || exit 0

python3 "$SCRIPT_DIR/_schedule.py" --root "$ROOT" due
