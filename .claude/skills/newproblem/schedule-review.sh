#!/usr/bin/env bash
# Spaced repetition review checker for TorchLeet.
# Reads .spaced-repetition.json from the project root and prints any
# problems that are due for review today (or overdue).
#
# Schedule format per entry:
# {
#   "problem": "llm/03-Attention",
#   "completed_date": "2025-05-20",
#   "reviews": ["2025-05-23"],
#   "schedule": [3, 7, 21, 60]
# }
#
# "schedule" = remaining intervals in days. After each review, the first
# element is popped and the review date is appended to "reviews".

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SCHEDULE_FILE="$PROJECT_ROOT/.spaced-repetition.json"

if [[ ! -f "$SCHEDULE_FILE" ]]; then
    exit 0
fi

# Bail if file is empty array
entry_count=$(python3 -c "
import json
data = json.load(open('$SCHEDULE_FILE'))
print(len(data))
" 2>/dev/null || echo "0")

if [[ "$entry_count" == "0" ]]; then
    exit 0
fi

# Check for due reviews — pass path via env var
SCHEDULE_FILE="$SCHEDULE_FILE" python3 -c "
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

schedule_file = Path(os.environ['SCHEDULE_FILE'])

with open(schedule_file) as f:
    entries = json.load(f)

today = datetime.now().date()
due = []

for entry in entries:
    schedule = entry.get('schedule', [])
    if not schedule:
        continue

    reviews = entry.get('reviews', [])
    completed = entry.get('completed_date', '')
    if not completed:
        continue

    # The anchor date is the last review date, or the completed date
    if reviews:
        anchor = datetime.strptime(reviews[-1], '%Y-%m-%d').date()
    else:
        anchor = datetime.strptime(completed, '%Y-%m-%d').date()

    next_interval = schedule[0]
    due_date = anchor + timedelta(days=next_interval)

    if due_date <= today:
        days_overdue = (today - due_date).days
        due.append((entry['problem'], due_date, days_overdue))

if due:
    print('--- SPACED REPETITION: Problems due for review ---')
    for problem, due_date, overdue in sorted(due, key=lambda x: x[2], reverse=True):
        if overdue > 0:
            print(f'  {problem}  (due {due_date}, {overdue}d overdue)')
        else:
            print(f'  {problem}  (due today)')
    print('Tip: Use /newproblem <topic> 10m to create a quick recall drill.')
    print('After reviewing, update .spaced-repetition.json: pop schedule[0], append today to reviews.')
    print('---------------------------------------------------')
"
