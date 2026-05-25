---
description: Creates a new practice problem (Question + Solution notebooks) from a topic and time budget. Use when the user wants to add a new problem to the repo.
---

Create a new practice problem based on: $ARGUMENTS

## Argument Parsing

Extract from `$ARGUMENTS`:
1. **Topic**: an arxiv link, paper name, concept name, or description
2. **Time budget**: a duration like `10m`, `30m`, `1h`, `2h` (default: `30m` if omitted)

If an arxiv link is provided, fetch the paper to understand its content before designing the problem.

## Step 0: Refresh the User-Implemented Functions List

CLAUDE.md was last updated:
!`git log -1 --format="%ci" -- CLAUDE.md`

If the last update was **today**, skip this step. Otherwise, scan all `*-Question.ipynb` files across the repo, extract the PyTorch/Python functions the student implemented themselves (from TODO/pass patterns in code cells), and update the "User-Implemented Functions/Methods (Running List)" section in CLAUDE.md. Only add, never remove. Commit the update before proceeding. Make sure it was the student that implemented the functions, not a past instance of you.

This ensures the scaffolding tiers in Step 2 are always based on current data.

## Step 1: Deconstruct the Skill

Before writing any notebook cells, perform this analysis:

1. **Identify the atomic sub-skills** the topic decomposes into (e.g., "TRAK" -> per-sample gradients, random projection matrices, projected gradient features, attribution scores)
2. **Rank sub-skills by learning value**: which ones transfer most broadly? Which are the core insight vs. boilerplate? Apply the 80/20 rule -- which 20% of sub-skills yield 80% of understanding?
3. **Check CLAUDE.md's "User-Implemented Functions" list** to determine which PyTorch functions/patterns the student has already used. This drives scaffolding tiers (see scaffolding-tiers.md).
4. **Fit to time budget** using the allocation algorithm below. If you can't fit everything, cut from the bottom of the learning-value ranking -- never water down the core.

### Time Allocation Algorithm

Each sub-skill's implementation time depends on the student's familiarity:

- **Tier 1** (first encounter with key functions): ~8-12 min per part
- **Tier 2** (has used key functions before, new combination): ~5 min per part
- **Tier 3** (has done this specific operation before): ~2 min per part

Sum the estimated times. If over budget, remove lowest-learning-value parts first. Priority order: **novel core concepts > composition/integration > recall of known concepts**.

**Recall embedding budget** (parts that re-implement something from a previous problem):
- **10m**: No recall parts (pure drill on one topic)
- **30m**: 0-1 recall parts (if time allows after core parts)
- **1h**: 1 recall part
- **2h**: 1-2 recall parts

### Time Budget -> Problem Type

The time budget is **continuous** -- the student can specify any duration (15m, 45m, 90m, etc.). The table below shows reference points; interpolate between them for intermediate values. A 15-minute problem is a slightly meatier recall drill (maybe 2-3 tight functions). A 45-minute problem is a focused implementation with one extra part. Use the time allocation algorithm above to determine exact part count.

| Budget | Type | Parts | Character |
|--------|------|-------|-----------|
| **~10m** | **Recall Drill** | 1-2 | No context. Pure retrieval. "Can you do X right now from memory?" |
| **~30m** | **Focused Implementation** | 3-5 | Minimal context. One concept, end-to-end with immediate feedback. |
| **~1h** | **Deep Implementation** | 5-7 | Moderate context. Core contribution of a paper/concept + integration test. |
| **~2h** | **Full Paper** | 7-12 | Full context. Reproduce key results. Interleave skills. Embed recall drills. |

These are **qualitatively different** problem types, not scaled versions of the same template. A 10-minute drill has zero opening lecture. A 2-hour paper has architecture diagrams. Interpolate the character/scaffolding smoothly between reference points.

## Step 2: Determine Scaffolding Tiers

For each part's implementation stub, choose a tier based on the student's history in CLAUDE.md. See scaffolding-tiers.md for the full templates.

**Rule**: NEVER hint a function the student has already used in a previous Question notebook. Check the "User-Implemented Functions" list in CLAUDE.md. If they've used `torch.func.vmap`, `einops.rearrange`, `F.softmax`, etc., those do not get hints -- ever. This forces retrieval, not recognition.

## Step 3: Build the Notebooks

Create TWO Jupyter notebooks in the appropriate directory:
1. **Question notebook** (`<name>-Question.ipynb`) -- the student-facing problem
2. **Solution notebook** (`<name>.ipynb`) -- complete working solution

Determine the correct directory from the topic (e.g., `llm/`, `torch/basic/`, `numerai/`, `practice/`, `papers/`). Create a new numbered subdirectory following the existing convention.

See notebook-structure.md for the full cell-by-cell template.

## Step 4: Solution Notebook

The solution notebook has the same structure but with:
- All TODOs replaced with working implementations
- All cells executed with outputs visible
- May use different idioms (einops, einsum) as long as correctness is identical

## Step 5: Validate and Finalize

1. **Run the solution notebook end-to-end**: `uv run jupyter nbconvert --to notebook --execute <solution>.ipynb`
2. **Verify the Question notebook parses**: ensure no syntax errors (all stubs end with `pass`)
3. **Verify every student-written variable is referenced by a subsequent test cell**
4. **Time-budget sanity check**: mentally walk through the parts. Would a student at the expected level finish in roughly the budgeted time? Adjust part count if not.
5. **Update README.md** to include the new problem
6. **Update CLAUDE.md**:
   - Add the problem to the appropriate section
   - Do NOT update "User-Implemented Functions" yet (that happens after the student solves it)
   - If the problem introduces a new category or pattern, note it
7. **Commit and push**: commit all new/changed files (notebooks, README.md, CLAUDE.md) and `git push` to keep the remote repo in sync.
8. **Log for spaced repetition**: Append an entry to `.spaced-repetition.json` in the project root:
   ```json
   {
     "problem": "<directory path, e.g. llm/14-LoRA>",
     "completed_date": "<today, YYYY-MM-DD>",
     "reviews": [],
     "schedule": [3, 7, 21, 60]
   }
   ```
   This schedules automatic review reminders at 3, 7, 21, and 60 days. The SessionStart hook in `.claude/settings.json` will surface due reviews when a session begins.

## Design Philosophy

These principles must hold for every problem generated:

- **Every line the student writes should require thinking, not typing.** If it's plumbing, scaffold it. If it requires understanding, leave it as a TODO.
- **Time on task = time learning.** Zero seconds reading boilerplate, zero seconds debugging imports, zero seconds loading data. All time on the core skill.
- **Force retrieval over recognition.** The student should pull knowledge from memory, not choose from visible options. Fewer hints. Sparser scaffolding. Especially for things they've done before.
- **Feedback is diagnostic, not just binary.** "Wrong" isn't helpful. "Your attention weights don't sum to 1 along dim=-1, which means your softmax is applied to the wrong dimension" is helpful.
- **The problem adapts to the student, not the other way around.** A student who has implemented attention 5 times gets `def attention(q, k, v): pass`. A first-timer gets full type annotations, docstrings, and step-by-step TODOs.
- **Different time budgets produce different experiences.** A 10-minute drill is not a baby version of a 2-hour paper. It is a fundamentally different exercise with a different purpose (retrieval practice vs. deep understanding).
