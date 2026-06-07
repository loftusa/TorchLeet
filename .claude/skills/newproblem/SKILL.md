---
description: Creates a new practice problem (Question + Solution notebooks) from a topic and time budget. Use when the user wants to add a new problem to the repo.
---

Create a new practice problem based on: $ARGUMENTS

## Argument Parsing

Extract from `$ARGUMENTS`:
1. **Topic**: an arxiv link, paper name, concept name, or description
2. **Time budget**: a duration like `10m`, `30m`, `1h`, `2h` (default: `30m` if omitted)
3. **Flags**:
   - `--deconstruct-first` (1h/2h only): the student decomposes the topic himself before seeing your decomposition. See Step 1.5.
   - `--mode {implement,debug}` (default `implement`): `debug` ships working code with planted bugs to find and fix, instead of stubs to fill. A varied-practice REVIEW modality, gated to already-known operations — see Step 3 "Debug modality" + notebook-structure.md. Best paired with a due spaced-rep review: `/newproblem <topic> 10m --mode debug`.

If an arxiv link is provided, fetch the paper to understand its content before designing the problem.

**No topic given?** If `$ARGUMENTS` is empty, or is just a budget, or is the literal `recommend`/`next`, do **Step -1: Recommend a Topic** instead of generating immediately.

## Step -1: Recommend a Topic (only when no topic was supplied)

Don't make the student invent a topic from a blank page — that metacognitive tax is itself a cost, and choosing what to practice is a coach's job. Surface an **unranked menu** from the only two grounded signals, with a one-line rationale each, then **ask the student to pick** (never silently choose):

1. **Due reviews** — run `python3 .claude/skills/newproblem/_schedule.py due`. These are the items actually decaying (Make It Stick: study what's being forgotten). Reviewing one is `review.sh <problem>`, not a new problem — offer that path.
2. **Coverage gaps** — topics in CLAUDE.md's "Key Interview Statistics" (Attention 70%, MHA 60%, BPE 50%, PE 45%, RMSNorm 35%, ...) for which **no problem directory exists yet**. List 2-3 highest-frequency uncovered topics.

Present as a short menu (e.g. "Due for review: …; Not yet covered: …") and ask which to do. Do **not** compute a value-per-minute score — the mastery/time terms aren't trustworthy. Once the student picks, proceed to Step 0 with that topic + budget.

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
4. **Tag each sub-skill on TWO independent axes** — they scaffold different things:
   - **function-novel**: are its primitives NOT in the User-Implemented Functions list? (drives *implementation* hint density — Tiers 1/2/3)
   - **idea-novel**: is the *composition/insight* conceptually new even if every primitive is already known? The canonical case is SAE angular-selectivity's "smallest enclosing arc = 2π − largest angular gap": every function (`sort`, `max`, `diff`) is known, yet the idea is genuinely new. These are invisible to the functions-list and get mis-scaffolded as bare Tier-3 stubs, blowing past the ~85% sweet spot into floundering. idea-novel sub-skills get the **Insight tier** conceptual nudge (see scaffolding-tiers.md), NOT extra code hints.
5. **Fit to time budget** using the allocation algorithm below. If you can't fit everything, cut from the bottom of the learning-value ranking -- never water down the core.

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

## Step 1.5: Deconstruct-First (only if `--deconstruct-first`, 1h/2h only)

The single most transferable meta-skill — decomposing a paper into atomic sub-skills and finding the 20% that yields 80% of understanding — is normally something you do *for* the student. With this flag, make him practice producing it. Below 1h, ignore the flag (no budget for it). Otherwise: still compute your Step 1 sub-skill list + 80/20 ranking, but do NOT emit it as the Cell-0 "Component Breakdown". Instead reserve it for the deconstruct cells (see notebook-structure.md "Deconstruct-First Cells"): the student writes his decomposition first, then a collapsible reveals yours as *one defensible split* to compare against. Zero extra generation cost (reuses Step 1 output).

## Step 2: Determine Scaffolding Tiers

For each part's implementation stub, choose a tier based on the student's history in CLAUDE.md. See scaffolding-tiers.md for the full templates.

**Rule**: NEVER hint a function the student has already used in a previous Question notebook. Check the "User-Implemented Functions" list in CLAUDE.md. If they've used `torch.func.vmap`, `einops.rearrange`, `F.softmax`, etc., those do not get hints -- ever. This forces retrieval, not recognition.

**Outcome-aware override (close the loop)**: before finalizing tiers, read the last ~30 lines of `.practice-log.jsonl` at the repo root (written by the Session Debrief — see notebook-structure.md). If a function/operation involved in this problem had a most-recent outcome of `solution` or `failed` (i.e. the student needed the answer or couldn't finish), scaffold it **one tier easier** (more support) than the functions-list would imply — it has decayed past the "known cold" assumption. This is the only signal that overrides the binary used/not-used rule; without it the tier model can only get harder, never accounting for forgetting. Degrade gracefully: if `.practice-log.jsonl` is missing or empty, use the functions-list rule unchanged.

## Step 3: Build the Notebooks

Create TWO Jupyter notebooks:
1. **Question notebook** (`<problem-dir>/<name>-Question.ipynb`) -- the student-facing problem
2. **Solution notebook** (`<problem-dir>/_solutions/<name>.ipynb`) -- complete working solution, in a `_solutions/` subfolder, **NOT beside the Question**

Determine the correct directory from the topic (e.g., `llm/`, `torch/basic/`, `numerai/`, `practice/`, `papers/`). Create a new numbered subdirectory following the existing convention.

**Why `_solutions/`**: retrieval is the highest-value mechanism in this skill (the Debrief calls it "where a large fraction of long-term retention is created"). An executed answer key one tab from the Question silently converts retrieval into recognition. The subfolder adds just enough friction that opening the solution is a deliberate act, not an accidental glance — while keeping it on disk as the ground-truth reference.

See notebook-structure.md for the full cell-by-cell template.

**Debug modality (`--mode debug`)**: instead of stub-filling, the Question's Cell B ships the *working* implementation with 1–3 planted bugs to locate and fix. This trains fault-localization-from-diagnostics — a skill the implement modality never exercises. Two hard gates:
- **Only for operations ALREADY in the User-Implemented Functions list** (Tier-3 territory). Showing the full body of a function the student has *never* implemented is recognition, not retrieval — it violates the core rule. If the target operation isn't in the list, refuse `--mode debug` and fall back to `implement` with a one-line note.
- The validation (Cell C) is **reused verbatim as the bug oracle** — plant bugs only where an existing assert/reference-diff will actually fire (no new validation logic, no no-oracle scavenger hunt).

Do NOT add `derive`/`predict` modes — the template already covers them (Predict-before-you-code, Key Formula header, Break-it-on-purpose extension).

## Step 4: Solution Notebook

The solution notebook (in `_solutions/`) has the same structure but with:
- All TODOs replaced with working implementations
- May use different idioms (einops, einsum) as long as correctness is identical
- **Output visibility is conditional on whether the Question already has a feedback floor:**
  - If the Question's validation cells contain a **reference-match** (`torch.allclose` against a PyTorch/library reference with EXPECTED diagnostics), the student already has graded numeric feedback, so **clear the solution's outputs** (`nbconvert --clear-output`) — execute it to prove correctness, then strip outputs so it's not a peek-able answer key.
  - If the Question's validation is **purely invariant/property-based** (no numeric reference — common for paper-reproduction problems), the executed solution is the *only* ground truth, so **keep its outputs**.

## Step 5: Validate and Finalize

1. **Run the solution notebook end-to-end**: `uv run jupyter nbconvert --to notebook --execute _solutions/<solution>.ipynb`. Then, per Step 4's rule, if the Question has a reference-match feedback floor, clear the solution's outputs: `uv run jupyter nbconvert --clear-output --inplace _solutions/<solution>.ipynb`.
2. **Verify the Question notebook parses**: ensure no syntax errors (all stubs end with `pass`)
3. **Verify every student-written variable is referenced by a subsequent test cell** (this also covers the committed prediction literals — see notebook-structure.md Cell A/C)
4. **Time-budget is a hard ceiling, not a target**: mentally walk through the parts. The budget is the MAXIMUM; deliberately under-fill it. If a student at the expected level might run *over*, cut parts (lowest learning-value first) — never pad to fill time. Running long is the failure mode, not running short.
5. **Update README.md** to include the new problem (link the Question; link the solution as `_solutions/<name>.ipynb`)
6. **Update CLAUDE.md**:
   - Add the problem to the appropriate section
   - Do NOT update "User-Implemented Functions" yet (that happens after the student solves it)
   - If the problem introduces a new category or pattern, note it
7. **Commit and push**: commit all new/changed files (notebooks incl. `_solutions/`, README.md, CLAUDE.md, `.spaced-repetition.json`) and `git push` to keep the remote repo in sync.
8. **Log for spaced repetition** (automated — never hand-edit JSON):
   ```bash
   python3 .claude/skills/newproblem/_schedule.py log "<directory path, e.g. llm/14-LoRA>"
   # debug-modality problem: add --mode debug so a later review can interleave modalities
   ```
   This appends a `[3, 7, 21, 60]`-day schedule entry (idempotent). The SessionStart hook surfaces due reviews; the student then reviews with `review.sh <problem> [grade]` (which advances the schedule for him — no manual edit).

### Review vs. generation (don't confuse them)

`/newproblem` creates **new material** (full pipeline above; new schedule entry). A **review** of already-learned material is the cheap path: `bash .claude/skills/newproblem/review.sh <problem> [grade 1-5]` re-surfaces the on-disk recall prompts (Debrief / Challenge / Anki) and advances that problem's schedule — O(read), no regeneration, no duplicate entry. Only reach for `/newproblem <topic> 10m` as a "review" when you deliberately want a *fresh, varied* drill (desirable difficulty / interleaving).

## Reference Architecture

Two references: one for **design choices** (what frontier models actually use), one for **runnable model** (what to load in notebooks).

### Frontier reference (last verified 2026-05-25): Kimi K2 (Moonshot AI)
Use this for architectural *decisions* — which attention variant, activation function, positional encoding, etc.
- 1T total params, 32B active (MoE: 384 experts, 8 selected per token, 1 shared)
- 61 layers, hidden dim 7168, 64 attention heads, MLA (Multi-head Latent Attention)
- SwiGLU activation, 160K vocab, 128K context
- Open-weight, Apache 2.0

### Runnable model (last verified 2026-05-25): Qwen3-0.6B
When a problem needs an actual LLM to run (e.g., extracting hidden states, generating completions, fine-tuning), use the **smallest model that still uses modern SOTA architecture**. Currently that's `Qwen/Qwen3-0.6B`:
- 0.6B params, 32 layers, hidden dim 1024, 16 attention heads, 8 KV heads (GQA)
- SwiGLU activation, RoPE, 128K context, 151K vocab
- Runs on any GPU in bf16, even on CPU in reasonable time

Use a larger model only when the problem specifically requires it (e.g., demonstrating emergent capabilities, MoE routing, or multi-GPU parallelism).

### Keeping current
At the start of each problem generation, if either reference is >3 months old, do a quick web search to check for newer open-weight models. Update this section if so.

## Design Philosophy

These principles must hold for every problem generated:

- **Every line the student writes should require thinking, not typing.** If it's plumbing, scaffold it. If it requires understanding, leave it as a TODO.
- **Time on task = time learning.** Zero seconds reading boilerplate, zero seconds debugging imports, zero seconds loading data. All time on the core skill.
- **Force retrieval over recognition.** The student should pull knowledge from memory, not choose from visible options. Fewer hints. Sparser scaffolding. Especially for things they've done before.
- **Feedback is diagnostic, not just binary.** "Wrong" isn't helpful. "Your attention weights don't sum to 1 along dim=-1, which means your softmax is applied to the wrong dimension" is helpful.
- **The problem adapts to the student, not the other way around.** A student who has implemented attention 5 times gets `def attention(q, k, v): pass`. A first-timer gets full type annotations, docstrings, and step-by-step TODOs.
- **Different time budgets produce different experiences.** A 10-minute drill is not a baby version of a 2-hour paper. It is a fundamentally different exercise with a different purpose (retrieval practice vs. deep understanding).
- **Worked examples are for novel IDEAS, never novel functions.** Scaffold a hard *concept* with an insight sentence or a worked→faded→full progression; never hand an expert a worked example of a function he knows cold — that is expertise-reversal, a net negative. Code-stub sparseness tracks function history; conceptual scaffolding tracks idea-novelty. They are independent (see Step 1.4 + the Insight tier).
- **Scaffold the idea, not the typing.** A genuinely new mechanism built from familiar primitives still deserves a conceptual nudge even though every line is the student's to write.
