Create a new practice problem based on: $ARGUMENTS

## Argument Parsing

Extract from `$ARGUMENTS`:
1. **Topic**: an arxiv link, paper name, concept name, or description
2. **Time budget**: a duration like `10m`, `30m`, `1h`, `2h` (default: `30m` if omitted)

If an arxiv link is provided, fetch the paper to understand its content before designing the problem.

## Step 0: Refresh the User-Implemented Functions List

Before doing anything else, check when CLAUDE.md's "User-Implemented Functions" section was last updated. Run:

```bash
git log -1 --format="%ci" -- CLAUDE.md
```

If the last update was **today**, skip this step. Otherwise, scan all `*-Question.ipynb` files across the repo, extract the PyTorch/Python functions the student implemented themselves (from TODO/pass patterns in code cells), and update the "User-Implemented Functions/Methods (Running List)" section in CLAUDE.md. Only add, never remove. Commit the update before proceeding. Make sure it was the student that implemented the functions, not a past instance of you.

This ensures the scaffolding tiers in Step 2 are always based on current data.

## Step 1: Deconstruct the Skill

Before writing any notebook cells, perform this analysis:

1. **Identify the atomic sub-skills** the topic decomposes into (e.g., "TRAK" → per-sample gradients, random projection matrices, projected gradient features, attribution scores)
2. **Rank sub-skills by learning value**: which ones transfer most broadly? Which are the core insight vs. boilerplate? Apply the 80/20 rule — which 20% of sub-skills yield 80% of understanding?
3. **Check CLAUDE.md's "User-Implemented Functions" list** to determine which PyTorch functions/patterns the student has already used. This drives scaffolding tiers (see below).
4. **Fit to time budget** using the allocation algorithm below. If you can't fit everything, cut from the bottom of the learning-value ranking — never water down the core.

### Time Allocation Algorithm

Each sub-skill's implementation time depends on the student's familiarity:

- **Tier 1** (first encounter with key functions): ~8-12 min per part
- **Tier 2** (has used key functions before, new combination): ~5 min per part
- **Tier 3** (has done this specific operation before): ~2 min per part

Sum the estimated times. If over budget, remove lowest-learning-value parts first. Priority order: **novel core concepts > composition/integration > recall of known concepts**.

### Time Budget → Problem Type

The time budget is **continuous** — the student can specify any duration (15m, 45m, 90m, etc.). The table below shows reference points; interpolate between them for intermediate values. A 15-minute problem is a slightly meatier recall drill (maybe 2-3 tight functions). A 45-minute problem is a focused implementation with one extra part. Use the time allocation algorithm above to determine exact part count.

| Budget | Type | Parts | Character |
|--------|------|-------|-----------|
| **~10m** | **Recall Drill** | 1-2 | No context. Pure retrieval. "Can you do X right now from memory?" |
| **~30m** | **Focused Implementation** | 3-5 | Minimal context. One concept, end-to-end with immediate feedback. |
| **~1h** | **Deep Implementation** | 5-7 | Moderate context. Core contribution of a paper/concept + integration test. |
| **~2h** | **Full Paper** | 7-12 | Full context. Reproduce key results. Interleave skills. Embed recall drills. |

These are **qualitatively different** problem types, not scaled versions of the same template. A 10-minute drill has zero opening lecture. A 2-hour paper has architecture diagrams. Interpolate the character/scaffolding smoothly between reference points.

## Step 2: Determine Scaffolding Tiers

For each part's implementation stub, choose a tier based on the student's history in CLAUDE.md:

### Tier 1 — First Encounter
The student has NOT used the key functions/patterns before. Provide full scaffolding.

```python
def function_name(
    arg1: Float[Tensor, "batch seq d_model"],
    arg2: Float[Tensor, "batch seq d_model"],
) -> Float[Tensor, "batch seq d_model"]:
    """What this computes, why it matters.

    Args:
        arg1: description
        arg2: description

    Returns:
        description with shape annotation
    """
    # Step 1: <what to compute>
    # Hint: <specific PyTorch function or formula>

    # Step 2: <what to compute>
    # Hint: <specific guidance>

    pass
```

### Tier 2 — Has Used Key Functions
The student has used the core functions but not in this combination. Compressed stub.

```python
def function_name(
    arg1: Float[Tensor, "batch seq d_model"],
    arg2: Float[Tensor, "batch seq d_model"],
) -> Float[Tensor, "batch seq d_model"]:
    """One-line description. See formula above."""
    # TODO: Implement
    pass
```

### Tier 3 — Recall Drill
The student has done this specific operation before. Bare minimum.

```python
def function_name(arg1, arg2):
    # Implement from memory
    pass
```

**Rule**: NEVER hint a function the student has already used in a previous Question notebook. Check the "User-Implemented Functions" list in CLAUDE.md. If they've used `torch.func.vmap`, `einops.rearrange`, `F.softmax`, etc., those do not get hints — ever. This forces retrieval, not recognition.

## Step 3: Build the Notebooks

Create TWO Jupyter notebooks in the appropriate directory:
1. **Question notebook** (`<name>-Question.ipynb`) — the student-facing problem
2. **Solution notebook** (`<name>.ipynb`) — complete working solution

Determine the correct directory from the topic (e.g., `llm/`, `torch/basic/`, `numerai/`, `practice/`, `papers/`). Create a new numbered subdirectory following the existing convention.

### Cell 0: Session Header (all time budgets)

**10-minute drill:**
```markdown
# [Title]

**Goal**: Implement [X] from memory.
**Time**: 10 minutes.
```

That's it. No background. No formulas. No references. The student either knows it or they don't — that's the point.

**30-minute focused:**
```markdown
# [Title]

**Goal**: After this session, you will be able to implement [specific capability] from memory.
**Time**: 30 minutes.

## What and Why
[1-3 sentences: what it is, where it appears in production, why it matters]

## Key Formula
$$[The core formula in LaTeX]$$

## References
[arxiv/paper link if applicable]
```

**1-hour deep:**
Same as 30-minute, plus:
```markdown
## Component Breakdown
[Bulleted list: 1 line per sub-skill, what each computes]
```

**2-hour full paper:**
Same as 1-hour, plus:
```markdown
## Architecture
[ASCII diagram or description of how components connect]

## Key Insight
[1-2 sentences: what makes this paper's approach novel vs. prior work]
```

### Cell 1: Imports + Setup (scaffolded, all budgets)

Provide ALL imports, data setup, model setup, and boilerplate. The student should be writing their first TODO within 60 seconds of opening the notebook. Every second spent on setup is a second not spent learning.

For papers: include synthetic data that demonstrates the concept without requiring large downloads. Keep it self-contained.

### Cells 2+: Per-Part Structure

For each part, exactly **three cells**:

**Cell A — Part Header (markdown):**
```markdown
## Part N: [Title]
[1 sentence: what to implement and why it matters in the larger system]

<details><summary>Hint 1: [category]</summary>[specific PyTorch function or formula fragment]</details>
```

Hint rules:
- **10m drills**: Zero hints. None.
- **30m**: 0-1 hints per part (only for Tier 1 sub-skills)
- **1h**: 1-2 hints per part (only for Tier 1 sub-skills)
- **2h**: 1-3 hints per part (only for Tier 1 sub-skills)
- **NEVER hint functions the student has used before** (check CLAUDE.md)
- Hints go IN the part, not in the opening section — help at point of need

**Cell B — Implementation Stub (code):**
Use the appropriate tier (1/2/3) based on Step 2. Use `jaxtyping` for tensor shape annotations. Constrain variable names by referencing them in the test cell below.

**Cell C — Validation (code):**

```python
# --- Part N Validation ---
torch.manual_seed(42)
# [test tensor setup]

result = student_function(test_input)

# Shape
assert result.shape == expected_shape, f"Shape: expected {expected_shape}, got {result.shape}"
print(f"  Shape: {result.shape} -- correct")

# Diagnostics (builds mental representation of what correct tensors look like)
print(f"  Range: [{result.min():.4f}, {result.max():.4f}]")
print(f"  Mean:  {result.mean():.4f}, Std: {result.std():.4f}")

# Property (domain invariant — explain WHY in the assertion message)
assert <invariant>, "<why this property must hold, e.g. 'attention weights must sum to 1 along key dimension'>"
print(f"  [Property name] -- correct")

# Reference match
ref = <pytorch_reference>(test_input)
max_diff = (result - ref).abs().max()
assert torch.allclose(result, ref, atol=1e-6), f"Max diff: {max_diff:.2e} (threshold: 1e-6)"
print(f"  Reference match -- correct (max diff: {max_diff:.2e})")

print(f"\nPart {N} complete.")
```

Key design choices:
- Print intermediate statistics (range, mean, std) so the student builds intuition for what correct tensors look like
- Assertion messages explain the invariant's rationale, not just state it
- Diagnostics run even on success — the student should learn what "correct" looks like
- No emoji in test output

### Recall Embedding (2-hour problems only)

For 2-hour problems, embed 1-2 parts that require the student to re-implement something from a previous problem, from memory. Instead of importing a helper:

```markdown
## Part 1: [Previously Learned Concept] (Recall)
You implemented this in [previous problem]. Reproduce it here from memory — do not look at your old solution.
```

Use Tier 3 scaffolding for these. This creates spaced repetition within the session and interleaves old + new skills.

### Final Cell: Session Debrief (all budgets except 10m)

```markdown
## Session Debrief

Without scrolling up, answer in your head:
1. What is the core formula for [topic]?
2. What is the key PyTorch function for [central operation]?
3. What shape does [key tensor] have and why?

**Challenge**: Close this notebook, open a blank one, and rewrite Part [hardest part number] from scratch without looking back.
```

This final retrieval practice cements the session. It's not optional decoration — it's where a large fraction of long-term retention is created.

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

## Design Philosophy Summary

These principles must hold for every problem generated:

- **Every line the student writes should require thinking, not typing.** If it's plumbing, scaffold it. If it requires understanding, leave it as a TODO.
- **Time on task = time learning.** Zero seconds reading boilerplate, zero seconds debugging imports, zero seconds loading data. All time on the core skill.
- **Force retrieval over recognition.** The student should pull knowledge from memory, not choose from visible options. Fewer hints. Sparser scaffolding. Especially for things they've done before.
- **Feedback is diagnostic, not just binary.** "Wrong" isn't helpful. "Your attention weights don't sum to 1 along dim=-1, which means your softmax is applied to the wrong dimension" is helpful.
- **The problem adapts to the student, not the other way around.** A student who has implemented attention 5 times gets `def attention(q, k, v): pass`. A first-timer gets full type annotations, docstrings, and step-by-step TODOs.
- **Different time budgets produce different experiences.** A 10-minute drill is not a baby version of a 2-hour paper. It is a fundamentally different exercise with a different purpose (retrieval practice vs. deep understanding).
