# Notebook Cell Structure

## Cell 0: Session Header (all time budgets)

**10-minute drill:**
```markdown
# [Title]

**Goal**: Implement [X] from memory.
**Time**: 10 minutes.
```

That's it. No background. No formulas. No references. The student either knows it or they don't -- that's the point.

**30-minute focused:**
```markdown
# [Title]

**Goal**: After this session, you will be able to implement [specific capability] from memory.
**Time**: 30 minutes.

## What and Why
[1-3 sentences: what it is, where it appears in production, why it matters]

## Key Formula
$$[The core formula in LaTeX]$$

**Where:**
- [Define every variable and symbol in the formula. No term left undefined.]

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

## Deconstruct-First Cells (only with `--deconstruct-first`, 1h/2h)

When SKILL.md Step 1.5 is active, insert these two cells immediately after the Session Header and BEFORE the imports/parts, so the student sees only the title/abstract/link first. Reuse your Step 1 ranking verbatim — zero extra cost. Do NOT also emit the Cell-0 "Component Breakdown" (that would give the answer away).

```markdown
## Decompose this first (before scrolling down)

This paper/topic breaks into a handful of atomic sub-skills. Before you read any
further: list the sub-skills you think it decomposes into, then rank them by
learning value (which 20% carry 80% of the understanding?). Write your answer,
then expand the reveal to compare — it's *one* defensible split, not ground truth.

<details><summary>One reasonable decomposition (compare, don't defer to it)</summary>

[Your Step 1 atomic sub-skill list + 80/20 ranking, verbatim, 1 line each]
[+ one line: "a stronger split might additionally surface ___"]
</details>
```

(The Part structure below still partially telegraphs the decomposition, so this is a useful-but-imperfect meta-rep — claim encoding-depth/generation benefit, not retention.)

## Cell 1: Imports + Setup (scaffolded, all budgets)

Provide ALL imports, data setup, model setup, and boilerplate. The student should be writing their first TODO within 60 seconds of opening the notebook. Every second spent on setup is a second not spent learning.

For papers: include synthetic data that demonstrates the concept without requiring large downloads. Keep it self-contained.

**Synthetic data rule:** Every synthetic tensor must have the **exact same shape, dtype, and semantic role** as the real tensor it stands in for. If the student's code would receive `(batch, seq_len, hidden_dim)` from a real base model, the synthetic data must be `(batch, seq_len, hidden_dim)` — not a lookup table or proxy that happens to produce the right shape after indexing. Name variables to match what they'd be called in production (e.g., `hidden_states` not `fake_embeddings`). Add a comment mapping each synthetic tensor to its real-world origin (e.g., `# same shape as base_model(tokens).hidden_states[-1]`).

**Synthetic oracle/loss rule:** When a problem needs an oracle function, ground-truth signal, or synthetic loss target, it must **structurally resemble the real thing**. For example, a reward model's oracle should be a fixed linear projection of hidden states (mimicking a learned reward head), not `tensor.mean()` — because the student should build intuition for what the real optimization landscape looks like. The oracle can be simple, but it should exercise the same structural pattern the model will learn in production. Ask: "would a practitioner recognize this setup as a simplified version of the real pipeline?"

## Cells 2+: Per-Part Structure

For each part, exactly **three cells**:

### Cell A — Part Header (markdown)

```markdown
## Part N: [Title]
[1 sentence: what to implement and why it matters in the larger system]

**Predict before you code**: What shape will the output tensor be? What range of values do you expect?

<details><summary>Hint 1: [category]</summary>[specific PyTorch function or formula fragment]</details>
```

The prediction prompt forces the student to form a mental model before writing code. Skip it for 10m drills (pure recall — no time for metacognition). Include it for 30m+ budgets.

**Commit quantifiable predictions (don't leave them as dead prose).** An unchecked prediction recruits almost none of the generation effect, and a wrong prediction never confronted with the truth wastes the highest-value learning moment (hypercorrection). So:
- For any prediction that resolves to a **number, shape, or count**, the FIRST line of Cell B (the stub) is a committed literal the student fills *before* coding, e.g. `predicted_xhat_std = ...  # commit a number before you code` or `predicted_shape = (...)`. Reference it in Cell C (see below) so Step 5.3's variable-reference check enforces it.
- For **qualitative/relational** predictions ("concentrated or spread?", "does the regime switch?"), keep them as prose — do NOT force them into a literal.
- Skip entirely for 10m drills.

Hint rules:
- **10m drills**: Zero hints. None.
- **30m**: 0-1 hints per part (only for Tier 1 sub-skills)
- **1h**: 1-2 hints per part (only for Tier 1 sub-skills)
- **2h**: 1-3 hints per part (only for Tier 1 sub-skills)
- **NEVER hint functions the student has used before** (check CLAUDE.md)
- Hints go IN the part, not in the opening section -- help at point of need
- **Insight tier is separate from function hints**: if the sub-skill is *idea-novel* (Step 1.4), also emit the one-sentence insight + self-explanation prompt from scaffolding-tiers.md "Insight Tier" — even when the function gets zero hints. The insight nudge describes the *idea*; it is not a function hint.

### Cell B — Implementation Stub (code)

Use the appropriate tier (1/2/3) from scaffolding-tiers.md. Use `jaxtyping` for tensor shape annotations. Constrain variable names by referencing them in the test cell below.

### Cell C — Validation (code)

```python
# --- Part N Validation ---
torch.manual_seed(42)
# [test tensor setup]

result = student_function(test_input)

# Prediction reveal (BEFORE any raising assert, so a buggy impl still gets confronted).
# Include one line per committed prediction from Cell A. No pass/fail — the gap IS the signal.
print(f"  You predicted x_hat std ~{predicted_xhat_std}; actual {result.std():.3f}"
      f"  (a large gap is the most valuable moment of the session)")

# Shape
assert result.shape == expected_shape, f"Shape: expected {expected_shape}, got {result.shape}"
print(f"  Shape: {result.shape} -- correct")

# Diagnostics (builds mental representation of what correct tensors look like)
print(f"  Range: [{result.min():.4f}, {result.max():.4f}]")
print(f"  Mean:  {result.mean():.4f}, Std: {result.std():.4f}")

# Property (domain invariant -- explain WHY in the assertion message)
assert <invariant>, "<why this property must hold, e.g. 'attention weights must sum to 1 along key dimension'>"
print(f"  [Property name] -- correct")

# Reference match (with diagnostic diff on failure)
ref = <pytorch_reference>(test_input)
max_diff = (result - ref).abs().max()
try:
    assert torch.allclose(result, ref, atol=1e-6), f"Max diff: {max_diff:.2e} (threshold: 1e-6)"
except AssertionError:
    print(f"  YOUR output:  {result.flatten()[:8].tolist()}")
    print(f"  EXPECTED:     {ref.flatten()[:8].tolist()}")
    diff = (result - ref).abs()
    print(f"  Max diff:     {diff.max():.2e} at flat index {diff.argmax().item()}")
    print(f"  Mean diff:    {diff.mean():.2e}")
    raise
print(f"  Reference match -- correct (max diff: {max_diff:.2e})")

print(f"\nPart {N} complete.")
```

Key design choices:
- **Reveal committed predictions before the raising asserts** — the prediction-vs-truth confrontation (generation + hypercorrection effect) is the point; if it printed after an assert that raises, a wrong implementation would swallow it in a traceback exactly when the student erred most.
- Print intermediate statistics (range, mean, std) so the student builds intuition for what correct tensors look like
- Assertion messages explain the invariant's rationale, not just state it
- Diagnostics run even on success -- the student should learn what "correct" looks like
- On failure, show a side-by-side comparison (first 8 elements, max/mean diff, location) so the student can self-correct without asking for help
- No emoji in test output

## Debug Modality (`--mode debug`)

A varied-practice alternative to stub-filling (see SKILL.md Step 3 for the gates: known-operations only; Cell C reused as oracle). Cells differ as follows:

- **Cell 0 (header)**: Goal becomes "Find and fix the bug(s)." State the count up front: "This implementation has **exactly N** bug(s) (N = 1–3). The tests below will tell you when it's correct."
- **Cell A (per part)**: replace "Predict before you code" with **"Predict before you run"**: "Read the implementation. Which invariant in the test do you expect to fail, and why?" (A committed guess at the failing invariant — same generation/hypercorrection benefit, now aimed at fault localization.)
- **Cell B**: ship the Solution's **working** implementation with the planted bug(s) inline. Plant bugs ONLY where Cell C's existing invariant assert or reference-match diff will actually fire — e.g. softmax over the wrong dim → "weights don't sum to 1" assert; a dropped `1/sqrt(d)` scale → reference-match diff; an off-by-one causal mask → a shape/property assert. Never plant a bug that no existing test catches.
- **Cell C**: **unchanged, verbatim** — it is the oracle. The student edits Cell B until Cell C passes.

The `_solutions/` notebook holds the correct (un-bugged) implementation as the answer key.

## Recall Embedding (30m+ problems)

For problems 30 minutes or longer, embed recall parts that require the student to re-implement something from a previous problem, from memory. Instead of importing a helper:

```markdown
## Part N: [Previously Learned Concept] (Recall)
You implemented this in [previous problem]. Reproduce it here from memory -- your old
worked solution is in that problem's `_solutions/` folder; opening it ends the rep, so
only consult it after you've produced or genuinely failed an attempt.
```

Use Tier 3 scaffolding for these. This creates spaced repetition within the session and interleaves old + new skills.

**Recall part budget by time:**
- **10m**: No recall parts (pure single-topic drill)
- **30m**: 0-1 recall parts (only if time allows after core parts)
- **1h**: 1 recall part
- **2h**: 1-2 recall parts

## Final Cell: Session Debrief (all budgets except 10m)

Two cells: a free-recall markdown cell, then a one-line session-log code cell.

**Debrief (markdown):**
```markdown
## Session Debrief

Write your answers into the code cell below (typing is overt retrieval — far
stronger than answering "in your head"). Don't scroll up.
1. What is the core formula for [topic]?
2. What is the key PyTorch function for [central operation]?
3. What shape does [key tensor] have and why?

**Check yourself**: your worked solution is in `_solutions/` — open it (and the
paper) to grade your answers. Opening it ends the retrieval rep, so answer first.

**Challenge**: Close this notebook, open a blank one, and rewrite Part [hardest
part number] from scratch without looking back.
```

```python
debrief = """
1.
2.
3.
"""  # type your recall here before checking _solutions/
```

**Session log (code) — closes the feedback loop.** The generator PRE-FILLS `problem`, `budget_min`, and each part's `n`/`tier`; the student fills only `actual_min`, each `outcome`, `difficulty`, and `stuck` (~5 s). The next `/newproblem` reads this file in Step 2 to scaffold decayed skills one tier easier.

```python
# --- Session log: fill the `___` then run (appends one line to .practice-log.jsonl) ---
import json, datetime, pathlib
_root = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
            if (p / ".git").exists() or (p / ".spaced-repetition.json").exists())
record = {
    "problem": "<dir path, pre-filled, e.g. papers/sae-concept-manifolds>",
    "date": datetime.date.today().isoformat(),
    "budget_min": 30,                                   # pre-filled
    "actual_min": ___,                                  # how long it really took
    "parts": [                                          # n + tier pre-filled; set outcome
        {"n": 1, "tier": 3, "outcome": "___"},          # unaided | hint | solution | failed
        {"n": 2, "tier": 2, "outcome": "___"},
    ],
    "difficulty": ___,                                  # 1 (trivial) .. 5 (over my head)
    "stuck": "___",                                     # one phrase: where you got stuck
}
with open(_root / ".practice-log.jsonl", "a") as f:
    f.write(json.dumps(record) + "\n")
print("logged ->", _root / ".practice-log.jsonl")
```

This final retrieval practice cements the session. It's not optional decoration -- it's where a large fraction of long-term retention is created. The log is append-only JSONL (crash-safe); a missing file just means the next problem falls back to the functions-list tier rule.

## Extension Cell (30m+ problems, after Session Debrief)

```markdown
## Extension (Optional)
Try one of these variations:
- [Variation 1: e.g., "Rewrite using einsum instead of matmul"]
- [Variation 2: e.g., "What happens if you remove the scaling factor? Run it and explain."]
- [Variation 3: e.g., "Implement a batched version"]
```

Provide 2-3 concrete variations that deepen understanding. Good extensions:
- **Change representation**: rewrite with different tools (einsum, einops, manual loops)
- **Break it on purpose**: remove a component and explain the failure mode
- **Scale it**: add batching, change dimensions, handle edge cases
- **Connect it**: use the output as input to another component

Skip for 10m drills. These are for students who finish early or want to push further.

## Anki Cards Cell (all budgets, final cell)

```markdown
## Anki Cards

Add these to your deck:

**Card 1**
Front: [question that mirrors a real recall context — when would you need this fact?]
Back: [1-5 words. Atomic. One fact per card.]

**Card 2**
Front: [different angle on the core concept]
Back: [1-5 words]

**Card 3**
Front: [operational/debugging angle — "what's wrong if you see X?"]
Back: [1-5 words]
```

Card design rules (per [LessWrong Anki guide](https://www.lesswrong.com/posts/7Q7DPSk4iGFJd8DRk)):
- **Atomic**: 1-5 words on the back. If you need more, split into two cards.
- **Real-context prompts**: Front should mirror when you'd actually need to recall this — in an interview, debugging a training run, reading a paper. Not "what is X?" but "you see Y happening, what's the cause?"
- **No info in the prompt**: Don't put to-be-learned material on the front.
- **Multiple angles**: The 3 cards should cover different retrieval paths to the same core knowledge — formula, intuition, and operational/diagnostic.
- **No cloze deletion**: Full question → short answer, not fill-in-the-blank.
