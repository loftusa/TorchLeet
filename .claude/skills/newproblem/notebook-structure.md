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

Hint rules:
- **10m drills**: Zero hints. None.
- **30m**: 0-1 hints per part (only for Tier 1 sub-skills)
- **1h**: 1-2 hints per part (only for Tier 1 sub-skills)
- **2h**: 1-3 hints per part (only for Tier 1 sub-skills)
- **NEVER hint functions the student has used before** (check CLAUDE.md)
- Hints go IN the part, not in the opening section -- help at point of need

### Cell B — Implementation Stub (code)

Use the appropriate tier (1/2/3) from scaffolding-tiers.md. Use `jaxtyping` for tensor shape annotations. Constrain variable names by referencing them in the test cell below.

### Cell C — Validation (code)

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
- Print intermediate statistics (range, mean, std) so the student builds intuition for what correct tensors look like
- Assertion messages explain the invariant's rationale, not just state it
- Diagnostics run even on success -- the student should learn what "correct" looks like
- On failure, show a side-by-side comparison (first 8 elements, max/mean diff, location) so the student can self-correct without asking for help
- No emoji in test output

## Recall Embedding (30m+ problems)

For problems 30 minutes or longer, embed recall parts that require the student to re-implement something from a previous problem, from memory. Instead of importing a helper:

```markdown
## Part N: [Previously Learned Concept] (Recall)
You implemented this in [previous problem]. Reproduce it here from memory -- do not look at your old solution.
```

Use Tier 3 scaffolding for these. This creates spaced repetition within the session and interleaves old + new skills.

**Recall part budget by time:**
- **10m**: No recall parts (pure single-topic drill)
- **30m**: 0-1 recall parts (only if time allows after core parts)
- **1h**: 1 recall part
- **2h**: 1-2 recall parts

## Final Cell: Session Debrief (all budgets except 10m)

```markdown
## Session Debrief

Without scrolling up, answer in your head:
1. What is the core formula for [topic]?
2. What is the key PyTorch function for [central operation]?
3. What shape does [key tensor] have and why?

**Challenge**: Close this notebook, open a blank one, and rewrite Part [hardest part number] from scratch without looking back.
```

This final retrieval practice cements the session. It's not optional decoration -- it's where a large fraction of long-term retention is created.

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
