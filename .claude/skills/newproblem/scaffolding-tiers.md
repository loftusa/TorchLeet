# Scaffolding Tier Templates

## Tier 1 — First Encounter

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

## Tier 2 — Has Used Key Functions

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

## Tier 3 — Recall Drill

The student has done this specific operation before. Bare minimum.

```python
def function_name(arg1, arg2):
    # Implement from memory
    pass
```

## Insight Tier — Idea-Novel Sub-Skills (orthogonal to Tiers 1–3)

Tiers 1–3 set *implementation* hint density off the student's **function** history. The Insight tier is **orthogonal**: it governs ONLY the conceptual nudge for a sub-skill whose *idea* is novel (Step 1.4 `idea-novel` tag), and **never touches the code stub** — an expert with every primitive known still gets the sparse stub his function-history earns and writes every line himself.

A sub-skill can be Tier-3 (function-known) *and* Insight-tier (idea-new) at once. The SAE max-gap arc trick is the canonical example.

**At ≤45m budgets** — emit, in the Part-N header (Cell A), exactly:
- ONE sentence naming the non-obvious insight (e.g. "the smallest enclosing arc is 2π minus the largest angular gap between consecutive sorted angles").
- A never-checked self-explanation prompt: "In one line, explain *why* that identity holds." (self-explanation effect; no answer key needed.)

The implementation stub itself stays at the function-history tier.

**At ≥1h budgets only** — you may optionally expand an idea-novel sub-skill into a full **worked → faded → full** progression (study a complete worked example, then a faded version with the key step blanked, then from-scratch). Only here is there part-budget to spend three parts on one idea.

**Hard gates (do not violate):**
- A worked-example cell is emitted ONLY at ≥1h AND ONLY for an idea-novel sub-skill.
- The faded + from-scratch follow-ups are MANDATORY — never let it degenerate into example-only reading.
- NEVER emit a worked example for a sub-skill that is merely **function-novel** — that stays Tier 1. For an expert on familiar material a worked example is **expertise-reversal** (a net negative): it replaces the generative reps the skill exists to create.
