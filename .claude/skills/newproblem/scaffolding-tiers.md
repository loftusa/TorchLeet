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
