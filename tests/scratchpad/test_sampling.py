"""Test Top-K and Top-P sampling implementation."""

import torch
import torch.nn.functional as F

torch.manual_seed(42)


def top_k_filtering(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Filter logits to keep only top-k values, setting others to -inf."""
    if k <= 0:
        raise ValueError("k must be positive")

    # Handle 1D input
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    vocab_size = logits.size(-1)

    # If k >= vocab_size, return original logits
    if k >= vocab_size:
        return logits.squeeze(0) if squeeze_output else logits

    # Get the k-th largest value as threshold
    top_k_values, _ = torch.topk(logits, k, dim=-1)
    threshold = top_k_values[..., -1:]

    # Create mask for values below threshold
    mask = logits < threshold

    # Set filtered positions to -inf
    filtered_logits = logits.masked_fill(mask, float('-inf'))

    if squeeze_output:
        filtered_logits = filtered_logits.squeeze(0)

    return filtered_logits


def top_p_filtering(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Filter logits using nucleus (top-p) sampling."""
    if p <= 0 or p > 1:
        raise ValueError("p must be in (0, 1]")

    # Handle 1D input
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    # If p == 1.0, return original
    if p == 1.0:
        return logits.squeeze(0) if squeeze_output else logits

    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)

    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    # Compute cumulative probabilities
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find tokens to remove (shift by 1 to keep token that pushes over p)
    sorted_mask = cumsum_probs - sorted_probs > p

    # Always keep at least the top token
    sorted_mask[..., 0] = False

    # Scatter the mask back to original positions
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(-1, sorted_indices, sorted_mask)

    # Apply mask
    filtered_logits = logits.masked_fill(mask, float('-inf'))

    if squeeze_output:
        filtered_logits = filtered_logits.squeeze(0)

    return filtered_logits


def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Sample from logits with temperature scaling and optional top-k/top-p filtering."""
    # Handle 1D input
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    # Step 1: Apply temperature
    if temperature != 1.0:
        temperature = max(temperature, 1e-8)
        logits = logits / temperature

    # Step 2: Apply top-k filtering
    if top_k > 0:
        logits = top_k_filtering(logits, k=top_k)

    # Step 3: Apply top-p filtering
    if top_p < 1.0:
        logits = top_p_filtering(logits, p=top_p)

    # Step 4: Convert to probabilities and sample
    probs = F.softmax(logits, dim=-1)

    if (probs.sum(dim=-1) == 0).any():
        token_idx = logits.argmax(dim=-1, keepdim=True)
    else:
        token_idx = torch.multinomial(probs, num_samples=1)

    if squeeze_output:
        token_idx = token_idx.squeeze(0)

    return token_idx


# ========== TESTS ==========
print("=== Testing Top-K Filtering ===")

# Simple test case
logits = torch.tensor([1.0, 4.0, 2.0, 5.0, 3.0])
print(f"Original logits: {logits}")

filtered_k2 = top_k_filtering(logits, k=2)
print(f"Top-2 filtered:  {filtered_k2}")

filtered_k3 = top_k_filtering(logits, k=3)
print(f"Top-3 filtered:  {filtered_k3}")

# Verify only k values remain finite
assert (filtered_k2 > float('-inf')).sum() == 2, "Should have exactly 2 finite values"
assert (filtered_k3 > float('-inf')).sum() == 3, "Should have exactly 3 finite values"

# Verify the top-k values are preserved
probs_k2 = F.softmax(filtered_k2, dim=-1)
print(f"Probabilities after Top-2: {probs_k2}")
assert probs_k2[1] > 0 and probs_k2[3] > 0, "Tokens 1 and 3 should have probability"
assert probs_k2[0] == 0 and probs_k2[2] == 0 and probs_k2[4] == 0, "Other tokens should have 0 probability"

print("\n✓ Top-K filtering tests passed!")

print("\n=== Testing Top-P (Nucleus) Filtering ===")

# Create logits where we know the probabilities
logits = torch.tensor([0.0, 1.0, 2.0, 3.0])
probs = F.softmax(logits, dim=-1)
print(f"Original logits: {logits}")
print(f"Original probs:  {probs}")
print(f"Cumulative:      {torch.cumsum(probs.sort(descending=True)[0], dim=-1)}")

filtered_p09 = top_p_filtering(logits, p=0.9)
probs_p09 = F.softmax(filtered_p09, dim=-1)
print(f"\nTop-p=0.9 filtered: {filtered_p09}")
print(f"Probs after p=0.9:  {probs_p09}")

filtered_p07 = top_p_filtering(logits, p=0.7)
probs_p07 = F.softmax(filtered_p07, dim=-1)
print(f"\nTop-p=0.7 filtered: {filtered_p07}")
print(f"Probs after p=0.7:  {probs_p07}")

kept_p09 = (probs_p09 > 0).sum().item()
kept_p07 = (probs_p07 > 0).sum().item()
print(f"\nTokens kept with p=0.9: {kept_p09}")
print(f"Tokens kept with p=0.7: {kept_p07}")
assert kept_p07 <= kept_p09, "Lower p should keep fewer or equal tokens"

# Test that probabilities still sum to 1
assert torch.allclose(probs_p09.sum(), torch.tensor(1.0)), "Probs should sum to 1"
assert torch.allclose(probs_p07.sum(), torch.tensor(1.0)), "Probs should sum to 1"

print("\n✓ Top-P filtering tests passed!")

print("\n=== Testing Combined Sampling ===")

torch.manual_seed(42)

vocab_size = 100
logits = torch.randn(vocab_size)

# Test 1: Pure sampling
samples_pure = [sample(logits, temperature=1.0).item() for _ in range(100)]
unique_pure = len(set(samples_pure))
print(f"Pure sampling (T=1.0): {unique_pure} unique tokens from 100 samples")

# Test 2: Low temperature
samples_low_t = [sample(logits, temperature=0.1).item() for _ in range(100)]
unique_low_t = len(set(samples_low_t))
print(f"Low temp (T=0.1): {unique_low_t} unique tokens from 100 samples")

# Test 3: Top-K sampling
samples_topk = [sample(logits, top_k=5).item() for _ in range(100)]
unique_topk = len(set(samples_topk))
print(f"Top-K (k=5): {unique_topk} unique tokens from 100 samples")
assert unique_topk <= 5, "Top-K should limit to at most K unique tokens"

# Test 4: Top-P sampling
samples_topp = [sample(logits, top_p=0.5).item() for _ in range(100)]
unique_topp = len(set(samples_topp))
print(f"Top-P (p=0.5): {unique_topp} unique tokens from 100 samples")

# Test 5: Combined
samples_combined = [sample(logits, temperature=0.7, top_k=40, top_p=0.9).item() for _ in range(100)]
unique_combined = len(set(samples_combined))
print(f"Combined (T=0.7, k=40, p=0.9): {unique_combined} unique tokens from 100 samples")

# Test 6: Batch processing
batch_logits = torch.randn(4, vocab_size)
batch_samples = sample(batch_logits, temperature=0.8, top_k=10)
assert batch_samples.shape == (4, 1), f"Expected shape (4, 1), got {batch_samples.shape}"
print(f"Batch sampling works: shape {batch_samples.shape}")

print("\n✓ Combined sampling tests passed!")

print("\n=== Adaptive Behavior Demo ===")

vocab_size = 100

# Scenario 1: High confidence
logits_confident = torch.zeros(vocab_size)
logits_confident[0] = 10.0
probs_confident = F.softmax(logits_confident, dim=-1)
print(f"HIGH CONFIDENCE - Top token prob: {probs_confident[0]:.4f}")

filtered_k10 = top_k_filtering(logits_confident, k=10)
n_kept_k = (F.softmax(filtered_k10, dim=-1) > 1e-6).sum().item()
print(f"  Top-K (k=10) keeps: {n_kept_k} tokens")

filtered_p09 = top_p_filtering(logits_confident, p=0.9)
n_kept_p = (F.softmax(filtered_p09, dim=-1) > 1e-6).sum().item()
print(f"  Top-P (p=0.9) keeps: {n_kept_p} tokens")

# Scenario 2: Low confidence
logits_uncertain = torch.randn(vocab_size) * 0.1
probs_uncertain = F.softmax(logits_uncertain, dim=-1)
print(f"\nLOW CONFIDENCE - Max token prob: {probs_uncertain.max():.4f}")

filtered_k10_unc = top_k_filtering(logits_uncertain, k=10)
n_kept_k_unc = (F.softmax(filtered_k10_unc, dim=-1) > 1e-6).sum().item()
print(f"  Top-K (k=10) keeps: {n_kept_k_unc} tokens")

filtered_p09_unc = top_p_filtering(logits_uncertain, p=0.9)
n_kept_p_unc = (F.softmax(filtered_p09_unc, dim=-1) > 1e-6).sum().item()
print(f"  Top-P (p=0.9) keeps: {n_kept_p_unc} tokens")

print("\n" + "="*50)
print("✓ All Top-K/Top-P Sampling tests passed!")
print("="*50)
