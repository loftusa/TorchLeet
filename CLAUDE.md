# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TorchLeet is a PyTorch learning resource with multiple problem sets:
1. **torch/** - PyTorch practice problems (Basic/Easy/Medium/Hard) covering fundamental deep learning concepts
2. **llm/** - Large Language Model implementation problems focusing on building transformer components from scratch
3. **numerai/** - Financial ML problems applying NLP/LLM techniques to quantitative finance (Numerai tournament context)
4. **papers/** - Paper reimplementations (research paper exercises)
5. **practice/** - Applied ML practice problems (LLM-as-Judge, Agent Eval, Codenames AI)

This is an educational repository where users solve incomplete problems by filling in missing code blocks marked with `...` or `#TODO` comments, then compare against solution files.

## Repository Structure

```
TorchLeet/
├── torch/
│   ├── basic/      # Beginner problems (linear regression, custom datasets, activation functions)
│   ├── easy/       # Intermediate problems (CNNs, RNNs, data augmentation, quantization)
│   ├── medium/     # Advanced problems (CNNs from scratch, LSTMs, 3D CNNs)
│   └── hard/       # Expert problems (custom autograd, GANs, Transformers, seq2seq)
├── llm/
│   ├── 01-RMS-Norm/
│   ├── 02-Sinusoidal-Positional-Embedding/
│   ├── 03-Implement-Attention-from-Scratch/
│   ├── 04-Multi-Head-Attention/
│   ├── 05-Byte-Pair-Encoder/
│   ├── 06-Rotary-Positional-Embedding/
│   ├── 07-Grouped-Query-Attention/
│   ├── 08-KV-Cache/
│   ├── 09-KL-Divergence-Loss/
│   ├── 10-Create-Embeddings-out-of-an-LLM/
│   ├── 11-Temperature-Sampling/
│   ├── 12-Top-K-Top-P-Sampling/
│   ├── 13-Beam-Search/
│   ├── 14-LoRA/
│   ├── 15-SmolLM/
│   ├── 17-Quantization/
│   ├── 18-VLM-Attention/
│   └── flash-attention.ipynb
├── numerai/
│   ├── 01-FinBERT-Sentiment-Pipeline/
│   ├── 02-Text-Embeddings-as-Features/
│   ├── 03-SEC-Filing-Features/
│   ├── 04-Company-Knowledge-Graphs/
│   ├── 05-Contrastive-Financial-Embeddings/
│   ├── 06-LLM-Probing-Financial-Signals/
│   ├── 07-Finetune-Financial-LLM/
│   └── 08-Common-Crawl-Pipeline/
├── papers/
│   ├── concept-influence/    # Concept Influence: training data attribution via interpretability
│   ├── influence-functions/  # Influence Functions: classical training data attribution (Koh & Liang 2017)
│   ├── trak/                 # TRAK: scalable training data attribution via random projections
│   ├── trackstar/            # TrackStar: scalable influence/fact tracing via EKFAC + preconditioner mixing (Chang et al. 2024)
│   ├── ekfac/                # EKFAC: Eigenvalue-corrected Kronecker-Factored Approximate Curvature (George et al. 2018)
│   ├── glp-meta-model/       # GLP meta-model paper implementation
│   ├── mfa-local-geometry/   # MFA local geometry paper implementation
│   └── jacobian-lens/        # Jacobian lens / global workspace: J_l = E[dh_final/dh_l], steering, coordinate patching, J-space pursuit (Transformer Circuits 2026)
├── practice/
│   ├── 01-LLM-as-Judge/      # LLM evaluation as judge
│   ├── 02-Agent-Eval-Harness/ # Agent evaluation harness
│   ├── 03-Codenames-AI/      # AI for Codenames board game
│   └── 04-Safety-Evaluation-Pipeline/  # Red-team campaign analysis (ISC paper)
├── rlhf/
│   └── 01-Reward-Model/       # Bradley-Terry reward model training
└── valsai_interview/          # Interview prep scripts
```

Each problem directory typically contains:
- `<problem-name>.ipynb` or `<problem-name>-Question.ipynb` - Problem statement with incomplete code
- `<problem-name>_SOLN.ipynb` or `<problem-name>.ipynb` (without -Question suffix) - Complete solution

## Working with Problems

### Problem Format Convention
- **Question files**: Named with `-Question` suffix or contain incomplete code blocks with `...` or `#TODO`
- **Solution files**: Named with `_SOLN` suffix or without `-Question` suffix
- All problems are Jupyter notebooks (`.ipynb` files)

### Key Patterns in Problem Files

1. **Incomplete Code Blocks**: Look for `...` or `#TODO` comments indicating where implementation is needed
2. **Test Assertions**: Most problems include test code that validates against PyTorch's built-in implementations
3. **Reference Comparisons**: Solutions are often validated with `torch.allclose()` against official PyTorch implementations

### Common Problem Types

**LLM Set Problems:**
- Loss functions (KL Divergence for distillation, discrete and continuous variants)
- Attention mechanisms (scaled dot-product, multi-head, grouped-query, VLM attention)
- Positional encodings (sinusoidal, RoPE)
- Normalization layers (RMSNorm)
- Tokenization (Byte-Pair Encoding)
- KV Cache implementation and optimization
- Decoding strategies (temperature sampling, top-k/top-p sampling, beam search)
- Parameter-efficient fine-tuning (LoRA)
- Quantization (INT8 per-channel)
- Full model implementations (SmolLM-135M architecture)
- Kernel optimizations (Flash Attention with Triton)
- Knowledge distillation (teacher-student training with temperature scaling)

**Torch Set Problems:**
- Custom layers and loss functions
- Model architectures from scratch
- Training loops and optimization
- Data loading and augmentation
- Model deployment (quantization, mixed precision)

**Numerai Set Problems:**
- Financial NLP pipelines (FinBERT sentiment, text embeddings as features)
- SEC filing feature extraction
- Company knowledge graphs
- Contrastive financial embeddings
- LLM probing for financial signals
- Financial LLM fine-tuning
- Common Crawl data pipelines

**Papers Set:**
- GLP meta-model reimplementation
- MFA local geometry reimplementation
- Doc-to-LoRA: Hypernetwork-based LoRA generation from documents (Perceiver aggregator, top-K context distillation loss, chunk-and-merge for long contexts)
- Concept Influence: Training data attribution via interpretability (linear probes as concept vectors, Vector Filter dot-product attribution, precision-recall evaluation, dataset filtering and retraining)
- TRAK: Scalable training data attribution via random projections (per-sample gradients with torch.func.vmap, Gaussian/Rademacher random projections, Johnson-Lindenstrauss lemma verification, projected gradient features, TRAK attribution scores as dot products)
- Influence Functions: Classical training data attribution (Koh & Liang 2017) — per-sample gradients, HVP via double backward trick, IHVP via Conjugate Gradient and LiSSA (Neumann series), influence scores validated against leave-one-out retraining (r=0.999 correlation)
- TrackStar: Scalable influence and fact tracing for LLM pretraining (Chang et al. 2024) — Adam/Adafactor gradient normalizers, EKFAC preconditioner computation via input/output covariance eigendecomposition, damped matrix power for numerical stability, two-sided (split) preconditioning equivalence proof, preconditioner mixing with automatic lambda via eigenvalue curve intersection, FAISS index building and retrieval, full pipeline combining all components
- EKFAC: Eigenvalue-corrected Kronecker-Factored Approximate Curvature (George et al. NeurIPS 2018) — activation covariance collection via forward hooks, gradient covariance via backward hooks, eigendecomposition of Kronecker factors, eigenvalue corrections in eigenbasis, EKFAC and KFAC inverse-Hessian vector products, comparison against full Fisher inversion on a small MLP. Based on bergson's implementation.
- SAE Concept Manifolds: Do Sparse Autoencoders capture concept manifolds? (Bhalla et al. 2026) — TopK Sparse Autoencoder (encoder + ReLU + top-k masking via scatter_, decoder with unit-norm column init), training on a 2-d circle embedded in 64-d ambient space, and a per-atom angular-selectivity diagnostic (smallest enclosing arc via sorted-gap trick with wraparound). Empirically reproduces the paper's "dilution" finding — atoms split between local-tile and global-basis regimes rather than cleanly committing to one.
- Role Confusion Probes: Prompt Injection as Role Confusion (Ye et al. ICML 2026) — linear "role probes" on a base model's (Qwen3-0.6B) hidden states. Controlled-dataset construction (wrap identical neutral content in `<user>` vs `<think>`/CoT tags so the probe learns tag geometry, not semantics), per-token feature extraction via `output_hidden_states` + offset-mapping content alignment, LogisticRegression probe, and the CoTness(t) := P(CoT|h_t) metric. Reproduces the headline result on a 0.6B model: content-disjoint role accuracy ~0.95 at layer 16, and CoT-styled text wrapped in `<user>` tags scores ~0.59 CoTness vs ~0.14 for genuine user text (gap +0.44) — style hijacks tags, the latent-space mechanism behind prompt injection.
- Probe Misalignment: Probing the Misaligned Thinking Process of Language Models (Zhou et al. 2026) — per-indicator linear logistic-regression probes on residual-stream activations (one direction per misalignment indicator), sentence-level mean pooling, OR-fusion of probes into a single turn-level monitor (flag if any probe fires on any sentence), per-probe threshold tuning at a fixed neutral false-positive rate (quantile operating point), and transcript-TPR / neutral-FPR / AUROC evaluation. Uses a synthetic oracle (planted unit directions in a 64-d residual stream) so it runs CPU-only. Demonstrates that OR-fusing N probes inflates turn-level FPR above any single probe's target — the motivation for the paper's probe→LLM-judge cascade.

**Practice Set:**
- LLM-as-Judge evaluation
- Agent evaluation harness
- Codenames AI (board game AI)
- Safety Evaluation Pipeline: Red-team campaign analysis based on ISC paper

**RLHF Set:**
- Reward Model: Bradley-Terry pairwise ranking loss, last-token pooling, preference pair training, reward distribution analysis (Wu et al., 2026) — harm taxonomy schemas, attack success rates, per-category/vector effectiveness matrices, policy compliance gap analysis, Cohen's kappa inter-rater reliability, bootstrap confidence intervals, model comparison radar plots, campaign dashboard heatmaps, temporal safety trends, and data-driven campaign prioritization scoring

## LLM Problem Ordering Philosophy

The LLM problem set is ordered to optimize for **interview preparation** and **learning progression**:

### Ordering Principles

1. **Interview Relevance First**: Most commonly asked interview questions appear early (Attention #3, Multi-Head Attention #4)
2. **Difficulty Progression**: Easier concepts before harder ones (RMS Norm → Attention → Full LLM)
3. **Logical Dependencies**: Build foundational concepts progressively (Positional Embeddings before Attention)
4. **Natural Groupings**: Related concepts together (all attention variants, all sampling methods)

### Current Ordering Rationale (Problems 1-18 Implemented, excluding #16)

**Tier 1: Fundamentals (Problems 1-2)**
- Start with simplest concepts to build confidence
- RMS Norm (#1): Easiest entry point, ~15-20 LOC
- Sinusoidal PE (#2): Mathematical foundation for position encoding

**Tier 2: Core Interview Questions (Problems 3-5)**
- Most critical for technical interviews
- Attention from Scratch (#3): Asked in ~70% of ML interviews, MOST IMPORTANT
- Multi-Head Attention (#4): Natural progression, asked in ~60% of interviews
- Byte Pair Encoding (#5): Tokenization is asked in ~50% of NLP interviews

**Tier 3: Modern Techniques (Problems 6-8)**
- Contemporary methods used in production models
- RoPE (#6): Modern positional encoding used in LLaMA
- Grouped Query Attention (#7): Efficiency optimization for inference
- KV Cache (#8): Practical optimization technique

**Tier 4: Training & Applications (Problems 9-10)**
- Training concepts and practical usage
- KL Divergence Loss (#9): Knowledge distillation and training
- Create Embeddings (#10): Practical application of LLMs

**Tier 5: Decoding Strategies (Problems 11-14)**
- Inference-time techniques for text generation
- Temperature Sampling (#11): Control randomness in generation
- Top-K/Top-P Sampling (#12): Filter token distributions for quality
- Beam Search (#13): Search-based decoding with length normalization
- LoRA (#14): Parameter-efficient fine-tuning

**Tier 6: Integration & Advanced (Problems 15-18+)**
- Full implementations and cutting-edge techniques
- SmolLM (#15): Integrates all previous concepts
- Flash Attention (#16): Advanced kernel optimization (unnumbered, standalone notebook)
- Quantization (#17): INT8 per-channel quantization
- VLM Attention (#18): Vision-Language Model attention mechanisms
- QLoRA, MoE, SFT/RLHF/DPO, etc.: Planned advanced topics

### Key Interview Statistics

Based on analysis of ML/NLP interview patterns:
- **Attention mechanisms**: 70% of Transformer-related interviews ask single-head attention
- **Multi-Head Attention**: 60% ask as natural follow-up
- **Byte Pair Encoding**: 50% of NLP interviews include tokenization questions
- **Positional Encodings**: 45% ask about sinusoidal or RoPE
- **RMS Norm**: 35% ask about normalization layers
- **Knowledge Distillation**: 20% discuss at conceptual level
- **Full LLM Implementation**: Rarely asked to code fully, but tests comprehensive understanding

### Learning Time Estimates

For interview preparation, recommended time per problem:
- Problems 1-2 (Fundamentals): 1-3 days each
- Problems 3-4 (Core Attention): 3-5 days each (practice extensively)
- Problem 5 (BPE): 2-3 days
- Problems 6-8 (Modern Techniques): 2-3 days each
- Problems 9-10 (Training/Apps): 1-3 days each
- Problem 15 (SmolLM): 5-7 days (integration practice)

**Total Interview Prep Time**: ~4-6 weeks for problems 1-10, practicing core attention mechanisms daily

## Critical Issues to Fix (Priority Order)

### 1. Multi-Head Attention Implementation (`llm/04-Multi-Head-Attention/`) ✅ FIXED
**Priority: CRITICAL**
- Location: `multi-head-attention-q5.ipynb`
- Status: **FIXED** - Implementation now passes validation
- Solution: Modified function to accept optional `weights` parameter; test now extracts weights from PyTorch's MultiheadAttention and passes them to custom implementation
- Result: Perfect match with max difference of 0.00e+00
- Key Learning: When validating custom implementations against reference implementations, use identical weights to verify logic correctness

### 2. Grouped-Query Attention Implementation (`llm/07-Grouped-Query-Attention/`) ✅ FIXED
**Priority: CRITICAL**
- Location: `grouped-query-attention.ipynb`
- Status: **FIXED** - Implementation now passes validation
- Issues Fixed:
  1. Added `num_query_heads` as a parameter (was incorrectly calculated as `d_model // 64`)
  2. Added `weights` parameter for weight copying from reference implementation
  3. Fixed test to properly compare GQA degenerating to MHA
- Result: Perfect match with max difference of 0.00e+00
- Bonus: Added demonstration of actual GQA with fewer KV heads than query heads

### 3. Sinusoidal Positional Embedding (`llm/02-Sinusoidal-Positional-Embedding/`) ✅ FIXED
**Priority: HIGH**
- Location: `sinusoidal-q7.ipynb`
- Status: **FIXED** - Now returns correct shape
- Issue: Parameters were swapped in the test call
- Root Cause: Class signature is `__init__(max_seq_len, d_model)` but test was calling `(d_model, max_seq_len)`
- Fix: Corrected parameter order in test cell
- Result: Now correctly outputs `(1, 50, 64)` with values bounded between -1 and 1

### 4. Rotary Positional Embedding (RoPE) (`llm/06-Rotary-Positional-Embedding/`) ✅ FIXED
**Priority: HIGH**
- Location: `rope-q8.ipynb`
- Status: **FIXED** - Implementation and test now work correctly
- Issues Fixed:
  1. Test was calling `Rotary(d_model, max_seq_len)` but constructor takes `(dim, base=10000)`
  2. Test was calling `apply_rotary_pos_emb(positions)` with 1 arg instead of 4: `(q, k, cos, sin)`
  3. Fixed cos/sin caching shape from `(seq_len, 1, 1, dim)` to `(1, 1, seq_len, dim)` for proper broadcasting
- Result: RoPE now correctly rotates Q and K tensors while preserving their magnitude
- Test demonstrates realistic usage with multi-head attention shapes

### 5. Create Embeddings from LLM (`llm/10-Create-Embeddings-out-of-an-LLM/`) ✅ FIXED
**Priority: MEDIUM**
- Location: `embeddings-q2.ipynb`
- Status: **FIXED** - Now successfully extracts embeddings and computes similarities
- Issues Fixed:
  1. Replaced dataset loading (deprecated API) with synthetic sample reviews
  2. Added `tokenizer.pad_token = tokenizer.eos_token` to enable batch tokenization
  3. Added missing `import torch.nn.functional as F` for cosine_similarity
  4. Added informative print statements to show progress and shapes
- Result: Successfully extracts 576-dimensional embeddings from SmolLM2-135M and computes cosine similarities with keywords
- Note: Uses synthetic reviews instead of Amazon dataset due to dataset API deprecation

### 6. README Accuracy ✅ FIXED
**Priority: LOW**
- Status: **FIXED** - README now accurately reflects which problems exist
- Added Flash Attention (#13) to the LLM problem list
- Marked all unimplemented problems with *(Coming Soon)*
- Updated problem #4 description to match actual implementation (embeddings extraction, not RAG)

### 7. KL Divergence Loss Implementation (`llm/09-KL-Divergence-Loss/`) ✅ NEW
**Priority: NEW FEATURE**
- Location: `llm/09-KL-Divergence-Loss/`
- Status: **IMPLEMENTED** - Comprehensive implementation now available as LLM problem #1
- Files Created:
  - `kl-divergence-Question.ipynb` - Problem statement with incomplete implementations
  - `kl-divergence.ipynb` - Complete solution with working code
- Features Implemented:
  1. **Discrete KL Divergence**: For classification tasks with temperature scaling
  2. **Gaussian KL Divergence**: Closed-form solution for continuous distributions
  3. **Distillation Loss**: Combined soft/hard target loss for knowledge distillation
  4. **Teacher-Student Training**: Practical demonstration with 10x model compression
  5. **Temperature Visualization**: Shows effect of temperature on probability distributions
- Architecture:
  - Teacher: 4-layer MLP (784→256→256→256→10) with ~270K parameters
  - Student: 2-layer MLP (784→64→10) with ~51K parameters
  - Compression ratio: 10x with minimal accuracy loss
- Testing Strategy:
  - Validates discrete KL against `F.kl_div()` with `atol=1e-6`
  - Validates Gaussian KL against `torch.distributions.kl_divergence()`
  - Tests multiple temperatures (T=1.0, T=3.0) to verify temperature scaling
  - Compares distilled vs baseline student to demonstrate distillation benefit
- Key Design Decisions:
  - Temperature T=3.0 for distillation (standard in literature)
  - Alpha=0.7 weighting (70% soft targets, 30% hard labels)
  - Synthetic MNIST-like dataset (500 samples, 10 classes) for self-contained demo
  - T² scaling factor for soft loss to balance gradient magnitudes
- Dependencies Added:
  - `matplotlib` for temperature visualization plots
  - `jupyter` and `nbconvert` for notebook execution testing
- Educational Value:
  - Comprehensive mathematical background with formulas
  - Practical application (knowledge distillation) alongside theory
  - Both discrete (classification) and continuous (Gaussian) variants covered
  - Demonstrates real model compression with measurable improvement

## Root Cause Analysis

**Weight Initialization Problem (Issues #1 and #2):**
The validation tests compare custom implementations against PyTorch's reference implementations, but:
- Custom implementations create **fresh random weights** using `nn.Linear()` inside the function
- PyTorch's `nn.MultiheadAttention` has its own random weights
- They will **never numerically match** unless weights are explicitly copied
- The implementations may be logically correct but fail numerical comparison

**Three possible fixes:**
1. Remove assertions and demonstrate correctness through shape validation and attention pattern visualization
2. Initialize custom implementation with weights copied from the reference module
3. Test the logic separately (e.g., verify attention weights sum to 1, output shapes correct, gradient flow works)

**Shape Issues (Issues #3 and #4):**
- Positional embedding tensors are being constructed or returned with wrong dimensions
- Likely indexing or concatenation bugs in the implementation
- These are straightforward debugging tasks once you trace through the tensor shapes

## Architecture Notes

### SmolLM-135M Structure
The complete LLM implementation follows this architecture:
```
smolLM
├── model (smolModel)
│   ├── embed_tokens: Embedding(49152, 576)
│   ├── layers: 30x LlamaDecoder
│   │   ├── self_attn: RopeAttention (Grouped-Query Attention with RoPE)
│   │   │   ├── W_query/W_key/W_value/W_output projections
│   │   │   └── rotary_emb: RotaryEmbedder
│   │   ├── mlp: MLP (SwiGLU-style with gate/up/down projections)
│   │   ├── pre_attn_rmsnorm: RMSNorm
│   │   └── pre_mlp_rmsnorm: RMSNorm
│   └── norm: RMSNorm
└── lm_head: Linear(576, 49152) [weight-tied with embed_tokens]
```

### Key Implementation Patterns

**Attention Mechanisms:**
- Standard pattern: Q, K, V projections → reshape to multi-head → scaled dot-product → concat → output projection
- GQA: Uses fewer K/V heads than Q heads, requires `repeat_interleave()` to match dimensions
- RoPE: Applied after projections but before attention computation, rotates query/key using sin/cos

**Testing Strategy:**
- Validate custom implementations against `torch.nn.MultiheadAttention` or `F.scaled_dot_product_attention`
- Use synthetic random tensors with fixed seeds for reproducibility
- Check with `torch.allclose()` using appropriate tolerances (typically `atol=1e-1, rtol=1e-2` for float16)

**Common Gotchas:**
- Weight initialization: Custom implementations create fresh random weights, causing mismatches with reference implementations
- To match PyTorch exactly, you'd need to copy weights from the reference module
- Attention mask shapes: Ensure proper broadcasting for batch/head dimensions
- Device placement: Move tensors to CUDA when available for performance testing

## Unimplemented Problems (Planned)

The README lists several problems marked as *(Coming Soon)* that don't yet have implementations:

**Torch Set - Missing:**
- Basic #8: Softmax from scratch
- Medium #4-6: AlexNet, Dense Retrieval System, KNN from scratch
- Hard #2-4: Neural Style Transfer, GNN, GCN
- Hard #8-9: Distributed training (DDP), Sparse Tensors
- Hard #11-14: CLIP Linear Probe, Cross-Modal Visualization, Vision Transformer, VAE

**LLM Set - Missing:**
- #16: Flash Attention is a standalone notebook (not in numbered directory)
- #18: QLoRA (Quantized LoRA)
- #19: Predictive Prefill with Speculative Decoding
- #20: Mixture of Experts
- #21-23: SFT, RLHF, DPO
- #24: Continuous Batching
- #25: Dense Passage Retrieval
- #26: 5D Parallelism

When adding new problems, follow the existing pattern:
- Create a directory in `torch/<difficulty>/`, `llm/`, `numerai/`, `papers/`, or `practice/`
- Include both `-Question.ipynb` and solution (`_SOLN.ipynb` or solution `.ipynb`) files
- Update README.md to link to the new problem
- Test that the solution works before committing

## Dependencies

Managed via `pyproject.toml` with `uv`. Key dependencies:
- Core: `torch>=2.9.1`, `numpy>=2.4.0`, `jaxtyping>=0.3.9`
- LLM/NLP: `transformers>=4.57.3`, `datasets>=4.4.2`, `huggingface-hub>=1.3.4`, `sentence-transformers>=5.2.2`, `nltk>=3.9.2`
- Interpretability: `nnsight>=0.5.15`, `nnterp>=1.2.2`
- Visualization: `matplotlib>=3.10.8`, `seaborn>=0.13.2`
- ML: `scikit-learn>=1.8.0`
- Utilities: `jupyter>=1.1.1`, `nbconvert>=7.16.6`

Install with: `uv sync`

## Development Philosophy

Per the README:
- Avoid using GPT to solve problems - learn by implementing yourself
- Test solutions against provided solution files
- Focus on understanding core PyTorch concepts deeply
- Problems are designed for hands-on practice, not just reading solutions

## User-Implemented Functions/Methods (Running List)

Functions the user has already implemented in Question notebooks. When creating new problems, do NOT hint these — force recall from memory.

**Torch Tensor Creation & Manipulation:**
`torch.arange`, `torch.ones`, `torch.zeros`, `torch.randn`, `torch.randn_like`, `torch.ones_like`, `torch.zeros_like`, `torch.full_like`, `torch.cat`, `torch.stack`, `torch.full`, `torch.linspace`, `torch.tensor`, `torch.where`, `torch.randint`, `torch.rand`, `torch.Generator`, `torch.manual_seed()`, `torch.isfinite`, `torch.normal()`, `torch.set_default_dtype()` (set global default floating-point dtype for new tensors, e.g. `torch.set_default_dtype(torch.float32)`), `.view()`, `.transpose()`, `.permute()`, `.unsqueeze()`, `.chunk()`, `.split()`, `.clone()`, `.flatten()`, `.flatten(start_dim=N)` (partial flatten from a specified dimension, e.g. `.flatten(2)` to collapse spatial dims in patch embedding: `proj(x).flatten(2).transpose(1, 2)`), `.reshape()`, `.to()` (dtype), `.squeeze()`, `.contiguous()`, `.detach()`, `.argmax()`, `.size()`, `.dim()` (number of dimensions), `.sum()`, `.scatter_()`, `.bool()`, `.float()`, `.int()`, `.item()`, `.expand()`, `.unbind()`, `.copy_()`, `.add_()`, `.sub_()`, `.data`, `.numel()`, `.element_size()`, `.numpy()`, `torch.flatten()`, `.eq()`, `.all()`, `.T` (2D transpose attribute), `.tolist()` (convert tensor to nested Python list), `.repeat()` (repeat tensor along dimensions, e.g. `x.repeat(1, n_kv_heads, 1, 1)` — distinct from `repeat_interleave` which repeats individual elements), `.flip()` (reverse a tensor along one or more dimensions, e.g. `x.flip(dims=[-1])` to reverse the last dim — used in Jacobian steering and coordinate patching), `.grad` (gradient tensor attribute on a leaf tensor, e.g. `param.grad` or `assert tensor.grad is not None` to verify gradient flow)

**Torch dtype Constants:**
`torch.float32`, `torch.float16`, `torch.bfloat16`, `torch.long`, `torch.int32`, `torch.int8`, `torch.uint8`, `torch.bool`

**Device Operations:**
`torch.cuda.is_available()`, `torch.device()`, `.cuda()`, `.cpu()`, `tensor.device` (property for accessing a tensor's device, e.g., `torch.arange(n, device=x.device)`)

**Torch Math Operations:**
`torch.sqrt`, `torch.rsqrt`, `torch.exp`, `torch.log`, `torch.sin`, `torch.cos`, `torch.round`, `torch.clamp`, `torch.einsum`, `torch.topk`, `torch.sort`, `torch.cumsum`, `torch.argsort`, `torch.masked_fill`, `.masked_fill_(mask, value)` (in-place), `torch.triu`, `torch.tril`, `torch.repeat_interleave`, `torch.multinomial`, `torch.gather`, `.gather(dim=..., index=...)` (tensor method form of `torch.gather`, e.g. `logits.gather(dim=-1, index=idx)` to select values at specified indices — same semantics as the function form, used in top-K context distillation loss), `torch.logsumexp`, `torch.cdist`, `torch.randperm`, `torch.diag`, `torch.mv`, `torch.trapezoid`, `torch.allclose`, `torch.isclose` (element-wise bool comparison), `torch.inf` (infinity constant), `torch.chunk` (function form), `.pow()`, `.mean()`, `.abs()`, `.max()`, `.min()`, `.amax()`, `.exp()`, `.log()`, `@` (matmul), `torch.matmul()`, `torch.tanh`, `torch.sigmoid`, `torch.softmax`, `torch.relu`, `torch.bmm`, `torch.outer`, `torch.max` (function form with dim arg), `torch.eye`, `torch.argmin`, `.norm()`, `.median()`, `.trace()`, `.unique()`, `.argsort()` (tensor method), `torch.all`, `torch.logical_or()` (or `|` tensor operator), `torch.logical_not()` (or `~` bool tensor operator), `torch.isnan`, `torch.pi` (π constant), `torch.diff()` (consecutive differences along a dimension), `torch.count_nonzero()` (count non-zero elements), `.cos()` (tensor method, e.g. `theta.cos()`), `.sin()` (tensor method, e.g. `theta.sin()`), `.sort().values` and `.sort().indices` (namedtuple attribute access from `torch.sort()`/`.sort()`), `.sort(descending=True)` (`descending` kwarg for descending sort, e.g. in beam search and top-k/top-p sampling), `.sqrt()` (tensor method form, e.g. `eigvals.sqrt()`), `torch.topk().values` and `torch.topk().indices` (namedtuple attribute access from `torch.topk()`, distinct from `.sort()`), `.max(dim=...).values` and `.min(dim=...).values` (namedtuple attribute access on `.max()/.min()` with a `dim` arg — analogous to `.topk().values`; e.g. `logits.max(dim=-1, keepdim=True).values` for numerically stable softmax), `.mean(dim=..., keepdim=True)` / `.sum(dim=..., keepdim=True)` (`keepdim=True` kwarg for dimension-reducing ops to preserve tensor rank), `torch.any()` (function form) and `.any()` (tensor method — checks if any element is True, e.g. `mask.any()` to test if an atom ever fires), `.nonzero()` / `torch.nonzero(tensor, as_tuple=True)` (returns indices of non-zero elements; often used as alternative to boolean fancy indexing), `torch.quantile()` (compute quantiles along a dimension, e.g. `torch.quantile(scores, 1 - target_fpr, dim=0)` for threshold tuning), `.std()` (tensor method for standard deviation, e.g. `x.std(dim=-1, keepdim=True)`), `.equal()` (tensor method for exact element-wise equality check returning a bool scalar, distinct from `torch.allclose` which has tolerances), `torch.sum()` (function form with dim arg, e.g. `torch.sum(X, dim=1)` — distinct from `.sum()` tensor method), `torch.amax()` (function form for reduction, e.g. `weight.amax(dim=reduce_dims, keepdim=True)` in per-channel quantization — distinct from `.amax()` tensor method), `.topk(largest=False)` (smallest-k via `largest=False` kwarg, e.g. `neuron_vals.topk(top_k, largest=False).indices`)

**torch.nn Modules:**
`nn.Linear`, `nn.Parameter`, `nn.ParameterList`, `nn.Embedding`, `nn.Embedding.from_pretrained()` (class method to construct an Embedding from a pretrained weight matrix), `nn.Conv2d`, `nn.Sequential`, `nn.ReLU`, `nn.GELU`, `nn.SiLU`, `nn.LayerNorm`, `nn.MultiheadAttention`, `nn.ModuleList`, `nn.ModuleDict`, `nn.CrossEntropyLoss`, `nn.init.kaiming_uniform_`, `self.register_buffer`, `nn.RNN`, `nn.LSTM`, `nn.MaxPool2d`, `nn.ConvTranspose2d`, `nn.Conv3d`, `nn.ConvTranspose3d`, `nn.LeakyReLU`, `nn.Sigmoid`, `nn.Tanh`, `nn.Softmax`, `nn.Flatten`, `nn.BCELoss`, `nn.MSELoss`, `nn.AdaptiveAvgPool2d`, `nn.Dropout`, `nn.init.kaiming_normal_`, `nn.init.xavier_normal_`, `nn.init.xavier_uniform_`, `nn.init.zeros_`, `nn.init.normal_`, `nn.init.constant_`, `nn.init.ones_`, `model.children()`, `nn.MultiheadAttention` cross-attention: `attn(query, key, value, key_padding_mask=mask)` where query≠key=value (Perceiver-style cross-attention), `module_dict["key_name"]` (string key access on `nn.ModuleDict`), `nn.MultiheadAttention(batch_first=True)` (input/output tensors in (batch, seq, d) order instead of default (seq, batch, d)), `multihead_attn.in_proj_weight` (concatenated Q/K/V weight matrix on `nn.MultiheadAttention`; sliced as `[:d_model]` for Q, `[d_model:2*d_model]` for K, `[2*d_model:]` for V), `multihead_attn.out_proj.weight` (output projection weight tensor on `nn.MultiheadAttention`)

**torch.nn.functional:**
`F.softmax`, `F.log_softmax`, `F.relu`, `F.leaky_relu()` (functional leaky ReLU, e.g. `F.leaky_relu(x, negative_slope=0.2)`), `F.gelu`, `F.silu`, `F.sigmoid`, `F.linear`, `F.normalize`, `F.normalize(..., dim=0)` (column-wise unit-norm, e.g. for SAE decoder weight init — distinct from the typical row-norm `dim=-1`), `F.kl_div`, `F.kl_div(..., log_target=True)` (when both inputs are log-probabilities; `F.kl_div(log_q, log_p, reduction='batchmean', log_target=True)`), `F.cross_entropy`, `F.mse_loss`, `F.binary_cross_entropy_with_logits`, `F.dropout`, `F.scaled_dot_product_attention`, `F.scaled_dot_product_attention(..., is_causal=True)` (`is_causal=True` kwarg to enable causal masking in a single call without passing an explicit mask), `F.layer_norm()` (functional form of layer norm — same math as `nn.LayerNorm` but stateless, useful inside custom modules), `F.unfold`, `F.pad`, `F.cosine_similarity`, `F.conv2d`, `F.embedding`, `F.interpolate`

**Autograd / Training:**
`torch.no_grad()`, `@torch.no_grad()` (as function decorator), `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`, `model.zero_grad()`, `torch.optim.Adam`, `torch.optim.AdamW`, `torch.optim.SGD`, `torch.save()`, `torch.load()`, `model.state_dict()`, `model.load_state_dict()`, `model.apply()`, `model.parameters()`, `model.named_parameters()`, `model.named_buffers()`, `model.named_children()`, `.requires_grad` (read attribute to check if a tensor participates in autograd, e.g. `if param.requires_grad:` — distinct from `.requires_grad_(True)` which is the in-place setter), `.requires_grad_(False)`, `.requires_grad_(True)` (in-place enable gradient tracking on a tensor, e.g. for activations that need gradients in a validation probe), `model.train()`, `model.eval()`, `torch.optim.lr_scheduler.CosineAnnealingLR`, `scheduler.step()`

**Custom Autograd (torch.autograd.Function):**
`torch.autograd.Function`, `ctx.save_for_backward()`, `ctx.saved_tensors`, `Function.apply()`

**Torch Distributions:**
`torch.distributions.Normal`, `torch.distributions.kl_divergence`

**Linear Algebra:**
`torch.linalg.eigvalsh`, `torch.linalg.eigh`, `torch.linalg.solve`, `torch.linalg.slogdet`, `torch.linalg.inv()`, `torch.linalg.qr()`, `torch.linalg.cholesky()` (Cholesky decomposition of a positive-definite matrix, e.g. for MFA covariance factors), `torch.linalg.pinv()` (Moore-Penrose pseudo-inverse, e.g. for least-squares projection in Jacobian-lens coordinate patching)

**Functional Transforms:**
`torch.func.grad`, `torch.func.vmap`, `torch.func.functional_call`

**Autograd (Low-level):**
`torch.autograd.grad`, `torch.autograd.grad(..., retain_graph=True)` (`retain_graph=True` kwarg to keep the computation graph alive so multiple backward passes can be taken through the same forward graph, e.g. computing per-layer Jacobians in a loop), `torch.autograd.functional.jacobian()` (compute the Jacobian matrix of a function with respect to its inputs, e.g. `jacobian(fn, inputs)` for Jacobian-lens steering and coordinate patching)

**Hooks:**
`module.register_forward_hook`, `module.register_full_backward_hook`, `module.register_backward_hook`

**Mixed Precision (torch.cuda.amp):**
`torch.cuda.amp.GradScaler()`, `torch.cuda.amp.autocast()`, `scaler.scale()`, `scaler.step()`, `scaler.update()`

**Quantization:**
`torch.quantization.quantize_dynamic`, `isinstance(module, nn.Linear)`, `setattr(model, name, module)`, `torch.qint8`, `torch.int8`, `torch.uint8`

**Torch JIT:**
`@torch.jit.script`

**torch.utils.data:**
`torch.utils.data.Dataset`, `torch.utils.data.DataLoader`, `torch.utils.data.TensorDataset`, `__len__()`, `__getitem__()`, `DataLoader(..., shuffle=True, drop_last=True)` (shuffle and drop_last params for training loops), `DataLoader(..., num_workers=N)` (parallel data loading with N worker processes for faster throughput)

**torch.utils.tensorboard:**
`SummaryWriter`, `writer.add_scalar()`, `writer.close()`

**torchvision:**
`transforms.Compose`, `transforms.ToTensor`, `transforms.Normalize`, `transforms.RandomHorizontalFlip`, `transforms.RandomCrop`, `transforms.Resize`, `transforms.ToPILImage`, `torchvision.datasets.CIFAR10`, `torchvision.datasets.MNIST`, `torchvision.datasets.FakeData`, `torchvision.models.resnet18`, `torchvision.models.resnet18(pretrained=True)` (load pretrained ImageNet weights; used in XAI/GradCAM and transfer learning notebooks), `torchvision.utils.make_grid`

**einops:**
`einops.rearrange`, `einops.repeat`, `einops.einsum`, `einops.reduce`

**HuggingFace:**
`AutoTokenizer.from_pretrained`, `AutoModelForSequenceClassification.from_pretrained`, `AutoModelForCausalLM.from_pretrained`, `AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)` (`dtype=` kwarg to control model weight precision at load time, e.g. `dtype=torch.float32` to force full precision for probe experiments), `pipeline()`, `pipeline(task, model=..., tokenizer=tokenizer)` (explicit `tokenizer=` kwarg form), `TrainingArguments`, `Trainer`, `trainer.train()`, `datasets.load_dataset`, `datasets.load_dataset(..., streaming=True)`, `SentenceTransformer`, `SentenceTransformer(model_name, prompts={"retrieval": "..."})` (prompts param), `SentenceTransformer.encode(texts)`, `SentenceTransformer.encode(texts, convert_to_tensor=True)`, `SentenceTransformer.encode(texts, normalize_embeddings=True)`, `SentenceTransformer.encode(texts, normalize_embeddings=True, batch_size=256)`, `datasets.Dataset.from_dict()`, `datasets.Dataset.from_list()` (construct a Dataset from a Python list of dicts, e.g. for fine-tuning), `dataset.map(batched=True)`, `dataset.map(batched=True, remove_columns=[...])`, `dataset.set_format("torch")`, `CLIPModel.from_pretrained`, `CLIPProcessor.from_pretrained`, `CLIPProcessor(images=pil_image, return_tensors="pt")` (callable form with image input), `tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=n)` (callable form), `tokenizer(text, return_tensors="pt", return_offsets_mapping=True)` (with `return_offsets_mapping=True` to get character-span offsets for aligning hidden states to source text), `tokenizer.pad_token`, `tokenizer.eos_token`, `tokenizer.decode()`, `model.config` (e.g. `model.config.num_hidden_layers`, `model.config.hidden_size`), `model.vision_model` (ViT submodule on CLIPModel), `model.lm_head` (LM head Linear layer attribute on causal LM), `model.lm_head.weight` (LM head weight matrix, e.g. for logit lens), `model.model.norm` (final RMSNorm layer of decoder LM), `model(..., output_hidden_states=True)`, `outputs.hidden_states`, `model(..., output_attentions=True)`, `outputs.attentions`

**Python Standard Library:**
`collections.Counter`, `collections.Counter.most_common()`, `collections.defaultdict`, `dataclasses.dataclass`, `dataclasses.field`, `functools.cache`, `copy.deepcopy`, `re`, `re.compile()`, `re.match()`, `re.sub`, `re.findall`, `re.search`, `re.split`, `enum.Enum`, `enum.auto`, `math.log`, `math.sqrt`, `math.ceil`, `math.inf`, `math.exp`, `math.pi`, `typing.Optional`, `typing.Tuple`, `typing.List`, `typing.Callable`, `typing.Dict`, `typing.Union`, `list[str]` / `dict[str, float]` / `tuple[str, float]` (PEP 585 lowercase generic type hints, as opposed to `typing.List` etc.), `X | None` / `X | Y` (PEP 604 union type syntax, as opposed to `typing.Optional`/`typing.Union`), `time.time()`, `json.load()`, `json.loads()`, `json.dumps()` (serialize Python object to JSON string, complement of `json.loads()`), `pathlib.Path`, `pathlib.Path.open()`, `pathlib.Path.cwd()` (classmethod returning current working directory as a Path), `pathlib.Path.parents` (`.parents` attribute — sequence of ancestor Paths, e.g. `Path.cwd().parents`), `path / "subdir"` (Path `/` operator for joining path segments, e.g. `root / ".practice-log.jsonl"`), `pathlib.Path.exists()` (instance method checking whether the path exists on disk; distinct from `os.path.exists()`), `urllib.parse.urlparse()`, `datetime.strptime()`, `datetime.fromisoformat()`, `datetime(year, month, day)` (datetime constructor), `datetime.strftime()`, `datetime.timedelta`, `datetime.date.today()` (returns today's date as a `datetime.date` object), `date.isoformat()` (converts a `datetime.date` to ISO-format string, e.g. `"2026-06-30"`), `random.seed()`, `random.shuffle()`, `random.randrange()`, `random.choice()`, `random.choices()`, `random.choices(population, weights=[...])` (weighted random sampling with `weights=` kwarg), `random.randint()`, `random.random()`, `random.gauss()`, `statistics.mode()`, `warnings.filterwarnings()`, `io.BytesIO`, `os.path.exists()`, `os.path.getsize()`

**Python Builtins:**
`isinstance()`, `hasattr()`, `setattr()`, `max(iterable, key=...)`, `max(iterable, key=dict_obj.get)` (using `dict.get` as a key callable, e.g. in BPE merge scoring), `min(iterable, key=...)`, `sorted(iterable, key=..., reverse=True)`, `property` (decorator), `classmethod` (decorator), `str.replace(old, new)`, `str.split()`, `str.join(iterable)` (e.g. `' '.join(tokens)`), `str.lower()`, `str.strip()`, `str.isalpha()`, `str.islower()`, `str.isupper()`, `str.isdigit()`, `str.startswith()`, `tuple(iterable)`, `set()`, `iter()` (Python builtin for getting an iterator from an iterable, e.g. `iter(dataloader)` to step through batches manually — distinct from the iterable itself), `next()`, `enumerate()`, `zip()`, `float('nan')` (explicit NaN sentinel value), `float("-inf")` (string-form negative infinity, used as attention mask fill value), `list.index(element)` (find index of first occurrence in a list, e.g. `col_labels.index("Direct Request")`), `list.pop(index)` (remove and return element at index, e.g. `word_list.pop(randrange(len(word_list)))` in board generation), `dict.get(key, default)` (safe lookup with fallback, e.g. `counts.get(r, 0)` in Cohen's kappa computation), `dict.pop(key)` (remove and return a key's value, e.g. `enc.pop("offset_mapping")` to strip before forwarding to model), `dict.items()` (key-value pair iterator, e.g. `{k: v.to(device) for k, v in enc.items()}`), `dict.values()` (value iterator, e.g. `list(cat_asr.values())`), `str.format(key=value)` (keyword-argument template interpolation, e.g. `TAG[role].format(c=content)` to fill named slots in a format string — distinct from f-strings), `str.index(substring)` (returns position of first occurrence of substring, raises `ValueError` on miss — distinct from `list.index` which operates on list elements), `any(iterable)` (Python builtin returning True if any element is truthy — distinct from `torch.any()` function and `.any()` tensor method), `sum(iterable)` (Python builtin summing an iterable — distinct from `.sum()` tensor method and `torch.sum()` function)

**NumPy:**
`np.ascontiguousarray`, `np.transpose`, `np.stack`, `np.argsort`, `np.argmax`, `np.argmin`, `np.argpartition`, `np.partition`, `np.zeros`, `np.zeros_like()` (create a zero array with the same shape and dtype as an existing array — distinct from `np.zeros()` which takes an explicit shape), `np.ones`, `np.array`, `np.mean`, `np.std`, `np.min`, `np.max`, `np.sum`, `np.abs`, `np.percentile`, `np.exp`, `np.log`, `np.log2`, `np.log10`, `np.cos`, `np.sin`, `np.average`, `np.linspace`, `np.concatenate`, `np.column_stack`, `np.clip`, `np.sort`, `np.sort(arr, axis=1)[:, ::-1]` (reverse-sort along an axis via NumPy slice `[::-1]`), `np.meshgrid`, `np.random.seed`, `np.random.rand`, `np.random.randn`, `np.random.choice`, `np.random.choice(arr, size=n, replace=True)` (bootstrap resampling with `replace=True`), `np.random.choice(arr, size=n, replace=False)` (sampling without replacement, e.g. drawing non-repeating negative samples in contrastive learning), `np.random.randint`, `np.random.normal`, `np.random.shuffle`, `np.random.RandomState(seed)`, `np.linalg.eigh`, `np.linalg.eig` (general eigendecomposition, non-symmetric matrices), `np.linalg.norm()`, `np.linalg.qr`, `np.trace`, `np.repeat`, `np.tile`, `np.arange`, `np.empty`, `np.random.permutation`, `np.append()`, `np.pi` (π constant), `np.isfinite()` (element-wise finiteness check, e.g. `np.isfinite(arr).all()` to validate activations contain no NaN/inf), `np.vstack()` (stack arrays vertically / row-wise, e.g. stacking per-example feature arrays into a 2-d matrix), `np.quantile()` (compute quantiles of an array, e.g. `np.quantile(neutral_scores[:, i], 1 - target_fpr)` for threshold tuning — distinct from `np.percentile` which takes percentage values 0–100 instead of fractions 0–1), `np.linspace(..., endpoint=False)` (`endpoint=False` kwarg to exclude the stop value, e.g. `np.linspace(0, 2*np.pi, N, endpoint=False)` for evenly-spaced radar plot angles that don't double-count 0 and 2π)

**Pandas:**
`pd.read_csv`, `pd.read_json()`, `pd.DataFrame`, `pd.to_datetime()`, `pd.concat()` (concatenate a list of DataFrames along an axis, e.g. `pd.concat([df1, df2], ignore_index=True)`), `df.groupby().agg()`, `df.groupby("col").agg(output_name=("source_col", func))` (named aggregation syntax — keyword-argument form that renames output columns inline, e.g. `agg(n_headlines=("headline", "count"), mean_sentiment=("sentiment_numeric", "mean"))`), `df.groupby().rolling()`, `df.groupby().apply()`, `.reset_index()`, `.dropna()`, `df["col"].map()`, `df["col"].apply()`, `df["col"].nunique()`, `df["col"].std()`, `df["col"].count()`, `df["col"].min()`, `df["col"].max()`, `df["col"].tolist()`, `df.to_csv()`, `df.diff()`, `pd.qcut()`, `pd.cut()`, `df["col"].fillna()`, `df.copy()`, `df.columns` (assignment), `df["col"].rank(pct=True)`, `df.to_string(index=False)`, `df.sort_values()`, `df.describe()`, `df.head()`, `df.tail()`, `df.loc[]`, `df.iloc[]`, `df.astype()`, `df.values` (underlying numpy array)

**Visualization:**
`plt.bar()`, `plt.subplots()`, `plt.figure()`, `plt.savefig()`, `plt.colorbar()`, `plt.tight_layout()`, `plt.rcParams.update()`, `plt.imshow()`, `plt.show()`, `plt.legend()`, `plt.title()`, `plt.xlabel()`, `plt.ylabel()`, `plt.xticks()`, `plt.grid()`, `plt.suptitle()`, `plt.axis("off")`, `plt.close()`, `plt.scatter()`, `plt.plot()`, `plt.barh()`, `plt.hist()`, `matplotlib.use("Agg")`, `matplotlib.patches.Patch` (legend patch elements), `plt.cm.tab10`, `plt.cm.Set3` (colormap objects), `ax.plot()`, `ax.fill()`, `ax.scatter()`, `ax.imshow()`, `ax.annotate()`, `ax.text()`, `ax.legend()`, `ax.set_title()`, `ax.set_xlabel()`, `ax.set_ylabel()`, `ax.set_xlim()`, `ax.set_ylim()`, `ax.set_xticks()`, `ax.set_xticklabels()`, `ax.spines[...].set_visible()`, `ax.axhline()`, `ax.axvline()`, `ax.bar()`, `ax.barh()`, `ax.grid()`, `ax.transAxes` (transform for relative positioning), `sns.heatmap()`, `sns.heatmap(..., annot=True, fmt=".2f")` (`annot=True` overlays cell values; `fmt=` controls display format, e.g. `fmt=".2f"` for two-decimal floats in campaign heatmaps), `sns.scatterplot()`, `sns.lineplot()`, `sns.despine()`, `sns.color_palette()`, `sns.set_palette()`, polar axes: `fig.add_subplot(polar=True)` (create polar axis), `ax.set_theta_offset()`, `ax.set_theta_direction()`, `ax.set_rlabel_position()`, `ax.set_thetagrids(angles, labels)` (set theta grid lines and tick labels on a polar axis, e.g. for radar charts — distinct from `ax.set_xticks()` which is for Cartesian axes), `ax.hist()`, `ax.imshow(..., aspect="auto")` (`aspect` kwarg to disable equal-axis locking, e.g. for non-square activation grids), `ax.imshow(..., extent=[xmin, xmax, ymax, ymin])` (`extent` kwarg to set axis coordinate ranges, e.g. `extent=[0, 2*math.pi, M, 0]` to label atom × angle heatmaps with real units), `ax.text(..., verticalalignment='top')` (`verticalalignment` / `va=` kwarg controlling vertical anchor of text annotations — `'top'` pins the top of the text bounding box to the y coordinate)

**Scientific Python:**
`sklearn.decomposition.PCA`, `pca.fit()`, `pca.fit_transform()`, `pca.transform()`, `pca.explained_variance_ratio_`, `sklearn.metrics.roc_auc_score`, `sklearn.metrics.silhouette_score`, `sklearn.metrics.cohen_kappa_score`, `sklearn.metrics.pairwise.cosine_similarity`, `sklearn.preprocessing.StandardScaler`, `scaler.fit_transform()`, `sklearn.linear_model.LogisticRegression`, `probe.fit()`, `probe.predict_proba()`, `probe.score(X, y)` (accuracy of fitted probe on a labelled array), `probe.coef_` (fitted weight matrix of shape `(n_classes, n_features)`; `probe.coef_[0]` for binary classification), `probe.classes_` (array of class labels in the order the probe learned them, e.g. `probe.classes_.tolist().index(COT)` to find the column index for a specific class in `predict_proba` output), `sklearn.model_selection.cross_val_score`, `scipy.stats.f_oneway`, `scipy.stats.ttest_ind`, `scipy.linalg.eigh()`, `scipy.cluster.hierarchy.linkage`, `scipy.cluster.hierarchy.fcluster`, `requests.get()`

**networkx:**
`nx.Graph()`, `nx.DiGraph()` (directed graph), `G.add_edge()`, `G.add_node()`, `nx.from_pandas_edgelist()`, `nx.from_dict_of_lists()`, `nx.pagerank()`, `nx.betweenness_centrality()`, `nx.clustering()`, `nx.degree_centrality()`, `nx.closeness_centrality()`, `nx.density()`, `nx.community.greedy_modularity_communities()`, `nx.connected_components()`, `nx.spring_layout()`, `nx.spectral_layout()`, `nx.draw(G, pos=..., node_color=..., with_labels=True)` (simple draw wrapper — distinct from `nx.draw_networkx()` which exposes all artist kwargs), `nx.draw_networkx()`, `nx.draw_networkx_nodes()`, `nx.draw_networkx_edges()`, `nx.draw_networkx_labels()`, `nx.laplacian_matrix()`, `nx.to_numpy_array()`, `G.degree(weight=...)` (weighted degree dict), `G.neighbors(n)`, `G.number_of_nodes()`, `G.number_of_edges()`

**NLTK:**
`nltk.download()`, `nltk.corpus.words.words()`, `nltk.corpus.wordnet.synsets()`

**PEFT (Hugging Face):**
`peft.LoraConfig`, `peft.TaskType`, `peft.get_peft_model()`, `model.print_trainable_parameters()`

**nnsight:**
`LanguageModel(model_name, device_map=..., dispatch=True)`, `model.trace(inputs)`, `model.trace(inputs, attention_mask=mask)` (trace with keyword args, e.g. in GLP/MFA notebooks), `layer.output.save()`, `nnsight.save()`, `model.tokenizer` (attribute), `model.lm_head` (attribute), `model.model` (underlying PyTorch model attribute), `model.model.layers[idx]` (indexed layer access for activation extraction), `model.model.layers[idx].output.save()` (save hidden state at specific layer)

**nnterp:**
`StandardizedTransformer(model_name, device_map=..., dispatch=True)`

**FAISS:**
`faiss.IndexFlatIP`

**tkinter:**
`tk.Tk()`, `tk.Button`, `tk.Label`, `tk.Frame`, `tk.Canvas()`, `tk.StringVar()` (tkinter observable string variable, used to bind widget text to a Python variable), `root.title(title)` (set the window title string on a Tk root window), `root.mainloop()`, `widget.config()`, `widget.grid()`, `widget.pack()`, `widget.bind()` (event binding)

**tqdm:**
`tqdm.auto.tqdm`

**PIL/Pillow:**
`PIL.Image.open()`, `PIL.Image.fromarray()`, `image.resize(size, resample=Image.BILINEAR)`

**jaxtyping:**
`Float[Tensor, "batch seq d_model"]` (runtime shape annotations), `jaxtyping.Float`

**Web/Data Collection:**
`warcio.ArchiveIterator()`, `trafilatura.extract()`, `BeautifulSoup()` (beautifulsoup4)
