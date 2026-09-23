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
│   ├── jacobian-lens/        # Jacobian lens / global workspace: J_l = E[dh_final/dh_l], steering, coordinate patching, J-space pursuit (Transformer Circuits 2026)
│   └── tdkps-perspective-shifts/  # TDKPS: monitoring behavioral drift in black-box multi-agent systems via omnibus MDS + permutation tests (Bridgeford & Helm 2025)
├── practice/
│   ├── 01-LLM-as-Judge/      # LLM evaluation as judge
│   ├── 02-Agent-Eval-Harness/ # Agent evaluation harness
│   ├── 03-Codenames-AI/      # AI for Codenames board game
│   ├── 04-Safety-Evaluation-Pipeline/  # Red-team campaign analysis (ISC paper)
│   ├── 05-Multi-Agent-Control-Collusion/  # AI-control safety metric, monitor collusion + paraphrase defense, prompt-infection propagation
│   └── 06-Control-Charts-Multi-Agent/  # Adaptive Shewhart chart on iso-mirror trajectory, slow-defection sleeper, adaptivity-vs-security tradeoff
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
- TDKPS Perspective Shifts: Detecting Perspective Shifts in Multi-agent Systems (Bridgeford & Helm 2025, arXiv:2512.05013) — monitoring behavioral drift in black-box agent fleets. Omnibus distance matrix over all (agent, time) snapshots (replicate-averaged response embeddings, Frobenius distances via `torch.cdist` on flattened matrices), classical MDS via Torgerson double-centering (B = −½ J D∘D J, `torch.linalg.eigh`, top-d eigenpairs with negative-eigenvalue clamping), agent-level drift scores δ_n = ‖ψ_n^(t1) − ψ_n^(t0)‖, and a group-level energy-distance statistic with a paired-swap permutation null (key insight: mean paired drift is swap-invariant so the test must use cross-agent terms). CPU-only synthetic ("temporal Gaussian blobs" with per-query orthogonal maps and stable agent effects); reproduces clean signal/null separation (drift ratio ~5, p_signal ≈ 0.002 vs p_null ≈ 0.53).

**Practice Set:**
- LLM-as-Judge evaluation
- Agent evaluation harness
- Codenames AI (board game AI)
- Safety Evaluation Pipeline: Red-team campaign analysis based on ISC paper
- Multi-Agent Safety (Control Evals, Collusion, Infection): synthesizes the AI-control / multi-agent-safety literature (Greenblatt et al. 2023 control evals; Shlegeris 2024 collusion + paraphrase defense; UK AISI Alignment Project 2025; Prompt Infection arXiv:2410.07283; position papers: DeepMind Distributional AGI Safety, Mila Coming Crisis, CAIF Multi-Agent Risks) — audit-budget quantile thresholding, catch-vs-ship Monte Carlo safety metric with the memoryless-race identity, signal-conditioned monitor collusion collapsing safety ~0.48→~0.02, paraphrase-defense recovery curve, and BFS steps-to-saturation across chain/star/ER agent topologies (reproduces the 50-agent <11-step saturation result). CPU-only synthetic, property-based validation with analytic references (star=2, chain=N-1).
- Control Charts for Multi-Agent Systems (Helm, Priebe & Duderstadt 2026, arXiv:2605.11135) — statistical process control for agent deployments: iso-mirror trajectory via classical MDS (double-centering B = -½JD⁽²⁾J, top eigenpair), adaptive Shewhart chart with rolling μ̂±kσ̂ limits validated against the analytic Gaussian false-alarm rate 2Φ(-k), normalized sigmoid slow-defection schedule (exact 0→p_max endpoints, p_max/2 midpoint by symmetry), and Monte-Carlo reproduction of both theorems: fast attacks fire an alarm burst in every deployment while slow attacks are indistinguishable from baseline (exactly so at k=5, Theorem 1/Fig. 4), and the static-vs-adaptive 2×2 where each chart fails exactly one scenario (adaptive misses slow drift, static false-alarms on legitimate learning — Theorem 2). CPU-only synthetic; the σ̂-inflation mechanism (a jump inflates the rolling std along with the deviation, so alarms last only ~W/(1+k²) absorption steps) is the notebook's quantitative twist on the paper.

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
`torch.arange` / `torch.arange(start, stop, step)` (3-argument step form — e.g. `torch.arange(0, d_model, 2)` to enumerate every-other index for sinusoidal PE even/odd channels and RoPE inv_freq; the step arg is required when you need strides other than 1), `torch.ones`, `torch.zeros`, `torch.randn`, `torch.randn_like`, `torch.ones_like`, `torch.zeros_like`, `torch.zeros_like(tensor, dtype=dtype)` (`dtype=` kwarg on `torch.zeros_like` — creates a zero tensor with the same shape but a different dtype, e.g. `torch.zeros_like(logits, dtype=torch.bool)` to build a boolean filter mask in top-k/top-p sampling; distinct from plain `torch.zeros_like(tensor)` which inherits the source dtype), `torch.full_like`, `torch.empty()` (create uninitialized tensor — used before init functions like `nn.init.kaiming_uniform_`, e.g. `nn.Parameter(torch.empty(out_features, in_features))` in custom linear layers and LoRA), `torch.zeros(..., device=device)` / `torch.ones(..., device=device)` / `torch.randn(..., device=device)` / `torch.linspace(..., device=device)` / `torch.full(..., device=device)` (`device=` kwarg on basic tensor creation functions — places the new tensor directly on the target device without a subsequent `.to(device)` call, e.g. `torch.zeros(batch, seq, device=x.device)` to create a mask on the same device as the input, or `torch.linspace(1.0, 0.0, steps + 1, device=device)` for a device-resident schedule in GLP flow matching), `torch.zeros(..., dtype=dtype)` / `torch.ones(..., dtype=dtype)` (`dtype=` kwarg on basic tensor creation functions with an explicit shape — creates a tensor of the specified dtype directly, e.g. `torch.zeros(n, m, dtype=torch.bool)` for a boolean mask in VLM attention or `torch.zeros(out, in, dtype=torch.int8)` for a quantized weight buffer — distinct from `torch.zeros_like(tensor, dtype=dtype)` which derives shape from an existing tensor, and from the `device=` variants above which only control placement), `torch.cat`, `torch.stack`, `torch.full`, `torch.linspace`, `torch.tensor`, `torch.where`, `torch.randint`, `torch.rand`, `torch.Generator`, `torch.manual_seed()`, `torch.Generator().manual_seed(seed)` (instance method — returns the generator after seeding, allowing chaining; e.g. `g = torch.Generator().manual_seed(seed)` for per-operation reproducibility — distinct from global `torch.manual_seed()`), `torch.isfinite`, `torch.normal()`, `torch.set_default_dtype()` (set global default floating-point dtype for new tensors, e.g. `torch.set_default_dtype(torch.float32)` or `torch.set_default_dtype(torch.float64)` for higher-precision Jacobian computations), `.view()`, `.view(*unpacked_list)` (starred-unpacking form when dimensions are stored in a list or tuple, e.g. `scales.view(*shape)` in per-channel quantization to restore spatial shape after per-axis reduction — same as `.view(d0, d1, ...)` but passes dims via `*`), `.transpose()`, `.permute()`, `.unsqueeze()`, `.chunk()`, `.split()`, `.clone()`, `.flatten()`, `.flatten(start_dim=N)` (partial flatten from a specified dimension, e.g. `.flatten(2)` to collapse spatial dims in patch embedding: `proj(x).flatten(2).transpose(1, 2)`), `.reshape()`, `.to()` (dtype), `.squeeze()`, `.squeeze(dim)` / `.squeeze(-1)` (integer-argument form — removes a specific singleton dimension rather than all of them, e.g. `.squeeze(-1)` to collapse the trailing size-1 dimension from a `(batch, 1)` reward tensor to `(batch,)` in the reward model; distinct from `.squeeze()` which removes every dimension of size 1), `.contiguous()`, `.detach()`, `.argmax()`, `.size()`, `.dim()` (number of dimensions), `.sum()`, `.scatter_()`, `.bool()`, `.float()`, `.double()` (convert to float64, e.g. `(mask == 0).double().mean().item()` for exact-precision fraction computations in Jacobian notebooks — distinct from `.float()` which gives float32), `.int()`, `.item()`, `.expand()`, `.unbind()`, `.copy_()`, `.add_()`, `.sub_()`, `.data`, `.numel()`, `.element_size()`, `.dtype` (read the data type of a tensor as a `torch.dtype` value, e.g. `assert w.dtype == torch.int8` to verify quantization produced the right type, or `scale.to(x.dtype)` to match precision — distinct from dtype constants like `torch.float32` which are used as arguments to `.to()` and other ops), `.numpy()`, `torch.flatten()`, `.eq()`, `.all()`, `.T` (2D transpose attribute), `.mT` (batched matrix transpose property — equivalent to `.transpose(-2, -1)` applied as an attribute, works on any ≥2-D tensor including batched ones, e.g. `J.mT @ u_t` in Jacobian-lens steering — distinct from `.T` which only works on exactly 2-D tensors), `.tolist()` (convert tensor to nested Python list), `.repeat()` (repeat tensor along dimensions, e.g. `x.repeat(1, n_kv_heads, 1, 1)` — distinct from `repeat_interleave` which repeats individual elements), `.flip()` (reverse a tensor along one or more dimensions, e.g. `x.flip(dims=[-1])` to reverse the last dim — used in Jacobian steering and coordinate patching), `.grad` (gradient tensor attribute on a leaf tensor, e.g. `param.grad` or `assert tensor.grad is not None` to verify gradient flow), `.unfold(dim, size, step)` (returns a rolling-window view of the tensor — e.g. `phi.unfold(0, w, 1)` produces shape `(T-w+1, w)` non-overlapping windows along dim 0, used in adaptive Shewhart control charts instead of an explicit loop), `.scatter(dim, index, src)` (out-of-place scatter — returns a new tensor with values from `src` placed at `index` positions along `dim`; distinct from `.scatter_()` which modifies in place; used in top-k filtering to copy top-k values back to original vocab positions), `tensor[None, :]` / `tensor[:, None]` (insert a size-1 dimension via `None` in an index — equivalent to `.unsqueeze(0)` / `.unsqueeze(-1)` but written as pure indexing, e.g. `lengths[:, None]` for broadcasting in padding mask creation), `tensor[:, 0::2]` / `tensor[:, 1::2]` (step-slice with stride 2 — Python `start:stop:step` indexing on a tensor, e.g. `pe[:, 0::2] = torch.sin(...)` and `pe[:, 1::2] = torch.cos(...)` in sinusoidal positional embedding to fill even/odd channels), `tensor[..., 1:]` / `tensor[..., :-1]` (Ellipsis `...` as a tensor index selects all preceding dimensions, enabling rank-agnostic trailing-dimension slicing — e.g. `mask[..., 1:] = mask[..., :-1].clone()` to shift a cumulative-probability mask one position right along the last dim without knowing the tensor's total rank; used in top-p sampling to implement nucleus filtering), `tensor[..., :tensor.shape[-1]//2]` / `tensor[..., tensor.shape[-1]//2:]` (first-half and second-half split via integer-division index — e.g. `x[..., :x.shape[-1]//2]` and `x[..., x.shape[-1]//2:]` in RoPE `rotate_half` to separate the two halves of the head dimension for rotation; generalizes to any even-dimension split without hard-coding the size), `tensor >= other` / `tensor > other` / `tensor < other` / `tensor <= other` (elementwise comparison operators returning bool tensors — used e.g. `torch.arange(max_len)[None, :] >= lengths[:, None]` to build a padding mask via broadcasting; complement to `.eq()` for `==`), `.repeat_interleave(n, dim=d)` (tensor method form, e.g. `K.repeat_interleave(num_query_heads // num_kv_heads, dim=1)` for GQA KV-head expansion — same semantics as the function form `torch.repeat_interleave()` already listed but called directly on the tensor), `tensor.shape` (tuple-of-ints property — no parentheses; used in assertions like `assert result.shape == (4,)` and in dimension arithmetic; distinct from `.size()` which is a method returning a `torch.Size` object), `tensor[bool_mask]` (boolean fancy indexing — select elements / rows where the bool tensor is True, e.g. `acts[b, valid]` or `data[mask].mean(dim=0)`; `.nonzero()` is documented as an alternative but the primary syntax was missing), `tensor[bool_mask] = value` (boolean mask in-place write assignment — the write form of boolean indexing; e.g. `result[swap_mask] = other[swap_mask]` to swap rows between tensors in a permutation test without modifying the source tensor; distinct from `tensor[bool_mask]` which reads, and from `.masked_fill_()` which fills with a scalar), `tensor[int_1d_tensor]` (integer tensor fancy indexing — index into a tensor with a 1-D integer index tensor to select rows, e.g. `data[indices].clone()` where `indices = torch.randperm(n)[:K]`; distinct from `.gather()` which requires shape-matching and `torch.index_select()` which requires an explicit dim), `torch.randn(..., generator=g)` (`generator=` kwarg on random tensor creation functions (`torch.randn`, `torch.rand`, etc.) — passes a seeded `torch.Generator` instance to control per-call randomness, e.g. `torch.randn(N, D, generator=g)` for reproducible per-seed trajectories in simulation helpers), `hidden_states[:, -1, :]` (last-token pooling via negative integer index on the sequence dimension — selects the final token's representation before a reward or classification head, e.g. `hidden[:, -1, :]` in `RewardModel.forward()`; common pooling strategy in decoder-only transformer applications where the last non-padding token carries accumulated context), `tensor[:, -1:, :]` (the "last token with kept sequence dimension" slice — unlike `[:, -1, :]` which squeezes the sequence dim to produce shape `(batch, features)`, `[:, -1:, :]` keeps it at size 1 producing shape `(batch, 1, features)`; used in autoregressive decoding to pass only the newest token through the model while preserving the 3-D shape expected by the forward pass, e.g. `W_q(x_current[:, -1:, :])` in KV-cache attention where the query is the current token only but the key/value include the full cached context), `.diagonal()` (extract the main diagonal of a 2D matrix as a 1D tensor — e.g. `D_omni.diagonal().abs().max()` to verify self-distances are zero in a pairwise distance matrix; distinct from `torch.diag()` which both extracts diagonals and constructs diagonal matrices from vectors), `tensor[i] = value` (single-integer-index element assignment to a tensor element in-place, e.g. `valid[0] = False` to zero out the BOS position in a bool mask — standard Python `__setitem__` applied to tensors; distinct from `tensor[bool_mask]` which selects elements and `tensor[int_1d_tensor]` which selects rows), `tensor[a:b, c:d] = value` (2-D slice-and-assign — in-place assignment using explicit row-and-column slice ranges on both dimensions simultaneously, e.g. `mask[:n_image, n_image:] = True` and `mask[n_image:, n_image:] = torch.triu(torch.ones(..., dtype=torch.bool), diagonal=1)` to fill the image→text and text→text regions of a VLM attention mask — distinct from `tensor[i] = value` which targets a single element and from `tensor[:, 0::2] = ...` which slices only one dimension), `tensor[a:b] = value` (1-D slice assignment — assigns a sub-tensor to a contiguous slice of a 1-D tensor in-place, e.g. `u[:D_AGENT] = torch.randn(D_AGENT, generator=g)` to fill the first D_AGENT elements of a zero vector with a displacement direction in the control-charts simulate() helper; distinct from `tensor[i] = value` which targets a single element and from `tensor[a:b, c:d] = value` which is 2-D), `tensor // other` / `tensor % other` (Python floor-division and modulo operators applied element-wise to tensors, e.g. `beam_idx = flat_idx // vocab_size` and `token_idx = flat_idx % vocab_size` to decode combined beam+vocabulary indices in beam search — `//` gives the quotient (which beam) and `%` gives the remainder (which token))

**Torch dtype Constants:**
`torch.float32`, `torch.float64` (double precision, e.g. `torch.set_default_dtype(torch.float64)` for Jacobian computations requiring higher numerical stability), `torch.float16`, `torch.bfloat16`, `torch.long`, `torch.int32`, `torch.int8`, `torch.uint8`, `torch.bool`

**Device Operations:**
`torch.cuda.is_available()`, `torch.device()`, `.cuda()`, `.cpu()`, `tensor.device` (property for accessing a tensor's device, e.g., `torch.arange(n, device=x.device)`)

**Torch Math Operations:**
`torch.sqrt`, `torch.rsqrt`, `torch.exp`, `torch.log`, `torch.sin`, `torch.cos`, `torch.round`, `.round(decimals=N)` (tensor method form with decimal precision kwarg, e.g. `tensor.round(decimals=2)` for display rounding — same as `torch.round()` but accepts `decimals=` for precision control), `torch.clamp`, `.clamp(min=val, max=val)` (tensor method form, e.g. `x.clamp(0.0, 1.0)` or `scale.abs().clamp(min=1e-6)` — same semantics as `torch.clamp()` but called on the tensor; used in control-charts and quantization notebooks), `torch.einsum`, `torch.einsum("i,j->ij", a, b)` (2-argument outer-product einsum — the two-tensor, two-subscript form that produces a rank-2 outer product, e.g. `torch.einsum("i,j->ij", t, inv_freq)` in RoPE to build the (seq_len, dim//2) position-frequency matrix; distinct from the 3-or-more subscript forms and from `torch.outer()` which is the non-einsum equivalent), `torch.einsum(subscripts, [tensor1, tensor2])` (list-argument form of torch.einsum — the second argument is a Python list of tensors rather than separate positional args, e.g. `torch.einsum("bsd,btd->bst", [q, k])` in attention-from-scratch; semantically identical to `torch.einsum("bsd,btd->bst", q, k)` but the list syntax is a valid alternative calling convention students encounter), `torch.topk`, `torch.sort`, `torch.cumsum`, `torch.argsort`, `torch.masked_fill`, `.masked_fill_(mask, value)` (in-place), `.masked_fill(mask, value)` (non-in-place tensor method — returns a new tensor with True-mask positions replaced by value; distinct from function form `torch.masked_fill()` and in-place `.masked_fill_()`), `torch.triu`, `torch.tril`, `.triu(diagonal=N)` (tensor method form of `torch.triu`, e.g. `attn_weights.triu(diagonal=1)` to extract the upper triangle — same semantics as the function form but called on the tensor; used to verify causal mask properties in attention notebooks), `torch.repeat_interleave`, `torch.multinomial`, `torch.multinomial(probs, num_samples=1)` (`num_samples=` kwarg — required positional-or-keyword argument specifying how many samples to draw; e.g. `torch.multinomial(probs, num_samples=1)` in temperature and top-k/top-p sampling to draw one token index per batch item from the filtered distribution — returns shape `(batch, num_samples)`), `torch.gather`, `.gather(dim=..., index=...)` (tensor method form of `torch.gather`, e.g. `logits.gather(dim=-1, index=idx)` to select values at specified indices — same semantics as the function form, used in top-K context distillation loss), `torch.index_select(tensor, dim, indices)` (select elements along `dim` using a 1-D index tensor, e.g. `torch.index_select(beam_tokens, 0, beam_indices)` to reorder beams by source-beam index in efficient beam search — distinct from `.gather()` which requires the index to match the output shape), `torch.logsumexp`, `torch.cdist`, `torch.randperm`, `torch.diag`, `torch.mv`, `torch.trapezoid`, `torch.allclose`, `torch.isclose` (element-wise bool comparison), `torch.inf` (infinity constant), `torch.finfo(dtype).min` (`torch.finfo(dtype)` returns float-type info; `.min` gives the most-negative finite value — used as a numerically safe alternative to `float('-inf')` when masking logits, e.g. `logits.masked_fill(mask, torch.finfo(logits.dtype).min)`), `torch.chunk` (function form), `.pow()`, `.mean()`, `.abs()`, `.max()`, `.min()`, `.amax()`, `.exp()`, `.log()`, `@` (matmul), `torch.matmul()`, `torch.tanh`, `torch.sigmoid`, `torch.softmax`, `torch.relu`, `.relu()` (tensor method form of relu, e.g. `heatmap.relu()` or `(weights * activations).sum(dim=1).relu()` in Grad-CAM heatmap generation — same as `torch.relu()` but called on the tensor; distinct from `F.relu()` and `nn.ReLU`), `.sigmoid()` (tensor method form of sigmoid, e.g. `defect_prob.sigmoid()` in control-chart defection schedules or `diff.sigmoid().log()` in Bradley-Terry reward loss — same as `torch.sigmoid()` but called on the tensor; distinct from `F.sigmoid()` which is functional and `nn.Sigmoid` which is a module), `torch.bmm`, `torch.outer`, `torch.max` (function form with dim arg), `values, indices = torch.max(tensor, dim)` (positional tuple-unpacking form of `torch.max` with a `dim` argument — e.g. `_, predicted = torch.max(outputs, dim=1)` to extract only the argmax class index while discarding the values; distinct from the already-documented `.max(dim=...).values` namedtuple form which uses attribute access), `torch.eye`, `torch.argmin`, `.norm()`, `.norm(dim=N)` (tensor method with `dim=` kwarg for reducing along a specific dimension, e.g. `.norm(dim=-1)` to compute per-row L2 norms from a `(batch, d)` tensor into a `(batch,)` vector — distinct from `.norm()` which returns a scalar global norm; used for per-atom correlation scoring in Jacobian-lens and per-agent drift in TDKPS), `.median()`, `.trace()`, `.unique()`, `.argsort()` (tensor method), `.argsort(descending=True)` (`descending=True` kwarg on the `.argsort()` tensor method for descending-order indices, e.g. `counts.argsort(descending=True)[:10]` to find the top-10 most frequent components in MFA — distinct from `.sort(descending=True)` which returns sorted values not indices), `torch.all`, `torch.logical_or()` (or `|` tensor operator), `torch.logical_not()` (or `~` bool tensor operator), `torch.logical_and()` (or `&` bool tensor operator — element-wise AND on bool tensors, e.g. `caught = backdoored & audited` in multi-agent safety notebooks to identify episodes where an attack was both made and detected), `torch.isnan`, `.isnan()` (tensor method form, e.g. `~atom_arcs.isnan()` to detect dead SAE atoms — same as `torch.isnan()` but called on the tensor; often combined with `~` for not-NaN masking), `torch.pi` (π constant), `torch.diff()` (consecutive differences along a dimension), `.diff()` (tensor method form, e.g. `phi.diff()` to compute consecutive differences — same as `torch.diff()` but called on the tensor), `torch.count_nonzero()` (count non-zero elements), `.cos()` (tensor method, e.g. `theta.cos()`), `.sin()` (tensor method, e.g. `theta.sin()`), `.sort().values` and `.sort().indices` (namedtuple attribute access from `torch.sort()`/`.sort()`), `.sort(descending=True)` (`descending` kwarg for descending sort, e.g. in beam search and top-k/top-p sampling), `.sqrt()` (tensor method form, e.g. `eigvals.sqrt()`), `torch.topk().values` and `torch.topk().indices` (namedtuple attribute access from `torch.topk()`, distinct from `.sort()`), `.max(dim=...).values` and `.min(dim=...).values` (namedtuple attribute access on `.max()/.min()` with a `dim` arg — analogous to `.topk().values`; e.g. `logits.max(dim=-1, keepdim=True).values` for numerically stable softmax), `.mean(dim=..., keepdim=True)` / `.sum(dim=..., keepdim=True)` (`keepdim=True` kwarg for dimension-reducing ops to preserve tensor rank), `.mean(dim=(d1, d2), keepdim=True)` (tuple-of-dims form for reducing multiple axes simultaneously, e.g. `gradients.mean(dim=(2, 3), keepdim=True)` in Grad-CAM to average over both spatial dims H and W — distinct from the single-int `dim=` form), `torch.any()` (function form) and `.any()` (tensor method — checks if any element is True, e.g. `mask.any()` to test if an atom ever fires), `.any(dim=N)` (tensor method with `dim=` kwarg for reducing along a specific dimension, e.g. `(scores >= thresholds).any(dim=-1)` in probe-misalignment for OR-fusion across probes per transcript — returns a bool tensor with one fewer dimension; distinct from `.any()` which reduces all elements to a scalar), `.nonzero()` / `torch.nonzero(tensor, as_tuple=True)` (returns indices of non-zero elements; often used as alternative to boolean fancy indexing), `torch.quantile()` (compute quantiles along a dimension, e.g. `torch.quantile(scores, 1 - target_fpr, dim=0)` for threshold tuning), `.quantile(q, dim=N)` (tensor method form of `torch.quantile()`, e.g. `scores.quantile(1 - target_fpr, dim=0)` for per-probe threshold tuning in probe-misalignment — same semantics as the function form but called directly on the tensor), `.std()` (tensor method for standard deviation, e.g. `x.std(dim=-1, keepdim=True)`), `.std(dim=N)` (tensor method for standard deviation along a specific dimension without keepdim — e.g. `windows.std(dim=-1)` where `windows = phi.unfold(0, w, 1)` to compute per-window rolling standard deviation for the adaptive Shewhart chart; complement to `.std()` scalar form and analogous to `.mean(dim=..., keepdim=True)` — distinct from that form in omitting keepdim), `.var()` (tensor method for variance, e.g. `activation.var(dim=0)` for per-feature variance in activation standardization — analogous to `.std()` but returns variance directly without taking the square root; used in GLP/MFA notebooks for `(x - mean) / (x.var(dim=0).sqrt() + eps)` standardization), `.equal()` (tensor method for exact element-wise equality check returning a bool scalar, distinct from `torch.allclose` which has tolerances), `torch.equal(a, b)` (function form — checks if two tensors have the same size and elements, returning a single bool; e.g. `torch.equal(out, expected)` in extension tests — distinct from `.equal()` tensor method which is semantically identical but called on the first tensor), `torch.sum()` (function form with dim arg, e.g. `torch.sum(X, dim=1)` — distinct from `.sum()` tensor method), `torch.amax()` (function form for reduction, e.g. `weight.amax(dim=reduce_dims, keepdim=True)` in per-channel quantization — distinct from `.amax()` tensor method), `.topk(largest=False)` (smallest-k via `largest=False` kwarg, e.g. `neuron_vals.topk(top_k, largest=False).indices`), `torch.nan` (PyTorch NaN constant, e.g. used to sentinel dead SAE atoms that never activate — analogous to `torch.inf` and `torch.pi`), `torch.bernoulli()` (draw binary samples from a Bernoulli distribution given a probability tensor, e.g. `torch.bernoulli(torch.full((n_episodes, t_tasks), p_attack))` for vectorized flag generation in multi-agent safety notebooks), `torch.mm(A, B)` (2D-only matrix multiply — distinct from `torch.matmul()` which handles any number of dimensions and `torch.bmm` which handles batched 3-D tensors; use when both inputs are exactly 2-D), `.argmax(dim=N)` (`dim=` kwarg on the already-documented `.argmax()` tensor method — e.g. `.float().argmax(dim=1)` to find the index of the first `True` in each row of a 2-D boolean tensor, used for vectorized first-event timing in multi-agent safety notebooks), `tensor ** N` / `tensor ** other` (Python `**` exponentiation operator applied element-wise to tensors, e.g. `sigma_p ** 2` to compute variance or `(mu_p - mu_q) ** 2` for squared difference — syntactically distinct from `.pow(N)` method but semantically equivalent; also used on plain ints for bit-width arithmetic like `2 ** (num_bits - 1) - 1`), `tensor / other` / `tensor / scalar` (Python `/` true-division operator applied element-wise to tensors, e.g. `gdir / gdir.norm()` to unit-normalize a steering direction, `(activations - act_mean) / (act_std + 1e-8)` for standardization, or `defect_prob / P_MAX` for fractional scaling — distinct from `//` floor-division which is already documented), `scalar / tensor` / `1.0 / tensor` (Python `__rtruediv__` — scalar divided by tensor element-wise, e.g. `psi_inv = 1.0 / self.psi` to compute per-diagonal noise-precision in the Woodbury identity for FactorAnalysis; syntactically `scalar / tensor` rather than `tensor / scalar` — the tensor is on the right, which invokes Python's reverse-division; equivalent to `torch.reciprocal(tensor) * scalar` but more readable as a mathematical formula), `tensor * other` / `tensor * scalar` (Python `*` element-wise multiplication operator, e.g. `NOISE_STD * torch.randn(T, D)` to scale noise, `(1 - t) * z0 + t * epsilon` for linear interpolation, or `ramp[:, None] * gdir` to broadcast a scalar ramp across a direction vector — syntactically distinct from `torch.mul()` but semantically equivalent), `.argmin()` / `.argmin(dim=N)` (tensor method form of argmin — e.g. `distances.argmin(dim=-1)` to assign each point to its nearest centroid in k-means, or `logits.argmin()` for the minimum-scoring token; distinct from `torch.argmin` function form already listed; complement to `.argmax(dim=N)` also documented)

**torch.nn Modules:**
`nn.Module` (base class for all custom nn.Module subclasses — inherited by every custom layer, attention block, and full model; defines `forward()`, `parameters()`, `train()`/`eval()`, and hook registration), `nn.Linear`, `nn.Linear(..., bias=False)` (disable learnable bias term, e.g. in attention projection matrices W_q/W_k/W_v/W_o where bias is typically omitted), `nn.Parameter`, `nn.ParameterList`, `nn.Embedding`, `nn.Embedding.from_pretrained()` (class method to construct an Embedding from a pretrained weight matrix), `nn.Conv2d`, `nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)` (patch embedding pattern where `stride=kernel_size` partitions the image into non-overlapping patches — used in ViT-style visual encoders and VLM patch projection layers; followed by `.flatten(2).transpose(1, 2)` to go from `(B, D, H/P, W/P)` to `(B, num_patches, D)` — distinct from standard conv where `stride < kernel_size` creates overlapping receptive fields), `nn.Sequential`, `nn.ReLU`, `nn.GELU`, `nn.SiLU`, `nn.LayerNorm`, `nn.MultiheadAttention`, `nn.MultiheadAttention(..., bias=False)` (disable key/value/output projection bias, e.g. in Perceiver cross-attention aggregators), `nn.ModuleList`, `nn.ModuleDict`, `nn.CrossEntropyLoss`, `nn.init.kaiming_uniform_`, `nn.init.kaiming_uniform_(weight, a=math.sqrt(5))` (the `a=math.sqrt(5)` form matches PyTorch's internal `nn.Linear` initialization — used in LoRA to init the frozen base weight identically to a pretrained linear layer), `self.register_buffer` (register a named buffer; also supports `None` to create an optional/absent buffer, e.g. `self.register_buffer('bias', None)` in QuantizedLinear when bias is disabled), `nn.RNN`, `nn.LSTM`, `nn.MaxPool2d`, `nn.ConvTranspose2d`, `nn.Conv3d`, `nn.ConvTranspose3d`, `nn.LeakyReLU`, `nn.Sigmoid`, `nn.Tanh`, `nn.Softmax`, `nn.Flatten`, `nn.BCELoss`, `nn.MSELoss`, `nn.AdaptiveAvgPool2d`, `nn.Dropout`, `nn.Dropout(p=dropout)` (`p=` kwarg for dropout probability — the primary constructor kwarg when probability is variable, e.g. `nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()` in LoRA to conditionally apply dropout; distinct from positional `nn.Dropout(0.1)` where the probability is a literal), `nn.Identity()` (no-op pass-through module that returns its input unchanged — used as the `else` branch of conditional module assignment, e.g. `self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()` to avoid an `if` in `forward()`; distinct from `nn.Dropout` which has trainable-but-zero-at-eval dropout behavior), `nn.init.kaiming_normal_`, `nn.init.xavier_normal_`, `nn.init.xavier_uniform_`, `nn.init.zeros_`, `nn.init.normal_`, `nn.init.constant_`, `nn.init.ones_`, `model.children()`, `nn.MultiheadAttention` cross-attention: `attn(query, key, value, key_padding_mask=mask)` where query≠key=value (Perceiver-style cross-attention), `module_dict["key_name"]` (string key access on `nn.ModuleDict`), `nn.MultiheadAttention(batch_first=True)` (input/output tensors in (batch, seq, d) order instead of default (seq, batch, d)), `multihead_attn.in_proj_weight` (concatenated Q/K/V weight matrix on `nn.MultiheadAttention`; sliced as `[:d_model]` for Q, `[d_model:2*d_model]` for K, `[2*d_model:]` for V), `multihead_attn.out_proj.weight` (output projection weight tensor on `nn.MultiheadAttention`), `nn.Embedding.weight` (direct access to the underlying `[n, d]` weight parameter of an `nn.Embedding` — e.g. `self.latent_queries.weight` to use the embedding table as learned query vectors in a Perceiver aggregator, without going through a forward integer index), `nn.ModuleList[start:end]` (slice a `nn.ModuleList` to iterate a prefix of blocks — e.g. `model.blocks[:layer_idx + 1]` to run only the first N transformer blocks for activation extraction; returns a Python list, not a new ModuleList), `linear.in_features` (integer attribute on an `nn.Linear` instance giving the input dimension, e.g. `cls(linear.in_features, linear.out_features, ...)` in `QuantizedLinear.from_float()`), `linear.out_features` (integer attribute on an `nn.Linear` instance giving the output dimension — same usage as `in_features`), `linear.weight` (the `weight` Parameter accessed directly as an instance attribute on `nn.Linear`, e.g. `linear.weight.data` to copy weights in `QuantizedLinear.from_float()` — distinct from `multihead_attn.in_proj_weight` and from iterating `named_parameters()`), `linear.bias` (the `bias` Parameter — or `None` — accessed as an instance attribute on `nn.Linear`, used with the guard `if linear.bias is not None:` to conditionally copy bias in quantization adapters)

**torch.nn.functional:**
`F.softmax`, `F.softmax(..., dtype=torch.float32).to(dtype)` (upcast-softmax-recast pattern for numerical stability in BF16/FP16 models — upcasts to float32 before softmax then recasts back, e.g. `F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)` in SmolLM attention), `F.log_softmax`, `F.relu`, `F.leaky_relu()` (functional leaky ReLU, e.g. `F.leaky_relu(x, negative_slope=0.2)`), `F.gelu`, `F.silu`, `F.sigmoid`, `F.logsigmoid` (log-sigmoid activation; the full Bradley-Terry pairwise ranking loss for reward models is `-F.logsigmoid(rewards_chosen - rewards_rejected).mean()` — the negation + mean of the log-sigmoid of the margin; used in rlhf/01-Reward-Model; identical to `-torch.log(torch.sigmoid(r_w - r_l)).mean()`), `F.linear`, `F.normalize`, `F.normalize(..., dim=0)` (column-wise unit-norm, e.g. for SAE decoder weight init — distinct from the typical row-norm `dim=-1`), `F.kl_div`, `F.kl_div(..., log_target=True)` (when both inputs are log-probabilities; `F.kl_div(log_q, log_p, reduction='batchmean', log_target=True)`), `F.cross_entropy`, `F.mse_loss`, `F.binary_cross_entropy_with_logits`, `F.dropout`, `F.scaled_dot_product_attention`, `F.scaled_dot_product_attention(..., is_causal=True)` (`is_causal=True` kwarg to enable causal masking in a single call without passing an explicit mask), `F.scaled_dot_product_attention(..., attn_mask=mask)` (float additive attention mask — values are added to attention logits before softmax, e.g. `F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask)` when `is_causal=False` but a precomputed mask is available), `F.layer_norm()` (functional form of layer norm — same math as `nn.LayerNorm` but stateless, useful inside custom modules), `F.unfold`, `F.pad`, `F.cosine_similarity`, `F.conv2d`, `F.embedding`, `F.interpolate`

**Autograd / Training:**
`torch.no_grad()`, `@torch.no_grad()` (as function decorator), `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`, `model.zero_grad()`, `torch.optim.Adam`, `torch.optim.AdamW`, `torch.optim.SGD`, `torch.save()`, `torch.load()`, `model.state_dict()`, `model.load_state_dict()`, `model.apply()`, `model.parameters()`, `model.named_parameters()`, `model.named_buffers()`, `model.named_children()`, `.requires_grad` (read attribute to check if a tensor participates in autograd, e.g. `if param.requires_grad:` — distinct from `.requires_grad_(True)` which is the in-place setter), `.requires_grad = False` (direct property assignment to freeze a parameter, e.g. `p.requires_grad = False` in LoRA to freeze pretrained weights without returning the tensor — equivalent to `.requires_grad_(False)` but does not support chaining), `.requires_grad_(False)`, `.requires_grad_(True)` (in-place enable gradient tracking on a tensor, e.g. for activations that need gradients in a validation probe), `model.train()`, `model.eval()`, `torch.optim.lr_scheduler.CosineAnnealingLR`, `scheduler.step()`, `torch.optim.AdamW(..., weight_decay=float)` (`weight_decay=` kwarg for L2 regularization on non-bias parameters, e.g. `torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)` in GLP meta-model training)

**Custom Autograd (torch.autograd.Function):**
`torch.autograd.Function`, `ctx.save_for_backward()`, `ctx.saved_tensors`, `Function.apply()`

**Torch Distributions:**
`torch.distributions.Normal`, `torch.distributions.kl_divergence`, `from torch.distributions import Normal, kl_divergence as torch_kl` (aliasing `kl_divergence` on import to avoid a naming collision when defining a custom `kl_divergence` function in the same notebook — the `as torch_kl` alias is the standard pattern in the KL divergence loss notebook; the bare-name `kl_divergence` clashes with student-defined functions of the same name)

**Linear Algebra:**
`torch.linalg.eigvalsh`, `torch.linalg.eigh`, `torch.linalg.solve`, `torch.linalg.slogdet`, `torch.linalg.inv()`, `torch.linalg.qr()`, `torch.linalg.qr(tensor).Q` (named attribute access for the orthogonal factor — e.g. `torch.linalg.qr(torch.randn(D, D)).Q[:, :2]` to get an orthonormal basis; analogous to `.values`/`.indices` on `torch.topk()` — distinct from tuple unpacking `Q, R = torch.linalg.qr(...)`), `torch.linalg.qr(tensor)[0]` (integer subscript `[0]` form to extract the Q factor — equivalent to `.Q` but via positional tuple indexing, e.g. `torch.linalg.qr(torch.randn(D, D))[0][:, :2]` in SAE concept-manifolds notebooks), `torch.linalg.cholesky()` (Cholesky decomposition of a positive-definite matrix, e.g. for MFA covariance factors), `torch.linalg.pinv()` (Moore-Penrose pseudo-inverse, e.g. for least-squares projection in Jacobian-lens coordinate patching), `torch.linalg.norm()` (Euclidean/Frobenius norm as a function, e.g. `torch.linalg.norm(Xbar[t1, n1] - Xbar[t2, n2])` for pairwise Frobenius distances in TDKPS drift scoring — distinct from `.norm()` tensor method which is the same but called on the tensor itself), `.logdet()` (tensor method computing the log-determinant of a square matrix — concise alternative to `torch.linalg.slogdet(M).logabsdet`, e.g. `M.logdet()` in MFA Woodbury log-likelihood)

**Functional Transforms:**
`torch.func.grad`, `torch.func.vmap`, `torch.func.functional_call`

**Autograd (Low-level):**
`torch.autograd.grad`, `torch.autograd.grad(..., retain_graph=True)` (`retain_graph=True` kwarg to keep the computation graph alive so multiple backward passes can be taken through the same forward graph, e.g. computing per-layer Jacobians in a loop), `torch.autograd.grad(..., create_graph=True)` (`create_graph=True` kwarg to build a higher-order computation graph over the gradient itself, enabling second-order derivatives; e.g. computing the Jacobian of activations w.r.t. hidden states in jacobian-lens — distinct from `retain_graph=True` which only preserves the existing graph for a re-run), `torch.autograd.functional.jacobian()` (compute the Jacobian matrix of a function with respect to its inputs, e.g. `jacobian(fn, inputs)` for Jacobian-lens steering and coordinate patching)

**Hooks:**
`module.register_forward_hook`, `module.register_full_backward_hook`, `module.register_backward_hook`

**Mixed Precision (torch.cuda.amp):**
`torch.cuda.amp.GradScaler()`, `torch.cuda.amp.autocast()`, `scaler.scale()`, `scaler.step()`, `scaler.update()`

**Quantization:**
`torch.quantization.quantize_dynamic`, `isinstance(module, nn.Linear)`, `setattr(model, name, module)`, `torch.qint8`, `torch.int8`, `torch.uint8`

**Torch JIT:**
`@torch.jit.script`

**torch.utils.data:**
`torch.utils.data.Dataset`, `torch.utils.data.DataLoader`, `torch.utils.data.TensorDataset`, `__len__()`, `__getitem__()`, `DataLoader(..., shuffle=True, drop_last=True)` (shuffle and drop_last params for training loops), `DataLoader(..., num_workers=N)` (parallel data loading with N worker processes for faster throughput), `len(dataloader)` (number of batches in one epoch — i.e. `ceil(dataset_size / batch_size)`; used in training loops to compute steps-per-epoch or log progress fractions, e.g. `for i, batch in enumerate(dataloader): ... if i % len(dataloader) == 0: log()`)

**torch.utils.tensorboard:**
`SummaryWriter`, `writer.add_scalar()`, `writer.close()`

**torchvision:**
`transforms.Compose`, `transforms.ToTensor`, `transforms.Normalize`, `transforms.RandomHorizontalFlip`, `transforms.RandomCrop`, `transforms.Resize`, `transforms.ToPILImage`, `torchvision.datasets.CIFAR10`, `torchvision.datasets.MNIST`, `torchvision.datasets.FakeData`, `torchvision.models.resnet18`, `torchvision.models.resnet18(pretrained=True)` (load pretrained ImageNet weights; used in XAI/GradCAM and transfer learning notebooks), `torchvision.utils.make_grid`

**einops:**
`einops.rearrange`, `einops.repeat`, `einops.einsum`, `einops.reduce`

**HuggingFace:**
`AutoTokenizer.from_pretrained`, `AutoModelForSequenceClassification.from_pretrained`, `AutoModelForCausalLM.from_pretrained`, `AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)` (`dtype=` kwarg to control model weight precision at load time, e.g. `dtype=torch.float32` to force full precision for probe experiments), `pipeline()`, `pipeline(task, model=..., tokenizer=tokenizer)` (explicit `tokenizer=` kwarg form), `TrainingArguments`, `Trainer`, `trainer.train()`, `datasets.load_dataset`, `datasets.load_dataset(..., streaming=True)`, `datasets.load_dataset("org/name", "config_name", split="train")` (second positional arg selects a named subset or configuration, e.g. `"wikitext-2-raw-v1"` for WikiText datasets — distinct from the no-subset form that loads the default configuration), `SentenceTransformer`, `SentenceTransformer(model_name, prompts={"retrieval": "..."})` (prompts param), `SentenceTransformer.encode(texts)`, `SentenceTransformer.encode(texts, convert_to_tensor=True)`, `SentenceTransformer.encode(texts, normalize_embeddings=True)`, `SentenceTransformer.encode(texts, normalize_embeddings=True, batch_size=256)`, `datasets.Dataset.from_dict()`, `datasets.Dataset.from_list()` (construct a Dataset from a Python list of dicts, e.g. for fine-tuning), `dataset.map(batched=True)`, `dataset.map(batched=True, remove_columns=[...])`, `dataset.set_format("torch")`, `CLIPModel.from_pretrained`, `CLIPProcessor.from_pretrained`, `CLIPProcessor(images=pil_image, return_tensors="pt")` (callable form with image input), `tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=n)` (callable form), `tokenizer(text, return_tensors="pt", return_offsets_mapping=True)` (with `return_offsets_mapping=True` to get character-span offsets for aligning hidden states to source text), `tokenizer.pad_token`, `tokenizer.eos_token`, `tokenizer.decode()`, `model.config` (e.g. `model.config.num_hidden_layers`, `model.config.hidden_size`), `model.config.num_attention_heads` (number of query heads), `model.config.num_key_value_heads` (number of KV heads for GQA), `model.config.head_dim` (per-head dimension), `model.config.rope_theta` (RoPE base frequency — used in SmolLM and GQA notebooks to build the rotary embedding with correct θ), `model.vision_model` (ViT submodule on CLIPModel), `model.lm_head` (LM head Linear layer attribute on causal LM), `model.lm_head.weight` (LM head weight matrix, e.g. for logit lens), `model.model.norm` (final RMSNorm layer of decoder LM), `model.model.norm.weight` (RMSNorm gain vector — e.g. `model.model.norm.weight.detach().cpu().float()` for logit-lens whitening in GLP and MFA notebooks), `model(..., output_hidden_states=True)`, `model(**enc, output_hidden_states=True)` (`**dict` unpacking to pass a tokenizer's output dict directly as model kwargs — e.g. after `enc = tok(text, return_tensors="pt")` and `enc.pop("offset_mapping")`, call `model(**enc, output_hidden_states=True)` rather than listing each key; common HuggingFace pattern when the tokenizer output and model signature align), `outputs.hidden_states`, `model(..., output_attentions=True)`, `outputs.attentions`, `outputs.hidden_states[layer]` (indexed access to a specific layer's hidden states from `output_hidden_states=True`, e.g. `outputs.hidden_states[target_layer]` — distinct from the tuple `outputs.hidden_states` which contains all layers), `outputs.hidden_states[-1]` (negative-index form to access the final hidden layer directly — equivalent to `outputs.hidden_states[num_layers]` but works without knowing the layer count, e.g. in FinBERT sentiment pipelines where the last layer's representation is used as a document embedding), `model.vision_model(pixel_values, output_attentions=True)` (calling the ViT submodule directly as a callable, e.g. to retrieve attention maps from a CLIP visual encoder)

**Python Standard Library:**
`collections.Counter`, `collections.Counter.most_common()`, `collections.defaultdict`, `collections.deque` (double-ended queue for BFS — `deque([start])` to initialize, `.append(node)` to enqueue, `.popleft()` to dequeue FIFO, `.appendleft(node)` to prepend; used for BFS infection propagation across agent topologies in multi-agent safety notebooks), `dataclasses.dataclass`, `dataclasses.field`, `dataclasses.field(default_factory=list)` (`default_factory=` kwarg — passes a zero-argument callable that produces the default value when the field is not explicitly initialized, e.g. `dataclasses.field(default_factory=list)` for a mutable list default in a `GameState` dataclass — required because mutable defaults are forbidden as bare literals in `@dataclass`), `functools.cache`, `copy.deepcopy`, `re`, `re.compile()`, `re.match()`, `re.sub`, `re.findall`, `re.search`, `re.split`, `re.IGNORECASE` (flag for case-insensitive matching, passed as third arg: `re.findall(pattern, text, re.IGNORECASE)` or `re.sub(pattern, repl, text, flags=re.IGNORECASE)`), `itertools.combinations(iterable, r)` (generate all r-length combinations from iterable without repetition, e.g. `combinations(company_list, 2)` to enumerate all company pairs for co-occurrence edge creation in knowledge graphs), `enum.Enum`, `enum.auto`, `EnumMember.value` (`.value` attribute on an enum member — accesses the integer or string value assigned to that member, e.g. `HarmSeverity.FULL_REFUSAL.value == 1` to verify enum values in tests; distinct from `enum.auto` which generates values automatically), `len(SomeEnum)` (Python builtin `len()` applied to an Enum class — returns the number of members, e.g. `assert len(HarmSeverity) == 5` to validate the taxonomy size; Python's `EnumMeta` implements `__len__` for this), `math.log`, `math.sqrt`, `math.ceil`, `math.inf`, `math.inf` as a return sentinel for unreachable graph nodes (e.g. `return math.inf` in `steps_to_saturation()` when any node is unreachable, signaling "never" rather than a finite step count — used in multi-agent-control BFS; complements the `math.inf` constant), `math.exp`, `math.pi`, `math.erf(x)`, `from math import inf, sqrt` (importing `math.inf` and `math.sqrt` as bare names for use directly in tight mathematical expressions — e.g. `from math import inf, sqrt` in attention-from-scratch so masking uses `value=-inf` and scaling uses `/ sqrt(d_k)` without the `math.` prefix; a common idiom when the prefix would clutter formulas) (error function, e.g. `math.erf(k / math.sqrt(2))` to compute the analytic Gaussian two-sided false-alarm rate `2*(1 - 0.5*(1 + math.erf(k/sqrt(2))))` for calibrating adaptive Shewhart control charts), `typing.Optional`, `typing.Tuple`, `typing.List`, `typing.Callable`, `typing.Dict`, `typing.Union`, `list[str]` / `dict[str, float]` / `tuple[str, float]` (PEP 585 lowercase generic type hints, as opposed to `typing.List` etc.), `X | None` / `X | Y` (PEP 604 union type syntax, as opposed to `typing.Optional`/`typing.Union`), `time.time()`, `json.load()`, `json.loads()`, `json.dumps()` (serialize Python object to JSON string, complement of `json.loads()`), `pathlib.Path`, `pathlib.Path.open()`, `pathlib.Path.cwd()` (classmethod returning current working directory as a Path), `pathlib.Path.parents` (`.parents` attribute — sequence of ancestor Paths, e.g. `Path.cwd().parents`), `path / "subdir"` (Path `/` operator for joining path segments, e.g. `root / ".practice-log.jsonl"`), `pathlib.Path.exists()` (instance method checking whether the path exists on disk; distinct from `os.path.exists()`), `urllib.parse.urlparse()` (parse a URL string into components; result has `.netloc` for the domain, e.g. `"reuters.com"`, and `.path` for the URL path component — used in Common Crawl URL filtering), `datetime.strptime()`, `datetime.fromisoformat()`, `datetime(year, month, day)` (datetime constructor), `datetime.strftime()`, `datetime.timedelta`, `timedelta.days` (`.days` integer attribute on a `datetime.timedelta` object — extracts the whole-day component from a time-delta, e.g. `(date2 - date1).days` to get an integer day count for exponential decay weighting in the Common Crawl pipeline — distinct from `timedelta.total_seconds()` which converts the full delta to float seconds), `datetime.date.today()` (returns today's date as a `datetime.date` object), `date.isoformat()` (converts a `datetime.date` to ISO-format string, e.g. `"2026-06-30"`), `random.seed()`, `random.shuffle()`, `random.randrange()`, `random.choice()`, `random.choices()`, `random.choices(population, weights=[...])` (weighted random sampling with `weights=` kwarg), `random.randint()`, `random.random()`, `random.gauss()`, `statistics.mode()`, `warnings.filterwarnings()`, `io.BytesIO`, `os.path.exists()`, `os.path.getsize()`

**Python Builtins:**
`isinstance()`, `hasattr()`, `setattr()`, `getattr()` (retrieve an attribute value by name, e.g. `getattr(model, layer_name)` to dynamically access a named submodule or attribute — complement to `hasattr()` which checks existence and `setattr()` which sets), `max(iterable, key=...)`, `max(iterable, key=dict_obj.get)` (using `dict.get` as a key callable, e.g. in BPE merge scoring), `min(iterable, key=...)`, `sorted(iterable, key=..., reverse=True)`, `property` (decorator), `classmethod` (decorator), `str.replace(old, new)`, `str.split()`, `str.join(iterable)` (e.g. `' '.join(tokens)`), `str.lower()`, `str.strip()`, `str.isalpha()`, `str.islower()`, `str.isupper()`, `str.isdigit()`, `str.startswith()`, `tuple(iterable)`, `set()`, `set.add(element)` (`.add()` method on a Python set — inserts one element in-place, e.g. `infected.add(neighbor)` in BFS propagation across agent topologies — distinct from `set()` which constructs and from `list.append()` which appends to a list), `iter()` (Python builtin for getting an iterator from an iterable, e.g. `iter(dataloader)` to step through batches manually — distinct from the iterable itself), `next()`, `next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if condition)` (generator-expression argument to `next()` for finding the first matching ancestor path — e.g. `next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / ".git").exists())` to locate the repo root; the `*pathlib.Path.cwd().parents` starred-unpack inside a list literal flattens the parents PurePath sequence alongside the cwd itself; used in jacobian-lens, tdkps, control-charts, and multi-agent-control session-log cells), `enumerate()`, `zip()`, `len(sequence)` (Python builtin returning the number of elements — works on lists, tensors (first dim), strings, and datasets; e.g. `len(sorted_scores)` for iteration bounds or `len(phi)` for trajectory length in control-chart notebooks — distinct from `.numel()` which gives total elements in a tensor across all dims), `float('nan')` (explicit NaN sentinel value), `float("-inf")` (string-form negative infinity, used as attention mask fill value), `float(x)` (Python builtin float() cast — converts a numpy scalar, tensor `.item()` result, or other numeric value to a Python float, e.g. `float(sim_score)` or `float(b[0])` to get a plain Python scalar for plotting; distinct from `.item()` which is a tensor method and `.float()` which keeps the result as a tensor), `int(x)` (Python builtin int() cast — truncates or converts a float/numpy/tensor scalar to a Python integer, e.g. `int(round((1 - AUDIT_BUDGET) * n))` for index arithmetic or `int(tensor.nonzero()[0])` to convert an index tensor to a scalar int; distinct from `.int()` tensor method and `math.ceil()`), `list.index(element)` (find index of first occurrence in a list, e.g. `col_labels.index("Direct Request")`), `list.pop(index)` (remove and return element at index, e.g. `word_list.pop(randrange(len(word_list)))` in board generation), `list.append(element)` (append an element to the end of a list in place, e.g. `merges.append(best)` in BPE to record merge operations), `dict.get(key, default)` (safe lookup with fallback, e.g. `counts.get(r, 0)` in Cohen's kappa computation), `dict.pop(key)` (remove and return a key's value, e.g. `enc.pop("offset_mapping")` to strip before forwarding to model), `dict.items()` (key-value pair iterator, e.g. `{k: v.to(device) for k, v in enc.items()}`), `dict.values()` (value iterator, e.g. `list(cat_asr.values())`), `dict.fromkeys(iterable)` (create a dict keyed by the iterable, deduplicating while preserving insertion order; `list(dict.fromkeys(seq))` is the idiomatic ordered-dedup pattern, e.g. `list(dict.fromkeys(decoded))[:20]` in Codenames board generation), `str.format(key=value)` (keyword-argument template interpolation, e.g. `TAG[role].format(c=content)` to fill named slots in a format string — distinct from f-strings), `str.index(substring)` (returns position of first occurrence of substring, raises `ValueError` on miss — distinct from `list.index` which operates on list elements), `any(iterable)` (Python builtin returning True if any element is truthy — distinct from `torch.any()` function and `.any()` tensor method), `all(iterable)` (Python builtin returning True if ALL elements are truthy — distinct from `torch.all()` function and `.all()` tensor method, e.g. `all(b.is_finished for b in beams)` to check beam search termination), `sum(iterable)` (Python builtin summing an iterable — distinct from `.sum()` tensor method and `torch.sum()` function), `abs(x)` (Python builtin absolute value — e.g. `abs(dist.sum() - 1.0) < 1e-6` to check normalization or `abs(a - b)` for scalar difference; distinct from `tensor.abs()` which operates on tensors and `np.abs()` for arrays), `round(x)` (Python builtin for rounding a number to the nearest integer, e.g. `int(round((1 - AUDIT_BUDGET) * n))` when computing integer indices from float fractions), `open(path, mode)` (Python builtin for opening a file, e.g. `open(root / ".practice-log.jsonl", "a")` with append mode — distinct from `pathlib.Path.open()` which is the method form), `file.write(string)` (write method on a file object, e.g. `f.write(json.dumps(record) + "\n")` inside a `with open(...) as f:` block)

**NumPy:**
`np.matmul(A, B)` / `A @ B` (numpy matrix multiply operator — e.g. `embed @ embed.T` to compute a full cosine similarity matrix from normalized embeddings in Codenames; distinct from PyTorch's `@` / `torch.matmul()` which operates on tensors), `np.ascontiguousarray`, `np.transpose`, `np.stack(arrays_list)` / `np.stack(arrays_list, axis=N)` (stack a sequence of arrays along a NEW axis — default `axis=0` produces shape `(n, ...)` from n arrays of shape `(...)`; distinct from `np.vstack()` which concatenates along the existing first axis without creating a new dimension, e.g. `np.stack(x).mean(axis=0)` in a groupby-apply to build per-ticker mean embeddings from a Series of 1-D arrays), `np.argsort`, `np.argsort(arr)[::-1]` (reverse-argsort via slice to get descending-order indices — distinct from `np.sort(arr)[::-1]` which returns sorted values), `np.argmax`, `np.argmin`, `np.argpartition`, `np.partition`, `np.zeros`, `np.zeros_like()` (create a zero array with the same shape and dtype as an existing array — distinct from `np.zeros()` which takes an explicit shape), `np.ones`, `np.array`, `np.mean`, `np.std`, `np.min`, `np.max`, `np.sum`, `np.abs`, `np.percentile`, `np.exp`, `np.log`, `np.log2`, `np.log10`, `np.cos`, `np.sin`, `np.degrees()` (convert radians to degrees, e.g. `np.degrees(angles)` for polar axis tick labels in radar charts — used with `ax.set_thetagrids(np.degrees(angles), labels)` in safety-eval notebooks), `np.average`, `np.linspace`, `np.concatenate`, `np.column_stack`, `np.clip`, `np.sort`, `np.sort(arr, axis=1)[:, ::-1]` (reverse-sort along an axis via NumPy slice `[::-1]`), `np.meshgrid`, `np.random.seed`, `np.random.rand`, `np.random.randn`, `np.random.choice`, `np.random.choice(arr, size=n, replace=True)` (bootstrap resampling with `replace=True`), `np.random.choice(arr, size=n, replace=False)` (sampling without replacement, e.g. drawing non-repeating negative samples in contrastive learning), `np.random.randint`, `np.random.normal`, `np.random.shuffle`, `np.random.RandomState(seed)`, `np.linalg.eigh`, `np.linalg.eig` (general eigendecomposition, non-symmetric matrices), `np.linalg.norm()`, `np.linalg.qr`, `np.trace`, `np.repeat`, `np.tile`, `np.arange`, `np.empty`, `np.random.permutation`, `np.append()`, `np.pi` (π constant), `np.isfinite()` (element-wise finiteness check, e.g. `np.isfinite(arr).all()` to validate activations contain no NaN/inf), `np.vstack()` (stack arrays vertically / row-wise, e.g. stacking per-example feature arrays into a 2-d matrix), `np.quantile()` (compute quantiles of an array, e.g. `np.quantile(neutral_scores[:, i], 1 - target_fpr)` for threshold tuning — distinct from `np.percentile` which takes percentage values 0–100 instead of fractions 0–1), `np.linspace(..., endpoint=False)` (`endpoint=False` kwarg to exclude the stop value, e.g. `np.linspace(0, 2*np.pi, N, endpoint=False)` for evenly-spaced radar plot angles that don't double-count 0 and 2π), `ndarray.astype(dtype)` (convert array element dtype in-place, e.g. `(xx * 255).astype(np.uint8)` to produce a uint8 array for `PIL.Image.fromarray()` — same semantics as constructing a new array with the target dtype)

**Pandas:**
`pd.read_csv`, `pd.read_json()`, `pd.DataFrame`, `pd.to_datetime()`, `pd.concat()` (concatenate a list of DataFrames along an axis, e.g. `pd.concat([df1, df2], ignore_index=True)`), `df.groupby().agg()`, `df.groupby("col")["col2"].agg(["mean", "std", "count"])` (single-column selection after groupby with a list of function strings — returns a DataFrame with one column per function, e.g. `df.groupby("quintile")["return"].agg(["mean", "std", "count"])` for per-quintile return stats; distinct from the named agg form below), `df.groupby("col").agg(output_name=("source_col", func))` (named aggregation syntax — keyword-argument form that renames output columns inline, e.g. `agg(n_headlines=("headline", "count"), mean_sentiment=("sentiment_numeric", "mean"))`), `df.groupby("col").rolling(N).agg(...)` (rolling window aggregation chained on a groupby result — e.g. `df.groupby("ticker").rolling(2).agg({"sentiment": "mean"})` for rolling per-ticker averages; distinct from `.agg()` directly on groupby which reduces each group to a scalar), `df.groupby().apply()`, `df.groupby("col")["col2"].apply(lambda x: np.stack(x).mean(axis=0))` (array-valued groupby aggregation — apply a function that stacks numpy arrays from each group and averages them, e.g. for aggregating per-ticker embedding arrays into a mean embedding; distinct from the string-aggregation forms using `agg()`), `.reset_index()`, `.dropna()`, `df["col"].map()`, `df["col"].apply()`, `df["col"].nunique()`, `df["col"].std()`, `df["col"].count()`, `df["col"].min()`, `df["col"].max()`, `df["col"].tolist()`, `df.to_csv()`, `df.diff()`, `pd.qcut()`, `pd.qcut(series, q=N, labels=[...]).astype(int)` (`labels=` kwarg to specify bin category labels, e.g. integer labels `[0, 1, 2, 3, 4]` for ordered quintile bins — distinct from the default which creates `Interval` category labels; `.astype(int)` converts the resulting Categorical to plain integers), `pd.cut()`, `df["col"].fillna()`, `df.copy()`, `df.columns` (assignment), `df["col"].rank(pct=True)`, `df.to_string(index=False)`, `df.sort_values()`, `df.describe()`, `df.head()`, `df.tail()`, `df.loc[]`, `df.iloc[]`, `df.astype()`, `df.values` (underlying numpy array)

**Visualization:**
`plt.bar()`, `plt.subplots()`, `plt.figure()`, `plt.savefig(path, dpi=N, bbox_inches='tight')` (`dpi=` controls pixel density; `bbox_inches='tight'` trims surrounding whitespace — both commonly used when saving figures for publication or reports, e.g. `plt.savefig("fig.png", dpi=150, bbox_inches='tight')`), `plt.colorbar()`, `plt.tight_layout()`, `plt.rcParams.update()`, `plt.imshow()`, `plt.imshow(..., alpha=float, cmap='colormap_name')` (`alpha=` blends the image over a background at the given opacity, e.g. `plt.imshow(heatmap, alpha=0.5, cmap='jet')` for Grad-CAM heatmap overlays — `cmap=` selects the colormap string; distinct from `ax.imshow()` which is the Axes-level method), `plt.show()`, `plt.legend()`, `plt.title()`, `plt.xlabel()`, `plt.ylabel()`, `plt.xticks()`, `plt.grid()`, `plt.suptitle()`, `plt.suptitle(..., y=float)` (`y=` positions the super-title vertically as a fraction of the figure height — e.g. `plt.suptitle("title", fontsize=14, y=1.02)` to push it above subplot axes and prevent overlap with `plt.tight_layout()`), `plt.axis("off")`, `plt.close()`, `plt.scatter()`, `plt.plot()`, `plt.barh()`, `plt.hist()`, `matplotlib.use("Agg")`, `matplotlib.patches.Patch` (legend patch elements), `plt.cm.tab10`, `plt.cm.Set3` (colormap objects), `ax.plot()`, `ax.fill()`, `ax.scatter()`, `ax.imshow()`, `ax.annotate()`, `ax.annotate(text, xy=(...), xycoords='axes fraction')` (`xycoords='axes fraction'` positions the annotation at a fixed fraction of the axes (0–1), independent of data coordinates — used for corner labels and subplot titles), `ax.annotate(..., bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"))` (`bbox=` adds a styled FancyBboxPatch background behind the annotation, e.g. rounded white box on MFA centroid labels), `ax.annotate("", xy=..., xytext=..., arrowprops=dict(arrowstyle="->", color=c, lw=1.1, alpha=0.85))` (the `arrowprops=` kwarg draws an arrow from `xytext` to `xy` — pass `""` as the text argument when only the arrow is needed, e.g. `ax.annotate("", xy=(float(b[0]), float(b[1])), xytext=(float(a[0]), float(a[1])), arrowprops=dict(arrowstyle="->", color=c, lw=1.1, alpha=0.85))` for per-agent drift arrows in TDKPS perspective-space trajectory plots), `ax.text()`, `ax.legend()`, `ax.legend(frameon=False)` (`frameon=False` kwarg to suppress the legend box border, e.g. for cleaner plots in MFA and SAE notebooks — common aesthetic choice alongside `sns.despine()`), `ax.set_title()`, `ax.set_xlabel()`, `ax.set_ylabel()`, `ax.set_xlim()`, `ax.set_ylim()`, `ax.set_xticks()`, `ax.set_xticklabels()`, `ax.set_yticklabels()` (set y-axis tick labels, e.g. `ax.set_yticklabels(row_labels)` for seaborn heatmap row annotations — symmetric counterpart to `ax.set_xticklabels()`), `ax.spines[...].set_visible()`, `ax.axhline()`, `ax.axvline()`, `ax.axvline(value, color=c, linestyle="--", label="...")` / `ax.axhline(value, color=c, linestyle="--", label="...")` (`linestyle=` kwarg controls dash pattern, e.g. `"--"` for dashed reference lines; `label=` adds the line to the legend — both kwargs work identically on `axhline`/`axvline`, e.g. `ax.axvline(threshold, color="C2", linestyle="--", label="threshold")` in SAE atom-arc plots), `ax.axhline(y, linewidth=N)` / `ax.axvline(x, linewidth=N)` (`linewidth=` kwarg sets the stroke width of the reference line in points, e.g. `ax.axhline(y=n_image-0.5, color='black', linewidth=2)` and `ax.axvline(x=n_image-0.5, color='black', linewidth=2)` to draw thick region-boundary lines on the VLM attention mask heatmap — distinct from the `linestyle=` and `label=` kwargs already documented), `ax.bar()`, `ax.barh()`, `ax.grid()`, `ax.grid(axis='y', alpha=N)` (`axis='y'` restricts gridlines to the y-axis only, e.g. `ax.grid(axis='y', alpha=0.3)` for horizontal-only reference lines in bar/distribution plots; `alpha=` sets transparency — distinct from `ax.grid(True)` which draws both axes), `ax.transAxes` (transform for relative positioning), `sns.heatmap()`, `sns.heatmap(..., annot=True, fmt=".2f")` (`annot=True` overlays cell values; `fmt=` controls display format, e.g. `fmt=".2f"` for two-decimal floats in campaign heatmaps), `sns.heatmap(..., cmap="RdYlGn_r")` (`cmap=` kwarg selects matplotlib colormap string; `_r` suffix reverses direction, e.g. `"RdYlGn_r"` makes red=high=bad for safety heatmaps), `sns.scatterplot()`, `sns.lineplot()`, `sns.despine()`, `sns.color_palette()`, `sns.set_palette()`, polar axes: `fig.add_subplot(polar=True)` (create polar axis), `ax.set_theta_offset()`, `ax.set_theta_direction()`, `ax.set_rlabel_position()`, `ax.set_thetagrids(angles, labels)` (set theta grid lines and tick labels on a polar axis, e.g. for radar charts — distinct from `ax.set_xticks()` which is for Cartesian axes), `ax.hist()`, `ax.hist(..., bins=N, edgecolor="black")` (`bins=N` integer bin count and `edgecolor=` border color kwargs on `ax.hist()` — e.g. `ax.hist(vals.cpu().numpy(), bins=20, edgecolor="black")` for activation distribution plots in SAE notebooks), `ax.imshow(..., aspect="auto")` (`aspect` kwarg to disable equal-axis locking, e.g. for non-square activation grids), `ax.imshow(..., extent=[xmin, xmax, ymax, ymin])` (`extent` kwarg to set axis coordinate ranges, e.g. `extent=[0, 2*math.pi, M, 0]` to label atom × angle heatmaps with real units), `ax.imshow(..., vmin=N, vmax=N)` (`vmin=`/`vmax=` clip the colormap range to [vmin, vmax], e.g. `ax.imshow(attn_weights, cmap='Blues', vmin=0, vmax=0.5)` to normalize attention heatmaps to a fixed scale — useful when comparing attention weights across layers), `ax.text(..., verticalalignment='top')` (`verticalalignment` / `va=` kwarg controlling vertical anchor of text annotations — `'top'` pins the top of the text bounding box to the y coordinate), `ax.text(..., va='center')` (the `'center'` value for `va=` / `verticalalignment=` — centers text vertically on the y coordinate; used alongside `rotation=90` for vertical axis labels), `ax.text(..., ha='center')` / `ax.text(..., horizontalalignment='center')` (`ha=` / `horizontalalignment=` kwarg controlling horizontal anchor of text annotations — `'center'` centers text on the x coordinate, e.g. `ax.text(n_image/2 - 0.5, -0.8, 'Image', ha='center', ...)` in VLM mask region labels), `ax.text(..., rotation=N)` (`rotation=` kwarg rotating the text string N degrees, e.g. `rotation=90` for vertical axis labels on the VLM attention mask heatmap), `ax.text(..., fontweight='bold')` (`fontweight=` kwarg for font weight, e.g. `fontweight='bold'` to emphasize region labels in attention mask visualizations), `ax.text(..., fontsize=N)` (`fontsize=` kwarg controlling the font size of text annotations in points, e.g. `ax.text(x, y, label, fontsize=8)` for small per-cell labels in heatmaps or compact scatter annotations — distinct from `fontweight=` which controls weight and from `plt.rcParams.update({'font.size': N})` which sets a global default), `ax.scatter(..., c=data_array, cmap='colormap_name')` (`c=` maps a numeric data array to point colors via the named colormap; returns a `PathCollection` mappable — distinct from `color=` which applies a single fixed color; e.g. `ax.scatter(x, y, c=df["quintile"], cmap="RdYlGn")` for return-quintile coloring), `ax.scatter(..., edgecolors='black', linewidth=0.3, s=60)` (`edgecolors=` sets the marker border color; `linewidth=` sets the border stroke width; `s=` sets the marker area in points² — distinct from `markersize=` used in `ax.plot()`; commonly used together with `c=`/`cmap=` for styled scatter plots), `ax.scatter(..., alpha=float)` (`alpha=` sets per-point transparency 0–1, e.g. `alpha=0.7` in PCA scatter plots to improve readability of overlapping points), `plt.colorbar(mappable, label='str')` (passing the return value of `ax.scatter(c=..., cmap=...)` as the first positional argument ties the colorbar to that specific plot element; `label=` kwarg sets the colorbar axis label — e.g. `plt.colorbar(scatter, label="Return Quintile")`), `plt.colorbar(mappable, ax=ax, shrink=float)` (`ax=` pins the colorbar to a specific Axes object instead of stealing space from the current axes — essential in multi-panel figures; `shrink=` scales the colorbar's length as a fraction of the parent axes height, e.g. `plt.colorbar(im, ax=ax, shrink=0.8)` in VLM and attention heatmap notebooks), `sns.scatterplot(..., hue=list_or_series)` (`hue=` kwarg maps a categorical sequence to distinct colors with an automatic legend, e.g. `sns.scatterplot(x=X[:,0], y=X[:,1], hue=[d["ticker"] for d in data])`)

**Scientific Python:**
`sklearn.decomposition.PCA`, `pca.fit()`, `pca.fit_transform()`, `pca.transform()`, `pca.explained_variance_ratio_`, `sklearn.metrics.roc_auc_score`, `sklearn.metrics.silhouette_score`, `sklearn.metrics.cohen_kappa_score`, `sklearn.metrics.pairwise.cosine_similarity`, `sklearn.metrics.pairwise_distances` (general pairwise distance matrix, e.g. `pairwise_distances(X, metric='euclidean')` — more general than `cosine_similarity`; used in TDKPS as a numpy-based alternative to `torch.cdist`), `sklearn.preprocessing.StandardScaler`, `scaler.fit_transform()`, `sklearn.linear_model.LogisticRegression`, `LogisticRegression(max_iter=N)` (`max_iter=` kwarg controlling maximum solver iterations — required to avoid `ConvergenceWarning` when fitting probes on high-dimensional activations, e.g. `LogisticRegression(max_iter=1000)`), `probe.fit()`, `probe.predict_proba()`, `probe.score(X, y)` (accuracy of fitted probe on a labelled array), `probe.coef_` (fitted weight matrix of shape `(n_classes, n_features)`; `probe.coef_[0]` for binary classification), `probe.classes_` (array of class labels in the order the probe learned them, e.g. `probe.classes_.tolist().index(COT)` to find the column index for a specific class in `predict_proba` output), `sklearn.model_selection.cross_val_score`, `scipy.stats.f_oneway`, `scipy.stats.ttest_ind`, `scipy.linalg.eigh()`, `scipy.linalg.sqrtm()` (matrix square root via Schur factorization — not a decomposition but the true square root of a matrix, e.g. `scipy.linalg.sqrtm(cov_g @ cov_r)` for the Fréchet distance computation in GLP meta-model), `scipy.cluster.hierarchy.linkage`, `scipy.cluster.hierarchy.linkage(embeddings, method='average', metric='cosine')` (`method=` and `metric=` kwargs for linkage-type and distance measure, e.g. average-linkage with cosine distance for embedding clusters), `scipy.cluster.hierarchy.linkage(arr, method='ward')` (Ward linkage — omits `metric=` since Ward requires Euclidean distance; used e.g. in MFA centroid clustering — distinct from `method='average', metric='cosine'` which is for non-Euclidean feature spaces), `scipy.cluster.hierarchy.fcluster`, `scipy.cluster.hierarchy.fcluster(Z, t=k, criterion='maxclust')` (`criterion='maxclust'` kwarg to cut the dendrogram into exactly k flat clusters), `scipy.sparse.csgraph.laplacian(adj)` (compute the graph Laplacian matrix from an adjacency matrix, e.g. for spectral graph features in company knowledge graphs), `scipy_sparse_matrix.toarray()` (convert a scipy sparse matrix to a dense numpy array — returned by `nx.laplacian_matrix(G)`, `nx.normalized_laplacian_matrix(G)`, and `scipy.sparse.csgraph.laplacian()` which all yield sparse format; call `.toarray()` before passing to numpy/sklearn operations, e.g. `nx.laplacian_matrix(G).toarray()` in knowledge-graph feature extraction), `requests.get()`, `requests.get(url, headers=dict)` (`headers=` kwarg to pass custom HTTP headers, e.g. `headers={"User-Agent": "Name email@email.com"}` required by the SEC EDGAR API — without it the API returns 403), `response.json()` (parse the JSON body of a `requests.Response` — e.g. `requests.get(url, headers=...).json()` to decode EDGAR API responses), `response.text` (raw string body of a `requests.Response`, e.g. for scraping HTML or plain-text filings)

**networkx:**
`nx.Graph()`, `nx.DiGraph()` (directed graph), `G.add_edge()`, `G.add_node()`, `nx.from_pandas_edgelist()`, `nx.from_dict_of_lists()`, `nx.pagerank()`, `nx.betweenness_centrality()`, `nx.clustering()`, `nx.degree_centrality()`, `nx.closeness_centrality()`, `nx.density()`, `nx.community.greedy_modularity_communities()`, `nx.community.louvain_communities()` (Louvain community detection — alternative to greedy modularity for larger graphs), `nx.connected_components()`, `nx.spring_layout()`, `nx.spectral_layout()`, `nx.draw(G, pos=..., node_color=..., with_labels=True)` (simple draw wrapper — distinct from `nx.draw_networkx()` which exposes all artist kwargs), `nx.draw_networkx()`, `nx.draw_networkx_nodes()`, `nx.draw_networkx_edges()`, `nx.draw_networkx_labels()`, `nx.draw_networkx_edge_labels()` (draw edge weight or label annotations on graph edges, e.g. for co-mention counts in company knowledge graphs — distinct from `nx.draw_networkx_labels()` which labels nodes), `nx.laplacian_matrix()`, `nx.to_numpy_array()`, `G.degree(weight=...)` (weighted degree dict), `G.neighbors(n)`, `G.number_of_nodes()`, `G.number_of_edges()`, `nx.path_graph(N)` (create a path/chain graph of N nodes — analytical chain topology gives steps=N-1 for BFS saturation), `nx.star_graph(N-1)` (create a star graph with node 0 as hub and N-1 leaves — infection from a leaf saturates in exactly 2 steps), `nx.erdos_renyi_graph(N, p, seed=...)` (generate an Erdős-Rényi random graph with edge probability `p` and reproducible `seed=` kwarg, e.g. 50-agent p=0.1 to reproduce <11-step saturation result), `nx.is_connected(G)` (check if an undirected graph is connected, e.g. `assert nx.is_connected(er)` to validate a test fixture before BFS diagnostics), `nx.eigenvector_centrality(G, weight="weight")` (eigenvector centrality — a node's importance is proportional to the sum of its neighbors' importances; `weight=` kwarg uses edge weights), `nx.normalized_laplacian_matrix(G)` (normalized Laplacian matrix where diagonal entries are 1 — distinct from `nx.laplacian_matrix()` which gives the unnormalized `D - A` form; used for spectral features with bounded eigenvalues in [0, 2])

**LightGBM:**
`lightgbm.LGBMRegressor` (gradient boosted tree regressor, e.g. for predicting returns from text embedding features in Numerai pipelines), `lgbm.fit(X_train, y_train)`, `lgbm.predict(X_test)`, `lgb.Dataset(X, label=y)` (LightGBM Dataset wrapper for the functional training API — wraps feature matrix `X` and label vector `y` into a dataset object; distinct from the scikit-learn API), `lgb.train(params, train_data, num_boost_round=...)` (functional training entry point — takes a `params` dict and an `lgb.Dataset`; distinct from `LGBMRegressor.fit()`)

**NLTK:**
`nltk.download()`, `nltk.download("words", quiet=True)` (`quiet=True` kwarg — suppresses download progress output, used when running notebooks in automated or test contexts, e.g. `nltk.download("punkt", quiet=True)`), `nltk.corpus.words.words()`, `nltk.corpus.wordnet.synsets()`

**PEFT (Hugging Face):**
`peft.LoraConfig`, `peft.TaskType`, `peft.TaskType.CAUSAL_LM` (enum member for causal language model tasks — passed as `task_type=peft.TaskType.CAUSAL_LM` in `LoraConfig` when fine-tuning decoder-only LLMs), `peft.get_peft_model()`, `model.print_trainable_parameters()`

**nnsight:**
`LanguageModel(model_name, device_map=..., dispatch=True)`, `model.trace(inputs)`, `model.trace(inputs, attention_mask=mask)` (trace with keyword args, e.g. in GLP/MFA notebooks), `layer.output.save()`, `nnsight.save()`, `model.tokenizer` (attribute), `model.lm_head` (attribute), `model.model` (underlying PyTorch model attribute), `model.model.layers[idx]` (indexed layer access for activation extraction), `model.model.layers[idx].output.save()` (save hidden state at specific layer)

**nnterp:**
`StandardizedTransformer(model_name, device_map=..., dispatch=True)`

**FAISS:**
`faiss.IndexFlatIP`

**tkinter:**
`tk.Tk()`, `tk.Button`, `tk.Label`, `tk.Frame`, `tk.Canvas()`, `tk.StringVar()` (tkinter observable string variable, used to bind widget text to a Python variable), `root.title(title)` (set the window title string on a Tk root window), `root.mainloop()`, `root.after(ms, callback)` (schedule a callback after `ms` milliseconds without blocking the event loop — tkinter's non-blocking delay mechanism), `widget.config()`, `widget.grid()`, `widget.pack()`, `widget.bind()` (event binding), `tk.messagebox.showinfo(title, message)` (display a simple info pop-up dialog — used in Codenames GUI to announce the winner), `tk.DISABLED` / `tk.NORMAL` (widget state constants passed to `widget.config(state=tk.DISABLED)` or `widget.config(state=tk.NORMAL)` to grey out / re-enable buttons)

**tqdm:**
`tqdm.auto.tqdm`

**PIL/Pillow:**
`PIL.Image.open()`, `PIL.Image.fromarray()`, `image.resize(size, resample=Image.BILINEAR)`

**jaxtyping:**
`Float[Tensor, "batch seq d_model"]` (runtime shape annotations), `jaxtyping.Float`, `Int[Tensor, "n"]` (integer tensor type annotation, e.g. for token index tensors), `jaxtyping.Int`, `Bool[Tensor, "T"]` (boolean tensor type annotation, e.g. for alarm masks — `adaptive_chart` returns `Bool[Tensor, "T"]` in the control-charts notebook), `jaxtyping.Bool`

**Web/Data Collection:**
`warcio.ArchiveIterator()`, `trafilatura.extract()`, `BeautifulSoup(html, "html.parser")` (beautifulsoup4 — parse an HTML string), `BeautifulSoup(...).get_text()` (extract all visible text from a parsed HTML document, stripping tags), `soup.find("title").text` (find the first matching tag and read its `.text` attribute, e.g. title extraction in Common Crawl pipeline)
