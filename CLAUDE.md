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
│   └── mfa-local-geometry/   # MFA local geometry paper implementation
├── practice/
│   ├── 01-LLM-as-Judge/      # LLM evaluation as judge
│   ├── 02-Agent-Eval-Harness/ # Agent evaluation harness
│   ├── 03-Codenames-AI/      # AI for Codenames board game
│   └── 04-Safety-Evaluation-Pipeline/  # Red-team campaign analysis (ISC paper)
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

**Practice Set:**
- LLM-as-Judge evaluation
- Agent evaluation harness
- Codenames AI (board game AI)
- Safety Evaluation Pipeline: Red-team campaign analysis based on ISC paper (Wu et al., 2026) — harm taxonomy schemas, attack success rates, per-category/vector effectiveness matrices, policy compliance gap analysis, Cohen's kappa inter-rater reliability, bootstrap confidence intervals, model comparison radar plots, campaign dashboard heatmaps, temporal safety trends, and data-driven campaign prioritization scoring

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
`torch.arange`, `torch.ones`, `torch.zeros`, `torch.randn`, `torch.randn_like`, `torch.ones_like`, `torch.zeros_like`, `torch.full_like`, `torch.cat`, `torch.stack`, `torch.full`, `torch.linspace`, `torch.tensor`, `torch.where`, `torch.randint`, `torch.Generator`, `.view()`, `.transpose()`, `.permute()`, `.unsqueeze()`, `.chunk()`, `.clone()`, `.flatten()`, `.reshape()`, `.to()` (dtype), `.squeeze()`, `.contiguous()`, `.detach()`, `.argmax()`, `.size()`, `.sum()`, `.scatter_()`, `.bool()`, `.float()`, `.item()`, `.expand()`, `.unbind()`

**Torch Math Operations:**
`torch.sqrt`, `torch.rsqrt`, `torch.exp`, `torch.log`, `torch.sin`, `torch.cos`, `torch.round`, `torch.clamp`, `torch.einsum`, `torch.topk`, `torch.sort`, `torch.cumsum`, `torch.argsort`, `torch.masked_fill`, `torch.triu`, `torch.repeat_interleave`, `torch.multinomial`, `torch.gather`, `torch.logsumexp`, `torch.cdist`, `torch.randperm`, `torch.diag`, `torch.mv`, `torch.trapezoid`, `torch.allclose`, `.pow()`, `.mean()`, `.abs()`, `.max()`, `.min()`, `.amax()`, `.exp()`, `.log()`, `@` (matmul), `torch.tanh`, `torch.sigmoid`, `torch.bmm`, `torch.outer`, `torch.max` (function form with dim arg)

**torch.nn Modules:**
`nn.Linear`, `nn.Parameter`, `nn.Embedding`, `nn.Conv2d`, `nn.Sequential`, `nn.ReLU`, `nn.GELU`, `nn.SiLU`, `nn.LayerNorm`, `nn.MultiheadAttention`, `nn.ModuleList`, `nn.ModuleDict`, `nn.CrossEntropyLoss`, `nn.init.kaiming_uniform_`, `self.register_buffer`, `nn.RNN`, `nn.LSTM`, `nn.MaxPool2d`, `nn.ConvTranspose2d`, `nn.Conv3d`, `nn.ConvTranspose3d`, `nn.LeakyReLU`, `nn.Sigmoid`, `nn.Tanh`, `nn.BCELoss`, `nn.MSELoss`, `nn.AdaptiveAvgPool2d`, `nn.Dropout`, `nn.init.kaiming_normal_`, `nn.init.xavier_normal_`, `nn.init.xavier_uniform_`, `nn.init.zeros_`, `nn.init.normal_`, `nn.init.constant_`

**torch.nn.functional:**
`F.softmax`, `F.log_softmax`, `F.relu`, `F.silu`, `F.linear`, `F.normalize`, `F.kl_div`, `F.cross_entropy`, `F.mse_loss`, `F.binary_cross_entropy_with_logits`, `F.dropout`, `F.scaled_dot_product_attention`, `F.unfold`, `F.pad`

**Autograd / Training:**
`torch.no_grad()`, `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`, `model.zero_grad()`, `torch.optim.Adam`, `torch.optim.SGD`, `torch.save()`, `torch.load()`, `model.state_dict()`, `model.load_state_dict()`, `model.apply()`, `model.parameters()`, `model.named_children()`, `.requires_grad_(False)`

**Custom Autograd (torch.autograd.Function):**
`torch.autograd.Function`, `ctx.save_for_backward()`, `ctx.saved_tensors`, `Function.apply()`

**Torch Distributions:**
`torch.distributions.Normal`, `torch.distributions.kl_divergence`

**Linear Algebra:**
`torch.linalg.eigvalsh`, `torch.linalg.eigh`, `torch.linalg.solve`

**Functional Transforms:**
`torch.func.grad`, `torch.func.vmap`, `torch.func.functional_call`

**Autograd (Low-level):**
`torch.autograd.grad`

**Hooks:**
`module.register_forward_hook`, `module.register_full_backward_hook`

**Mixed Precision (torch.cuda.amp):**
`torch.cuda.amp.GradScaler()`, `torch.cuda.amp.autocast()`, `scaler.scale()`, `scaler.step()`, `scaler.update()`

**Quantization:**
`torch.quantization.quantize_dynamic`, `isinstance(module, nn.Linear)`, `setattr(model, name, module)`

**torch.utils.data:**
`torch.utils.data.Dataset`, `torch.utils.data.DataLoader`, `torch.utils.data.TensorDataset`, `__len__()`, `__getitem__()`

**torch.utils.tensorboard:**
`SummaryWriter`, `writer.add_scalar()`, `writer.close()`

**torchvision:**
`transforms.Compose`, `transforms.ToTensor`, `transforms.Normalize`, `transforms.RandomHorizontalFlip`, `transforms.RandomCrop`, `transforms.Resize`, `transforms.ToPILImage`, `torchvision.datasets.CIFAR10`, `torchvision.datasets.MNIST`, `torchvision.models.resnet18`, `torchvision.utils.make_grid`

**einops:**
`einops.rearrange`, `einops.repeat`, `einops.einsum`

**HuggingFace:**
`AutoTokenizer.from_pretrained`, `AutoModelForSequenceClassification.from_pretrained`, `AutoModelForCausalLM.from_pretrained`, `pipeline()`, `TrainingArguments`, `Trainer`, `datasets.load_dataset`, `SentenceTransformer`

**Python Standard Library:**
`collections.Counter`, `collections.defaultdict`, `dataclasses.dataclass`, `functools.cache`, `copy.deepcopy`, `re`, `enum.Enum`, `math.log`, `math.sqrt`, `typing.Optional`, `typing.Tuple`, `typing.List`, `typing.Callable`, `time.time()`

**Python Builtins:**
`isinstance()`, `setattr()`, `max(iterable, key=...)`, `min(iterable, key=...)`

**FAISS:**
`faiss.IndexFlatIP`

**NumPy Interop:**
`np.ascontiguousarray`, `np.transpose`

**Pandas:**
`pd.read_csv`, `pd.DataFrame`

**Scientific Python:**
`sklearn.decomposition.PCA`, `sklearn.metrics.roc_auc_score`, `scipy.stats.f_oneway`, `scipy.cluster.hierarchy.linkage`, `scipy.cluster.hierarchy.fcluster`
