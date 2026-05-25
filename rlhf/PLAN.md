# RLHF Interview Prep: 10-Day Plan

**Goal**: Build enough depth in RL/RLHF internals to reason credibly about training pipeline tradeoffs, RL environment operational health, and compute planning — the core gaps for the Anthropic TPM Research role.

**Format**: One 30-minute `/newproblem` session per day. Each builds on the previous day.

**Prerequisite knowledge you already have**: LLM forward passes, loss functions (KL divergence, cross-entropy), attention, LoRA, training loops, mixed precision.

---

## Day 1: Reward Model
**Topic**: Train a reward model that scores (prompt, completion) pairs.
**Why first**: The reward model is the bridge between your LLM knowledge and RL. It's just a classifier on top of a language model — familiar territory. Everything downstream depends on understanding what the reward signal looks like.
**Core skills**: Last-token pooling, pairwise ranking loss (Bradley-Terry), handling chosen/rejected pairs.
**Interview relevance**: "How does the reward model affect training stability?" is a common question. Reward hacking, overoptimization, and reward model quality are major operational concerns.

## Day 2: Policy Gradient Fundamentals (REINFORCE)
**Topic**: Implement REINFORCE on a toy text generation task.
**Why**: PPO is just REINFORCE with variance reduction and clipping. You need the base algorithm first. This is where RL intuition gets built — why gradients are noisy, why baselines help, what the log-probability trick does.
**Core skills**: Log-prob extraction from LLM outputs, advantage estimation, policy gradient loss.
**Interview relevance**: Understanding *why* RL training is unstable (high variance gradients) lets you reason about operational monitoring.

## Day 3: PPO Core — Clipped Surrogate Objective
**Topic**: Implement the PPO clipped objective and value loss.
**Why**: PPO is the workhorse of RLHF. The clipping mechanism is the key innovation — it prevents catastrophic policy updates. This is the most important single concept for understanding RLHF training dynamics.
**Core skills**: Probability ratios, clipped surrogate loss, value function loss, entropy bonus.
**Interview relevance**: "Why does the training run diverge?" almost always traces back to PPO hyperparameters (clip range, KL coefficient, learning rate).

## Day 4: GAE and the Full PPO Loop
**Topic**: Implement Generalized Advantage Estimation and wire up the complete PPO training loop.
**Why**: GAE controls the bias-variance tradeoff in advantage estimation. The full loop (generate → score → compute advantages → update) is what actually runs during RLHF training. You need to see the whole thing once.
**Core skills**: GAE (lambda-returns), minibatch PPO updates, multiple epochs per batch, KL penalty.
**Interview relevance**: "Walk me through what happens during one RLHF training step" — you should be able to trace data through every component.

## Day 5: KL Penalty and Reward Hacking
**Topic**: Implement KL-penalized reward and demonstrate reward hacking on a toy setup.
**Why**: The KL divergence between the policy and reference model is *the* central operational metric in RLHF. Too low = not learning. Too high = reward hacking / mode collapse. This is what the TPM would be monitoring.
**Core skills**: KL divergence between two LLM output distributions, adaptive KL controller, detecting reward hacking.
**Interview relevance**: "How do you know if a training run is going well?" — KL dynamics, reward curves, and generation quality are the answer.

## Day 6: DPO — Direct Preference Optimization
**Topic**: Implement DPO loss and training loop.
**Why**: DPO eliminates the reward model and PPO entirely — it directly optimizes the policy from preference pairs. It's simpler, more stable, and increasingly preferred. Understanding *why* it works (implicit reward model) and *when* it fails gives you a complete picture.
**Core skills**: DPO loss (log-ratio trick), reference model management, beta parameter.
**Interview relevance**: "When would you use DPO vs PPO?" is a live debate. Being able to articulate the tradeoffs (stability vs. flexibility, compute cost, online vs. offline data) is high-signal.

## Day 7: Compute Efficiency in RL Training
**Topic**: Implement generation-side optimizations: KV cache reuse during rollouts, gradient accumulation across RL minibatches, selective parameter freezing.
**Why**: RL training is ~4x more expensive than SFT because each step requires generation + scoring + training. Understanding where the compute goes is directly relevant to the "compute resource planning" responsibility.
**Core skills**: Memory profiling, generation vs. training compute split, gradient checkpointing in RL context.
**Interview relevance**: "Why is RLHF so expensive and what can we do about it?" — you should know the bottlenecks.

## Day 8: Eval for Alignment — Building an Eval Harness
**Topic**: Build a mini eval pipeline: run a model on a benchmark, compute pass rates, bootstrap confidence intervals, compare two checkpoints with statistical tests.
**Why**: "Drive eval readiness for model launches" is a core responsibility. You need to know what eval infrastructure looks like from the inside — not just running lm-eval-harness, but understanding what it's doing and why standardization matters.
**Core skills**: Batch inference, scoring functions, bootstrap CIs, paired significance tests, results reporting.
**Interview relevance**: "How would you standardize eval reporting across research teams?" — you should have opinions grounded in implementation experience.

## Day 9: Training Run Monitoring and Diagnostics
**Topic**: Implement a training monitor that tracks gradient norms, loss curves, KL divergence, reward statistics, and generation quality over a short PPO run. Detect anomalies (gradient explosion, reward hacking, mode collapse).
**Why**: "Operational health of RL environments across major training runs" means monitoring. You need to know what healthy vs. unhealthy training looks like and which metrics to watch.
**Core skills**: Gradient norm tracking, moving-average baselines, anomaly detection heuristics, W&B logging.
**Interview relevance**: "A training run is showing reward increasing but KL is spiking — what's happening?" You should diagnose this instantly.

## Day 10: End-to-End RLHF Pipeline
**Topic**: Wire together: SFT base → reward model → PPO training → eval → comparison against DPO baseline. All on a small model (SmolLM-135M or similar).
**Why**: Integration test for everything you've learned. The TPM role requires seeing how all the pieces connect and where things break at the seams. This is your capstone.
**Core skills**: Pipeline orchestration, checkpoint management, comparing training strategies, summarizing results for stakeholders.
**Interview relevance**: You can now walk through the entire RLHF pipeline from data to deployment, point to where operational risks live, and explain tradeoffs at each stage.

---

## How to Use This Plan

Each day, run:
```
/newproblem <topic from that day> 30m
```

After completing each problem, the spaced repetition system will schedule review reminders. When a review comes due, use:
```
/newproblem <topic> 10m
```
for a quick recall drill.

## What "Done" Looks Like

After 10 days you should be able to:
1. Whiteboard the full RLHF pipeline (data → RM → PPO → eval) with no notes
2. Explain why PPO training is unstable and what metrics to monitor
3. Articulate DPO vs PPO tradeoffs with concrete technical reasons
4. Reason about compute costs and where optimization effort should go
5. Describe what an eval pipeline needs to be launch-ready
6. Diagnose common training failures from metric patterns
