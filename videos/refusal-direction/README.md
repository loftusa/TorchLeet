# Refusal Direction — 3blue1brown-style explainer video

A manim video explaining **"Refusal in Language Models Is Mediated by a Single
Direction"** (Arditi et al., 2024 — [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)).

## Contents

- `refusal_direction.py` — six manim scenes (Manim Community v0.18):
  1. `S1_Intro` — the refusal phenomenon and the paper's claim
  2. `S2_ResidualStream` — activations as vectors in the residual stream
  3. `S3_DifferenceInMeans` — finding r = μ − ν from harmful/harmless prompt clusters
  4. `S4_Ablation` — directional ablation x′ = x − r̂r̂ᵀx, the projection geometry
  5. `S5_Addition` — activation addition x′ = x + r induces refusal (two-way causality)
  6. `S6_Orthogonalization` — weight orthogonalization W′ = W − r̂r̂ᵀW + method summary
- `NARRATION.md` — voiceover script timed to the scenes
- `build_narration.py` — ElevenLabs TTS pipeline: generates per-scene narration
  (Feynman/3b1b register, "Adam" voice), holds each scene's faded-out last frame
  until its narration finishes, and muxes everything together. Needs
  `ELEVENLABS_API_KEY` in the environment.
- `refusal_direction_full.mp4` — the silent assembled video (~2m15s, 1080p60)
- `refusal_direction_narrated.mp4` — the narrated version (~2m29s)

## Rebuild

```bash
manim -qh -a refusal_direction.py
# then concat the six scene mp4s in S1..S6 order, e.g. with ffmpeg -f concat
ELEVENLABS_API_KEY=... python3 build_narration.py   # narrated version
```

`media/` (manim's render cache), `audio/` (TTS clips), and `build/`
(intermediate muxes) are not checked in.
