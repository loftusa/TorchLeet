# Narration script — "Refusal in Language Models Is Mediated by a Single Direction"

Voiceover text for the manim video (`refusal_direction_full.mp4`), timed to the six scenes.
Total runtime ≈ 2m15s. Paper: Arditi et al., 2024 — arXiv:2406.11717.

## Scene 1 — Intro (~23s)

> Every chat model you've used has been trained to refuse. Ask it how to pick a
> lock, and it declines. Ask it how to bake a pie, and it happily complies.
> Somewhere between those two responses, the model made a decision. The question
> this paper asks is: where, inside the network, does that decision live?
> And the answer turns out to be almost absurdly simple: refusal is mediated by
> one single direction in activation space.

## Scene 2 — The residual stream (~19s)

> First, a bit of anatomy. As a prompt flows through a transformer, every layer
> reads from and writes to a shared channel called the residual stream. At each
> layer, and at each token position, the stream holds one vector — about four
> thousand dimensions for a 7-billion-parameter model. If refusal is computed
> anywhere, it has to pass through here. So let's go looking for it in
> activation space.

## Scene 3 — Difference in means (~21s)

> The recipe is the simplest thing you could try. Run a batch of harmful prompts
> through the model and record the activations. Do the same for harmless
> prompts. Each cluster has a mean — mu for harmful, nu for harmless — and the
> arrow between them, r equals mu minus nu, is the candidate "refusal
> direction." You get one candidate per layer and token position; a small
> validation set picks the single best one.

## Scene 4 — Directional ablation (~27s)

> Now the causal test. Take any activation x, measure its component along
> r-hat, and subtract it off. Geometrically, this projects x onto the
> hyperplane orthogonal to the refusal direction. Do this at every layer and
> every token position, and the model becomes incapable of representing the
> direction at all. The result: all thirteen chat models tested comply with
> harmful requests — no fine-tuning, no adversarial prompts, just a projection.

## Scene 5 — Activation addition (~13s)

> And it works in reverse. Add the direction back in — at just one layer — and
> the model refuses everything, even a request to bake a pie. Removing the
> direction bypasses refusal; adding it induces refusal. That two-way control
> is what makes r causal, not just a correlate.

## Scene 6 — Weight orthogonalization + closing (~32s)

> One last trick. Instead of intervening at inference time, take every matrix
> that writes into the residual stream and orthogonalize it against r-hat:
> W prime equals W minus r-hat r-hat-transpose W. Mathematically identical to
> ablation — but now it's just an ordinary set of weights.
>
> So that's the whole method. A difference of two means finds the direction.
> Subtracting it removes refusal. Adding it induces refusal. And one rank-one
> weight edit makes it permanent. One direction, found with a handful of
> prompts, removed with a vector subtraction — which tells us something
> uncomfortable about how shallow current safety fine-tuning really is.
