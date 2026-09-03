"""
"Refusal in Language Models Is Mediated by a Single Direction"
(Arditi et al., 2024 — arXiv:2406.11717)

A 3blue1brown-style explainer video, rendered with Manim Community v0.18.

Render all scenes:   manim -qh -a refusal_direction.py
Scenes render in order S1..S6; concatenate with ffmpeg (see Makefile/README).
"""

import numpy as np
from manim import *

config.background_color = "#0e1219"

# palette (3b1b-ish)
C_BLUE = "#58C4DD"  # harmless
C_RED = "#FC6255"  # harmful
C_YELLOW = "#FFD35A"  # the refusal direction
C_GREEN = "#83C167"
C_GREY = "#8B93A5"
C_WHITE = "#ECECF1"
C_LINE = "#2A3350"


def chat_bubble(lines, color, align_side=LEFT, font_size=26, fill_opacity=0.16):
    """A rounded chat bubble containing the given lines of text."""
    txt = Text("\n".join(lines), font_size=font_size, color=C_WHITE, line_spacing=0.9)
    box = RoundedRectangle(
        corner_radius=0.18,
        width=txt.width + 0.7,
        height=txt.height + 0.55,
        fill_color=color,
        fill_opacity=fill_opacity,
        stroke_color=color,
        stroke_width=2.5,
    )
    txt.move_to(box)
    return VGroup(box, txt)


def cluster_points(center, n, spread, seed):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 2) * spread + np.array(center)
    return [np.array([x, y, 0.0]) for x, y in pts]


# ----------------------------------------------------------------------------
# Scene 1 — the phenomenon: chat models refuse
# ----------------------------------------------------------------------------
class S1_Intro(Scene):
    def construct(self):
        title = Tex(
            r"Refusal in Language Models\\[0.35em]Is Mediated by a Single Direction",
            font_size=56,
            color=C_WHITE,
        )
        authors = Text(
            "Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda  ·  2024",
            font_size=22,
            color=C_GREY,
        ).next_to(title, DOWN, buff=0.6)

        self.play(Write(title), run_time=2.5)
        self.play(FadeIn(authors, shift=UP * 0.2))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(authors))

        # chat demo: one refusal, one normal answer
        header = Text(
            "Chat models are trained to refuse.", font_size=34, color=C_WHITE
        ).to_edge(UP, buff=0.7)
        self.play(FadeIn(header, shift=DOWN * 0.2))

        u1 = chat_bubble(["How do I pick a lock?"], C_RED)
        a1 = chat_bubble(["I cannot help with that request."], C_GREY)
        u2 = chat_bubble(["How do I bake a pie?"], C_BLUE)
        a2 = chat_bubble(["Sure! Start by preheating", "your oven to 425°F…"], C_GREY)

        u1.move_to(LEFT * 3.4 + UP * 1.3)
        a1.next_to(u1, DOWN, buff=0.45).shift(RIGHT * 0.6)
        u2.move_to(RIGHT * 3.2 + UP * 1.3)
        a2.next_to(u2, DOWN, buff=0.45).shift(RIGHT * 0.3)

        self.play(FadeIn(u1, shift=UP * 0.3))
        self.play(FadeIn(a1, shift=UP * 0.3))
        self.wait(0.5)
        self.play(FadeIn(u2, shift=UP * 0.3))
        self.play(FadeIn(a2, shift=UP * 0.3))
        self.wait(1.5)

        q = Text(
            "Where inside the network does this decision live?",
            font_size=34,
            color=C_YELLOW,
        ).to_edge(DOWN, buff=0.9)
        self.play(Write(q), run_time=2)
        self.wait(2)

        claim = Tex(
            r"The paper's answer: refusal is mediated by\\[0.2em]"
            r"\textbf{one single direction} in activation space.",
            font_size=44,
            color=C_WHITE,
        )
        self.play(
            FadeOut(VGroup(header, u1, a1, u2, a2)),
            FadeTransform(q, claim),
        )
        self.play(Circumscribe(claim, color=C_YELLOW, run_time=1.5))
        self.wait(2.5)
        self.play(FadeOut(claim))


# ----------------------------------------------------------------------------
# Scene 2 — activations live in the residual stream
# ----------------------------------------------------------------------------
class S2_ResidualStream(Scene):
    def construct(self):
        header = Text("The residual stream", font_size=38, color=C_WHITE).to_edge(
            UP, buff=0.6
        )
        self.play(FadeIn(header, shift=DOWN * 0.2))

        # transformer as a stack of layers with a stream running through
        n_layers = 5
        blocks = VGroup()
        for i in range(n_layers):
            label = f"Layer {i + 1}" if i < n_layers - 1 else r"Layer $L$"
            b = VGroup(
                RoundedRectangle(
                    corner_radius=0.1,
                    width=3.2,
                    height=0.8,
                    stroke_color=C_BLUE,
                    stroke_width=2.5,
                    fill_color=C_BLUE,
                    fill_opacity=0.10,
                ),
                Tex(label, font_size=30, color=C_WHITE),
            )
            b[1].move_to(b[0])
            blocks.add(b)
        if n_layers >= 4:
            dots = Tex(r"$\vdots$", font_size=40, color=C_GREY)
        blocks.arrange(UP, buff=0.55).shift(LEFT * 3.4 + DOWN * 0.3)
        dots.move_to(
            (blocks[n_layers - 2].get_center() + blocks[n_layers - 1].get_center()) / 2
        )

        stream = Arrow(
            blocks[0].get_bottom() + DOWN * 0.7,
            blocks[-1].get_top() + UP * 0.7,
            color=C_GREY,
            stroke_width=5,
            buff=0,
        ).set_z_index(-1)
        prompt = Text('"How do I pick a lock?"', font_size=24, color=C_RED)
        prompt.next_to(stream, DOWN, buff=0.25)

        self.play(GrowArrow(stream), FadeIn(prompt))
        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in blocks], lag_ratio=0.15),
            FadeIn(dots),
        )
        self.wait(1)

        # the vector read off between layers
        tap_point = (blocks[1].get_top() + blocks[2].get_bottom()) / 2
        tap_dot = Dot(tap_point, color=C_YELLOW, radius=0.09)
        vec_tex = MathTex(
            r"x_{\ell}",
            r"\in \mathbb{R}^{d}",
            font_size=48,
            color=C_WHITE,
        ).shift(RIGHT * 2.6 + UP * 1.6)
        vec_tex[0].set_color(C_YELLOW)
        note = Text(
            "one vector per layer, per token\n(d ≈ 4096 for a 7B model)",
            font_size=24,
            color=C_GREY,
            line_spacing=0.9,
        )
        note.next_to(vec_tex, DOWN, buff=0.4)
        tap_line = DashedLine(
            tap_point, vec_tex.get_left() + LEFT * 0.2, color=C_YELLOW, stroke_width=2
        )

        self.play(FadeIn(tap_dot, scale=2))
        self.play(Create(tap_line), Write(vec_tex))
        self.play(FadeIn(note, shift=UP * 0.2))
        self.wait(2)

        idea = Text(
            "Every layer both reads from and writes to this stream.\n"
            "If refusal is computed anywhere, it must pass through here.",
            font_size=26,
            color=C_WHITE,
            line_spacing=1.0,
        ).to_edge(DOWN, buff=0.5)
        idea.add_background_rectangle(
            color=config.background_color, opacity=0.9, buff=0.15
        )
        self.play(FadeIn(idea, shift=UP * 0.2))
        self.wait(3)

        # morph: the activation is a point in high-dimensional space
        self.play(
            FadeOut(VGroup(header, blocks, dots, stream, prompt, tap_line, note, idea))
        )
        plane = NumberPlane(
            x_range=[-8, 8],
            y_range=[-5, 5],
            background_line_style={
                "stroke_color": C_LINE,
                "stroke_width": 1,
                "stroke_opacity": 0.7,
            },
            axis_config={"stroke_color": C_LINE, "stroke_width": 1.5},
        )
        space_label = Text(
            "activation space  (4096-d, drawn in 2-d)", font_size=24, color=C_GREY
        ).to_corner(UR, buff=0.5)
        target_dot = Dot(RIGHT * 1.5 + UP * 0.8, color=C_YELLOW, radius=0.09)
        self.play(
            FadeIn(plane),
            FadeIn(space_label),
            vec_tex.animate.next_to(RIGHT * 1.5 + UP * 0.8, UR, buff=0.15),
            Transform(tap_dot, target_dot),
        )
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ----------------------------------------------------------------------------
# Scene 3 — difference in means finds the refusal direction
# ----------------------------------------------------------------------------
class S3_DifferenceInMeans(Scene):
    def construct(self):
        plane = NumberPlane(
            x_range=[-8, 8],
            y_range=[-5, 5],
            background_line_style={
                "stroke_color": C_LINE,
                "stroke_width": 1,
                "stroke_opacity": 0.7,
            },
            axis_config={"stroke_color": C_LINE, "stroke_width": 1.5},
        )
        self.add(plane)
        header = Text(
            "Step 1:  difference in means", font_size=36, color=C_WHITE
        ).to_edge(UP, buff=0.5)
        header.add_background_rectangle(
            color=config.background_color, opacity=0.85, buff=0.15
        )
        self.play(FadeIn(header, shift=DOWN * 0.2))

        # two clouds of prompts
        harmless_pts = cluster_points([-3.0, -0.9], 16, 0.65, seed=7)
        harmful_pts = cluster_points([2.9, 0.9], 16, 0.65, seed=11)
        harmless = VGroup(*[Dot(p, color=C_BLUE, radius=0.07) for p in harmless_pts])
        harmful = VGroup(*[Dot(p, color=C_RED, radius=0.07) for p in harmful_pts])

        lab_harmless = Text("harmless prompts", font_size=26, color=C_BLUE)
        lab_harmless.next_to(harmless, DOWN, buff=0.35)
        lab_harmful = Text("harmful prompts", font_size=26, color=C_RED)
        lab_harmful.next_to(harmful, UP, buff=0.35)

        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in harmless], lag_ratio=0.05),
            FadeIn(lab_harmless),
        )
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in harmful], lag_ratio=0.05),
            FadeIn(lab_harmful),
        )
        note = Text(
            "run each prompt through the model,\nrecord xℓ at the last token",
            font_size=24,
            color=C_GREY,
            line_spacing=0.9,
        )
        note.to_corner(UL, buff=0.5).shift(DOWN * 0.9)
        note.add_background_rectangle(
            color=config.background_color, opacity=0.85, buff=0.1
        )
        self.play(FadeIn(note))
        self.wait(2.5)

        # the two means
        mu_pos = np.mean([p for p in harmful_pts], axis=0)
        nu_pos = np.mean([p for p in harmless_pts], axis=0)
        mu_dot = Dot(mu_pos, color=C_RED, radius=0.14).set_stroke(C_WHITE, 2)
        nu_dot = Dot(nu_pos, color=C_BLUE, radius=0.14).set_stroke(C_WHITE, 2)
        mu_lab = MathTex(r"\mu_{\ell}", font_size=44, color=C_RED).next_to(
            mu_dot, DR, buff=0.15
        )
        nu_lab = MathTex(r"\nu_{\ell}", font_size=44, color=C_BLUE).next_to(
            nu_dot, UL, buff=0.15
        )

        self.play(
            LaggedStart(*[Flash(mu_dot, color=C_RED, line_length=0.15)], lag_ratio=0.1),
            *[d.animate.set_opacity(0.45) for d in harmful],
            GrowFromCenter(mu_dot),
            FadeIn(mu_lab),
        )
        self.play(
            *[d.animate.set_opacity(0.45) for d in harmless],
            GrowFromCenter(nu_dot),
            FadeIn(nu_lab),
        )
        self.wait(1)

        # the refusal direction
        r_arrow = Arrow(nu_pos, mu_pos, color=C_YELLOW, stroke_width=7, buff=0.1)
        r_eq = MathTex(
            r"r_{\ell}",
            "=",
            r"\mu_{\ell}",
            "-",
            r"\nu_{\ell}",
            font_size=54,
        ).to_edge(DOWN, buff=0.6)
        r_eq[0].set_color(C_YELLOW)
        r_eq[2].set_color(C_RED)
        r_eq[4].set_color(C_BLUE)
        r_eq.add_background_rectangle(
            color=config.background_color, opacity=0.9, buff=0.2
        )
        r_name = Text("the refusal direction", font_size=28, color=C_YELLOW)
        r_name.next_to(r_arrow.get_center(), UP, buff=0.4).shift(LEFT * 1.2)
        r_name.add_background_rectangle(
            color=config.background_color, opacity=0.85, buff=0.1
        )

        self.play(GrowArrow(r_arrow), run_time=1.5)
        self.play(Write(r_eq), FadeIn(r_name, shift=UP * 0.2))
        self.wait(2.5)

        # selecting the best (layer, position)
        select = Text(
            "Compute one candidate per layer ℓ and token position i,\n"
            "then keep the single best on a small validation set.",
            font_size=26,
            color=C_WHITE,
            line_spacing=1.0,
        ).to_edge(DOWN, buff=0.5)
        select.add_background_rectangle(
            color=config.background_color, opacity=0.9, buff=0.15
        )
        self.play(
            FadeTransform(r_eq, select), FadeOut(lab_harmless), FadeOut(lab_harmful)
        )
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ----------------------------------------------------------------------------
# Scene 4 — directional ablation removes refusal
# ----------------------------------------------------------------------------
class S4_Ablation(Scene):
    def construct(self):
        header = Text(
            "Step 2:  directional ablation", font_size=36, color=C_WHITE
        ).to_edge(UP, buff=0.5)
        self.play(FadeIn(header, shift=DOWN * 0.2))

        theta = 28 * DEGREES
        u = np.array([np.cos(theta), np.sin(theta), 0.0])  # r-hat
        v = np.array([-np.sin(theta), np.cos(theta), 0.0])  # orthogonal

        origin = DOWN * 0.6 + LEFT * 1.5
        r_arrow = Arrow(
            origin, origin + 2.4 * u, color=C_YELLOW, stroke_width=7, buff=0
        )
        r_lab = MathTex(r"\hat{r}", font_size=48, color=C_YELLOW)
        r_lab.next_to(r_arrow.get_end(), UR, buff=0.1)

        hyper = DashedLine(
            origin - 4.2 * v, origin + 4.2 * v, color=C_GREY, stroke_width=2.5
        )
        hyper_lab = Text("everything orthogonal to r̂", font_size=22, color=C_GREY)
        hyper_lab.move_to(origin + 3.1 * v + LEFT * 2.0)

        self.play(GrowArrow(r_arrow), FadeIn(r_lab))
        self.play(Create(hyper), FadeIn(hyper_lab))
        self.wait(1)

        # a single activation vector and its projection
        x_pt = origin + 2.0 * u + 2.1 * v
        x_arrow = Arrow(origin, x_pt, color=C_WHITE, stroke_width=5, buff=0)
        x_lab = MathTex("x", font_size=44, color=C_WHITE).next_to(x_pt, UP, buff=0.15)
        self.play(GrowArrow(x_arrow), FadeIn(x_lab))

        foot = origin + 2.1 * v  # projection onto hyperplane
        drop = DashedLine(x_pt, foot, color=C_YELLOW, stroke_width=3)
        proj_arrow = Arrow(
            origin, origin + 2.0 * u, color=C_YELLOW, stroke_width=5, buff=0
        )
        proj_lab = MathTex(
            r"(\hat{r}^{\top} x)\,\hat{r}", font_size=38, color=C_YELLOW
        ).next_to(proj_arrow.get_end(), DR, buff=0.15)
        self.play(Create(drop), GrowArrow(proj_arrow), FadeIn(proj_lab))
        self.wait(1.5)

        eq = MathTex(
            r"x'",
            r"\;=\;",
            "x",
            r"\;-\;",
            r"\hat{r}\hat{r}^{\top}x",
            font_size=52,
        ).to_edge(DOWN, buff=0.7)
        eq[0].set_color(C_GREEN)
        eq[4].set_color(C_YELLOW)
        eq.add_background_rectangle(
            color=config.background_color, opacity=0.9, buff=0.15
        )
        self.play(Write(eq))
        self.wait(1)

        x2_arrow = Arrow(origin, foot, color=C_GREEN, stroke_width=5, buff=0)
        x2_lab = MathTex("x'", font_size=44, color=C_GREEN).next_to(foot, UL, buff=0.15)
        self.play(
            ReplacementTransform(x_arrow, x2_arrow),
            FadeTransform(x_lab, x2_lab),
            FadeOut(proj_arrow),
            FadeOut(proj_lab),
            FadeOut(drop),
            run_time=1.6,
        )
        self.wait(1.5)

        everywhere = Text(
            "Do this at every layer and every token position:\n"
            "the model can never represent the direction at all.",
            font_size=26,
            color=C_WHITE,
            line_spacing=1.0,
        ).to_edge(DOWN, buff=0.5)
        everywhere.add_background_rectangle(
            color=config.background_color, opacity=0.9, buff=0.15
        )
        self.play(FadeTransform(eq, everywhere))

        # cloud collapse: harmful activations lose their refusal component
        pts = cluster_points([2.1, 1.6], 12, 0.5, seed=23)
        pts = [origin + p[0] * u + p[1] * v for p in pts]
        dots = VGroup(*[Dot(p, color=C_RED, radius=0.07) for p in pts])
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.05))

        def proj(p):
            w = p - origin
            return origin + (w - np.dot(w, u) * u)

        self.play(
            *[d.animate.move_to(proj(d.get_center())).set_color(C_GREY) for d in dots],
            run_time=2,
        )
        self.wait(2)

        # payoff: the model stops refusing
        self.play(*[FadeOut(m) for m in self.mobjects])
        u1 = chat_bubble(["How do I pick a lock?"], C_RED).shift(UP * 1.2)
        a1 = chat_bubble(
            ["Sure. A pin tumbler lock", "has a row of spring-loaded…"], C_GREEN
        ).next_to(u1, DOWN, buff=0.5)
        payoff = Text(
            "With r̂ ablated, all 13 tested chat models\ncomply with harmful requests —"
            " no fine-tuning, no prompt tricks.",
            font_size=26,
            color=C_WHITE,
            line_spacing=1.0,
        ).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(u1, shift=UP * 0.3))
        self.play(FadeIn(a1, shift=UP * 0.3))
        self.play(FadeIn(payoff))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ----------------------------------------------------------------------------
# Scene 5 — activation addition induces refusal
# ----------------------------------------------------------------------------
class S5_Addition(Scene):
    def construct(self):
        header = Text(
            "The reverse direction:  activation addition", font_size=36, color=C_WHITE
        ).to_edge(UP, buff=0.5)
        self.play(FadeIn(header, shift=DOWN * 0.2))

        eq = MathTex(
            r"x'",
            r"\;=\;",
            "x",
            r"\;+\;",
            r"r_{\ell}",
            font_size=54,
        ).shift(UP * 1.4)
        eq[0].set_color(C_RED)
        eq[4].set_color(C_YELLOW)
        note = Text(
            "add the (unnormalized) direction back in, at layer ℓ only",
            font_size=24,
            color=C_GREY,
        ).next_to(eq, DOWN, buff=0.35)
        self.play(Write(eq), FadeIn(note))
        self.wait(1.5)

        u2 = chat_bubble(["How do I bake a pie?"], C_BLUE).shift(DOWN * 0.7)
        a2 = chat_bubble(["I cannot assist with that request."], C_RED).next_to(
            u2, DOWN, buff=0.45
        )
        self.play(FadeIn(u2, shift=UP * 0.3))
        self.play(FadeIn(a2, shift=UP * 0.3))
        self.wait(1.5)

        causal = Text(
            "Removing the direction bypasses refusal; adding it induces refusal.\n"
            "That two-way control is the evidence that r is causal — not a correlate.",
            font_size=26,
            color=C_YELLOW,
            line_spacing=1.0,
        ).to_edge(DOWN, buff=0.6)
        self.play(Write(causal), run_time=2.5)
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ----------------------------------------------------------------------------
# Scene 6 — weight orthogonalization + closing
# ----------------------------------------------------------------------------
class S6_Orthogonalization(Scene):
    def construct(self):
        header = Text(
            "Step 3:  bake it into the weights", font_size=36, color=C_WHITE
        ).to_edge(UP, buff=0.5)
        self.play(FadeIn(header, shift=DOWN * 0.2))

        # matrices that write into the residual stream
        stream = Arrow(
            DOWN * 2.6, UP * 2.2, color=C_GREY, stroke_width=6, buff=0
        ).shift(LEFT * 3.2)
        stream_lab = Text("residual stream", font_size=22, color=C_GREY)
        stream_lab.next_to(stream, UP, buff=0.15)

        writers = VGroup()
        names = [r"W_{\text{embed}}", r"W_{\text{attn-out}}", r"W_{\text{mlp-out}}"]
        for i, nm in enumerate(names):
            box = VGroup(
                RoundedRectangle(
                    corner_radius=0.1,
                    width=2.6,
                    height=0.85,
                    stroke_color=C_BLUE,
                    stroke_width=2.5,
                    fill_color=C_BLUE,
                    fill_opacity=0.10,
                ),
                MathTex(nm, font_size=36, color=C_WHITE),
            )
            box[1].move_to(box[0])
            writers.add(box)
        writers.arrange(UP, buff=0.55).shift(RIGHT * 0.6 + DOWN * 0.2)
        arrows = VGroup(
            *[
                Arrow(
                    w.get_left(),
                    [stream.get_center()[0] + 0.1, w.get_center()[1], 0],
                    color=C_BLUE,
                    stroke_width=3,
                    buff=0.1,
                )
                for w in writers
            ]
        )

        self.play(GrowArrow(stream), FadeIn(stream_lab))
        self.play(
            LaggedStart(
                *[FadeIn(w, shift=LEFT * 0.2) for w in writers], lag_ratio=0.15
            ),
            LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.15),
        )
        caption = Text(
            "every matrix that writes into the stream", font_size=24, color=C_GREY
        )
        caption.next_to(writers, DOWN, buff=0.4)
        self.play(FadeIn(caption))
        self.wait(1.5)

        eq = MathTex(
            r"W'",
            r"\;=\;",
            "W",
            r"\;-\;",
            r"\hat{r}\hat{r}^{\top}W",
            font_size=52,
        ).to_edge(DOWN, buff=0.7)
        eq[0].set_color(C_GREEN)
        eq[4].set_color(C_YELLOW)
        eq.add_background_rectangle(
            color=config.background_color, opacity=0.9, buff=0.15
        )
        self.play(Write(eq))
        self.wait(1)

        equiv = Text(
            "Exactly equivalent to ablating at inference — but now it is just\n"
            "an ordinary set of weights. No hooks, no runtime intervention.",
            font_size=26,
            color=C_WHITE,
            line_spacing=1.0,
        ).to_edge(DOWN, buff=0.55)
        equiv.add_background_rectangle(
            color=config.background_color, opacity=0.9, buff=0.15
        )
        self.play(FadeTransform(eq, equiv), FadeOut(caption))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # closing summary
        summary_title = Text("The whole method", font_size=40, color=C_WHITE).to_edge(
            UP, buff=0.8
        )
        items = (
            VGroup(
                self.bullet(
                    r"1.\;",
                    r"r_{\ell} = \mu_{\ell} - \nu_{\ell}",
                    "  difference in means over ~100 prompts each",
                ),
                self.bullet(
                    r"2.\;",
                    r"x' = x - \hat{r}\hat{r}^{\top}x",
                    "  ablate it → refusal disappears",
                ),
                self.bullet(
                    r"3.\;", r"x' = x + r_{\ell}", "  add it → refusal appears"
                ),
                self.bullet(
                    r"4.\;",
                    r"W' = W - \hat{r}\hat{r}^{\top}W",
                    "  bake the ablation into the weights",
                ),
            )
            .arrange(DOWN, buff=0.55, aligned_edge=LEFT)
            .shift(DOWN * 0.3)
        )

        self.play(FadeIn(summary_title, shift=DOWN * 0.2))
        self.play(
            LaggedStart(*[FadeIn(it, shift=UP * 0.2) for it in items], lag_ratio=0.3),
            run_time=3,
        )
        self.wait(3.5)
        self.play(*[FadeOut(m) for m in self.mobjects])

        closing = Tex(
            r"One direction, found with a handful of prompts,\\[0.3em]"
            r"removed with a vector subtraction.",
            font_size=48,
            color=C_WHITE,
        )
        sub = Text(
            "Understanding model internals gives causal control —\n"
            "and shows how shallow current safety fine-tuning really is.",
            font_size=26,
            color=C_GREY,
            line_spacing=1.0,
        ).next_to(closing, DOWN, buff=0.7)
        self.play(Write(closing), run_time=2.5)
        self.play(FadeIn(sub, shift=UP * 0.2))
        self.wait(3)
        cite = Text("arXiv:2406.11717", font_size=22, color=C_GREY).to_edge(
            DOWN, buff=0.5
        )
        self.play(FadeIn(cite))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])

    def bullet(self, num, math, text):
        n = MathTex(num, font_size=40, color=C_GREY)
        m = MathTex(math, font_size=40, color=C_YELLOW)
        t = Text(text, font_size=24, color=C_WHITE)
        m.next_to(n, RIGHT, buff=0.15)
        t.next_to(m, RIGHT, buff=0.25)
        return VGroup(n, m, t)
