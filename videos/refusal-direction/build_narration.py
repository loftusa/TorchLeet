"""Add ElevenLabs TTS narration to the refusal-direction video.

Feynman/3b1b-register narration, one clip per scene. For each scene the
video is held on its final frame (scenes end faded out, so this is
invisible) until the narration finishes, then everything is concatenated
into refusal_direction_narrated.mp4.

Usage:
    ELEVENLABS_API_KEY=... uv run build_narration.py

Idempotent: existing audio clips in audio/ are reused (delete to regenerate).
"""

import json
import os
import subprocess
from pathlib import Path

import requests

ROOT = Path(__file__).parent
HQ = ROOT / "media" / "videos" / "refusal_direction" / "1080p60"
AUDIO = ROOT / "audio"
BUILD = ROOT / "build"

VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")  # Adam
MODEL_ID = "eleven_multilingual_v2"

# Spoken narration, Feynman/3b1b register. Keyed by scene file stem.
NARRATION = {
    "S1_Intro": (
        "Here's a little puzzle. Ask a chat model something dangerous, and it "
        "says no. Ask it something innocent, and it happily answers. Somewhere "
        "inside billions of numbers, it decided to refuse. So — where does that "
        "decision live? Here's the surprise: it's not complicated at all. It's "
        "one single direction — one arrow — in activation space."
    ),
    "S2_ResidualStream": (
        "First, the anatomy. As your prompt flows up through the transformer, "
        "every layer reads from and writes to one shared channel — the residual "
        "stream. At each layer it's just a vector, about four thousand numbers. "
        "So think of it as a point, in a very big space."
    ),
    "S3_DifferenceInMeans": (
        "Now here's the beautifully dumb idea. Feed the model a pile of harmful "
        "prompts, and a pile of harmless ones, and look where they land. Two "
        "clouds. Take the mean of each, draw the arrow between them — and "
        "that's it. Subtract two averages, and you've found the refusal "
        "direction."
    ),
    "S4_Ablation": (
        "But is it real? Well — do surgery. Take any activation, measure how "
        "much of it points along that direction, and subtract exactly that much "
        "off. Geometrically, you're flattening the whole space onto the plane "
        "where refusal can't exist. Do it at every layer, every token, and the "
        "model literally cannot represent the concept. And sure enough — "
        "thirteen different chat models, and every single one just… answers."
    ),
    "S5_Addition": (
        "And it runs in reverse! Add the direction back in, and the model "
        "refuses everything — even baking a pie. Take it out, refusal vanishes. "
        "Put it in, refusal appears. That's causality."
    ),
    "S6_Orthogonalization": (
        "One last trick, and it's a good one. Instead of intervening while the "
        "model runs, reach into the weights themselves. Every matrix that "
        "writes into the stream gets orthogonalized against r-hat — one "
        "rank-one edit — and now no computation, anywhere, can ever write that "
        "direction again. So that's the whole method. Two averages find it. A "
        "subtraction removes refusal. An addition restores it. One direction, "
        "hiding in plain sight — which tells you something about how thin that "
        "layer of safety training really is."
    ),
}

SCENES = list(NARRATION)


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return float(out)


def tts(text: str, out_path: Path, api_key: str) -> None:
    resp = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}",
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        json={
            "text": text,
            "model_id": MODEL_ID,
            "voice_settings": {
                "stability": 0.40,
                "similarity_boost": 0.75,
                "style": 0.30,
            },
        },
        timeout=120,
    )
    assert resp.ok, f"TTS failed ({resp.status_code}): {resp.text[:500]}"
    assert resp.content[:3] != b"{", "expected audio bytes, got JSON"
    out_path.write_bytes(resp.content)


def main() -> None:
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    assert api_key, "set ELEVENLABS_API_KEY"
    for s in SCENES:
        assert (HQ / f"{s}.mp4").exists(), f"missing render: {s}.mp4 (run manim first)"

    AUDIO.mkdir(exist_ok=True)
    BUILD.mkdir(exist_ok=True)

    # 1. generate narration clips
    for s in SCENES:
        clip = AUDIO / f"{s}.mp3"
        if clip.exists():
            print(f"[tts] {s}: reusing existing clip")
        else:
            print(f"[tts] {s}: generating…")
            tts(NARRATION[s], clip, api_key)
        print(
            f"      audio {probe_duration(clip):5.1f}s / video "
            f"{probe_duration(HQ / f'{s}.mp4'):5.1f}s"
        )

    # 2. per-scene mux: hold last (faded-out) frame until narration ends
    parts = []
    for s in SCENES:
        vid, aud = HQ / f"{s}.mp4", AUDIO / f"{s}.mp3"
        v_dur, a_dur = probe_duration(vid), probe_duration(aud)
        target = max(v_dur, a_dur + 0.6)  # 0.6s breathing room after speech
        part = BUILD / f"{s}_narrated.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(vid),
                "-i",
                str(aud),
                "-filter_complex",
                f"[0:v]tpad=stop_mode=clone:stop_duration={max(0, target - v_dur) + 1:.3f}[v];"
                f"[1:a]apad=pad_dur={max(0, target - a_dur) + 1:.3f},"
                f"aresample=44100[a]",
                "-map",
                "[v]",
                "-map",
                "[a]",
                "-t",
                f"{target:.3f}",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                "medium",
                "-pix_fmt",
                "yuv420p",
                "-r",
                "60",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(part),
            ],
            check=True,
        )
        print(f"[mux] {s}: {target:5.1f}s")
        parts.append(part)

    # 3. concatenate
    concat_list = BUILD / "concat.txt"
    concat_list.write_text("".join(f"file '{p}'\n" for p in parts))
    final = ROOT / "refusal_direction_narrated.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(final),
        ],
        check=True,
    )
    print(f"[done] {final.name}: {probe_duration(final):.1f}s")
    print(json.dumps({s: probe_duration(AUDIO / f"{s}.mp3") for s in SCENES}, indent=2))


if __name__ == "__main__":
    main()
