"""Prepare a reference VIDEO for MiniMax-H3 exactly like ai-toolkit sampling.

Mirrors ai-toolkit's reference treatment — training cache AND sampling now
use the same recipe (real-time pacing, tail trim) — so a LoRA trained in
ai-toolkit sees the identical reference at inference in ComfyUI:

  - frame count  = int(total / src_fps * 24) capped by max_length, snapped
                   DOWN to the model's 17n+5 grid
  - frame picks  = real-time pacing from frame 0: frame i <- source frame
                   round(i * src_fps / 24), tail trimmed (any source fps works;
                   ComfyUI's H3 node itself assumes 24 fps input and would
                   otherwise produce stretched / half-speed motion)
  - soundtrack   = trimmed to the same window: round(length / 24 * sr) samples
  - size         = the reference's own aspect at target_megapixels on the /32
                   grid (ai-toolkit sizes refs to the TARGET's pixel area; the
                   generation canvas derives from the ref's aspect here, so
                   ref block == generated size exactly)

Outputs feed MiniMaxH3ReferenceToVideo directly: frames -> ref_video_i,
audio -> ref_video_audio_i, width/height/length -> the node's inputs.
"""

import math

import torch
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

H3_FPS = 24


def snap_down_17n5(n: int) -> int:
    n = max(5, int(n))
    return ((n - 5) // 17) * 17 + 5


class AIToolkitMiniMaxH3RefVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AIToolkitMiniMaxH3RefVideo",
            display_name="AI-Toolkit H3 Reference Video",
            category="ai-toolkit/minimax",
            description=(
                "Treats a reference video exactly like ai-toolkit's minimax_h3_ref2va "
                "sampling: resamples to 24 fps from the real source fps (real-time "
                "pacing, tail trimmed), snaps the frame count DOWN to 17n+5, trims the "
                "soundtrack to the same window, and sizes the frames to the reference's "
                "own aspect at the target pixel area (/32 grid). Wire frames, audio, "
                "width, height and length straight into MiniMaxH3ReferenceToVideo."
            ),
            inputs=[
                io.Video.Input("video", tooltip="The reference video (any fps)"),
                io.Float.Input(
                    "target_megapixels",
                    default=0.258,
                    min=0.01,
                    max=4.0,
                    step=0.001,
                    tooltip=(
                        "Pixel area of the generation. 0.258 MP = 672x384 = "
                        "ai-toolkit res-512 bucket; ~0.59 MP = res 768."
                    ),
                ),
                io.Int.Input(
                    "max_length",
                    default=0,
                    min=0,
                    max=3600,
                    tooltip="Optional cap on the 24 fps frame count (0 = whole clip)",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="frames"),
                io.Audio.Output(display_name="audio"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.Int.Output(display_name="length"),
            ],
        )

    @classmethod
    def execute(cls, video, target_megapixels, max_length=0) -> io.NodeOutput:
        comp = video.get_components()
        frames = comp.images  # (T, H, W, C) float 0..1
        src_fps = float(comp.frame_rate)
        total = int(frames.shape[0])
        if total < 1:
            raise ValueError("Reference video decoded 0 frames")

        # ai-toolkit treatment (training AND sampling): duration-based count
        # snapped DOWN to 17n+5, real-time pacing from frame 0, tail trimmed
        n = int(total / src_fps * H3_FPS)
        if max_length and max_length > 0:
            n = min(n, int(max_length))
        n = snap_down_17n5(n)

        ratio = src_fps / H3_FPS
        idx = [min(round(i * ratio), total - 1) for i in range(n)]
        frames = frames[idx]

        # reference sizing: own aspect at the target pixel area on /32
        h, w = int(frames.shape[1]), int(frames.shape[2])
        scale = math.sqrt(float(target_megapixels) * 1_000_000.0 / (w * h))
        out_w = max(32, round(w * scale / 32) * 32)
        out_h = max(32, round(h * scale / 32) * 32)
        if (out_h, out_w) != (h, w):
            frames = (
                torch.nn.functional.interpolate(
                    frames.permute(0, 3, 1, 2),
                    size=(out_h, out_w),
                    mode="bicubic",
                    antialias=True,
                    align_corners=False,
                )
                .clamp(0.0, 1.0)
                .permute(0, 2, 3, 1)
                .contiguous()
            )

        # soundtrack trimmed to the SAME real-time window the frames cover
        audio = comp.audio
        if (
            audio is not None
            and audio.get("waveform") is not None
            and audio["waveform"].numel() > 0
        ):
            sr = int(audio["sample_rate"])
            keep = int(round(n / H3_FPS * sr))
            audio = {"waveform": audio["waveform"][..., :keep], "sample_rate": sr}
        else:
            # silent placeholder so the output is always a valid AUDIO; leave
            # the H3 node's ref_video_audio input unconnected to skip audio
            audio = {
                "waveform": torch.zeros(1, 2, int(round(n / H3_FPS * 44100))),
                "sample_rate": 44100,
            }

        print(
            f"[aitk-h3-ref] {w}x{h}@{src_fps:g}fps x{total} -> "
            f"{out_w}x{out_h}, {n} frames @24fps ({n / H3_FPS:.2f}s), real-time pacing"
        )
        return io.NodeOutput(frames, audio, out_w, out_h, n)


class AIToolkitMiniMaxH3Extension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [AIToolkitMiniMaxH3RefVideo]


async def comfy_entrypoint() -> AIToolkitMiniMaxH3Extension:
    return AIToolkitMiniMaxH3Extension()
