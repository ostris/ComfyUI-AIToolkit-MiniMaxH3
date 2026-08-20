# ComfyUI-AIToolkit-MiniMaxH3

Reference-video preprocessing for **MiniMax-H3 ref2va** that matches
[ai-toolkit](https://github.com/ostris/ai-toolkit) exactly, so LoRAs trained
there behave the same in ComfyUI.

## Node: AI-Toolkit H3 Reference Video

Takes a `VIDEO` (any fps) and outputs everything
`MiniMaxH3ReferenceToVideo` needs:

| output  | wire to                     | what it is |
|---------|-----------------------------|------------|
| frames  | `ref_video_1`               | 24 fps frames, real-time paced from frame 0, tail trimmed, count snapped DOWN to the model's 17n+5 grid, resized to the ref's own aspect at `target_megapixels` on the /32 grid |
| audio   | `ref_video_audio_1`         | soundtrack trimmed to the exact window the kept frames cover (disconnect if the clip is silent — a silent placeholder is emitted) |
| width   | `width`                     | generation width (= ref aspect at the target area) |
| height  | `height`                    | generation height |
| length  | `length`                    | the snapped 24 fps frame count |

### Inputs

- **video** – the reference clip. Any source fps works; frames are resampled
  to 24 fps by frame selection (`frame i <- source round(i * src_fps / 24)`).
  The stock H3 node assumes 24 fps input; feeding it a 48/60/90 fps clip
  directly produces stretched / half-speed motion.
- **target_megapixels** – pixel area of the generation. `0.258` = 672x384 =
  ai-toolkit's res-512 bucket; `~0.59` = res 768.
- **max_length** – optional cap on the 24 fps frame count (0 = whole clip).

## Example

`example_workflow.json` — load video → this node → H3 Reference to Video →
KSampler (cfg 1.0, euler — the model is guidance-distilled, positive is wired
to both conditioning inputs) → separate AV latent → video+audio decode → save.
Sigma shifts 12 (video) / 3 (audio). Add a `LoraLoaderModelOnly` between the
UNET loader and `ModelSamplingMiniMaxH3` to use an ai-toolkit LoRA.
