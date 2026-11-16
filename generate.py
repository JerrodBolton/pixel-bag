"""Local text-to-image with Hugging Face diffusers on Apple Silicon.

Usage:
    uv run generate.py "a cozy cabin in a snowy forest, golden hour"
    uv run generate.py "a corgi astronaut" --steps 4 --seed 42 --out corgi.png
"""

import argparse
import time
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image

# Any text-to-image model on the Hub works here. SDXL-Turbo is fast (1-4 steps)
# and fits comfortably in 16 GB. Weights download once to ~/.cache/huggingface.

# this is the model ID on the Hugging Face Hub
MODEL_ID = "stabilityai/sdxl-turbo"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an image from a text prompt.")
    parser.add_argument("prompt")
    parser.add_argument("--steps", type=int, default=2, help="1-4 is plenty for Turbo")
    parser.add_argument("--seed", type=int, default=None, help="fix for reproducible output")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = pick_device()
    print(f"Loading {MODEL_ID} on {device}...")

    # A "pipeline" bundles everything a diffusion model needs: the text encoder
    # (prompt -> embeddings), the UNet (denoiser), the VAE (latents -> pixels),
    # and the scheduler (how noise is removed step by step).
    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,  # half precision: half the memory, faster
        variant="fp16",             # download the fp16 weight files only
    ).to(device)

    # A seeded generator makes results repeatable.
    generator = None
    if args.seed is not None:
        generator = torch.Generator("cpu").manual_seed(args.seed)

    start = time.time()
    image = pipe(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=0.0,  # Turbo models are trained without classifier-free guidance
        generator=generator,
    ).images[0]
    print(f"Generated in {time.time() - start:.1f}s")

    out = args.out or Path("outputs") / f"{int(time.time())}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
