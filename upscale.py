# Description: Upscale images using the StableDiffusionUpscalePipeline from the diffusers library.
# This example uses the A100 GPU to upscale images.
# The model is loaded from the Hugging Face model hub.
# The model is cached in a shared volume to avoid downloading it on every run.
# The upscaled images are saved to a shared volume. The shared volume can be accessed through the modal CLI (see [modal docs](https://modal.com/docs/reference/cli/volume)).
# To retrieve the upscaled images, run the following command: `modal volume get image_upscaling_vol /cache/output/*.png [DESTINATION_DIR]`.

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from PIL import Image

from fastapi import FastAPI

import modal

web_app = FastAPI()
assets_path = Path(__file__).parent / "assets"
stub = modal.Stub(name="image_upscaling")
# Commit in `diffusers` to checkout from.
GIT_SHA = "ed616bd8a8740927770eebe017aedb6204c6105f"

image = (
    modal.Image.debian_slim()
    .pip_install(
        "accelerate",
        "datasets",
        "ftfy",
        "gradio~=3.10",
        "smart_open",
        "transformers",
        "torch",
        "torchvision",
        "triton",
    )
    .pip_install("xformers", pre=True)
    .apt_install("git")
    .run_commands(
        f"cd /root && git clone https://github.com/huggingface/diffusers && cd diffusers && git checkout {GIT_SHA} && pip install -e ."
    )
)
stub.image = image

# A persistent shared volume will store model artifacts across Modal app runs.

volume = modal.SharedVolume().persist("image_upscaling_vol")
CACHE_DIR = Path("/cache")
OUTPUT_DIR = CACHE_DIR / Path("output")

# Set up a "secret" to set the environment variable `TRANSFORMERS_CACHE`.
stub["MODEL_SECRET"] = modal.Secret({"TRANSFORMERS_CACHE": str(CACHE_DIR)})

if stub.is_inside(image):
    from diffusers import StableDiffusionUpscalePipeline
    import torch
    import os

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model.
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    # Specify cache directory to use shared volume.
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, cache_dir=CACHE_DIR
    )
    pipeline = pipeline.to("cuda")
    # Must reduce GPU VRAM usage to fit on A100
    pipeline.enable_attention_slicing()
    pipeline.enable_xformers_memory_efficient_attention()
    MODEL = pipeline


@stub.function(
    image=image,
    gpu="A100",
    shared_volumes={
        str(CACHE_DIR): volume,
    },
    timeout=60,
    secret=stub["MODEL_SECRET"],
    mounts=[modal.Mount(local_dir="./assets", remote_dir="/assets")],
)
def upscale(img_path: str):
    low_res_img = Image.open("/assets/" + img_path).convert("RGB")
    upscaled_image = MODEL(prompt="", image=low_res_img).images[0]
    OUTPATH = OUTPUT_DIR / f"upscaled-{img_path}"
    upscaled_image.save(OUTPATH)
    print(f"Saved upscaled image to {OUTPATH}")
    return OUTPATH


if __name__ == "__main__":
    # The images to be upscaled are stored in the assets directory.
    # The images need to be quadratic, and the size should be a multiple of 4. Eg. 128x128, 256x256, 512x512.
    img_paths = os.listdir(assets_path)
    print(img_paths)
    with stub.run() as app:
        results = list(upscale.map(img_paths))
        print(results)
