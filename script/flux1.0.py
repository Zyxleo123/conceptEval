import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE"] = "cpu"
import torch
from diffusers import FluxPipeline
from huggingface_hub import login

login(token='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')

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev").to("cpu")

prompt = "Design an image that captures the height differences between a pencil, a giraffe, and the Eiffel Tower. Use a perspective view where the pencil is lying flat, the giraffe is standing next to it, and the Eiffel Tower looms in the background, bathed in sunset light.Realistic."
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)  # Ensure the generator uses the CPU
).images[0]
image.save("flux-dev.png")
