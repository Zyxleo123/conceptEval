import requests
import os
import json
import io
from PIL import Image
from tqdm import tqdm

OUTPUT_ROOT = os.path.join('output', 'flux1.0_images')

API_URL = "https://api-inference.hf-mirror.com/models/black-forest-labs/FLUX.1-dev"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.content

with open(os.path.join('data', 'vicuna_layout_prompts.json')) as f:
    prompts = json.load(f)['asdf']

for i, prompt in tqdm(enumerate(prompts)):
    image_bytes = query({
        "inputs": prompt,
    })
    image = Image.open(io.BytesIO(image_bytes))
    with open(os.path.join(OUTPUT_ROOT, f"{i}.png"), "wb") as f:
        image.save(f, format="png")