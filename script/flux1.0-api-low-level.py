import requests
import os
import json
import io
from PIL import Image
from tqdm import tqdm

OUTPUT_DIR = os.path.join('output', 'flux1.0_images', 'low_level_test')

API_URL = "https://api-inference.hf-mirror.com/models/black-forest-labs/FLUX.1-dev"
# API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.content

prompts = []
with open(os.path.join('data', 'low_level.json')) as f:
    concepts = json.load(f)
    for concept in concepts:
        concept_name = concept['name']
        knowledges = concept['knowledges']
        for knowledge_name in knowledges:
            knowledge = knowledges[knowledge_name]
            for i, instance in enumerate(knowledge):
                prompt_t2i = instance['prompt_t2i']
                image = query({
                    "inputs": prompt_t2i,
                })
                image = Image.open(io.BytesIO(image))
                image.save(os.path.join(OUTPUT_DIR, f"{concept_name}_{knowledge_name}_{i}.png"), format="png")
                print(f"Saved image for {concept_name}_{knowledge_name}_{i}")

        
            
