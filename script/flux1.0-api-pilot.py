import requests

API_URL = "https://api-inference.hf-mirror.com/models/black-forest-labs/FLUX.1-dev"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content
image_bytes = query({
	"inputs": "A towering giraffe stands beside a small pencil, both dwarfed by the majestic Eiffel Tower in the background. Realistic lighting highlights the height disparity amidst an urban setting, blending nature and architecture."
})
# You can access the image with PIL.Image for example
import io
from PIL import Image
image = Image.open(io.BytesIO(image_bytes))
image.save("flux1.0-dev.png", format="png")