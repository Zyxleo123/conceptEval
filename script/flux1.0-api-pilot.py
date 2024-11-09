import requests

API_URL = "https://api-inference.hf-mirror.com/models/black-forest-labs/FLUX.1-dev"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content
image_bytes = query({
	"inputs": "A view of Lake michigan, a notebook and a football field in a realistic style, " + \
 			"with the lake michigan being MUCH larger than the notebook and the notebook being MUCH smaller than the football field.",
})

import io
from PIL import Image
image = Image.open(io.BytesIO(image_bytes))
image.save("notebook.png", format="png")