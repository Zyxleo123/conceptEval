import requests

response = requests.post(
    f"https://api.stability.ai/v2beta/stable-image/generate/sd3",
    headers={
        "authorization": f"Bearer sk-vGmfVmhF86f4KwVXnNv3x6HdzmKlefwIxtsra3AsHZBe76ms",
        "accept": "image/*"
    },
    files={"none": ''},
    data={
        "prompt": "Einstein with ((tennis wear)) on",
        "output_format": "jpeg",
    },
)

if response.status_code == 200:
    with open("./einstein_sd3.jpeg", 'wb') as file:
        file.write(response.content)
else:
    raise Exception(str(response.json()))