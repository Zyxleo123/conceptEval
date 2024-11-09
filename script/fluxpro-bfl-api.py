from time import sleep
import requests
import json
from PIL import Image
from io import BytesIO

post_url = "https://api.bfl.ml/v1/flux-pro"

headers = {
    "Content-Type": "application/json",
    "X-Key": "eecef31e-fca6-4f57-a3ec-614f09dda0bf"
}

def fluxpro_bfl_api(prompt, output_file_path):
    data = {
        "prompt": prompt,
        "width": 1024,
        "height": 768,
        "steps": 40,
        "prompt_upsampling": True,
        "seed": 42,
        "guidance": 5,
        "safety_tolerance": 5,
        "output_format": "jpeg"
    }


    print("Posting request to BFL API...")
    post_response = requests.post(post_url, headers=headers, data=json.dumps(data))

    if post_response.status_code == 200:
        id = post_response.json()["id"]
        print("Response received. ID:", id)
    else:
        print(post_response.text)
        exit(1)

    print("Getting result from BFL API...")

    get_url = f"https://api.bfl.ml/v1/get_result?id={id}"
    status = None
    while status != "Ready":
        get_response = requests.get(get_url)
        if get_response.status_code == 200:
            status = get_response.json()["status"]
            print("Status:", status)
        else:
            print(get_response.text)
            exit(1)
        sleep(0.5)

    result_url = get_response.json()["result"]["sample"]
    img = requests.get(result_url)
    img = Image.open(BytesIO(img.content))
    img.save(output_file_path, "PNG")

if __name__ == "__main__":
    choice = input("One image/Selected subconcepts? (o/s/n): ").lower()
    if choice == "o":
        prompt = input("Prompt: ")
        output_file_path = input("Output File Path: ")
        fluxpro_bfl_api(prompt, output_file_path)
        exit(0)
    elif choice == "s":
        selected_sub_concept_names = input("Input selected subconcept names, seperated with spaces.")
        selected_sub_concept_names = selected_sub_concept_names.split(' ')
    
    import os
    with open(os.path.join('/home', 'zyx_21307130052', 'conceptEval', 'data', 'low_level_prompt_enhanced.json')) as f:
        concepts = json.load(f)
    for concept in concepts:
        concept_name = concept['concept']
        for sub_concept in concept['sub_concepts']:
            sub_concept_name = sub_concept['name']
            if choice == 's' and sub_concept_name not in selected_sub_concept_names:
                continue
            for i, instance in enumerate(sub_concept['samples']):
                prompt = instance['t2i_prompt']
                fluxpro_bfl_api(prompt, f'output/flux11pro_prompt_1024_768/{concept_name}_{sub_concept_name}_{i+1}.png')
