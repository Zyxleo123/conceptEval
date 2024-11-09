import os, json
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"), 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def t2i_prompt_enhance(t2i_prompt):
    base_prompt = "你是提示词工程专家。修改提示词，增加原提示词中物品的细节以及图像整体的描述。确保没有物品/场景在背景被虚化。原提示词中没有要求magnitude的object的magnitude在剩余2个object的magnitude之间。" + \
    "示例输入：A realistic image with a Red Traffic Light, a desert, and a storm cloud. With the magnitude of SATURATION significantly higher for the Red Traffic Light, and the magnitude of SATURATION significantly lower for the storm cloud." + \
    "示例输出：A vibrant red traffic light glows intensely against a stark, arid desert landscape, with dusty beige sands stretching wide. Dark, ominous storm clouds loom above, rendered in muted greys, creating a dramatic contrast." + \
    "示例输入：A realistic image with a basketball, a cloud, and a spoon. With the magnitude of CURVATURE significantly more uniform for the basketball, and the magnitude of CURVATURE significantly less uniform for the cloud." + \
    "示例输出：A hyper-realistic scene featuring a glossy basketball with smooth, uniform curvature, juxtaposed against a wispy, unevenly shaped cloud in a bright blue sky, with a shiny silver spoon resting on soft grass below." + \
    "请修改下面的提示词：" + t2i_prompt
    completion = client.chat.completions.create(
        model="qwen-max",
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': base_prompt}],
        )
    result = json.loads(completion.model_dump_json())['choices'][0]['message']['content']
    return result

if __name__ == "__main__":
    choice = input("One prompt/Selected subconcepts? (o/s/n): ").lower()
    if choice == "o":
        prompt = input("Prompt: ")
        print(t2i_prompt_enhance(prompt))
        exit(0)
    elif choice == "s":
        selected_sub_concept_names = input("Input selected subconcept names, seperated with spaces: ")
        selected_sub_concept_names = selected_sub_concept_names.split(' ')

    with open(os.path.join('/home', 'zyx_21307130052', 'conceptEval', 'data', 'low_level.json')) as f:
        concepts = json.load(f)

    for concept in concepts:
        concept_name = concept['concept']
        for sub_concept in concept['sub_concepts']:
            sub_concept_name = sub_concept['name']
            if choice == 's' and sub_concept_name not in selected_sub_concept_names:
                continue
            for i, instance in enumerate(sub_concept['samples']):
                prompt = instance['t2i_prompt']
                instance['t2i_prompt'] = t2i_prompt_enhance(prompt)
    with open(os.path.join('/home', 'zyx_21307130052', 'conceptEval', 'data', 'low_level_prompt_enhanced3_texture_only.json'), 'w') as f:
        json.dump(concepts, f, indent=4)
        

