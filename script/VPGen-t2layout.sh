base_model_path='models/vicuna_13b_checkpoint'

# LoRA checkpoint path
lora_model_path='models/lora_checkpoint/vicuna13B_GPU4_flickr30k_coco_paintskills_epoch2_mbatch32_lora16_cutoff256'

# where to load prompts
prompts_path='data/vicuna_layout_prompts.json'

# Where to save the generated layouts
layout_dump_path='output/vicuna_layout.json'

echo $base_model_path
echo $lora_model_path
echo $prompts_path
echo $layout_dump_path

python VPGen/text2layout_inference.py \
	--llm_device "cpu" \
	--base_model $base_model_path \
	--lora_model $lora_model_path \
	--data_path $prompts_path \
	--layout_dump_path $layout_dump_path