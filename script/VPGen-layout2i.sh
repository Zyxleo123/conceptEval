model='gligen'

# layout generated by Vicuna
layout_path='output/vicuna_layout.json'

# Where to save the images
image_dump_dir='output/vpgen_images'

# Where to save the bounding box images
layout_image_dump_dir='output/vpgen_bb_images'

echo $layout_path
echo $image_dump_dir
echo $layout_image_dump_dir

python VPGen/inference_images.py \
    --model $model \
    --layout_path $layout_path \
    --image_dump_dir $image_dump_dir \
    --layout_image_dump_dir $layout_image_dump_dir \