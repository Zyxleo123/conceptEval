import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.question import get_prompts_from_sample
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import requests
import torch
import json
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(os.path.join('answer', 'paligemma-3b-mix-224-mid-level.log'))
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

logger.info('\n' + '='*200 + '\n' + '='*200 + '\n')

OUTPUT_ROOT = os.path.join('output')

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    from huggingface_hub import login 
    login(token='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')
    model_id = "google/paligemma-3b-mix-224"

    logger.info("Loading model...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, device_map=DEVICE).eval()
    logger.info("Model loaded.")
    processor = AutoProcessor.from_pretrained(model_id)

    dummy_image = Image.new('RGB', (224, 224), color = (73, 109, 137))

    with open('output/imagenette2_concept.jsonl') as f:

        ##### Concept Loop #####
        for k, line in enumerate(f):
            sample = json.loads(line)
            logger.info(f"Evaluating sample {sample['pos_offset']}(#{k}); Type {sample['type']}")

            textual_questions, vision_questions = get_prompts_from_sample(sample)
            knowledge_num = len(sample['questions'])
            vision_correct_count_concept = 0
            text_correct_count_concept = 0

            ##### Knowledge Loop #####
            for i, (text_question_variants, vision_question_variations) in enumerate(zip(textual_questions, vision_questions)):
                logger.info(f"\tKnowledge {i+1}/{knowledge_num} - Relation: {sample['questions'][i]['relation']}")

                ##### Text #####
                logger.info("\tTextual Evaluation")

                variant_num = len(text_question_variants)
                question_correct_count_knowledge = 0
                for prompt_variation in text_question_variants:
                    model_inputs = processor(text=prompt_variation, images=dummy_image, return_tensors="pt").to(DEVICE)
                    input_len = model_inputs["input_ids"].shape[-1]
                    with torch.inference_mode():
                        logger.info(f"\t\tUser: {prompt_variation.split('(yes/no): ')[1]}")
                        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
                        generation = generation[0][input_len:]
                        decoded = processor.decode(generation, skip_special_tokens=True)

                    logger.info(f"\t\tPaliGemma: {decoded}")
                    if "yes" in decoded.lower():
                        question_correct_count_knowledge += 1
                        text_correct_count_concept += 1
                logger.info(f"\tTextual Accuracy: {question_correct_count_knowledge}/{variant_num}={question_correct_count_knowledge/variant_num:.2f}")

                ##### Vision #####
                variant_num = len(vision_question_variations)
                question_correct_count_knowledge = 0
                for j, (prompt, image_variation) in enumerate(vision_question_variations):
                    model_inputs = processor(text=prompt, images=image_variation, return_tensors="pt").to(DEVICE)
                    input_len = model_inputs["input_ids"].shape[-1]
                    with torch.inference_mode():
                        logger.info(f"\t\tUser: <{sample['images'][j].split('/')[-1]}>{prompt.split('(yes/no): ')[1]}")
                        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
                        generation = generation[0][input_len:]
                        decoded = processor.decode(generation, skip_special_tokens=True)

                    logger.info(f"\t\tPaliGemma: {decoded}")
                    if "yes" in decoded.lower():
                        question_correct_count_knowledge += 1
                        vision_correct_count_concept += 1
                logger.info(f"\tVision Accuracy: {question_correct_count_knowledge}/{variant_num}={question_correct_count_knowledge/variant_num:.2f}") 
            logger.info(f"Evaluation complete for sample {sample['pos_offset']}." + "-"*100)
    logger.info("Evaluation complete.")