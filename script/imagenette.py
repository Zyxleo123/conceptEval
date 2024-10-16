import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import pandas as pd
from nltk.corpus import wordnet as wn
from src.utils.question import generate_positiveQ_from_conceptnet

DATA_ROOT = os.path.join('data', 'imagenette2')
OUTPUT_ROOT = os.path.join('output')

import logging
generator_logger = logging.getLogger(name='generate_sample_from_pos_offset')
main_logger = logging.getLogger(name='main')
logging.basicConfig(level=logging.INFO)
file_handler = logging.FileHandler(os.path.join(OUTPUT_ROOT, 'imagenette2.log'))
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
generator_logger.addHandler(file_handler)
main_logger.addHandler(file_handler)


main_logger.info("-"*100)

main_logger.info("Loading ConceptNet...")
conceptnet = pd.read_csv('data/conceptnet/conceptnet_big_weight_lower.csv', sep='\t')
main_logger.info("ConceptNet loaded.")

def generate_sample_from_pos_offset(pos_offset, synset_image_dir):
    synset = wn.synset_from_pos_and_offset(pos_offset[0], int(pos_offset[1:]))
    data = {"pos_offset": pos_offset, "type": "object", 
            "questions": [], "images": [], "synonyms": []}
    for lemma in synset.lemmas():
        generator_logger.info(f"Generating questions for {lemma.name()}...")
        name = lemma.name()
        qs = generate_positiveQ_from_conceptnet(conceptnet, name)
        data["questions"].extend(qs)
        data["synonyms"].append(name)
    image_dir = os.path.join(DATA_ROOT, pos_offset)
    data["images"] = [os.path.join(image_dir, img) for img in os.listdir(image_dir)[:5]]
    return data

if __name__ == '__main__':
    pos_offsets = os.listdir(DATA_ROOT)
    pos_offsets = [pos_offset for pos_offset in pos_offsets if os.path.isdir(os.path.join(DATA_ROOT, pos_offset))]
    output_file = os.path.join(OUTPUT_ROOT, 'imagenette2_concept.jsonl')
    if os.path.exists(output_file):
        os.remove(output_file)
    for pos_offset in pos_offsets:
        main_logger.info(f"Generating sample for {pos_offset}...")
        sample = generate_sample_from_pos_offset(pos_offset, os.path.join(DATA_ROOT, pos_offset))
        if len(sample["questions"]) < 5:
            main_logger.warning(f"Only {len(sample['questions'])} relations found for <{pos_offset}>. Skipping...")
            continue
        with open(output_file, 'a') as f:
            json.dump(sample, f)
            f.write('\n')
