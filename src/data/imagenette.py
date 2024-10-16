import os, json
import pandas as pd
from nltk.corpus import wordnet as wn
from utils.conceptnet import generate_positive_Qs

DATA_ROOT = os.path.join('data', 'imagenette2')
OUTPUT_FILE = os.path.join('output', 'imagenette2')

print("Loading ConceptNet...")
conceptnet = pd.read_csv('data/conceptnet/conceptnet_big_weight_lower.csv', sep='\t')
print("ConceptNet loaded.")

def generate_sample_from_pos_offset(pos_offset, synset_image_dir):
    synset = wn.synset_from_pos_and_offset(pos_offset[0], int(pos_offset[1:]))
    data = {"questions": [], "images": [], "synonyms": []}
    for lemma in synset.lemmas():
        print(f"\tGenerating questions for {lemma.name()}...")
        name = lemma.name()
        qs = generate_positive_Qs(conceptnet, name)
        data["questions"].extend(qs)
        data["synonyms"].append(name)
    data["images"] = os.listdir(synset_image_dir)[:5]
    return data

if __name__ == '__main__':
    pos_offsets = os.listdir(DATA_ROOT)
    pos_offsets = [pos_offset for pos_offset in pos_offsets if os.path.isdir(os.path.join(DATA_ROOT, pos_offset))]
    output_file = open(OUTPUT_FILE, 'w')
    for pos_offset in pos_offsets:
        print(f"Generating sample for {pos_offset}...")
        sample = generate_sample_from_pos_offset(pos_offset, os.path.join(DATA_ROOT, pos_offset))
        json.dump(sample, output_file)
    output_file.close()