from .conceptnet import search, noun_inflect, RELATION_DISC
from PIL import Image 

def generate_positiveQ_from_conceptnet(df, concept):
    """ 
    For a concept, first find all relations with the concept formed as "A <relation> <concept>" or "<concept> <relation> B".
    Replace <concept> with "the object in the image" and <relation> with the corresponding relation description.
    Prepend "Is the following statement true, answer with yes/no:" to the beginning of the sentence. """

    questions = []
    relation_count = {}

    concept_forms = noun_inflect(concept)
    search_result = search(df, concept)
    for i, row in search_result.iterrows():
        question = "Is the following statement true? Answer with one word(yes/no): "
        relation = row['relation']
        if relation in RELATION_DISC and (relation not in relation_count or relation_count[relation] < 1):
            clause = RELATION_DISC[relation]
            if relation in relation_count:
                relation_count[relation] += 1
            else:
                relation_count[relation] = 1
        else:
            continue
        A = row['start'].split('/')[3]
        B = row['end'].split('/')[3]
        weight = row['info']

        if A in concept_forms:
            clause = clause.replace('<B>', f'"{B.replace("_", " ")}"')
            is_start = True
        elif B in concept_forms:
            clause = clause.replace('<A>', f'"{A.replace("_", " ")}"')
            is_start = False
        else:
            raise ValueError(f"Word not found in relation. Start: {A}, End: {B}, Word: {concept}")
        question += f"{clause}"
        question = {
            'relation': relation,
            'start': A.replace('_', ' '),
            'end': B.replace('_', ' '),
            'question': question,
            'is_start': is_start,
            'answer': 1,
            'weight': weight,
        }
        questions.append(question)
        if len(questions) >= 10:
            break
    return questions

def get_prompts_from_sample(sample):
    """ Generate a prompt from a sample. 
    Sample structure:
        "questions": [
            {
                "relation": str,
                "start": str,
                "end": str,
                "question": str,
                "is_start": bool,
                "answer": int,
                "weight": float
            }
        ],
        "images": [str],
        "synonyms": [str],
        "pos_offset": str
    Replace '<A>'(if is_start) or '<B>'(if not is_start) with "the object in the image" for vision prompts,
        and each of the synonyms for text prompts. 
    Return a doubly nested list. 
        The outer list contains the different knowledge questions;
        The inner list contains the different variants of the question. """
    text_prompts = []
    vision_prompts = []
    for question in sample['questions']:
        prompt = question['question']
        vision_prompt = prompt.replace('<A>' if question['is_start'] else '<B>',
                                       f'"the {sample["type"]} in the image"')
        
        text_variants = []
        vision_variants = []
        for synonym in sample['synonyms']:
            text_prompt = prompt.replace('<A>' if question['is_start'] else '<B>',
                                        f'"{synonym.replace("_", " ")}"')
            text_variants.append(text_prompt)
        for image in sample['images']:
            vision_variants.append((vision_prompt, Image.open(image)))
        text_prompts.append(text_variants)
        vision_prompts.append(vision_variants)
    return text_prompts, vision_prompts
    
