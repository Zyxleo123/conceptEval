import inflect, inflection
import pandas as pd
import numpy as np

INFLECT_ENGINE = inflect.engine()

RELATION_DISC = {
    "/r/RelatedTo": "<A> and <B> have some positive relationship",
    # "/r/FormOf": "<A> is an inflected form of <B>; <B> is the root word of <A>",
    "/r/IsA": "<A> is a subtype or a specific instance of <B>; every <A> is a <B>",
    "/r/PartOf": "<A> is a part of <B>",
    "/r/HasA": "<B> belongs to <A>, either as an inherent part or due to a social construct of possession",
    "/r/UsedFor": "<A> is used for <B>; the purpose of <A> is <B>",
    "/r/CapableOf": "<A> can typically do <B>",
    "/r/AtLocation": "<B> is a typical or inherent location for <A>",
    "/r/Causes": "<A> typically causes <B>",
    "/r/HasSubevent": "<B> happens as a subevent of <A>",
    "/r/HasFirstSubevent": "<A> begins with subevent <B>",
    "/r/HasLastSubevent": "<A> concludes with subevent <B>",
    "/r/HasPrerequisite": "In order for <A> to happen, <B> needs to happen",
    "/r/HasProperty": "<A> has <B> as a property; <A> can be described as <B>",
    "/r/MotivatedByGoal": "<A> is done to achieve <B>",
    "/r/ObstructedBy": "<A> is a goal that can be prevented by <B>",
    "/r/Desires": "<A> typically wants <B>",
    "/r/CreatedBy": "<B> creates <A>",
    # "/r/Synonym": "<A> and <B> have very similar meanings",
    # "/r/Antonym": "<A> and <B> are opposites in some relevant way",
    # "/r/DistinctFrom": "<A> and <B> are distinct members of a set; <A> is not <B>",
    # "/r/DerivedFrom": "<A> is a word or phrase that appears within <B> and contributes to <B>'s meaning",
    "/r/SymbolOf": "<A> symbolically represents <B>",
    # "/r/DefinedAs": "<A> overlaps with <B> in meaning, and <B> is a more explanatory version of <A>",
    "/r/MannerOf": "<A> is a specific way to do <B>",
    "/r/LocatedNear": "<A> and <B> are typically found near each other",
    "/r/HasContext": "<A> is used in the context of <B>, such as a topic area or regional dialect",
    "/r/SimilarTo": "<A> is similar to <B>",
    # "/r/EtymologicallyRelatedTo": "<A> and <B> have a common origin",
    # "/r/EtymologicallyDerivedFrom": "<A> is derived from <B>",
    "/r/CausesDesire": "<A> makes someone want <B>",
    "/r/MadeOf": "<A> is made of <B>",
    "/r/ReceivesAction": "<B> can be done to <A>",
    # "/r/ExternalURL": "Points to a URL outside of ConceptNet for further information"
}

def noun_inflect(noun):
    """ 
    In: noun; style does not matter.
    Return: '_' separated inflections of a noun. """

    noun = noun.lower().replace('_', ' ')
    inflections =  [
        INFLECT_ENGINE.plural(noun),
        INFLECT_ENGINE.a(inflection.singularize(noun)),
        inflection.singularize(noun),
    ]
    inflections = [infl.replace(' ', '_') for infl in inflections]
    return inflections


def search(df, concept, tgt_relations=None, return_other=False):
    concept_forms = noun_inflect(concept)
    concept_forms_set = set(concept_forms)

    # Vectorized filtering
    start_split = df['start'].str.split('/').str[3]
    end_split = df['end'].str.split('/').str[3]
    mask = start_split.isin(concept_forms_set) | end_split.isin(concept_forms_set)
    result = df[mask]

    if tgt_relations is None:
        tgt_relations = RELATION_DISC.keys()
    result = result[result['relation'].isin(tgt_relations)]
    result = result.drop_duplicates().sort_values('info', ascending=False)
    
    if len(result) < 5:
        print(f"Only {len(result)} relations found for <{concept}>")

    if return_other:
        # Vectorized operation to return the other part of the relation
        result_start_split = result['start'].str.split('/').str[3]
        result_end_split = result['end'].str.split('/').str[3]
        other = np.where(result_start_split.isin(concept_forms_set), result_end_split, result_start_split)
        return result, other
    
    return result
