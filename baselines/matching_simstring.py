# https://pypi.org/project/simstring-fast/

import tqdm
import fire
import numpy as np
from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.measure.dice import DiceMeasure
from simstring.measure.jaccard import JaccardMeasure
from simstring.database.dict import DictDatabase
from simstring.searcher import Searcher


def main(
    data_path: str = 'datasets/mimic-ibkh.npz',
    sim_measure: str = 'cosine',
):
    data = np.load(data_path, allow_pickle=True)

    # add db concepts
    db = DictDatabase(CharacterNgramFeatureExtractor(2))
    db_concepts = data['concept_names'].tolist()
    print(f'Adding {len(db_concepts)} concepts to database...')
    for concept in db_concepts:
        db.add(concept)
    # do search
    print(f'Using {sim_measure} measure...')
    if sim_measure == 'cosine':
        searcher = Searcher(db, CosineMeasure())
        threshold = 0.6
    elif sim_measure == 'dice':
        searcher = Searcher(db, DiceMeasure())
        threshold = 0.3
    elif sim_measure == 'jaccard':
        searcher = Searcher(db, JaccardMeasure())
        threshold = 0.5
    else:
        raise NotImplementedError(f'{sim_measure=} not implemented')
    acc_top1 = 0
    for mention, gold_concept in tqdm.tqdm(zip(data['mention_names'], data['mention_labels'])):
        results = searcher.search(mention, threshold)
        # cal accuracy on top1
        if len(results) > 0:
            if gold_concept == results[0]:
                acc_top1 += 1
    print(f'acc_top1: {acc_top1 / len(data["mention_names"])}')


if __name__ == '__main__':
    fire.Fire()