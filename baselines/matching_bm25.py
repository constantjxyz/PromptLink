import tqdm
import fire
import numpy as np
from rank_bm25 import BM25Okapi


def main(
    data_path: str = 'datasets/mimic-ibkh.npz',
):
    data = np.load(data_path, allow_pickle=True)
    db_concepts = data['concept_names'].tolist()
    tokenized_db_concepts = [c.split(" ") for c in db_concepts]
    bm25 = BM25Okapi(tokenized_db_concepts)

    acc_top1 = 0
    for mention, gold_concept in tqdm.tqdm(zip(data['mention_names'], data['mention_labels'])):
        results = bm25.get_top_n(mention.split(" "), db_concepts, n=1)
        # cal accuracy on top1
        if len(results) > 0:
            if gold_concept == results[0]:
                acc_top1 += 1
    print(f'acc_top1: {acc_top1 / len(data["mention_names"])}')


if __name__ == '__main__':
    fire.Fire()