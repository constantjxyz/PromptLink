import os 
import re
import numpy as np


def top_k_accuracy(preds, labs, query_idxs=range(3), k=1, queries=[]):
    count = 0
    sum = len(query_idxs)
    right_idxs = []
    wrong_idxs = []
    for query_idx in query_idxs:
        # print(query_idx)
        prediction = preds[query_idx][:k]
        prediction = [re.sub(r'[^\w\s]', '', item.lower()) for item in prediction]
        # lowercase and remove punctuation 
        result = re.sub(r'[^\w\s]', '', labs[query_idx].lower()) in prediction
        # lowercase and remove punctuation 
        if result == False:
            wrong_idxs.append(query_idx)
        else:
            right_idxs.append(query_idx)
        count += result
    print(f'top:{k}, count: {count}, sum:{sum}, accuracy:{count/sum}')
    return np.array(count/sum), np.array(right_idxs), np.array(wrong_idxs)
