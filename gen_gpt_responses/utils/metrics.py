import os 
import re
import numpy as np
import string
from utils.others import preprocess_np
 

def top_k_accuracy(preds, labs, query_idxs=range(3), k=1, queries=[]):
    count = 0
    sum = len(query_idxs)
    right_idxs = []
    wrong_idxs = []
    preds = preprocess_np(np.array(preds))
    labs = preprocess_np(np.array(labs))
    for query_idx in query_idxs:
        prediction = preds[query_idx][:k]
        result = labs[query_idx] in prediction
        if result == False:
            wrong_idxs.append(query_idx)
        else:
            right_idxs.append(query_idx)
        count += result
    print(f'top:{k}, count: {count}, sum:{sum}, accuracy:{count/sum}')
    return np.array(count/sum), np.array(right_idxs), np.array(wrong_idxs)

