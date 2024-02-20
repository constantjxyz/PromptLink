import os
import numpy as np
from scipy import spatial
import time
import torch.nn.functional as F
import torch
import pandas as pd
import editdistance
import textdistance
  
def generate_predictions_cosine_torch_cpu(mimic_embedding, ibkh_embedding, ibkh_names, query_idxs=range(3), k=10):
    predictions = []
    values = []
    ibkh_embedding = torch.tensor(ibkh_embedding)
    for query_idx in query_idxs:
        query = torch.tensor(mimic_embedding[query_idx])
        similarity = F.cosine_similarity(query, ibkh_embedding, dim=1)
        searched_idxs = list(torch.argsort(similarity, descending=True)[:k])    # find the top k largest value's indexes in dists
        values.append(torch.sort(similarity, descending=True)[0][:k])
        predictions.append([ibkh_names[idx] for idx in searched_idxs])
        if query_idx % 100 == 0:
            print(f'Current query index: {query_idx}')
    return predictions, values

def generate_predictions_cosine_torch_slice(mimic_embedding, ibkh_embedding, ibkh_names, query_idxs=range(3), k=10, slice_num=3, device='cuda:6'):
    predictions = []
    ibkh_slice_sample_num = int(len(ibkh_embedding) / slice_num)
    for query_idx in query_idxs:
        query = torch.tensor(mimic_embedding[query_idx])
        query = query.to(device)
        searched_idxs = []
        searched_values = []
        for slice_id in range(slice_num):
            ibkh_start_id = slice_id * ibkh_slice_sample_num
            if slice_id == slice_num - 1:
                ibkh_end_id = len(ibkh_embedding)
            else:
                ibkh_end_id = ibkh_start_id + ibkh_slice_sample_num
            ibkh_slice_embedding = ibkh_embedding[ibkh_start_id:ibkh_end_id]
            ibkh_slice_embedding = torch.tensor(ibkh_slice_embedding).to(device)
            similarity = F.cosine_similarity(query, ibkh_slice_embedding, dim=1)
            top_k_idxs = list(np.array(torch.argsort(similarity, descending=True)[:k].cpu()))
            searched_values.extend(np.array(similarity[top_k_idxs].cpu()))
            top_k_idxs = [(t + ibkh_start_id) for t in top_k_idxs ]
            searched_idxs.extend(top_k_idxs)   # find the top k largest value's indexes in dists
        aux_idxs = np.argsort(np.array(searched_values))[::-1][:k]
        final_searched_idxs = np.array(searched_idxs)[aux_idxs]
        predictions.append([ibkh_names[idx] for idx in final_searched_idxs])
        if query_idx % 100 == 0:
            print(f'Current query index: {query_idx}')
    return predictions


def generate_predictions_levenshtein(mimic_names, ibkh_names, query_idxs=range(5), k=5,):
    predictions = [] 
    values = []
    for query_idx in query_idxs:
        query = mimic_names[query_idx]
        distances = np.zeros(len(ibkh_names))
        for i, str2 in enumerate(ibkh_names):
            distances[i] = editdistance.eval(query, str2)
        searched_idxs = list(np.argsort(distances)[:k])    # find the top k smallest value's indexes in dists
        values.append(np.sort(distances)[:k])
        predictions.append([ibkh_names[idx] for idx in searched_idxs])
        if query_idx % 100 == 0:
            print(f'Current query index: {query_idx}')
    return predictions, values 

def generate_predictions_jaro_winkler(mimic_names, ibkh_names, query_idxs=range(5), k=5,):
    predictions = [] 
    values = []
    for query_idx in query_idxs:
        query = mimic_names[query_idx]
        distances = np.zeros(len(ibkh_names))
        for i, str2 in enumerate(ibkh_names):
            distances[i] = textdistance.jaro_winkler(query, str2)
        searched_idxs = list(np.argsort(distances)[::-1][:k])    # find the top k largest value's indexes in dists
        values.append(np.sort(distances)[::-1][:k])
        predictions.append([ibkh_names[idx] for idx in searched_idxs])
        if query_idx % 100 == 0:
            print(f'Current query index: {query_idx}')
    return predictions, values 




