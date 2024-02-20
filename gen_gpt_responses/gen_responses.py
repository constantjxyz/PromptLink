'''
    generate gpt responses and store in numpy arrays
'''

import os
import numpy as np
import pandas as pd
import openai
import re
from collections import Counter
import string
import time
from utils.others import load_dataset, load_candidates, preprocess_np
from utils.metrics import top_k_accuracy
from utils.prompts import generate_responses_prompt0, generate_responses_prompt1

# gpt4; most of the gpt settings are ommited
deployment_name='gpt-4-0613'
gpt_setting = {'api_key':openai.api_key, 'api_base':openai.api_base, 'api_type':openai.api_type, 'api_version':openai.api_version, 'deployment_name':deployment_name}


# load the dataset
candidates_dir = 'dataset/mimic_ibkh/mimic_diagnoses_ibkh_diseases/response/sapbert_pubmed/predictions100.npy'
candidates = load_candidates(file_dir=candidates_dir, candidate_num=10)
candidates = preprocess_np(candidates) # lowercase the array and remove 
dataset_dir = 'dataset/mimic_ibkh/mimic_diagnoses_ibkh_diseases/embedding/sapbert_pubmed/embedding.npz'

_, _, labels, concepts, mentions = load_dataset(dataset_dir)
labels, concepts, mentions = preprocess_np(labels), preprocess_np(concepts), preprocess_np(mentions) # lowercase the array and remove punctuation
acc, sapbert_top1_right_idxs, sapbert_top1_wrong_idxs = top_k_accuracy(candidates, labels, query_idxs=range(len(labels)), k=1, queries=mentions)
acc, sapbert_top1_right_idxs, sapbert_top1_wrong_idxs = top_k_accuracy(candidates, labels, query_idxs=range(len(labels)), k=3, queries=mentions)
acc, sapbert_top5_right_idxs, sapbert_top5_wrong_idxs = top_k_accuracy(candidates, labels, query_idxs=range(len(labels)), k=5, queries=mentions)
acc, sapbert_top5_right_idxs, sapbert_top5_wrong_idxs = top_k_accuracy(candidates, labels, query_idxs=range(len(labels)), k=7, queries=mentions)
acc, sapbert_top10_right_idxs, sapbert_top10_wrong_idxs = top_k_accuracy(candidates, labels, query_idxs=range(len(labels)), k=10, queries=mentions)


# generate responses
repeat_times = 5
start = 0
end = len(mentions)
prompt = 'prompt1'
if prompt == 'prompt1':
    responses, input_list = generate_responses_prompt1(mentions, candidates, query_idx=range(start, end), candidate_number=candidates.shape[1], repeat_times=repeat_times, gpt_setting=gpt_setting)
elif prompt == 'prompt0':
    responses, input_list = generate_responses_prompt0(mentions, candidates, query_idx=range(start, end), candidate_number=candidates.shape[1], repeat_times=repeat_times, gpt_setting=gpt_setting)


# save the responses
response_save_dir = 'dataset/mimic_ibkh/mimic_diagnoses_ibkh_diseases/response/sapbert_pubmed/gpt4_candidate10_only_prompt1_' + str(start) + '_' + str(end) + '.npy'
np.save(response_save_dir, np.array(responses, dtype=object), allow_pickle=True)
print(f'Responses saved in: {response_save_dir}')
print('End')
