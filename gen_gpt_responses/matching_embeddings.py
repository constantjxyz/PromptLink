from utils.others import *
from utils.metrics import *
from utils.prompts import *
from utils.retrieval import *
import re
import os
import numpy as np


# -----------------   modify here ---------------------------
print('Start matching embeddings')
response_save_dir = 'dataset/mimic_ibkh/mimic_diagnoses_ibkh_diseases/response/sapbert_pubmed/gpt4_candidate10_only_prompt1.npy'
responses = np.load(response_save_dir, allow_pickle=True)
file_dir = 'dataset/mimic_ibkh/mimic_diagnoses_ibkh_diseases/embedding/sapbert_pubmed/embedding.npz'
candidate_dir = 'dataset/mimic_ibkh/mimic_diagnoses_ibkh_diseases/response/sapbert_pubmed/predictions100.npy'
candidates = load_candidates(file_dir = candidate_dir, candidate_num=10)
save_excel = ''
save_candidates = ''

mention_embeddings, concept_embeddings, labels, concepts, mentions = load_dataset(file_dir)
print(f'File_dir: {file_dir}')
print(f'Shapes: Mention Embeddings {mention_embeddings.shape}, Labels {labels.shape}, Concept Embeddings {concept_embeddings.shape}, Concepts {concepts.shape}, Mentions {mentions.shape}')


experiment_ids = range(len(mentions))
# mode = 'prompt0'
mode = 'prompt1'
# mode = 'prompt0_filter'
    

# -----------------   no need to modify ---------------------------
start_time = time.time()
print(f'Experiment ids: start{experiment_ids[0]}, end:{experiment_ids[-1]}')


if mode == 'prompt0':
    new_predictions = retrieve_prediction_prompt0(responses, response_idx=range(responses.shape[0]), answer_number=responses.shape[1])
    new_predictions = new_predictions.reshape(int(new_predictions.shape[0]/candidates.shape[1]), candidates.shape[1], responses.shape[1])
    retrieved_answer = pick_prompt0(new_predictions, candidates, prediction_idx=range(new_predictions.shape[0]), candidate_num=new_predictions.shape[1], answer_num=new_predictions.shape[2])
    top_1_acc, right_idxs, wrong_idxs = top_k_accuracy(retrieved_answer, labels, query_idxs=range(len(retrieved_answer)), k=1, queries=mentions)

elif mode == 'prompt0_filter':
    new_predictions = retrieve_prediction_prompt0(responses, response_idx=range(responses.shape[0]), answer_number=responses.shape[1])
    new_predictions = new_predictions.reshape(int(new_predictions.shape[0]/candidates.shape[1]), candidates.shape[1], responses.shape[1])
    retrieved_answer = pick_prompt0(new_predictions, candidates, prediction_idx=range(new_predictions.shape[0]), candidate_num=new_predictions.shape[1], answer_num=new_predictions.shape[2])
    new_candidates = filter_candidates(new_predictions, candidates, prediction_idx=range(new_predictions.shape[0]), candidate_num=new_predictions.shape[1], answer_num=new_predictions.shape[2], low=0, high=new_predictions.shape[2]*0.8)
    top_1_acc, right_idxs, wrong_idxs = top_k_accuracy(retrieved_answer, labels, query_idxs=range(len(retrieved_answer)), k=1, queries=mentions)

elif mode == 'prompt1':
    new_predictions = retrieve_prediction_prompt1(responses, response_idx=range(responses.shape[0]), answer_number=responses.shape[1])
    retrieved_answer = pick_prompt1(new_predictions, candidates, prediction_idx=range(new_predictions.shape[0]), candidate_num=candidates.shape[1], answer_num=new_predictions.shape[1])
    top_1_acc, right_idxs, wrong_idxs = top_k_accuracy(retrieved_answer, labels, query_idxs=range(len(retrieved_answer)), k=1, queries=mentions)
     


if save_candidates != '':
    np.save(save_candidates, new_candidates, allow_pickle=True)
if save_excel != '':
    prediction_excel(retrieved_answer, labels, query_idxs=experiment_ids, k=1, queries=mentions, output_dir=save_excel)
    
end_time = time.time()
print_running_time(start_time=start_time)
print('End matching embeddings')