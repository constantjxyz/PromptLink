from utils.others import *
from utils.distances import *
from utils.metrics import *
import re
import os
import numpy as np


# -----------------   modify here ---------------------------
print('Start matching embeddings')
file_dir = 'dataset/mimic_ibkh/mimic_otherconcepts_ibkh_diseases/embedding/embedding.npz'
save_excel, save_prediction = '', 'dataset/mimic_ibkh/mimic_otherconcepts_ibkh_diseases/candidate/candidates100.npy'

mention_embeddings, concept_embeddings, labels, concepts, mentions = load_dataset(file_dir)
print(f'File_dir: {file_dir}')
print(f'Shapes: Mention Embeddings {mention_embeddings.shape}, Labels {labels.shape}, Concept Embeddings {concept_embeddings.shape}, Concepts {concepts.shape}, Mentions {mentions.shape}')


    
# device = torch.device('cuda')
device = torch.device('cpu')
slice_num = 1
experiment_ids = range(len(mentions))
# experiment_ids = range(100)
    

# -----------------   no need to modify ---------------------------
start_time = time.time()
print(f'Experiment ids: start{experiment_ids[0]}, end:{experiment_ids[-1]}')
mention_predictions, prediction_values = generate_predictions_cosine_torch_cpu(mention_embeddings, concept_embeddings, concepts, query_idxs=experiment_ids, k=100,)

top_1_acc, _, _ = top_k_accuracy(mention_predictions, labels, query_idxs=experiment_ids, k=1, queries=mentions)
top_5_acc, _, _ = top_k_accuracy(mention_predictions, labels, query_idxs=experiment_ids, k=5, queries=mentions)
top_10_acc, _, _ = top_k_accuracy(mention_predictions, labels, query_idxs=experiment_ids, k=10, queries=mentions)
top_20_acc, _, _ = top_k_accuracy(mention_predictions, labels, query_idxs=experiment_ids, k=20, queries=mentions)
top_50_acc, _, _ = top_k_accuracy(mention_predictions, labels, query_idxs=experiment_ids, k=50, queries=mentions)
top_100_acc, _, _ = top_k_accuracy(mention_predictions, labels, query_idxs=experiment_ids, k=100, queries=mentions)
print(f'Linking finished, accuracies:[{top_1_acc} (top 1), {top_5_acc} (top 5), {top_10_acc} (top 10), {top_20_acc}(top 20), {top_50_acc}(top 50), {top_100_acc}(top 100)]')
    
    
if save_prediction != '':
    prediction_npy(mention_predictions, output_dir=save_prediction)
if save_excel != '':
    prediction_excel(mention_predictions, labels, query_idxs=experiment_ids, k=5, queries=mentions, output_dir=save_excel)
    
end_time = time.time()
print_running_time(start_time=start_time)
print('End matching embeddings')