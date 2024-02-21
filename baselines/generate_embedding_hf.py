import os
import numpy as np
import pandas as pd
import torch
# from transformers import *
from transformers import AutoTokenizer
from transformers import pipeline
import time
from transformers import BioGptForCausalLM, BioGptTokenizer, AutoModel

print('Start generating embedding')
start_time = time.time()

# '''  ---------------  modify the configuration here  ------------------------'''
tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")  
model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
device = torch.device('cuda')
# device = torch.device('cpu')
npz_load_dir = 'dataset/mimic_ibkh/mimic_diagnoses_ibkh_diseases/preprocessed_data/data.npz'
npz_save_dir = 'embedding.npz'
# mode = 'try'
# mode = 'medium'
mode = 'generate'
# '''  ---------------  modify the configuration here  ------------------------'''

pipeline = pipeline('feature-extraction', model=model, tokenizer=tokenizer, device=device)
feature_extraction = pipeline
embeddings = feature_extraction(['Hello, world!', 'This is a test sentence.'])
sentence_embeddings = [np.mean(embedding, axis=0) for embedding in embeddings]
print(sentence_embeddings[1].shape)
print(f'Running time {time.time()-start_time} seconds')

dataset = np.load(npz_load_dir, allow_pickle=True)
if mode == 'try':
    mention_mid_embeddings = feature_extraction(list(dataset['mention_names'][:5]))
elif mode == 'medium':
    mention_mid_embeddings = feature_extraction(list(dataset['mention_names'][:30]))
elif mode == 'generate':
    mention_mid_embeddings = feature_extraction(list(dataset['mention_names']))
mention_embeddings = [np.mean(embedding, axis=0).mean(axis=0) for embedding in mention_mid_embeddings]
print(f'Running time {time.time()-start_time} seconds')

if mode == 'try':
    concept_mid_embeddings = feature_extraction(list(dataset['concept_names'][:5]))
elif mode == 'medium':
    concept_mid_embeddings = feature_extraction(list(dataset['concept_names'][:30]))
elif mode == 'generate':
    concept_mid_embeddings = feature_extraction(list(dataset['concept_names']))
concept_embeddings = [np.mean(embedding, axis=0).mean(axis=0) for embedding in concept_mid_embeddings]

np.savez(npz_save_dir, 
         mention_embeddings=np.array(mention_embeddings), 
         concept_embeddings=np.array(concept_embeddings), mention_labels=np.array(dataset['mention_labels'], dtype=object), concept_names=np.array(dataset['concept_names'], dtype=object), mention_names=np.array(dataset['mention_names'], dtype=object))

print(f'Running time {time.time()-start_time} seconds')
print('End generating embedding')


