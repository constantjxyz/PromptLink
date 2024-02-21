import os
import numpy as np
import pandas as pd
import torch
import time
import openai
deployment_name='text-embedding-ada-002'

print('Start generating embedding')
start_time = time.time()

# '''  ---------------  modify the configuration here  ------------------------'''

npz_load_dir = 'dataset/mimic_ibkh/mimic_diagnoses_ibkh_diseases/preprocessed_data/data.npz'
npz_save_dir = 'dataset/mimic_ibkh/mimic_diagnoses_ibkh_diseases/embedding/ada2/embedding.npz'
# mode = 'try'
# mode = 'medium'
mode = 'generate'
# '''  ---------------  modify the configuration here  ------------------------'''

output = openai.Embedding.create(
    input="Your text goes here", engine=deployment_name
)
print(np.array(output['data'][0]['embedding']).shape)
print(f'Running time {time.time()-start_time} seconds')

def get_embedding(text, model=deployment_name):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], engine=model)['data'][0]['embedding']

dataset = np.load(npz_load_dir, allow_pickle=True)


mention_embeddings = []
if mode == 'try':
   for i in range(min(5, len(dataset['mention_names']))):
       print(i)
       embedding = get_embedding(dataset['mention_names'][i], model=deployment_name)
       mention_embeddings.append(embedding)
elif mode == 'medium':
    for i in range(min(30, len(dataset['mention_names']))):
        embedding = get_embedding(dataset['mention_names'][i], model=deployment_name)
        mention_embeddings.append(embedding)
elif mode == 'generate':
    for i in range(len(dataset['mention_names'])):
        embedding = get_embedding(dataset['mention_names'][i], model=deployment_name)
        mention_embeddings.append(embedding)
mention_embeddings = np.array(mention_embeddings)
print(f'Running time {time.time()-start_time} seconds')


concept_embeddings = []
if mode == 'try':
   for i in range(min(5, len(dataset['concept_names']))):
       print(i)
       embedding = get_embedding(dataset['concept_names'][i], model=deployment_name)
       concept_embeddings.append(embedding)
elif mode == 'medium':
    for i in range(min(30, len(dataset['concept_names']))):
        embedding = get_embedding(dataset['concept_names'][i], model=deployment_name)
        concept_embeddings.append(embedding)
elif mode == 'generate':
    for i in range(len(dataset['concept_names'])):
        embedding = get_embedding(dataset['concept_names'][i], model=deployment_name)
        concept_embeddings.append(embedding)
concept_embeddings = np.array(concept_embeddings)
print(f'Running time {time.time()-start_time} seconds')

np.savez(npz_save_dir, 
         mention_embeddings=np.array(mention_embeddings), 
         concept_embeddings=np.array(concept_embeddings), mention_labels=np.array(dataset['mention_labels'], dtype=object), concept_names=np.array(dataset['concept_names'], dtype=object), mention_names=np.array(dataset['mention_names'], dtype=object))

print(f'Running time {time.time()-start_time} seconds')
print('End generating embedding')


