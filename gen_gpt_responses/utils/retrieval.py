'''
retrieve predictions from the gpt responses

'''

from utils.others import preprocess_np
import numpy as np
import re
from collections import Counter


def retrieve_prediction_prompt1(responses, response_idx=range(3), answer_number=5):
    new_predictions = [[[''] for j in range(answer_number)] for i in response_idx]
    for i in response_idx:
        for j in range(answer_number):
            response = responses[i][j]
            index_of_answer = response.find("Answer:")   # get the answer strings
            if index_of_answer != -1:  # avoid there are no strings
                answer_part = response[index_of_answer + len("Answer:"):].split('\n')[0].strip().replace("'", "")
                new_predictions[i][j] = answer_part.lower()
    return np.char.mod('%s', np.array(new_predictions, dtype=object))

from utils.others import preprocess_np

def pick_prompt1(predictions, candidates, prediction_idx=range(3), candidate_num=5, answer_num=5):
    # ensure the quality of the retrieved answers, lowercase the array and remove punctuation
    predictions = preprocess_np(predictions)
    retrieved_answer = []
    for i in prediction_idx: 
        prediction = predictions[i]  # get the prediction for query i
        # print(prediction)
        candidate = candidates[i]   # get the candidates for query i
        # print(candidate)
        
        candidate = np.append(candidate, 'nothing')
        prediction = [item for item in prediction if item in candidate]
        
        if len(prediction) == 0:
            most_common_element = candidate[0]
        else:
            counter = Counter(prediction)
            if counter['nothing'] / len(prediction) > 0.5:    # set a threshold value for nothing
                most_common_element = 'nothing' 
            else:
                del counter['nothing']
                candidate = np.delete(candidate, -1)
            # print(counter)
            counter[candidate[0]] += 1 
        most_common_element = ''
        most_count = 0
        for item in candidate:
            if counter[item] > most_count:
                most_common_element = item
                most_count = counter[item]   # output top candidates selected by sapbert
        retrieved_answer.append(most_common_element)
    return np.char.mod('%s', np.array(retrieved_answer)).reshape(len(retrieved_answer), 1)
        
def retrieve_prediction_prompt0(responses, response_idx=range(3), answer_number=5):
    new_predictions = [[[''] for j in range(answer_number)] for i in response_idx]
    for i in response_idx:
        for j in range(answer_number):
            response = responses[i][j]
            index_of_answer = response.find("Answer:")   # get the answer strings
            if index_of_answer != -1:  # avoid there are no strings
                answer_part = response[index_of_answer + len("Answer:"):].split('\n')[0].strip().replace("'", "")
                new_predictions[i][j] = answer_part.lower()
    return np.char.mod('%s', np.array(new_predictions))

def pick_prompt0(predictions, candidates, prediction_idx=range(3), candidate_num=5, answer_num=5):
    # ensure the quality of the retrieved answers, lowercase the array and remove punctuation
    predictions = preprocess_np(predictions)
    retrieved_answer = []
    for i in prediction_idx: 
        prediction = predictions[i]  # get the prediction for query i
        # print(prediction)
        candidate = candidates[i]   # get the candidates for query i
        # print(candidate)
        beliefs = np.zeros(shape=candidate.shape)  # initialize the belief for candidates
        for j in range(len(candidate)):
            beliefs[j] = np.count_nonzero(prediction[j] =='yes')  # calculate the belief for each candidate according to appearance frequency of 'yes'
        # print(beliefs)
        # if max(beliefs) <= 0:
        if max(beliefs) < 0:
            retrieved_answer.append('nothing')
        else:
            retrieved_answer.append(candidate[np.argmax(beliefs)])
        # print(retrieved_answer)
        # retrieved_answer.append(candidate[np.argmax(beliefs)])
    return np.char.mod('%s', np.array(retrieved_answer)).reshape(len(retrieved_answer), 1)


def filter_candidates(predictions, candidates, prediction_idx=range(3), candidate_num=5, answer_num=5, low=0, high=4):
    # ensure the quality of the retrieved answers, lowercase the array and remove punctuation
    predictions = preprocess_np(predictions)
    retrieved_answer = []
    filtered_candidates = np.copy(candidates)   
    for i in prediction_idx: 
        prediction = predictions[i]  # get the prediction for query i
        # print(prediction)
        candidate = candidates[i]   # get the candidates for query i
        # print(candidate)
        beliefs = np.zeros(shape=candidate.shape)  # initialize the belief for candidates
        for j in range(len(candidate)):
            beliefs[j] = np.count_nonzero(prediction[j] =='yes')  # calculate the belief for each candidate according to appearance frequency of 'yes'
        # print(beliefs)
        if np.min(beliefs)<= low and np.max(beliefs)>= high:
            zero_indices = np.where(beliefs <= low)
            new_candidate = np.copy(candidate)
            new_candidate[zero_indices] = ''
            filtered_candidates[i] = new_candidate
        
    return np.char.mod('%s', np.array(filtered_candidates))
