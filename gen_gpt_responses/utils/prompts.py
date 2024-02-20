import numpy as np
import time
import openai

'''generate responses for different kinds of prompts 
    input: mentions, candidates, query_idx, candidate_number, repeat_times, gpt_engine
    loop for each idx in query idx
    print: gpt_times, token numbers
    output: numpy arrays of responses, prompts
'''

def generate_responses_prompt0(queries, all_items, query_idx = range(3), candidate_number=5, repeat_times=1, gpt_setting=dict()):
    # set gpt
    openai.api_key = gpt_setting['api_key']
    openai.api_base = gpt_setting['api_base'] 
    openai.api_type = gpt_setting['api_type']
    openai.api_version = gpt_setting['api_version']
    deployment_name= gpt_setting['deployment_name']
    
    # set containers to store variables which need to be returned or printed finally
    prompt_input_list = []
    all_responses = []
    all_prompt_tokens = []
    all_completion_tokens = []
    all_gpt_times = []
    
    for i in query_idx:  # loop for each mention
        # set variables to calculate token numbers and time
        print(f'Generating responses for mention index {i}')
        prompt_tokens = 0
        completion_tokens = 0
        gpt_times = 0
        
        # loop for each candidate
        for j in range(candidate_number):
            
            # initiate variables 
            gpt_start_time = time.time()
            item = all_items[i][j]
            
            # generate prompt
            prompt_input =   [{"role":"user", "content":f"'{item}' and '{queries[i]}' refer to the same item, is it correct? Give your answer in the form: Answer: <one of: YES, NO>"}]
            
            # generate response
            response = openai.ChatCompletion.create(messages=prompt_input, engine=deployment_name, n=repeat_times)
            
            # retrieve useful information from the prompt
            prompt_input_list.append(prompt_input)  # initiate prompts
            gpt_times += float(time.time() - gpt_start_time)  # time
            prompt_tokens += int(response.usage['prompt_tokens'])  # prompt tokens
            completion_tokens += int(response.usage['completion_tokens']) # completion tokens
            multiple_responses = []  # answers for each repeat time
            for repeat_id in range(repeat_times):
                valid_response_dict = response.choices[repeat_id].message
                if "content" in valid_response_dict.keys():
                    # avoid cases that answers are sensored
                    multiple_responses.append(valid_response_dict.content)
                else:
                    multiple_responses.append('')
            all_responses.append(multiple_responses) 
            time.sleep(1)
        
        # after answer for each mention
        all_prompt_tokens.append(prompt_tokens)
        all_completion_tokens.append(completion_tokens)
        all_gpt_times.append(gpt_times)
        # time.sleep(3)  # avoid load limit
        
    # print out the statistics
    print(f'Total mention numbers: {len(query_idx)}')
    print(f'Total running time for generating gpt answers:{sum(all_gpt_times)} seconds')
    print(f'Total prompt token counts: {sum(all_prompt_tokens)}')
    print(f'Total completion token counts: {sum(all_completion_tokens)}')
    # return the responses and prompts
    return np.char.mod('%s', np.array(all_responses)), np.array(prompt_input_list)


def generate_responses_prompt1(queries, all_items, query_idx = range(3), candidate_number=5, repeat_times=1, gpt_setting=dict()):
    # set gpt
    openai.api_key = gpt_setting['api_key']
    openai.api_base = gpt_setting['api_base'] 
    openai.api_type = gpt_setting['api_type']
    openai.api_version = gpt_setting['api_version']
    deployment_name= gpt_setting['deployment_name']
    
    # set containers to store variables which need to be returned or printed finally
    prompt_input_list = []
    all_responses = []
    all_prompt_tokens = []
    all_completion_tokens = []
    all_gpt_times = []
    
    for i in query_idx:  # loop for each mention
        # set variables to calculate token numbers and time
        print(f'Generating responses for mention index {i}')
        prompt_tokens = 0
        completion_tokens = 0
        gpt_times = 0
        

        # initiate variables 
        gpt_start_time = time.time()
        items = [all_items[i][j] for j in range(candidate_number) if all_items[i][j] != '']
        # print(items)
        if len(items) == 1:
            all_responses.append([f'Answer:{items[0]}' for t in range(repeat_times)]) 
            # print(all_responses)
            continue
            
        # generate prompt
        prompt_input =   [{"role":"user", "content":f"What is the relationship between the query item '{queries[i]}' and the candidate items in the list {items}. Output the relationship result for each candidate in the candidate list.\n Give your answer in the form:\n Candidate 1: <candidate 1>\ncategory: <one of: EXACT_MATCH, RELATED_TO, DIFFERENT>\nCandidate 2: <candidate 2>\ncategory: <one of: EXACT_MATCH, RELATED_TO, DIFFERENT>. Make use of all provided information, including the concept names, definitions, and relationships. Looking at your previous generated results. Pick the items with EXACT_MATCH or RELATED_TO relationship. The item in the picked items that you think closest to '{queries[i]}' is the output chosen item. If all the items are with DIFFERENT relationship, output 'Nothing'.\nGive your answer in the form:\nAnswer: <the chosen item or NOTHING>"}]
            
        # generate response
        response = openai.ChatCompletion.create(messages=prompt_input, engine=deployment_name, n=repeat_times)
            
        # retrieve useful information from the prompt
        gpt_times = float(time.time() - gpt_start_time)  # time
        prompt_tokens = int(response.usage['prompt_tokens'])  # prompt tokens
        completion_tokens = int(response.usage['completion_tokens']) # completion tokens
        multiple_responses = []  # answers for each repeat time
        for repeat_id in range(repeat_times):
            valid_response_dict = response.choices[repeat_id].message
            if "content" in valid_response_dict.keys():
                # avoid cases that answers are sensored
                multiple_responses.append(valid_response_dict.content)
            else:
                multiple_responses.append('')
        
        # after answer for each mention
        prompt_input_list.append(prompt_input)  # initiate prompts
        all_responses.append(multiple_responses)
        # print(all_responses)
        all_prompt_tokens.append(prompt_tokens)
        all_completion_tokens.append(completion_tokens)
        all_gpt_times.append(gpt_times)
        # time.sleep(5)  # avoid load limit
        
    # print out the statistics
    
    print(f'Total mention numbers: {len(query_idx)}')
    print(f'Total running time for generating gpt answers:{sum(all_gpt_times)} seconds')
    print(f'Total prompt token counts: {sum(all_prompt_tokens)}')
    print(f'Total completion token counts: {sum(all_completion_tokens)}')
    # return the responses and prompts
    return np.char.mod('%s', np.array(all_responses)), np.array(prompt_input_list)

