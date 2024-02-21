# Linking Prediction Using LLM

This folder shows how PromptLink leverages the LLM (specifically GPT-4 [Link](https://openai.com/gpt-4)) and two-stage prompts to retrieve the final linking prediction answer. 

## Run the Code

* Obtain LLM's responses for the first-stage prompt: Run the "gen_responses.py" file and set the "prompt" parameter as "prompt0". This prompt checks whether a concept pair should be linked.

* Retrieve the responses for the first-stage prompt and filter candidates: In PromptLink, we run the "matching_embeddings.py" file and set the "mode" parameter as "prompt0_filter". In this way, belief scores are calculated and candidates are filtered. If you want to obtain the final linking result from the first-stage responses and omit the second-stage, you could simply set the "mode" parameter as "prompt0".

* Obtain LLM's responses for the second-stage prompt: Run the "gen_responses.py" file and set the "prompt" parameter as "prompt1".

* Retrieve the responses for the second-stage prompt: Run the "matching_embeddings.py" file and set the "mode" parameter as "prompt1". Then the final linking results could be retrieved from the second-stage prompts' responses.

## File Details

* "gen_responses.py": Leverages the GPT-4 LLM to generate responses based on specific prompts.

* "matching_embeddings.py": Processes the LLM's responses to filter candidates and obtain linking results.

* "utils/prompts.py": Details the specific two-stage prompts used.

* "utils/retrieval.py": Describes how we process the LLM's responses, including calculating belief scores and frequencies.

* "utils/metrics.py": Describes how we calculate the linking accuracy results.

* "utils/others.py": Contains additional utility functions for data input/output (I/O).