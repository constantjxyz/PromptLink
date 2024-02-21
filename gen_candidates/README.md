# Concept Representation and Candidate Generation
This folder contains the code for PromptLink's concept representation and candidate generation process.

## Run the Code

* Concept generation: Run the "generate_embeddings.py" file.

* Candidate generation: Run the "matching_embeddings.py" file.

## File Details

* File "generate_embeddings.py": This file uses a pre-trained language model (specifically SAPBERT from [Link](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext)) to create embeddings for biomedical concepts. For concepts that span multiple tokens, the token-level embeddings are averaged to create the concept embedding. 

* File "matching_embeddings.py": This file calculates the cosine similarity between concept embedding pairs and identifies the top-K candidates with the highest similarities as candidates for further GPT-based linking prediction.

* "utils/metrics.py": Describes how we calculate the linking accuracy results.

* "utils/distances.py": Details the method for calculating the similarity between embedding pairs.

* "utils/others.py": Contains other utility functions for data input/output (I/O).