# Running Compared Baseline Methods
This folder contains the code for running all compared baseline methods, including BM25, Levenshtein Distance, BioBERT, and SAPBERT.

## Run the code
* Cosine Distance: Run the "matching_simstring.py" file and specify the "sim_measure" parameter as "cosine". More details [here](https://github.com/nullnull/simstring).

* Jaccard Distance: Run the "matching_simstring.py" file and specify the "sim_measure" parameter as "jaccard". More details [here](https://github.com/nullnull/simstring).

* Levenshtein Distance: Run the "matching_strings.py" file and use the "generate_predictions_levenshtein" function. More details [here](https://pypi.org/project/python-Levenshtein/).

* Jaro-Winkler Distance: Run the "matching_strings.py" file and use the "generate_predictions_jaro_winkler" function. More details [here](https://pypi.org/project/textdistance/).

* BM25: Run the "matching_bm25.py" file. More details [here](https://github.com/dorianbrown/rank_bm25/tree/master).

* ada002: Run the "generate_embedding_openai" file to generate concept embeddings using OPENAI's "text-embedding-ada-002" model [Link](https://openai.com/blog/new-and-improved-embedding-model). Then run the "matching_embeddings.py" file to calculate pairwise embedding similarities and generate linking prediction results.

* BioGPT: Run the "generate_embedding_biogpt" file to generate concept embeddings using BioGPT model from [here](microsoft/biogpt). Then run the "matching_embeddings.py" file to calculate pairwise embedding similarities and generate linking prediction results.

* SAPBERT: Run the "generate_embedding_hf" file to generate concept embeddings using SAPBERT model from the Hugging Face platform [Link](cambridgeltl/SapBERT-from-PubMedBERT-fulltext). Then run the "matching_embeddings.py" file to calculate pairwise embedding similarities and generate linking prediction results.

Other baseline methods' pipelines are similar to the SAPBERT's pipeline. Only the utilized models are replaced.

* BioBERT: Utilize the BioBERT model from [here](https://huggingface.co/dmis-lab/biobert-v1.1).

* BioClinicalBERT: Utilize the BioClinicalBERT model from [here](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT).

* BioDistilBERT: Utilize the BioDistilBERT model from [here](https://huggingface.co/nlpie/bio-distilbert-uncased).

* KRISSBERT: Utilize the KRISSBERT model from [here](https://huggingface.co/microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL).

## File Details

* File "generate_embedding_openai.py": Generates embeddings for the compared "ada002" method.

* File "generate_embedding_biogpt.py": Generates embeddings for the compared "BioGPT" method.

* File "generate_embedding_hf.py": Generates embeddings for the compared "SAPBERT", "BioBERT", "BioClinicalBERT", "BioDistilBERT", and "KRISSBERT" methods.

* File "matching_simstring.py": Identifies the top-K candidates for "Cosine Distance" method and "Jaccard Distance" method.

* File "matching_bm25.py": Identifies the top-K candidates for "BM25" method.

* File "matching_strings.py": Identifies the top-K candidates for "Levenshtein Distance" method and "Jaro-Winkler Distance" method.

* File "matching_embeddings.py": Identifies the top-K candidates for other embedding-based compared methods.

* File "utils/metrics.py": Describes how we calculate the linking accuracy results.

* File "utils/distances.py": Details the method for calculating the similarity between embedding pairs.

* File "utils/others.py": Contains other utility functions for data input/output (I/O).