# Data Mining Technology for Business and Society
Link: https://docs.google.com/document/d/1EO8zx8LVET59JNosQvASvcsOX-d7jciv4bRjpDk0HuI/edit#heading=h.z8i2cb69entx

## Data and software are available at:
Sentence-Transformers: https://www.sbert.net/

GENRE: https://github.com/facebookresearch/GENRE

Dataset(s): https://github.com/facebookresearch/KILT

To test on leaderboard: https://eval.ai/web/challenges/challenge-page/689/overview

The main idea is to fact check claims using the knowledge that modern language models, such as BERT, acquire during pre-training.
FEVER will be used as the main dataset, sentence-transformers and GENRE as the main packages.

## Part 1
In this part of the homework, you have to preprocess the FEVER dataset, use an embedding model on each sentence and train a classifier over the generated embeddings.

### Part 1.1 
Download the train, dev and test set from https://github.com/facebookresearch/KILT. In particular:\
http://dl.fbaipublicfiles.com/KILT/fever-train-kilt.jsonl \
http://dl.fbaipublicfiles.com/KILT/fever-dev-kilt.jsonl \
http://dl.fbaipublicfiles.com/KILT/fever-test_without_answers-kilt.jsonl 


A claim should have the following structure:\
{'id': 33078,\
 'input': 'The Boston Celtics play their home games at TD Garden.',\
 'output': [{'answer': 'SUPPORTS',\
   'provenance': [{'bleu_score': 0.517310804423051,\
     'end_character': 404,\
     'end_paragraph_id': 1,\
     'section': 'Section::::Abstract.',\
     'start_character': 227,\
     'start_paragraph_id': 1,\
     'title': 'Boston Celtics',\
     'wikipedia_id': '43376'}]}]} 

Preprocess all claims in the FEVER train, dev and test set as follows:\
1. Keep just the first entry in the output list and discard all keys from the “output” list, except for “answer”. Each datapoint should now have this structure:\
{'id': 33078,\
 'input': 'The Boston Celtics play their home games at TD Garden.',\
 'output': [{'answer': 'SUPPORTS'}]}\
For the test set there is no “output” list 

2. Use the sentence-transformers python library to get sentence embeddings for the 'input' field of each claim. It is mandatory to use the 'paraphrase-distilroberta-base-v1' model. Append the embedding to the claim dict:\
{'id': 33078,\
 'input': 'The Boston Celtics play their home games at TD Garden.',\
 'output': [{'answer': 'SUPPORTS'}],\
"claim_embedding": array(...)} 

3. Save the results in the files "emb_train.jsonl", "emb_dev.jsonl", "emb_test.jsonl"\
### Part 1.2 
Using the claim embeddings as input vectors, representing the labels as 0 ("REFUTES") and 1 ("SUPPORTS"), you must train at most two binary classifiers (e.g. SVM) on the train set, then get the performance on the dev set. It is mandatory to finetune the hyperparameters of the models (using e.g. GridSearch or RandomSearch).

Choose a valid metric to evaluate the performance of the classifier(s). 

## Part 2
In this part of the homework, you have to improve the predictions using the functionalities of the GENRE python package to get additional information about the main entity of each claim.
### Part 2.1
This part can take a long time to run. It is feasible to complete it by the deadline, especially if you follow the recommendations of the fifth lab; but if you encounter too many problems, perform it only on a portion of the data (use only the first X% of the train records; use the whole dev set and the whole test set anyway) and don't leave part 2.2 undone. If only a fraction of the train set is used, specify this in the report.

Retrieve the KILT knowledge source from the following link:
abstract_kilt_knowledgesource.json

For train, dev and test set:\
1. Use GENRE End-to-End Entity Linking on the 'input' field of each claim. It is mandatory to use the "e2e_entity_linking_wiki_abs" pretrained model, either the transformers (recommended) or fairseq one. For each claim, using only the first result of the method, extract the Wikipedia page name found by the method for each entity.
Be careful to follow the recommendations in Lab 5 regarding the use of GENRE and its possible issues.

Example of a GENRE result for one sentence: \
[{'logprob': tensor(-0.1509),\ 
   'text': ' The { Boston Celtics } [ Boston Celtics ] play their { home } [ Home (sports) ] games at { TD Garden } [ TD Garden ]'},\
  {'logprob': tensor(-0.1911),\ 
   'text': ' The { Boston Celtics } [ Boston Celtics ] play their { home } [ Home advantage ] games at { TD Garden } [ TD Garden ]'},\ 
  {'logprob': tensor(-0.2203),\ 
   'text': ' The { Boston Celtics } [ Boston Celtics ] play their { home games } [ Home (sports) ] at { TD Garden } [ TD Garden ]'},\ 
...]\ 
--- The result highlighted in yellow is the one to keep.

Append the results to the claim dictionary.\ 
{'id': 33078,\ 
 'input': 'The Boston Celtics play their home games at TD Garden.',\ 
 'output': [{'answer': 'SUPPORTS'}],\ 
"claim_embedding": array(...),\ 
"wikipedia_pages": ["Boston Celtics", "Home (sports)", "TD Garden"]}

2. For each claim, retrieve all the abstracts in the knowledge base related to the "wikipedia_pages". Order the abstracts according to their length in ascending order (from shortest to longest), then concatenate them separated by a whitespace " ". Add the resulting string to the claim dictionary. If there is no page in the "wikipedia pages" list or none of the pages are found in the KILT knowledge source, consider as if you’ve retrieved a string with only one blank space (" ").
{'id': 33078,
 'input': 'The Boston Celtics play their home games at TD Garden.',
 'output': [{'answer': 'SUPPORTS'}],
"claim_embedding": array(...),
"wikipedia_pages": ["Boston Celtics", "Home (sports)", "TD Garden"],
"wikipedia_abstract": "In sports, home is the place and venue identified… The Boston Celtics are an American professional basketball team… TD Garden, sometimes colloquially referred to as simply the Garden, is a…"}


3. Use the sentence-transformers python library to get sentence embeddings for the "wikipedia_abstract". It is mandatory to use the 'paraphrase-distilroberta-base-v1' model. Since the abstract is usually longer than the claim, set the embedding model max_seq_length to 256 (must be shown in the code). Append the embedding to the claim dict:
{'id': 33078,
 'input': 'The Boston Celtics play their home games at TD Garden.',
 'output': [{'answer': 'SUPPORTS'}],
"claim_embedding": array(...),
"wikipedia_pages": ["Boston Celtics", "Home (sports)", "TD Garden"],
"wikipedia_abstract": "In sports, home is the place and venue identified… The Boston Celtics are an American professional basketball team… TD Garden, sometimes colloquially referred to as simply the Garden, is a…",
"abstract_embedding": array(...)}

4. Save the results in the files "emb_train2.jsonl", "emb_dev2.jsonl", "emb_test2.jsonl". Alternatively, you can overwrite the previous "emb_train.jsonl", "emb_dev.jsonl", "emb_test.jsonl" files.

### Part 2.2
Repeat what you have done in Part1.2, but now use as input vectors the concatenation of claim_embeddings and abstract embeddings. If the previous input matrix had size (number_of_claims, embedding_size), the new concatenated matrix should have size (number_of_claims, embedding_size*2). It is mandatory to use the same classifier(s) as Part1.2, as well as the same evaluation metric.

It is important to remark that it is not requested to find a unique hyperparameter configuration that is good for both Part1.2 and Part2.2, but, what is requested, is to find a good ad-hoc hyperparameter configuration for each part. While it is mandatory to perform the hyperparameter configuration on the same hyperparameters as Part1.2, the parameter space may differ between the two parts.




