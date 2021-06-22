from genre.hf_model import GENRE
from sentence_transformers import SentenceTransformer
from datetime import datetime
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_hf as get_prefix_allowed_tokens_fn
from genre.utils import get_entity_spans_hf as get_entity_spans
import os
import sys
import re
import numpy as np
# to measure exec time
from timeit import default_timer as timer
import json

path = '../'
path_data = '../data/DMT/'
data_json = 'emb_train_58k.json/part-00000-3f205b34-0256-4a93-b78f-075f0b22758d-c000.json'
output_name = 'emb_train'
knowledge_data_path = path + 'data/' + 'abstract_kilt_knowledgesource.json'
save_path = '../part2_res/'
#batch = [1, 50]  # First included, last not
batch_size = 500
num_batches = 10
model_chunks = 10
start_batch = 50
batches = [[i*batch_size, (i+1)*batch_size] for i in range(start_batch, start_batch + num_batches)]
#batches = [[57000, 57026]]
print(batches)

print('Loading Models and getting Knowledge Data')
genre_model = GENRE.from_pretrained(path + "hf_e2e_entity_linking_wiki_abs").eval()
sentence_model = SentenceTransformer('paraphrase-distilroberta-base-v1')
sentence_model.max_seq_length = 256
with open(knowledge_data_path) as f:
    knowledge_data = json.load(f)
print('Models are loaded')


# We will first define several functions and then execute them below
def load_data_batch(batch_size, path, data):
    """
    :param batch_size: (list) [1, 100] --> First is included, last not, so in the next batch we would need [100, 200], [200, 300], ...
    :param path:
    :param data:
    :return:
    """

    data_list = []
    sentences = []
    # with open(path_data + data, 'r', encoding='utf-8') as json_file:
    with open(path + data, 'r', encoding='utf-8') as json_file:
        j = 1
        for line in json_file:
            if j >= batch_size[0]:
                temp = json.loads(line.rstrip('\n|\r'))
                data_list.append(temp)
                sentences.append(temp['input'])
            j += 1
            if j == batch_size[1]:
                break

    return data_list, sentences


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')

# Loop over all the sentences in the initial batch and reduce them into a smaller
# chunk so that we can provide them to the GENRE model
print('Start the big loop')
t_0 = datetime.now()
print(batches)
for batch in batches:
    final_res = []
    print('*************************')
    print('Processing Batch ', batch)
    print('*************************')
    print('Loading Data')
    t_0 = datetime.now()
    data, sentences = load_data_batch(batch, path_data, data_json)
    print(f'Loading time required: {(datetime.now() - t_0).total_seconds()} seconds')
    for chunk in range(0, int(np.ceil(len(sentences)/model_chunks))):
        # Get the Sentence Batch we will provide to the model
        data_chunk = data[chunk*model_chunks:(chunk+1)*model_chunks]
        sentence_batch = sentences[chunk*model_chunks:(chunk+1)*model_chunks]

        # Add initial space to the sentence and remove punctuation
        sentence_batch = [' ' + string if string[-1] != '.' else ' ' + string[:-1] for string in sentence_batch]

        # Run the model
        print('Running Model')
        t = datetime.now()
        model_res = genre_model.sample(sentence_batch)
        print(f'Model {chunk} run in: {(datetime.now() - t).total_seconds()} seconds')

        # Get the best performing
        print('Extracting Results')
        t = datetime.now()
        model_res = np.reshape(model_res, (len(sentence_batch), 5)).tolist()
        wikipedia_pages = [re.findall('\[ (.*?) \]', i[0]['text']) for i in model_res]
        print(f'Getting the actual wikipedia pages in: {(datetime.now() - t).total_seconds()} seconds')

        # Generate Wikipedia Abstract
        print('Getting Wikipedia Pages')
        t = datetime.now()
        wikipedia_abstract = []
        for wikipedia_page in wikipedia_pages:
            temp = []
            for keyword in wikipedia_page:
                try:
                    t1 = datetime.now()
                    temp.append(knowledge_data[keyword])
                except:
                    temp.append(' ')
            temp.sort(key=lambda x: len(x.split()), reverse=False)
            wikipedia_abstract.append(' '.join(temp))
        print(f'Getting the actual wikipedia pages in: {(datetime.now() - t).total_seconds()} seconds')

        # Generate Abstract Embedding
        print('Running Sentence Embedding Model')
        t = datetime.now()
        abstract_embedding = sentence_model.encode(wikipedia_abstract).tolist()
        print(f'Sentence Embedding Model on Wikipedia Abstract: {(datetime.now() - t).total_seconds()} seconds')

        # Store data into final_res list
        print('Store Data Accordingly')
        t = datetime.now()
        for i in range(0, len(data_chunk)):
            data_chunk[i]['wikipedia_pages'] = wikipedia_pages[i]
            data_chunk[i]['wikipedia_abstract'] = wikipedia_abstract[i]
            data_chunk[i]['abstract_embedding'] = abstract_embedding[i]
            final_res.append(data_chunk[i])
        print(f'Time required to store data: {(datetime.now() - t).total_seconds()} seconds')

        print('========================')

    print('========================')

    print(f'Time to run over batch {batch}: {(datetime.now() - t_0).total_seconds()/60} minutes')

    # Save data into separate file
    print('Saving Data')
    t = datetime.now()
    output_string = save_path + str(batch[0]) + '_' + str(batch[1]) + '_' + output_name + '.jsonl'
    dump_jsonl(data=final_res, output_path=output_string, append=False)
    print(f'Time required to save data: {(datetime.now() - t).total_seconds()} seconds')

