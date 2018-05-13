import numpy as np
import json
import re

import sys
sys.path.append('./../')
from config import *
import config

from tqdm import tqdm

_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]

def clean_data(text):
	return text.translate ({ord(c): " " for c in '!@#$%^&*()[]{};:,./<>?\|`~-=_+"'})

def preprocess_data(text):
	# text = clean_data(text)
	text = text.lower()
	text = text.strip()
	return text

def build_dataset(words):
  words += ['UNK','<pad>']

  dictionary = dict()

  for word in tqdm(words):
    if dictionary.get(word, -1)==-1:
        dictionary[word] = len(dictionary)

  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

  return dictionary, reverse_dictionary

def get_num_words(words):
  return len(words)

def read_data(file_path):
	data = []

	with open(file_path, 'r') as f:
		lines = f.readlines()
		for line in tqdm(lines):
			data.append(preprocess_data(line))

	return data


def get_words(sentences):
	words = []
	for sent in tqdm(sentences):
		words += basic_tokenizer(sent)

	return words

def build_vocab(dataDir, load=False):
	print('dataDir')
	print(dataDir)
	print('Getting sentences...')
	sentences = read_data(dataDir)
	print('Getting words...')
	words = get_words(sentences)
	print('Building dictionary...')
	dictionary, reverse_dictionary = build_dataset(words)
	vocabulary_size = len(dictionary)
	
	print("Vocabulary_size")
	print(vocabulary_size)

	if load:
		# with open(data_json_path,'r') as f:
		# 	data_json = json.load(f)
		pass
	else:
		#write the dictionary to a file
		with open(dictionaryPath, 'w') as f:
			json.dump(dictionary,f,indent=4)

	return sentences, words, dictionary, reverse_dictionary, vocabulary_size

def build_vocab_parsed(words, load=False):
	dictionary, reverse_dictionary = build_dataset(words)
	vocabulary_size = len(dictionary)
	
	print("Vocabulary_size")
	print(vocabulary_size)

	if load:
		# with open(data_json_path,'r') as f:
		# 	data_json = json.load(f)
		pass
	else:
		#write the dictionary to a file
		with open(dictionaryPathParsed, 'w') as f:
			json.dump(dictionary,f,indent=4)

	return dictionary, reverse_dictionary, vocabulary_size

def build_vocab_parsed_2(words, load=False):
	dictionary, reverse_dictionary = build_dataset(words)
	vocabulary_size = len(dictionary)
	
	print("Vocabulary_size")
	print(vocabulary_size)

	if load:
		# with open(data_json_path,'r') as f:
		# 	data_json = json.load(f)
		pass
	else:
		#write the dictionary to a file
		with open(dictionaryPathParsed_2, 'w') as f:
			json.dump(dictionary,f,indent=4)

	return dictionary, reverse_dictionary, vocabulary_size

def loadVocabParsed(dictionary_path=dictionaryPathParsed):
	with open(dictionary_path,'r') as f:
		dictionary = json.load(f)

	vocab_size = len(dictionary)

	return dictionary, vocab_size

def getTrainingDataParsed(data_json, dictionary, reverse_dictionary):
	if len(data_json)==0:
		return data_json

	# print('data_json')
	# print(data_json)
	for data_item in data_json:
		if data_item.get('def_word',-1) != -1:
			data_item['def_word'] = dictionary.get(data_item['def_word'],dictionary['UNK'])
		if data_item.get('next',-1) != -1:
			data_item['word'] = dictionary.get(data_item['word'],dictionary['UNK'])
			data_item['next'] = getTrainingDataParsed(data_item['next'],dictionary,reverse_dictionary)
		else:
			if str(type(data_item['dep_tree'])) == "<class 'dict'>":
				data_item['dep_tree']['word'] = dictionary.get(data_item['dep_tree']['word'],dictionary['UNK'])
				data_item['dep_tree']['next'] = getTrainingDataParsed(data_item['dep_tree']['next'],dictionary,reverse_dictionary)
			else:
				data_item['dep_tree'] = getTrainingDataParsed(data_item['dep_tree'],dictionary,reverse_dictionary)

	return data_json

def prepare_sentence_tokens(dataNoise, embedding_out_names_path, loadOne=False):
	print('Tokenizing sentences...')
	in_sent_arr = []
	in_token_arr = []
	out_word_arr = []
	out_token_arr = []
	out_arr = []
	max_seq_len = 0

	print('Loading data...')
	with open(dataNoise,'r') as f:
		data_json = json.load(f)

	with open(embedding_out_names_path,'r') as f:
		out_names_arr = json.load(f)
	print('Loaded data')
	# word_count = 0
	if not loadOne:
		for word,def_arr in tqdm(data_json.items()):
			tokenized_out = basic_tokenizer(word)
			for line in def_arr:
				tokenized_in = basic_tokenizer(line)

				if len(tokenized_in)>max_seq_len:
					max_seq_len = len(tokenized_in)

				in_sent_arr.append(line)
				in_token_arr.append(tokenized_in)

				out_word_arr.append(word)
				out_token_arr.append(tokenized_out)

				# out_arr.append(word_count)
				out_arr.append(out_names_arr.index(word))
	else:
		for name in tqdm(out_names_arr):
			word = name
			tokenized_out = basic_tokenizer(word)
			
			line = data_json[word][0]

			tokenized_in = basic_tokenizer(line)

			if len(tokenized_in)>max_seq_len:
				max_seq_len = len(tokenized_in)

			in_sent_arr.append(line)
			in_token_arr.append(tokenized_in)

			out_word_arr.append(word)
			out_token_arr.append(tokenized_out)

			# out_arr.append(word_count)
			out_arr.append(out_names_arr.index(word))			
			
		# word_count += 1
	print("Done Tokenizing")
	print("max_seq_len")
	print(max_seq_len)

	return in_sent_arr, in_token_arr, out_word_arr, out_token_arr, out_arr, max_seq_len, data_json



def getNegatives(_class, num_classes):
	negative_classes = []
	while len(negative_classes) < NUM_NEGATIVES:
		rand_class = random.randint(0,num_classes)
		if rand_class != _class:
			negative_classes.append(rand_class)

	return negative_classes

def get_training_data(in_token_arr, out_arr, out_token_arr, seq_len, dictionary):
	inp = []
	out = []

	for index,sent in tqdm(enumerate(in_token_arr)):
		tokenized_in = in_token_arr[index]
		tokenized_out = out_token_arr[index]
		assert len(tokenized_out)==1, "There must be only one output word"
		cursor = 0

		padding_len = (seq_len - len(tokenized_in))
		assert padding_len>=0, "padding length must be >= 0"

		tokenized_in = tokenized_in+['<pad>']*padding_len

		seq_in = []
		for token in tokenized_in:
			# if index==0:
			# 	print("token")
			# 	print(token)
			# 	print(dictionary.get(token))
			cur_in_token = dictionary.get(token, -1)
			seq_in.append(cur_in_token if cur_in_token>-1 else dictionary["UNK"])

		#reverse the sequence of tokens for input
		inp.append(seq_in[::-1])
		out.append([out_arr[index]])
		# negatives.append(getNegatives(index,NUM_CLASSES))

	return inp, out


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)

    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def saveDataInFiles(training_in_data, training_out_data, in_sent_arr, out_word_arr):
	with open('../data/sample/training_in_data.txt', 'w') as f:
		json.dump(training_in_data,f,indent=4)
	with open('../data/sample/training_out_data.txt', 'w') as f:
		json.dump(training_out_data,f,indent=4)
	with open('../data/sample/training_in_data_text.txt', 'w') as f:
		json.dump(in_sent_arr,f,indent=4)
	with open('../data/sample/training_out_data_text.txt', 'w') as f:
		json.dump(out_word_arr,f,indent=4)

#load the embedding matrix for words
def loadDictEmbedings(out_embedding_arr_path):
  with open(out_embedding_arr_path, 'r') as f:
    print()
    print('Loading dictEmbeddings from {}'.format(out_embedding_arr_path))
    dictEmbedding = json.load(f)
    dictEmbedding = np.array(dictEmbedding)
    return dictEmbedding

def loadVocab(dictionaryFile, dataNoise):
	#read the dictionary from a file
	with open(dictionaryFile, 'r') as f:
		dictionary = json.load(f)
	vocabulary_size = len(dictionary)

	#load the out json
	with open(dataNoise,'r') as f:
	  data_json = json.load(f)


	#get the reverse dictionary
	rev_dict = np.empty(vocabulary_size,dtype=object)
	for key,value in dictionary.items():
		rev_dict[value] = key

	return dictionary, rev_dict, vocabulary_size, data_json

def readTestData(testDataPath):
	#read the test data
	docs_raw = []
	answers_raw = []
	with open(testDataPath,'r') as f:
	  lines = f.readlines()
	  for index,line in enumerate(lines):
	    if index%2==0:
	      docs_raw.append(line.strip('\n'))
	    else:
	      answers_raw.append(line.strip('\n'))

	return answers_raw, docs_raw

def process_test_data(answers_raw, docs_raw):
	docs = []
	answers = []
	#preprocess the docs
	for doc in docs_raw:
	  docs.append(preprocess_data(doc))
	for answer in answers_raw:
	  assert len(answer.split(' '))==1, "The answer in test data must be a single word"
	  answers.append(preprocess_data(answer))

	return docs, answers

def tokenize_test_sentences(docs):
	#training data preprocessor
	print('Tokenizing sentences...')
	in_sent_arr = []
	in_token_arr = []
	for index,line in enumerate(docs):
	  tokenized = basic_tokenizer(line)

	  in_sent_arr.append(line)
	  in_token_arr.append(tokenized)
	print("Done Tokenizing")

	return in_sent_arr, in_token_arr

#prepare the test data
def get_test_data_words(in_sent_arr, in_token_arr, dictionary):
  inp = []
  out = []
  print('in token arr')
  print(in_token_arr)
  for index,sent in enumerate(in_sent_arr):
    tokenized_in = dictionary.get(in_token_arr[index][0],dictionary['UNK'])
    inp.append(tokenized_in)

  return inp, [-1 for _ in inp]


def loadOutNames(out_names_path):
	with open(out_names_path, 'r') as f:
	  out_names = json.load(f)

	return out_names

def getOutNamesParsed(data_json_path=parsedJsonInputFile):
	with open(data_json_path,'r') as f:
		data_json = json.load(f)

	out_names = []
	for data in data_json:
		out_names.append(data['def_word'])
	
	return out_names

def getOutWordsDictNum(out_names, dictionary):
	out_words_dict_num = []

	for name in out_names:
		out_words_dict_num.append(dictionary[name])

	return out_words_dict_num

# def getPosWords(data_json_path=config.parsedJsonInputFile):
# 	with open(data_json_path,'r') as f:
# 		data_json = json.load(f)

# 	for data in data_json:

def getPosWords(node_arr):
	if len(node_arr)<=0:
		return []

	pos_words_arr = []
	for node in node_arr:
		if node.get('dep_tree',-1)!=-1 and\
		 str(type(node['dep_tree']))=="<class 'list'>":
		 node_next = node['dep_tree']
		 pos_val = None
		if node.get('dep_tree',-1)!=-1 and\
		 str(type(node['dep_tree']))=="<class 'dict'>":
			node_next = node['dep_tree']['next']
			pos_val = node['dep_tree']['pos']
		if node.get('next',-1)!=-1:
			node_next = node['next']
			pos_val = node['pos']

		if pos_val:
			pos_words_arr.append(pos_val)

		pos_words_arr += getPosWords(node_next)

	return pos_words_arr


def makePOSDict(pos_words, save_path=config.posDictPath):
	pos_dict = {}
	for word in pos_words:
		pos_dict[word] = len(pos_dict)

	with open(save_path,'w') as f:
		json.dump(pos_dict,f,indent=4)

	return pos_dict

def getPosDict(save_path=config.posDictPath):
	with open(save_path,'r') as f:
		pos_dict = json.load(f)

	return pos_dict

def getTestWordsData(data_path):
	with open(data_path,'r') as f:
		data = f.read()

	data_arr = data.split('\n\n')

	x = []
	y = []
	for data in data_arr:
		lines = data.split('\n')
		x.append(lines[0])

		y.append(lines[1:])

	return x, y

def process_test_data_words(answers_raw, docs_raw):
	docs = []
	answers = []
	#preprocess the docs
	for doc in docs_raw:
		assert len(doc.split(' '))==1, "All test data must be single words"
		docs.append(preprocess_data(doc))
	for answer_arr in answers_raw:
		answers_tmp = []
		for answer in answer_arr:
			assert len(answer.split(' '))==1, "All test data must be single words"
			answers_tmp.append(preprocess_data(answer))
		answers.append(answers_tmp)

	return docs, answers

def getAllTestWords(answers, docs, dictionary):
	all_words = []
	all_words_dict = []

	for answer in answers:
		all_words += answer
	for doc in docs:
		all_words.append(doc)

	for word in all_words:
		all_words_dict.append(dictionary.get(word,dictionary['UNK']))

	return all_words, all_words_dict