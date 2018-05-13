import sys
sys.path.append('./../')
sys.path.append('./')
import config
import data_helpers
import json
import numpy as np

def load_data_plain(data_path):
	x_arr = []
	y_arr = []

	# sentence_id = -1
	with open(data_path,'r') as f:
		lines = f.readlines()[1:]

		for line in lines:
			line = line.strip()

			cols = line.split('	')

			# if sentence_id != cols[1]:
			sent = cols[2]
			sentiment = int(cols[3])
			sentence_id = cols[1]

			sent = data_helpers.preprocess_data(sent)

			x_arr.append(sent)
			y_arr.append(sentiment)
			# else:
			# 	continue


	return x_arr, y_arr

def load_test_data_plain(data_path):
	x_arr = []
	phrase_arr = []

	# sentence_id = -1
	with open(data_path,'r') as f:
		lines = f.readlines()[1:]

		for index,line in enumerate(lines):
			line = line.strip()

			cols = line.split('	')

			# if sentence_id != cols[1]:
			# print(line)
			# print(cols)
			# print(index)
			# print()
			if len(cols)==3:
				sent = cols[2]
			else:
				sent = ""
			sentence_id = cols[1]
			phrase_id = cols[0]

			sent = data_helpers.preprocess_data(sent)

			x_arr.append(sent)
			phrase_arr.append(phrase_id)
			# else:
			# 	continue

	return x_arr, phrase_arr

def prepare_sentence_tokens(sentences):
	print('Tokenizing sentences...')
	in_sent_arr = []
	in_token_arr = []

	max_seq_len = 0


	for line in sentences:
		tokenized_in = data_helpers.basic_tokenizer(line)

		if len(tokenized_in)>max_seq_len:
			max_seq_len = len(tokenized_in)

		in_sent_arr.append(line)
		in_token_arr.append(tokenized_in)

	print("Done Tokenizing")
	print("max_seq_len")
	print(max_seq_len)

	return in_sent_arr, in_token_arr, max_seq_len

def get_training_data(in_token_arr, seq_len, dictionary):
	inp = []

	for index,sent in enumerate(in_token_arr):
		tokenized_in = in_token_arr[index]

		padding_len = (seq_len - len(tokenized_in))
		assert padding_len>=0, "padding length must be >= 0"

		tokenized_in = tokenized_in+['<pad>']*padding_len

		seq_in = []
		for token in tokenized_in:
			cur_in_token = dictionary.get(token, -1)
			seq_in.append(cur_in_token if cur_in_token>-1 else dictionary["UNK"])

		#reverse the sequence of tokens for input
		inp.append(seq_in[::-1])

	return inp

def saveDataInFiles(training_in_data, in_sent_arr):
	with open('../data/sample/mr/training_in_data.txt', 'w') as f:
		json.dump(training_in_data,f,indent=4)
	with open('../data/sample/mr/training_in_data_text.txt', 'w') as f:
		json.dump(in_sent_arr,f,indent=4)

def loadVocab(dictionaryFile):
	#read the dictionary from a file
	with open(dictionaryFile, 'r') as f:
		dictionary = json.load(f)
	vocabulary_size = len(dictionary)

	#get the reverse dictionary
	rev_dict = np.empty(vocabulary_size,dtype=object)
	for key,value in dictionary.items():
		rev_dict[value] = key

	return dictionary, rev_dict, vocabulary_size

def saveTestedSentiments(phrase_ids, sentiments):
	data_to_write = ''
	data_to_write += 'PhraseId,Sentiment\n'
	for index,phrase_id in enumerate(phrase_ids):
		data_to_write += str(phrase_id)+','+str(sentiments[index])+'\n'
	with open(config.mr_save_test_path, 'w') as f:
		f.write(data_to_write)

	return