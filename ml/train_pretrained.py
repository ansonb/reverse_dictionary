import tensorflow as tf
import collections

import os
import json, xmljson
from lxml.etree import fromstring, tostring
import re

from tensorflow.python.ops import rnn
import datetime

import numpy as np

import pdb

dataDir = '/home/anson/Desktop/hackathons/money_control/word2vec/word2vec/GST/train'
words = []
skip_window = 2

embedding_size = 32

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
	text = clean_data(text)
	text = text.lower()
	return text

def build_dataset(words):
  count = ['UNK']
  count.extend(collections.Counter(words))

  dictionary = dict()

  for word in count:
    dictionary[word] = len(dictionary)
  data = list()
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
    data.append(index)
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

  return data, count, dictionary, reverse_dictionary

# Step 2: Build the dictionary and replace rare words with UNK token.
def get_num_words(words):
  return len(words)


#read data for wor2vec
# Read the data into a list of strings.
def read_data(dataDir,skip_window=skip_window):
  data = []
  padding = ["<eos>"]*int(skip_window)
  for file in os.listdir(dataDir):
    file_path = os.path.join(dataDir,file)

    jsonData = json.load(open(file_path,'r'))

    header = jsonData['header'].split()
    summary = jsonData['summary'].split()

    body = jsonData['body']
    body = preprocess_data(body)
    body = body.split()

    # data += body + padding
    data += body

  return data

words = read_data(dataDir)
data, count, dictionary, reverse_dictionary = build_dataset(words)
vocabulary_size = len(dictionary)
print("Vocabulary_size")
print(vocabulary_size)

#write the dictionary to a file
with open('dictionary.json', 'w') as f:
	json.dump(dictionary,f,indent=4)


#training data preprocessor
docs = []
max_seq_len = 0
data_dir = "/home/anson/Desktop/hackathons/money_control/data/hackathon/GST"
for file in os.listdir(data_dir):
	file_path = os.path.join(data_dir,file)
	with open(file_path, 'r', encoding='latin') as f:
		data_xml = ""
		lines = f.readlines()
		for line in lines:
			data_xml += line

		data_xml.replace("","")

		data_to_strip = '<?xml version="1.0" encoding="utf-8"?>'
		strippedData = data_xml[len(data_to_strip):]

		jsonData = fromstring(strippedData)
		jsonData=json.loads(json.dumps(xmljson.badgerfish.data(jsonData)))

		header = jsonData['news']['article']['Heading']['$']
		summary = jsonData['news']['article']['Summary']['$']
		body = jsonData['news']['article']['Body']['$']
		_docs = body.split('<p></p>')
		for doc in _docs:
			tokenized = basic_tokenizer(doc)
			if len(tokenized)>max_seq_len:
				max_seq_len = len(tokenized)

			doc = preprocess_data(doc)
			docs.append(doc)

window = 10
seq_len = 10
hop = 1
#prepare the training data
def get_training_data(docs):
	inp,inp_text = [],[]
	out,out_text = [],[]

	for doc in docs:
		cursor = 0
		tokenized = basic_tokenizer(doc)
		#if sentence length is less than window length then will not enter for loop
		for _ in range(len(tokenized)-window-hop):
			seq_in = ["<eos>" for _ in range(seq_len)]
			seq_out = ["<eos>" for _ in range(seq_len)]
			seq_in_text = ["<eos>" for _ in range(seq_len)]
			seq_out_text = ["<eos>" for _ in range(seq_len)]

			seq_in[0:window] = tokenized[cursor:cursor+window]
			seq_out[0:window] = tokenized[cursor+hop:cursor+hop+window]
			seq_in_text[0:window] = tokenized[cursor:cursor+window]
			seq_out_text[0:window] = tokenized[cursor+hop:cursor+hop+window]

			for index,token in enumerate(seq_in):
				seq_in[index] = dictionary.get(token, None) or dictionary["UNK"]
			for index,token in enumerate(seq_out):
				seq_out[index] = dictionary.get(token, None) or dictionary["UNK"]

			#reverse the sequence of tokens for both input and output
			inp.append(seq_in[::-1])
			out.append(seq_out[::-1])
			inp_text.append(seq_in_text[::-1])
			out_text.append(seq_out_text[::-1])

			cursor += 1

	return inp, out, inp_text, out_text

training_in_data, training_out_data, training_in_data_text, training_out_data_text = get_training_data(docs)

with open('training_in_data.txt', 'w') as f:
	json.dump(training_in_data,f,indent=4)
with open('training_out_data.txt', 'w') as f:
	json.dump(training_out_data,f,indent=4)
with open('training_in_data_text.txt', 'w') as f:
	json.dump(training_in_data_text,f,indent=4)
with open('training_out_data_text.txt', 'w') as f:
	json.dump(training_out_data_text,f,indent=4)

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    overflow_size = data_size%batch_size
    padding_size = batch_size-overflow_size

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            if batch_num == num_batches_per_epoch-1:
                start_index = batch_num * batch_size - padding_size
            else: 
                start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def lstm_cell(lstm_size, output_keep_prob=1.0):
  # return tf.contrib.rnn.BasicLSTMCell(lstm_size)
  encoDecoCell = tf.contrib.rnn.BasicLSTMCell(lstm_size, state_is_tuple=True)  # Or GRUCell, LSTMCell(args.hiddenSize)
  #Only for training output_keep_prob is 0.5
  encoDecoCell = tf.contrib.rnn.DropoutWrapper(encoDecoCell, input_keep_prob=1.0, output_keep_prob=output_keep_prob)  # TODO: Custom values
  return encoDecoCell

def train_step(x_batch, y_batch, input_x, input_y, train_step_op, loss, step_num):
    """
    A single training step
    """
    feed_dict = {
      input_x: x_batch,
      input_y: y_batch,
    }
    _, loss = sess.run(
        [train_step_op, loss],
        feed_dict)

    print("{}: step, {} loss".format(step_num, loss))


graph = tf.Graph()
num_epochs = 200
save_every = 100
checkpoint_dir = "models_w2vecPretrained"
checkpoint_prefix = os.path.join(checkpoint_dir, "model")

checkpoint_dir_w2vec = './../../word2vec/word2vec/model_seperate_arr'
checkpoint_file_w2vec = tf.train.latest_checkpoint(checkpoint_dir_w2vec)

ops_to_not_train = ['my_embedding','encoder_outputs_x','encoder_outputs_y']

with graph.as_default():
	sess = tf.Session(graph=graph)
	with sess.as_default():
		sequence_length = seq_len
		vocab_size = len(dictionary)
		output_keep_prob = 1.0

		batch_size = 10

		hiddenSize = 256

		input_x = tf.placeholder(tf.int32, shape=(batch_size, sequence_length), name="input_x")
		input_y = tf.placeholder(tf.int32, shape=(batch_size, sequence_length), name="input_y")

		# input_x = tf.Variable(np.zeros((batch_size, sequence_length)), dtype=tf.int32,name="input_var")
		# input_x = tf.assign(input_x,input_x_plh)

		# Load the saved meta graph and restore variables
		# with tf.variable_scope("saver_w2vec"):
		saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file_w2vec))
		saver.restore(sess, checkpoint_file_w2vec)

		# Get the placeholders from the graph by name
		embedding = graph.get_operation_by_name("my_embedding").outputs[0]
		# print("embedding shape")
		# print(embedding.shape)
		embedding_var = tf.Variable(embedding,trainable=False,name="my_embedding_2")
		# embedding_var = tf.Variable(
  		#tf.random_uniform([vocabulary_size, embedding_size], minval=-1.0, maxval=1.0),name="my_embedding")

		with tf.variable_scope("lstm_par2vec"):
			embedded_chars_x = tf.nn.embedding_lookup(embedding_var, input_x)
			embedded_chars_x = tf.transpose(embedded_chars_x,perm=[0,2,1])	
			embedded_chars_x_list = [tf.reshape(tf.slice(embedded_chars_x,[0,0,i],[batch_size,embedding_size,1]),[batch_size,embedding_size]) for i in range(sequence_length)]


			embedded_chars_y = tf.nn.embedding_lookup(embedding_var, input_y)
			embedded_chars_y = tf.transpose(embedded_chars_y,perm=[0,2,1])
			embedded_chars_y_list = [tf.reshape(tf.slice(embedded_chars_y,[0,0,i],[batch_size,embedding_size,1]),[batch_size,embedding_size]) for i in range(sequence_length)]


			encoDecoCell1 = lstm_cell(hiddenSize, output_keep_prob=output_keep_prob)
			encoDecoCell2 = lstm_cell(hiddenSize, output_keep_prob=output_keep_prob)
			# print("encodeco cell 1")
			# print(encoDecoCell1.name)
			# print()
			# print("encodeco cell 2")
			# print(encoDecoCell2.name)

			encoDecoCell = tf.contrib.rnn.MultiRNNCell([encoDecoCell1,encoDecoCell2], state_is_tuple=True)
			# sess.run(tf.global_variables_initializer())

			encoder_outputs_x, encoder_state_x = rnn.static_rnn(encoDecoCell, embedded_chars_x_list,dtype=tf.float32)
			# output_x = tf.Variable(encoder_outputs_x[-1], name="encoder_outputs_x")
			# print("encoder outputs x")
			# print(encoder_outputs_x)
			encoder_outputs_y, encoder_state_y = rnn.static_rnn(encoDecoCell, embedded_chars_y_list,dtype=tf.float32)
			# output_y = tf.Variable(encoder_outputs_y[-1], name="encoder_outputs_y")

			loss = tf.losses.mean_squared_error(encoder_outputs_y[-1],encoder_outputs_x[-1])
			train_step_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
		batches = batch_iter(
                    list(zip(training_in_data, training_out_data)), batch_size, num_epochs)

		print("*************Initialising variables*****************")
		for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
			if v.op.name in ops_to_not_train:
				continue
			print("Initialising " + v.op.name)
			sess.run(v.initializer)
		print("Uninitialised varaiables")
		print(tf.report_uninitialized_variables())
		# with tf.variable_scope("saver"):
		# 	saver = tf.train.Saver(max_to_keep=5)
		saver.save(sess,checkpoint_prefix,global_step=0)
		try:
			for batch_num,batch in enumerate(batches):
				x_batch, y_batch = zip(*batch)
				train_step(x_batch,y_batch,input_x,input_y,train_step_op,loss,batch_num)
				if batch_num%save_every == 0:
					saver.save(sess,checkpoint_prefix,global_step=batch_num)
					print("Saving checkpoint to {}-{}".format(checkpoint_prefix,batch_num))
		except Exception as e:
			print("***********Exception******************")
			print(e)
			saver.save(sess,checkpoint_prefix,global_step=batch_num)
			print("Saving checkpoint to {}-{}".format(checkpoint_prefix,batch_num))


# # import tensorflow as tf

# # model_folder

# # sequence_length = 100
# # vocab_size = 
# # embedding_size = 128
# # output_keep_prob = 1.0

# # input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
# # input_y = tf.placeholder(tf.int32, [None, sequence_length], name="input_y")

# # with tf.device('/cpu:0'), tf.name_scope("embedding"):
# #     W = tf.Variable(
# #         tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
# #         name="W")

# #     embedded_chars = tf.nn.embedding_lookup(W, input_x)

# # encoDecoCell1 = lstm_cell(args.hiddenSize, output_keep_prob=output_keep_prob)
# # encoDecoCell2 = lstm_cell(args.hiddenSize, output_keep_prob=output_keep_prob)
# # encoDecoCell = tf.contrib.rnn.MultiRNNCell([encoDecoCell1,encoDecoCell2], state_is_tuple=False)

# # encoder_outputs_x, encoder_state_x = rnn.static_rnn(encoDecoCell, input_x, dtype=dtype)
# # encoder_outputs_x, encoder_state_x = rnn.static_rnn(encoDecoCell, input_x, dtype=dtype)

# import tensorflow as tf
# import collections

# import os
# import json, xmljson
# from lxml.etree import fromstring, tostring
# import re

# from tensorflow.python.ops import rnn
# import datetime

# import numpy as np

# dataDir = '/home/anson/Desktop/hackathons/money_control/word2vec/word2vec/GST/train'
# words = []
# skip_window = 2




# def build_dataset(words):
#   count = [['UNK', -1]]
#   # count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
#   count.extend(collections.Counter(words))
#   dictionary = dict()
#   print("count len")
#   print(len(count))
#   for word, _ in count:
#     dictionary[word] = len(dictionary)
#   data = list()
#   unk_count = 0
#   for word in words:
#     if word in dictionary:
#       index = dictionary[word]
#     else:
#       index = 0  # dictionary['UNK']
#       unk_count += 1
#     data.append(index)
#   count[0][1] = unk_count
#   reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
#   print("reverse_dictionary")
#   print(len(reverse_dictionary))
#   # print(reverse_dictionary)
#   return data, count, dictionary, reverse_dictionary

# # Step 2: Build the dictionary and replace rare words with UNK token.
# def get_num_words(words):
#   return len(words)
# # vocabulary_size = 50000
# # vocabulary_size = 20000
# # print(vocabulary_size)


# #read data for wor2vec
# # Read the data into a list of strings.
# def read_data(dataDir,skip_window=skip_window):
#   """Extract the first file enclosed in a zip file as a list of words"""
#   data = []
#   padding = ["<eos>"]*int(skip_window)
#   for file in os.listdir(dataDir):
#     file_path = os.path.join(dataDir,file)

#     jsonData = json.load(open(file_path,'r'))

#     header = jsonData['header'].split()
#     summary = jsonData['summary'].split()
#     tmpBody = jsonData['body']
#     body_cleaned = tmpBody.translate ({ord(c): "" for c in '!@#$%^&*()[]{};:,./<>?\|`~-=_+"'})

#     body = body_cleaned.split()
#     data += body + padding
#   return data

# words = read_data(dataDir)
# data, count, dictionary, reverse_dictionary = build_dataset(words)

# _WORD_SPLIT = re.compile("([.,!?\"':;)(])")
# def basic_tokenizer(sentence):
#   """Very basic tokenizer: split the sentence into a list of tokens."""
#   words = []
#   for space_separated_fragment in sentence.strip().split():
#     words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
#   return [w for w in words if w]

# def clean_data(text):
# 	return text.translate ({ord(c): "" for c in '!@#$%^&*()[]{};:,./<>?\|`~-=_+"'})

# def preprocess_data(text):
# 	text = clean_data(text)
# 	text = text.lower()
# 	return text

# #training data preprocessor
# docs = []
# max_seq_len = 0
# data_dir = "/home/anson/Desktop/hackathons/money_control/data/hackathon/GST"
# for file in os.listdir(data_dir):
# 	file_path = os.path.join(data_dir,file)
# 	with open(file_path, 'r', encoding='latin') as f:
# 		data_xml = ""
# 		lines = f.readlines()
# 		for line in lines:
# 			data_xml += line

# 		data_xml.replace("","")

# 		data_to_strip = '<?xml version="1.0" encoding="utf-8"?>'
# 		strippedData = data_xml[len(data_to_strip):]

# 		print('processing ' + file)
# 		jsonData = fromstring(strippedData)
# 		jsonData=json.loads(json.dumps(xmljson.badgerfish.data(jsonData)))

# 		# header = jsonData['news']['article']['Heading']['$']
# 		# summary = jsonData['news']['article']['Summary']['$']
# 		body = jsonData['news']['article']['Body']['$']
# 		_docs = body.split('<p></p>')
# 		for doc in _docs:
# 			tokenized = basic_tokenizer(doc)
# 			if len(tokenized)>max_seq_len:
# 				max_seq_len = len(tokenized)

# 			doc = preprocess_data(doc)
# 			docs.append(doc)

# print('docs length')
# print(len(docs))

# print("seq len")
# print(max_seq_len)

# window = 8
# seq_len = 10
# #prepare the training data
# def get_training_data(docs):
# 	inp = []
# 	out = []

# 	for doc in docs:
# 		cursor = 0
# 		tokenized = basic_tokenizer(doc)
# 		for _ in range(len(tokenized)-window-1):
# 			seq_in = ["<eos>" for _ in range(seq_len)]
# 			seq_out = ["<eos>" for _ in range(seq_len)]

# 			seq_in[0:window] = tokenized[cursor:cursor+window]
# 			seq_out[0:window] = tokenized[cursor+1:cursor+1+window]

# 			inp.append(seq_in[::-1])
# 			out.append(seq_out[::-1])

# 			cursor += 1

# 	return inp, out

# training_in_data, training_out_data = get_training_data(docs)
# print("training data")
# print(len(training_in_data))
# print(len(training_out_data))

# def batch_iter(data, batch_size, num_epochs, shuffle=True):
#     """
#     Generates a batch iterator for a dataset.
#     """
#     data = np.array(data)
#     data_size = len(data)
#     num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
#     for epoch in range(num_epochs):
#         # Shuffle the data at each epoch
#         if shuffle:
#             shuffle_indices = np.random.permutation(np.arange(data_size))
#             shuffled_data = data[shuffle_indices]
#         else:
#             shuffled_data = data
#         for batch_num in range(num_batches_per_epoch):
#             start_index = batch_num * batch_size
#             end_index = min((batch_num + 1) * batch_size, data_size)
#             yield shuffled_data[start_index:end_index]

# def lstm_cell(lstm_size, output_keep_prob=1.0):
#   # return tf.contrib.rnn.BasicLSTMCell(lstm_size)
#   encoDecoCell = tf.contrib.rnn.BasicLSTMCell(lstm_size, state_is_tuple=False)  # Or GRUCell, LSTMCell(args.hiddenSize)
#   #Only for training output_keep_prob is 0.5
#   encoDecoCell = tf.contrib.rnn.DropoutWrapper(encoDecoCell, input_keep_prob=1.0, output_keep_prob=output_keep_prob)  # TODO: Custom values
#   return encoDecoCell

# def train_step(x_batch, y_batch, input_x, input_y, train_step_op, loss):
#     """
#     A single training step
#     """
#     feed_dict = {
#       input_x: x_batch,
#       input_y: y_batch,
#     }
#     _, loss = sess.run(
#         [train_step_op, loss],
#         feed_dict)
#     time_str = datetime.datetime.now().isoformat()
#     print("{}: step {}, loss {:g}, acc {:g}".format(time_str, loss))

# checkpoint_dir = './../../word2vec/word2vec/model'
# checkpoint_file_w2vec = tf.train.latest_checkpoint(checkpoint_dir)
# graph = tf.Graph()
# num_epochs = 200
# save_every = 100
# checkpoint_dir = "models"
# checkpoint_prefix = os.path.join(checkpoint_dir, "model")
# with graph.as_default():
# 	sess = tf.Session()
# 	with sess.as_default():
# 		# Load the saved meta graph and restore variables
# 		saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file_w2vec))
# 		saver.restore(sess, checkpoint_file_w2vec)

# 		# Get the placeholders from the graph by name
# 		embedding = graph.get_operation_by_name("my_embedding").outputs[0]
# 		print("embedding shape")
# 		print(embedding.shape)
# 		embeding_var = tf.Variable(embedding,trainable=False)


# 		sequence_length = seq_len
# 		vocab_size = len(dictionary)
# 		embedding_size = 32
# 		output_keep_prob = 1.0

# 		batch_size = 10

# 		hiddenSize = 256

# 		# decoderInputs  = [tf.placeholder(tf.float32, [batch_size,num_decoder_symbols], name='inputs') for _ in range(num_words)]
# 		input_x = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_x")
# 		input_y = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_y")

# 		embedded_chars_x = tf.nn.embedding_lookup(embeding_var, input_x)
# 		embedded_chars_x = tf.transpose(embedded_chars_x,perm=[0,2,1])
# 		print("embedded_chars_x.shape")
# 		print(embedded_chars_x.shape)		
# 		embedded_chars_x_list = [tf.reshape(tf.slice(embedded_chars_x,[0,0,i],[batch_size,embedding_size,1]),[batch_size,embedding_size]) for i in range(sequence_length)]
# 		print("embedded_chars_x_list.shape")
# 		print(embedded_chars_x_list)

# 		embedded_chars_y = tf.nn.embedding_lookup(embeding_var, input_y)
# 		embedded_chars_y = tf.transpose(embedded_chars_y,perm=[0,2,1])
# 		print("embedded_chars_y.shape")
# 		print(embedded_chars_y.shape)
# 		embedded_chars_y_list = [tf.reshape(tf.slice(embedded_chars_y,[0,0,i],[batch_size,embedding_size,1]),[batch_size,embedding_size]) for i in range(sequence_length)]
# 		print("embedded_chars_y_list.shape")
# 		print(embedded_chars_y_list)


# 		encoDecoCell1 = lstm_cell(hiddenSize, output_keep_prob=output_keep_prob)
# 		encoDecoCell2 = lstm_cell(hiddenSize, output_keep_prob=output_keep_prob)
# 		encoDecoCell = tf.contrib.rnn.MultiRNNCell([encoDecoCell1,encoDecoCell2], state_is_tuple=False)

# 		encoder_outputs_x, encoder_state_x = rnn.static_rnn(encoDecoCell, embedded_chars_x_list,dtype=tf.float32)
# 		encoder_outputs_y, encoder_state_y = rnn.static_rnn(encoDecoCell, embedded_chars_y_list,dtype=tf.float32)

# 		vector = tf.Variable(encoder_outputs_x,name="lstm_outputs")
# 		# print("encoder state x")
# 		# print(encoder_state_x)
# 		# print("encoder state x")
# 		# print(encoder_outputs_x)

# 		loss = tf.losses.mean_squared_error(encoder_outputs_y[-1],encoder_outputs_x[-1])
# 		train_step_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
# 		batches = batch_iter(
#                     list(zip(training_in_data, training_out_data)), batch_size, num_epochs)
# 		# Training loop. For each batch...
# 		for batch_num,batch in enumerate(batches):
# 			x_batch, y_batch = zip(*batch)
# 			train_step(x_batch,y_batch,input_x,input_y,train_step_op)
# 			if batch_num%save_every:
# 				saver.save(sess,checkpoint_prefix)