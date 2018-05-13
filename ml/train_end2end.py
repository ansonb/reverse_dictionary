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

dataDir = '../data/one_word.txt'
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
	# text = clean_data(text)
	text = text.lower()
	return text

def build_dataset(words):
  words += ['UNK','<pad>']

  dictionary = dict()

  for word in words:
    if dictionary.get(word, -1)==-1:
        dictionary[word] = len(dictionary)

  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

  return dictionary, reverse_dictionary

# Step 2: Build the dictionary and replace rare words with UNK token.
def get_num_words(words):
  return len(words)


#read data file; do some preprocessing.
def read_data(file_path,skip_window=skip_window):
	data = []

	with open(file_path, 'r') as f:
		lines = f.readlines()
		for line in lines:
			data.append(preprocess_data(line))

	return data

#read data for wor2vec
# Read the data into a list of strings
def get_words(sentences):
	words = []
	for sent in sentences:
		words += basic_tokenizer(sent)

	return words

sentences = read_data(dataDir)
words = get_words(sentences)
dictionary, reverse_dictionary = build_dataset(words)
vocabulary_size = len(dictionary)
print("Vocabulary_size")
print(vocabulary_size)

#write the dictionary to a file
with open('dictionary_end2end.json', 'w') as f:
	json.dump(dictionary,f,indent=4)


#training data preprocessor
print('Tokenizing sentences...')
in_sent_arr = []
in_token_arr = []
out_word_arr = []
out_token_arr = []
max_seq_len = 0
for index,line in enumerate(sentences):
	tokenized = basic_tokenizer(line)
	# if index==0:
		# print("tokenized")
		# print(tokenized)
	if len(tokenized)>max_seq_len:
		max_seq_len = len(tokenized)

	if index%2==0:
		in_sent_arr.append(line)
		in_token_arr.append(tokenized)
	else:
		out_word_arr.append(line)
		out_token_arr.append(tokenized)
print("Done Tokenizing")
print("max_seq_len")
print(max_seq_len)

# window = 10
seq_len = max_seq_len
hop = 1
#prepare the training data
def get_training_data(in_sent_arr, out_word_arr, in_token_arr):
	inp = []
	out = []

	for index,sent in enumerate(in_sent_arr):
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
		cur_out_token = dictionary.get(tokenized_out[0], -1)
		seq_out = cur_out_token if cur_out_token>-1 else dictionary["UNK"]

		#reverse the sequence of tokens for input
		inp.append(seq_in[::-1])
		# out.append([seq_out])
		out.append([index])


	return inp, out

NUM_CLASSES = len(out_word_arr)
print("number of classes")
print(NUM_CLASSES)
training_in_data, training_out_data = get_training_data(in_sent_arr,out_word_arr,in_token_arr)

with open('../data/training_in_data.txt', 'w') as f:
	json.dump(training_in_data,f,indent=4)
with open('../data/training_out_data.txt', 'w') as f:
	json.dump(training_out_data,f,indent=4)
with open('../data/training_in_data_text.txt', 'w') as f:
	json.dump(in_sent_arr,f,indent=4)
with open('../data/training_out_data_text.txt', 'w') as f:
	json.dump(out_word_arr,f,indent=4)

#define the loas function
def getLoss(params):
	logits = None
	if params['mode'] == "train":
		loss = tf.nn.nce_loss(
	      weights=params['weights'],
	      biases=params['biases'],
	      labels=params['labels'],
	      inputs=params['inputs'],
	      num_sampled=params['num_sampled'],
	      num_classes=params['num_classes'],
	      num_true=params['num_true'],
	      partition_strategy="div")
	elif params['mode'] == "eval":
		logits = tf.matmul(params['inputs'], tf.transpose(params['weights']))
		logits = tf.nn.bias_add(logits, params['biases'])
		labels_reshaped = tf.reshape(params['labels'],[-1])
		labels_one_hot = tf.one_hot(labels_reshaped, params['num_classes'])
		loss = tf.nn.sigmoid_cross_entropy_with_logits(
	      labels=labels_one_hot,
	      logits=logits)
	
		loss = tf.reduce_sum(loss, axis=1)

	return loss, logits

def getAccuracy(labels,logits):
	# print("The shape of labels is {}, but the shpe of logits is {}".format(len(labels),logits.shape))
	assert(len(labels)==logits.shape[0]), "The number of labels and logits must be the same"
	# print("labels")
	# print(labels)
	# print("logits")
	# print(logits)
	labels_reshaped = np.reshape(labels,[-1])
	# print(np.argmax(logits,axis=1))
	# print(labels_reshaped)
	return np.sum(np.equal(labels_reshaped,np.argmax(logits,axis=1)))/logits.shape[0]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    overflow_size = data_size%batch_size
    padding_size = (batch_size-overflow_size)%batch_size

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

def train_step(x_batch, y_batch, input_x, input_y, loss, train_step_op, step_num):
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

    loss = np.sum(loss)/loss.shape[0]
    print("{}: step, {} loss".format(step_num, loss))

def eval_step(x_batch, y_batch, input_x, input_y, eval_loss, logits, step_num):
    """
    eval step
    """
    feed_dict = {
      input_x: x_batch,
      input_y: y_batch,
    }

    loss, logits = sess.run(
        [eval_loss,logits],
        feed_dict)
    print('shape of logits')
    print(logits.shape)
    accuracy = getAccuracy(y_batch,logits)
    print()
    print("{}: step, {} accuracy".format(step_num, accuracy))
    print()	


graph = tf.Graph()
num_epochs = 2000
save_every = 100
eval_after = 100
checkpoint_dir = "model"
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
restore = True
# model_to_restore_from = os.path.join(checkpoint_dir, "model-468572.ckpt")
with graph.as_default():
	sess = tf.Session(graph=graph)
	with sess.as_default():
		sequence_length = seq_len
		vocab_size = len(dictionary)
		output_keep_prob = 1.0

		batch_size = 10

		hiddenSize = 32

		num_true = 1
		num_sampled = 32
		num_classes = NUM_CLASSES

		input_x_plh = tf.placeholder(tf.int32, shape=(batch_size, sequence_length), name="input_x")
		input_y = tf.placeholder(tf.int32, shape=(batch_size,num_true), name="input_y")

		input_x = tf.Variable(np.zeros((batch_size, sequence_length)), dtype=tf.int32)
		input_x = tf.assign(input_x,input_x_plh)

		#This will be the weights variable for the nce loss
		embedding_var_inp = tf.Variable(
        	tf.random_uniform([vocabulary_size, embedding_size], minval=-1.0, maxval=1.0),name="embedding_inp")
		embedding_var_out = tf.Variable(
        	tf.random_uniform([num_classes, embedding_size], minval=-1.0, maxval=1.0),name="embedding_out")
		#This will be the bias variable for the nce loss
		bias_var = tf.Variable(tf.zeros(num_classes),name="emb_bias")

		embedded_chars_x = tf.nn.embedding_lookup(embedding_var_inp, input_x)
		embedded_chars_x = tf.transpose(embedded_chars_x,perm=[0,2,1])	
		embedded_chars_x_list = [tf.reshape(tf.slice(embedded_chars_x,[0,0,i],[batch_size,embedding_size,1]),[batch_size,embedding_size]) for i in range(sequence_length)]


		embedded_chars_y = tf.nn.embedding_lookup(embedding_var_out, input_y)
		# embedded_chars_y = tf.transpose(embedded_chars_y,perm=[0,2,1])
		# embedded_chars_y_list = [tf.reshape(tf.slice(embedded_chars_y,[0,0,i],[batch_size,embedding_size,1]),[batch_size,embedding_size]) for i in range(sequence_length)]


		encoDecoCell1 = lstm_cell(hiddenSize, output_keep_prob=output_keep_prob)
		encoDecoCell2 = lstm_cell(hiddenSize, output_keep_prob=output_keep_prob)
		encoDecoCell = tf.contrib.rnn.MultiRNNCell([encoDecoCell1,encoDecoCell2], state_is_tuple=True)
		sess.run(tf.global_variables_initializer())

		encoder_outputs_x, encoder_state_x = rnn.static_rnn(encoDecoCell, embedded_chars_x_list, dtype=tf.float32)
		# encoder_outputs_y, encoder_state_y = rnn.static_rnn(encoDecoCell, embedded_chars_y_list,dtype=tf.float32)

		train_loss,_ = getLoss({
	    	'mode': 'train',
	    	'weights': embedding_var_out,
			'biases': bias_var,
			'labels': input_y,
			'inputs': encoder_outputs_x[-1],
			'num_sampled': num_sampled,
			'num_classes': num_classes,
			'num_true': num_true
    	})
		eval_loss,logits = getLoss({
	    	'mode': 'eval',
	    	'weights': embedding_var_out,
			'biases': bias_var,
			'labels': input_y,
			'inputs': encoder_outputs_x[-1],
			'num_sampled': num_sampled,
			'num_classes': num_classes,
			'num_true': num_true
    	})
		# loss = tf.losses.mean_squared_error(embedded_chars_y,encoder_outputs_x[-1])
		# train_step_op = tf.train.AdamOptimizer(1e-3).minimize(train_loss)
		optimizer = tf.train.AdamOptimizer(1e-3)
		gradients, variables = zip(*optimizer.compute_gradients(loss))
		gradients = [
    		None if gradient is None else tf.clip_by_norm(gradient, 5.0)
    		for gradient in gradients]
		train_step_op = optimizer.apply_gradients(zip(gradients, variables))

		batches = batch_iter(
                    list(zip(training_in_data, training_out_data)), batch_size, num_epochs)

		print("*************Initialising variables*****************")
		for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
			print("Initialising " + v.op.name)
			sess.run(v.initializer)
		print("Uninitialised varaiables")
		print(tf.report_uninitialized_variables())
		saver = tf.train.Saver(max_to_keep=5)

		#restore the model if restore is set to true
		if restore:
			latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
			print('latest_checkpoint')
			print(latest_checkpoint)
			saver.restore(sess,latest_checkpoint)
			start_index_model = latest_checkpoint.rfind('-') + 1
			step = int(latest_checkpoint[start_index_model:])
			print('step')
			print(step)
		else:
			step = 0
		try:
			for batch in batches:
				x_batch, y_batch = zip(*batch)
				for _ in range(1):
					step += 1
					train_step(x_batch,y_batch,input_x,input_y,train_loss,train_step_op,step)
					if step%save_every == 0:
						saver.save(sess,checkpoint_prefix,global_step=step)
						print("Saving checkpoint to {}-{}".format(checkpoint_prefix,step))
					if step%eval_after==0:
						eval_step(x_batch,y_batch,input_x,input_y,eval_loss,logits,step)

					
		except KeyboardInterrupt:
			print("***********KeyboardInterrupt******************")
			saver.save(sess,checkpoint_prefix,global_step=step)
			print("Saving checkpoint to {}-{}".format(checkpoint_prefix,step))