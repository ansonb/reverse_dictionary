# import tensorflow as tf

# model_folder

# sequence_length = 100
# vocab_size = 
# embedding_size = 128
# output_keep_prob = 1.0

# input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
# input_y = tf.placeholder(tf.int32, [None, sequence_length], name="input_y")

# with tf.device('/cpu:0'), tf.name_scope("embedding"):
#     W = tf.Variable(
#         tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
#         name="W")

#     embedded_chars = tf.nn.embedding_lookup(W, input_x)

# encoDecoCell1 = lstm_cell(args.hiddenSize, output_keep_prob=output_keep_prob)
# encoDecoCell2 = lstm_cell(args.hiddenSize, output_keep_prob=output_keep_prob)
# encoDecoCell = tf.contrib.rnn.MultiRNNCell([encoDecoCell1,encoDecoCell2], state_is_tuple=False)

# encoder_outputs_x, encoder_state_x = rnn.static_rnn(encoDecoCell, input_x, dtype=dtype)
# encoder_outputs_x, encoder_state_x = rnn.static_rnn(encoDecoCell, input_x, dtype=dtype)

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


embedding_size = 32

#read the dictionary from a file
with open('dictionary.json', 'r') as f:
	dictionary = json.load(f)
vocabulary_size = len(dictionary)

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

docs = [
	'the dream of one nation became a reality on july',
	'world is merely an illusion albeit a persistent one <eos>',
	'the gst council may on Friday approve tax cuts <eos>',
	'when will the gst council approve tax cuts <eos> <eos>',
	'the panel was headed by union finance minister arun jaitley',
	'who headed the panel <eos> <eos> <eos> <eos> <eos> <eos>',
	'diesel and petrol are currently exempt from the gst <eos>',
	'products that are exempt from gst <eos> <eos> <eos> <eos>',
	'growth is expected to slow from 8 to 7 <eos>',
	'what will be the effect of gst on growth <eos>'
]


window = 8
seq_len = 10
#prepare the training data
def get_training_data(docs):
	inp = []
	out = []

	for doc in docs:
		cursor = 0
		tokenized = basic_tokenizer(doc)
		for _ in range(len(tokenized)-window-1):
			seq_in = ["<eos>" for _ in range(seq_len)]
			seq_out = ["<eos>" for _ in range(seq_len)]

			seq_in[0:window] = tokenized[cursor:cursor+window]
			seq_out[0:window] = tokenized[cursor+1:cursor+1+window]
			for index,token in enumerate(seq_in):
				seq_in[index] = dictionary.get(token, None) or dictionary["UNK"]
			for index,token in enumerate(seq_out):
				seq_out[index] = dictionary.get(token, None) or dictionary["UNK"]

			inp.append(seq_in[::-1])
			out.append(seq_out[::-1])

			cursor += 1

	return inp, out

training_in_data, training_out_data = get_training_data(docs)
print("len(training_in_data)")
print(len(training_in_data))

def batch_iter(data, batch_size, num_epochs, shuffle=False):
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

def lstm_cell(lstm_size, output_keep_prob=1.0):
  # return tf.contrib.rnn.BasicLSTMCell(lstm_size)
  encoDecoCell = tf.contrib.rnn.BasicLSTMCell(lstm_size, state_is_tuple=True)  # Or GRUCell, LSTMCell(args.hiddenSize)
  #Only for training output_keep_prob is 0.5
  encoDecoCell = tf.contrib.rnn.DropoutWrapper(encoDecoCell, input_keep_prob=1.0, output_keep_prob=output_keep_prob)  # TODO: Custom values
  return encoDecoCell

def train_step(x_batch, y_batch, input_x, input_y, train_step_op, loss):
    """
    A single training step
    """
    feed_dict = {
      input_x: x_batch,
      input_y: y_batch,
    }
    loss = sess.run(
        [loss],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{} loss".format(loss))

def eval_step(x_batch, input_x, encoder_outputs_x):
    """
    A single training step
    """
    feed_dict = {
      input_x: x_batch,
    }
    encoder_outputs = sess.run(
        [encoder_outputs_x],
        feed_dict)
    print("{} encoder_outputs".format(encoder_outputs))
    return encoder_outputs[0][-1]

graph = tf.Graph()
num_epochs = 1
save_every = 100
checkpoint_dir = "models_w2vecPretrained"
checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

checkpoint_dir_w2vec = './../../word2vec/word2vec/model_seperate_arr'
checkpoint_file_w2vec = tf.train.latest_checkpoint(checkpoint_dir_w2vec)

with graph.as_default():
  sess = tf.Session(graph=graph)
  with sess.as_default():
    
    sequence_length = seq_len
    vocab_size = len(dictionary)
    output_keep_prob = 1.0

    batch_size = 10

    hiddenSize = 256

    # saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint))
    # saver.restore(sess, checkpoint)
    # print("=========================")
    # for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    #     print(v.op.name)
    # print("=========================")

    # input_x = graph.get_tensor_by_name("input_x:0")
    # print("input_x")
    # print(input_x)
    # encoder_outputs_x = graph.get_tensor_by_name("encoder_outputs_x:0")
    # print("encoder_outputs")
    # print(encoder_outputs_x)

    input_x_plh = tf.placeholder(tf.int32, shape=(batch_size, sequence_length), name="input_x")
    input_y = tf.placeholder(tf.int32, shape=(batch_size, sequence_length), name="input_y")

    input_x = tf.Variable(np.zeros((batch_size, sequence_length)), dtype=tf.int32)
    input_x = tf.assign(input_x,input_x_plh)

    # Load the saved meta graph and restore variables
    # with tf.variable_scope("saver_w2vec"):
    # saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file_w2vec))
    # saver.restore(sess, checkpoint_file_w2vec)

    # Get the placeholders from the graph by name
    # embedding = graph.get_operation_by_name("my_embedding").outputs[0]
    # print("embedding shape")
    # print(embedding.shape)
    # embedding_var = tf.Variable(embedding,trainable=False,name="my_embedding_2")
    embedding_var = graph.get_tensor_by_name("my_embedding_2:0")
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
      sess.run(tf.global_variables_initializer())

      encoder_outputs_x, encoder_state_x = rnn.static_rnn(encoDecoCell, embedded_chars_x_list,dtype=tf.float32)
      # output_x = tf.Variable(encoder_outputs_x[-1], name="encoder_outputs_x")
      # print("encoder outputs x")
      # print(encoder_outputs_x)
      # encoder_outputs_y, encoder_state_y = rnn.static_rnn(encoDecoCell, embedded_chars_y_list,dtype=tf.float32)
      # output_y = tf.Variable(encoder_outputs_y[-1], name="encoder_outputs_y")

      # loss = tf.losses.mean_squared_error(encoder_outputs_y[-1],encoder_outputs_x[-1])

    batches = batch_iter(
                    list(zip(training_in_data, training_out_data)), batch_size, num_epochs)

    saver = tf.train.Saver() 
    modelDir = "models_w2vecPretrained"
    MODEL_NAME_BASE = tf.train.latest_checkpoint(modelDir)

    # MODEL_NAME_BASE = 'models_w2vecPretrained/model-35'
    print("Restoring checkpoint from " + MODEL_NAME_BASE)
    saver.restore(sess,MODEL_NAME_BASE)

    for batch_num,batch in enumerate(batches):
    	x_batch, y_batch = zip(*batch)
    	par2vec = eval_step(x_batch,input_x,encoder_outputs_x)


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  par2vec = np.array(par2vec)
  print("par2vec.shape")
  print(par2vec.shape)
  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 10
  low_dim_embs = tsne.fit_transform(par2vec[:plot_only, :])
  # labels = [docs[i] for i in range(plot_only)]
  labels = [str(i) for i in range(plot_only)]
  plot_with_labels(low_dim_embs, labels)

  print('ditance between 0,1')
  print(np.sum((par2vec[0,:]-par2vec[1,:])**2))
  # print('ditance between 3,2')
  # print(np.sum((par2vec[3,:]-par2vec[2,:])**2))
#output
# ditance between 3,6
# 1.60006e-08
# ditance between 3,2
# 2.86644e-08

except ImportError:
  print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")