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

import sys
sys.path.append('./../data_preprocess')
sys.path.append('./../ml')
sys.path.append('./../')
from data_helpers import *
from model import *
from config import *

sentences, words, dictionary, reverse_dictionary, vocabulary_size = build_vocab(dataDir,load=True)


in_sent_arr, in_token_arr, out_word_arr, out_token_arr, out_arr, seq_len, data_json = prepare_sentence_tokens(dataNoise,embedding_out_names_path,loadOne=True)

print('in_token_arr')
print(in_token_arr)
print('out_token_arr')
print(out_token_arr)

NUM_CLASSES = len(data_json)
print('NUM_CLASSES')
print(NUM_CLASSES)

training_in_data, training_out_data = get_training_data(in_token_arr,out_arr,out_token_arr,seq_len, dictionary)

print('training_in_data')
print(training_in_data)
print('training_out_data')
print(training_out_data)

def getOutNames(out_arr, out_word_arr):
	result = []
	for out in out_arr:
		word = out_word_arr[out[0]]
		result.append(word)
	return result

graph = tf.Graph()
num_epochs = 1

with graph.as_default():
	sess = tf.Session(graph=graph)
	with sess.as_default():
		input_x,\
	    input_y,\
	    train_loss,\
	    train_step_op,\
	    batch_size_tensor,\
	    eval_loss,\
	    logits,\
	    embedded_chars_y,\
	    encoder_outputs_x = build_graph(sess,
				  seq_len,
				  dictionary,
				  batch_size,
				  NUM_CLASSES,
				  vocabulary_size,
				  embedding_size)

		batches = batch_iter(
                    list(zip(training_in_data, training_out_data)), batch_size, num_epochs, shuffle=False)

		saver = tf.train.Saver()
		print('checkpoint_dir')
		print(checkpoint_dir)
		restoreModel(checkpoint_dir, sess, saver)

		embedding_arr = np.array([])
		# out_names = []
		for batch_num,batch in enumerate(batches):
			x_batch, y_batch = zip(*batch)
			batch_size = len(x_batch)
			
			if len(embedding_arr)==0:
				embedding_arr = np.reshape(np.array(eval_out_embedding(y_batch,input_y,batch_size,batch_size_tensor,embedded_chars_y,sess)),(-1,32))
				# out_names = getOutNames(y_batch,out_word_arr)
			else:
				embedding_arr = np.concatenate((embedding_arr,np.reshape(eval_out_embedding(y_batch,input_y,batch_size,batch_size_tensor,embedded_chars_y,sess),(-1,32))),axis=0)
				# out_names += getOutNames(y_batch,out_word_arr)

json.dump(embedding_arr.tolist(),open(out_embedding_arr_path,'w'))
# json.dump(out_names,open(out_names_path,'w'))
