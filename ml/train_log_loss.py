import tensorflow as tf
import collections

import os
import json, xmljson
from lxml.etree import fromstring, tostring
import re

# from tensorflow.python.ops import rnn
from tensorflow.contrib import rnn
import datetime

import numpy as np

import pdb

import sys
sys.path.append('./../data_preprocess')
sys.path.append('./')
sys.path.append('./../')
from data_helpers import *
from model import *
from config import *

from tqdm import tqdm

# dataDir = '../data/sample/one_word_sample.txt'
# dataNoise = '../data/sample/one_word_noise_sample.json'

# embedding_size = 32

sentences, words, dictionary, reverse_dictionary, vocabulary_size = build_vocab(dataDir)

print('Preparing tokens...')
in_sent_arr, in_token_arr, out_word_arr, out_token_arr, out_arr, seq_len, data_json = prepare_sentence_tokens(dataNoise,embedding_out_names_path)

NUM_CLASSES = len(data_json.items())
print("number of classes")
print(NUM_CLASSES)
print('Building training data...')
training_in_data, training_out_data = get_training_data(in_token_arr,out_arr,out_token_arr,seq_len,dictionary)

saveDataInFiles(training_in_data, training_out_data, in_sent_arr, out_word_arr)

out_names = loadOutNames(out_names_path)
# print('out_names')
# print(out_names)
out_names_dict_indices = getOutWordsDictNum(out_names,dictionary)
print('len(out_names_dict_indices)')
print(len(out_names_dict_indices))

def onTrainingCompletion(saver, step, checkpoint_prefix, sess):
	saver.save(sess,checkpoint_prefix,global_step=step)
	print("Saving checkpoint to {}-{}".format(checkpoint_prefix,step))

graph = tf.Graph()
# num_epochs = 2000
# save_every = 100
# eval_after = 100
# checkpoint_dir = "../model/sample"
# checkpoint_prefix = os.path.join(checkpoint_dir, "model")
# restore = False

with graph.as_default():
	sess = tf.Session(graph=graph)
	with sess.as_default():
		batch_size = 10

		input_x,\
		input_y,\
		train_loss,\
		train_step_op,\
		batch_size_tensor,\
		eval_loss,\
		logits,\
		embedded_chars_y,\
	    encoder_outputs_x,\
	    embedded_chars_x,\
	    embedding_var_inp = build_graph(sess,
			seq_len,
			dictionary,
			batch_size,
			NUM_CLASSES,
			vocabulary_size,
			embedding_size,
			out_names_dict_indices)

		saver = tf.train.Saver(max_to_keep=5)
		if restore:
			step = restoreModel(checkpoint_dir,sess,saver)
		else:
			step = 0

		batches = batch_iter(
                  list(zip(training_in_data,training_out_data)),
                  batch_size,
                  num_epochs)

		try:
			for batch in batches:
				x_batch, y_batch = zip(*batch)
				for _ in range(1):
					step += 1
					batch_size = len(x_batch)
					train_step(x_batch,
						y_batch,
						input_x,
						input_y,
						train_loss,
						train_step_op,
						step,
						batch_size,
						batch_size_tensor,
						sess)

					if step%save_every == 0:
						saver.save(sess,checkpoint_prefix,global_step=step)
						print("Saving checkpoint to {}-{}".format(checkpoint_prefix,step))
					
					if step%eval_after==0:
						batch_size = len(x_batch)
						eval_step(x_batch,
							y_batch,
							input_x,
							input_y,
							eval_loss,
							logits,
							step,
							batch_size,
							batch_size_tensor,
							sess)

					
		except KeyboardInterrupt:
			print("***********KeyboardInterrupt******************")
			onTrainingCompletion(saver, step, checkpoint_prefix, sess)