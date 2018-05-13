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
sys.path.append('./../ml')
sys.path.append('./../')
import data_helpers
import mr_model
import config
import mr_data_helpers

sentences, phrase_ids = mr_data_helpers.load_test_data_plain(config.mr_data_path_test)
dictionary, reverse_dictionary, vocabulary_size = mr_data_helpers.loadVocab(config.mrDictionaryPath)

print("Vocabulary_size")
print(vocabulary_size)

in_sent_arr, in_token_arr, seq_len = mr_data_helpers.prepare_sentence_tokens(sentences)

NUM_CLASSES = 5
print("number of classes")
print(NUM_CLASSES)
training_in_data = mr_data_helpers.get_training_data(in_token_arr,seq_len,dictionary)

graph = tf.Graph()

with graph.as_default():
	sess = tf.Session(graph=graph)
	with sess.as_default():
		input_x,\
	    input_y,\
	    train_loss,\
	    train_step_op,\
	    batch_size_tensor,\
	    eval_loss,\
	    encoder_outputs_x,\
	    embedded_chars_x,\
	    embedding_var_inp,\
	    fc2_out = mr_model.build_graph(sess,
			seq_len,
			dictionary,
			NUM_CLASSES,
			vocabulary_size,
			config.embedding_size)

		saver = tf.train.Saver()
		mr_model.restoreModel(config.mr_checkpoint_dir,sess,saver)

		batches = data_helpers.batch_iter(
                  list(zip(training_in_data,phrase_ids)),
                  config.batch_size,
                  1,
                  shuffle=False)

		evaluated_logits = np.array([])
		for index,batch in enumerate(batches):
			x_batch, phrase_id_batch = zip(*batch)
			for _ in range(1):
				batch_size = len(x_batch)
				if len(evaluated_logits)==0:
					evaluated_logits = mr_model.test_step(x_batch,
						input_x,
						fc2_out,
						batch_size,
						batch_size_tensor,
						sess)
				else:
					evaluated_logits = np.concatenate(
						(evaluated_logits,
						mr_model.test_step(x_batch,
							input_x,
							fc2_out,
							batch_size,
							batch_size_tensor,
							sess)),
						axis=0
					)

				
		sentiments = mr_model.getSentimentsFromLogits(evaluated_logits)
		mr_data_helpers.saveTestedSentiments(phrase_ids,sentiments)