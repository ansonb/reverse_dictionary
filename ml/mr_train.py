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
import data_helpers
import mr_model
import config
import mr_data_helpers

sentences, sentiments = mr_data_helpers.load_data_plain(config.mr_data_path_train)
# sentences = data_helpers.preprocess_data(sentences)
words = data_helpers.get_words(sentences)
dictionary, reverse_dictionary = data_helpers.build_dataset(words)
vocabulary_size = len(dictionary)

print("Vocabulary_size")
print(vocabulary_size)

#write the dictionary to a file
with open(config.mrDictionaryPath, 'w') as f:
	json.dump(dictionary,f,indent=4)

in_sent_arr, in_token_arr, seq_len = mr_data_helpers.prepare_sentence_tokens(sentences)

NUM_CLASSES = 5
print("number of classes")
print(NUM_CLASSES)
training_in_data = mr_data_helpers.get_training_data(in_token_arr,seq_len,dictionary)

mr_data_helpers.saveDataInFiles(training_in_data, in_sent_arr)

def onTrainingCompletion(saver, step, checkpoint_prefix, sess):
	saver.save(sess,checkpoint_prefix,global_step=step)
	print("Saving checkpoint to {}-{}".format(checkpoint_prefix,step))

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

		saver = tf.train.Saver(max_to_keep=5)
		if config.mr_restore:
			step = mr_model.restoreModel(config.mr_checkpoint_dir,sess,saver)
		else:
			step = 0

		batches = data_helpers.batch_iter(
                  list(zip(training_in_data,sentiments)),
                  config.batch_size,
                  config.num_epochs)

		try:
			for batch in batches:
				x_batch, y_batch = zip(*batch)
				for _ in range(1):
					batch_size = len(x_batch)
					mr_model.train_step(x_batch,
						y_batch,
						input_x,
						input_y,
						train_loss,
						train_step_op,
						step,
						batch_size,
						batch_size_tensor,
						sess)

					if step%config.save_every == 0:
						saver.save(sess,config.mr_checkpoint_prefix,global_step=step)
						print("Saving checkpoint to {}-{}".format(config.mr_checkpoint_prefix,step))
					
					if step%config.eval_after==0:
						batch_size = len(x_batch)
						mr_model.eval_step(x_batch,
							y_batch,
							input_x,
							input_y,
							eval_loss,
							fc2_out,
							step,
							batch_size,
							batch_size_tensor,
							sess)
				
				step += 1
					
		except KeyboardInterrupt:
			print("***********KeyboardInterrupt******************")
			onTrainingCompletion(saver, step, config.mr_checkpoint_prefix, sess)