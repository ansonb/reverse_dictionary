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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import sys
sys.path.append('./../data_preprocess')
sys.path.append('./../ml')
sys.path.append('./../')
from data_helpers import *
from model import *
from config import *
import config

dictionary, rev_dict, vocabulary_size, data_json = loadVocab(dictionaryFile,dataNoise)
NUM_CLASSES = len(data_json.items())
print("number of classes")
print(NUM_CLASSES)


docs_raw, answers_raw = getTestWordsData(wordTestDataPath)
docs, answers = process_test_data_words(answers_raw,docs_raw)
all_words, all_words_dict = getAllTestWords(answers, docs, dictionary)

in_sent_arr, in_token_arr = tokenize_test_sentences(docs)

print('len in sent arr')
print(len(in_sent_arr))

training_in_data, training_out_data = get_test_data_words(in_sent_arr,in_token_arr,dictionary)
print('training_in_data[0]')
print(training_in_data[0])
print("len(training_in_data)")
print(len(training_in_data))

out_names = loadOutNames(out_names_path)
print('out_names')
print(out_names)
out_names_dict_indices = getOutWordsDictNum(out_names,dictionary)

def getWords(input_word_embeddings,evaluated_embedding_inp):
  print('input_embeddings shape')
  print(input_word_embeddings.shape)
  print('evaluated embeddings shape')
  print(evaluated_embedding_inp.shape)
  sim = np.matmul(input_word_embeddings,np.transpose(evaluated_embedding_inp))
  
  sorted_word_match_indices = np.argsort(sim,axis=1)

  top3_arr = []
  for word_indices in sorted_word_match_indices:
    top3_arr.append([rev_dict[word_index] for word_index in word_indices[::-1][:4] ])

  return top3_arr

def savePredictions(docs, answers, top3):
  testedJson = []

  for index, pred_word in enumerate(top3):
    doc = docs[index]
    answer = answers[index]
    
    preds = top3[index]

    obj = {
      'word_to_test': doc,
      'expected_matches': answer,
      'computed_matches': preds
    }
    testedJson.append(obj)

  with open('../test/output/wordTestOutput.json','w') as f:
    json.dump(testedJson,f,indent=4)

def getAccuracy(answers, top3_arr):
  count = 0
  total = 0

  for index, top3 in enumerate(top3_arr):
    print('top3: ',top3)
    print('answer: ',answers[index])
    print()
    for word in top3:
      if word in answers[index]:
        count += 1

      total += 1

  return count/total


graph = tf.Graph()
num_epochs = 1
batch_size = config.batch_size
# save_every = 100
with graph.as_default():
  sess = tf.Session(graph=graph)
  with sess.as_default():
    # input_x,\
    # input_y,\
    # train_loss,\
    # train_step_op,\
    # batch_size_tensor,\
    # eval_loss,\
    # logits,\
    # embedded_chars_y,\
    # encoder_outputs_x,\
    # embedded_chars_x,\
    # embedding_var_inp = build_graph(sess,
    #     seq_len,
    #     dictionary,
    #     batch_size,
    #     NUM_CLASSES,
    #     vocabulary_size,
    #     embedding_size,
    #     out_names_dict_indices)
    meta_file_path = config.checkpoint_prefix+'-63400.meta'
    loadTestGraph(sess,
      meta_file_path,
      config.checkpoint_dir)

    embedding_var_inp = tf.get_default_graph().get_tensor_by_name("embedding_inp:0")

    batches = batch_iter(
                    list(zip(training_in_data, training_out_data)), batch_size, num_epochs, shuffle=False)

    saver = tf.train.Saver()
    restoreModel(checkpoint_dir, sess, saver)

    input_word_embeddings = np.array([])
    for batch_num,batch in enumerate(batches):
      x_batch, y_batch = zip(*batch)
      batch_size = len(x_batch)
      # y_batch = np.array(y_batch)
      # y_batch = np.reshape(y_batch,(batch_size,1))
      
      if len(input_word_embeddings)==0:
        input_word_embeddings = eval_in_embedding(x_batch,
            embedding_var_inp,
            sess)

      else:
        input_word_embeddings = np.concatenate((input_word_embeddings,
          eval_in_embedding(x_batch,
            embedding_var_inp,
            sess)),axis=0)

    evaluated_embedding_inp = sess.run(embedding_var_inp)

    top3 = getWords(input_word_embeddings,evaluated_embedding_inp)
    accuracy = getAccuracy(answers,top3)
    print('Accuracy: {}'.format(accuracy))
    savePredictions(docs,answers,top3)


    #save all the word embeddings in a png
    op = tf.nn.embedding_lookup(embedding_var_inp,all_words_dict)
    embedded_all_test_words = sess.run(op) 
    embedded_projection = TSNE(n_components=2).fit_transform(embedded_all_test_words)
    # print('embedded_projection')
    # print(embedded_projection)
    x = embedded_projection[:,0]
    y = embedded_projection[:,1]
    plt.scatter(x,y)
    for index,word in enumerate(all_words):
      plt.annotate(word,(x[index],y[index]))
    plt.show()