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
import config

dictionary, rev_dict, vocabulary_size, data_json = loadVocab(dictionaryFile,dataNoise)
NUM_CLASSES = len(data_json.items())
print("number of classes")
print(NUM_CLASSES)


answers_raw, docs_raw = readTestData(testDataPath)
docs, answers = process_test_data(answers_raw,docs_raw)


in_sent_arr, in_token_arr = tokenize_test_sentences(docs)

print('in_sent_arr[0]')
print(in_sent_arr[0])

training_in_data, training_out_data = get_test_data(in_sent_arr,in_token_arr,dictionary)
print('training_in_data[0]')
print(training_in_data[0])
print("len(training_in_data)")
print(len(training_in_data))

out_names = loadOutNames(out_names_path)
print('out_names')
print(out_names)
out_names_dict_indices = getOutWordsDictNum(out_names,dictionary)

def getWords(evaluated_logits):
  # print('vec_arr[0]')
  # print(vec_arr[0])
  # print()
  # print('vec_arr shape')
  # print(vec_arr.shape)

  # print('dictEmbeddings')
  # print(dictEmbedding[0])
  # sim = np.matmul(vec_arr,np.transpose(dictEmbedding))
  # print('sim dim')
  # print(sim.shape)
  
  # sorted_word_match_indices = np.argsort(sim,axis=1)
  print('evaluated_logits shape')
  print(evaluated_logits.shape)
  sorted_word_match_indices = np.argsort(evaluated_logits,axis=1)
  # print(sorted_word_match_indices)
  # print('best_word_match_index shape')
  # print(sorted_word_match_indices.shape)

  words_arr = []
  top3_arr = []
  for word_indices in sorted_word_match_indices:
    words_arr.append(out_names[word_indices[::-1][0]])
    top3_arr.append([out_names[word_index] for word_index in word_indices[::-1][:3] ])

  return words_arr, top3_arr

def savePredictions(preds, docs, answers, top3, measure=0):
  testedJson = []
  preds_arr = preds if measure==0 else top3
  for index, pred_word in enumerate(preds_arr):
    doc = docs[index]
    answer = answers[index]
    if measure == 0:
      pred_def = data_json[pred_word][0]
    elif measure == 1:
      pred_def = [data_json[word][0] for word in pred_word]

    obj = {
      'sentence': doc,
      'actual_def': answer,
      'predicted_word': pred_word,
      'predicted_words_def': pred_def
    }
    testedJson.append(obj)

  with open('../test/output/testOutput.json','w') as f:
    json.dump(testedJson,f,indent=4)

def getAccuracy(preds, answers, top3, measure=0):
  count = 0
  for index, pred in enumerate(preds):
    if measure==0: #measure accuracy based on top 1
      if pred == answers[index]:
        count += 1
    elif measure==1: #measure accuracy based on top 3
      print('pred: ',pred)
      print('top3: ',top3[index])
      print('answer: ',answers[index])
      print()
      if answers[index] in top3[index]:
        count += 1

  return count/len(answers)

dictEmbedding = loadDictEmbedings(out_embedding_arr_path)


graph = tf.Graph()
num_epochs = 1
batch_size = config.batch_size
# save_every = 100
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
        embedding_size,
        out_names_dict_indices)


    batches = batch_iter(
                    list(zip(training_in_data, training_out_data)), batch_size, num_epochs, shuffle=False)

    saver = tf.train.Saver()
    restoreModel(checkpoint_dir, sess, saver)

    evaluated_logits = np.array([])
    for batch_num,batch in enumerate(batches):
      x_batch, y_batch = zip(*batch)
      batch_size = len(x_batch)
      # y_batch = np.array(y_batch)
      # y_batch = np.reshape(y_batch,(batch_size,1))
      
      if len(evaluated_logits)==0:
        evaluated_logits = test_step(x_batch,
                    y_batch,
                    input_x,
                    input_y,
                    logits,
                    batch_size,
                    batch_size_tensor,
                    sess)[0]
        print('evaluated_logits')
        print(evaluated_logits.shape)
      else:
        evaluated_logits = np.concatenate((evaluated_logits,
          test_step(x_batch,
            y_batch,
            input_x,
            input_y,
            logits,
            batch_size,
            batch_size_tensor,
            sess)[0]),axis=0)
    
    words, top3 = getWords(evaluated_logits)
    accuracy = getAccuracy(words,answers,top3,measure=measure)
    print('Accuracy: {}'.format(accuracy))
    savePredictions(words,docs,answers,top3,measure=measure)
    print()
    print('best match words')
    print(words)