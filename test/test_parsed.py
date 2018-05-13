import tensorflow as tf

import sys
sys.path.append('./../')
sys.path.append('./../data_preprocess')
sys.path.append('./../ml')

import config
import model_recursive_nn
import data_helpers
import model

import json

import numpy as np

in_data_path = config.parsedJsonInputFileTest

def find_max_num_children_per_level(_input):
	pass

def bfs(arr,index):
	if index>=len(arr):
		return

	count = index
	while True:
		if arr[count] == '$':
			break

		if arr[count] == '|':
			count += 1
			continue

		if arr[count].get('dep_tree', -1) == -1:
			if len(arr[count]['next']) > 0:
				arr += (arr[count]['next'])
				arr += '|'
		else:
			if len(arr[count]['dep_tree']['next']) > 0:
				arr += (arr[count]['dep_tree']['next'])
				arr += '|'
		
		count += 1

	if arr[-1] != '$':
		arr.append('$')
	bfs(arr,count+1)

def getNumberOfLevels(arr, terminatingChar):
	num_levels = 0

	for item in arr:
		if item == terminatingChar:
			num_levels += 1

	return num_levels

def findMaxNodesInLevel(arr):
	num_levels = getNumberOfLevels(arr,'$')

	cur_level = 0
	max_num_nodes = [0 for _ in range(num_levels)]
	num_nodes = 0
	for item in arr:
		if item == '$':
			max_num_nodes[cur_level] = max_num_nodes[cur_level] or num_nodes
			num_nodes = 0
			cur_level += 1
			continue

		if item == '|':
			if num_nodes>max_num_nodes[cur_level]:
				max_num_nodes[cur_level] = num_nodes
			num_nodes = 0
			continue

		num_nodes += 1


	return max_num_nodes

def makeUniformTree(node, level_arr, cur_level, _dict):
	if cur_level>=len(level_arr):
		if node.get('dep_tree',-1) != -1:
			node['dep_tree']['val'] = _dict.get(node['dep_tree']['word'],_dict['UNK'])
		else:
			if node.get('word',-1) != -1:
				node['val'] = _dict.get(node['word'],_dict['UNK'])
			else:
				node['val'] = -1
		return

	# print(type(node))
	if node.get('dep_tree',-1) != -1:
		if str(type(node['dep_tree'])) == "<class 'dict'>":
				node['dep_tree']['val'] = _dict.get(node['dep_tree']['word'],_dict['UNK'])

				num_children_to_append = level_arr[cur_level] - len(node['dep_tree']['next'])

				for _ in range(num_children_to_append):
					node['dep_tree']['next'].append({
						'next': []
						})

				for child in node['dep_tree']['next']:
					makeUniformTree(child,level_arr,cur_level+1,_dict)
		else:
			for child in node['dep_tree']:
				makeUniformTree(child,level_arr,cur_level+1,_dict)
	else:
		if node.get('word',-1) != -1:
			node['val'] = _dict.get(node['word'],_dict['UNK'])
		else:
			node['val'] = -1

		num_children_to_append = level_arr[cur_level] - len(node['next'])

		for _ in range(num_children_to_append):
			node['next'].append({
				'next': []
				})

		for child in node['next']:
			makeUniformTree(child,level_arr,cur_level+1,_dict)

mtCount = 0
def makeTensor(node):

	# global mtCount
	# mtCount += 1

	if node.get('dep_tree',-1)!=-1 and\
	 str(type(node['dep_tree']))=="<class 'list'>" and\
	 len(node['dep_tree']) == 0:
		return []
	if node.get('dep_tree',-1)!=-1 and\
	 str(type(node['dep_tree']))=="<class 'dict'>" and\
	 len(node['dep_tree']['next']) == 0:
		return node['dep_tree']['val']
	if node.get('next',-1)!=-1 and len(node['next']) == 0:
		return node['val']

	if node.get('dep_tree',-1)!=-1 and\
	 str(type(node['dep_tree']))=="<class 'list'>":
	 node_next = node['dep_tree']
	 node_val = None
	if node.get('dep_tree',-1)!=-1 and\
	 str(type(node['dep_tree']))=="<class 'dict'>":
		node_next = node['dep_tree']['next']
		node_val = node['dep_tree']['val']
	if node.get('next',-1)!=-1:
		node_next = node['next']
		node_val = node['val']

	# print(mtCount)
	#combine child tensors
	arr = []
	for child in node_next:
		arr.append(makeTensor(child))

	if node_val==None:
		return np.array(arr)

	#make the parent tensor
	arr = np.array(arr)
	arr_copy = np.ones(arr.shape)
	arr_copy = np.multiply(arr_copy,-1)
	arr_copy = np.reshape(arr_copy,[-1])
	arr_copy[0] = node_val
	arr_copy = np.reshape(arr_copy,arr.shape)

	#combine the two
	arr_2 = []
	arr_2.append(arr_copy)
	arr_2.append(arr)

	return arr_2


def getWords(data):
	words_arr = []
	for _def in data:
		sentence = _def['sent']
		sentence = data_helpers.preprocess_data(sentence)
		words_arr += data_helpers.basic_tokenizer(sentence)

		def_word = _def['def_word']
		def_word = data_helpers.preprocess_data(def_word)
		words_arr += [def_word]

	return words_arr

def loadData(data_path):
	with open(data_path,'r') as f:
		data = json.load(f)

	words = getWords(data)
	return words, data

def makeLevelsArrEqual(level_arr_arr, max_num_levels):
	print(level_arr_arr)
	print('max_num_levels')
	print(max_num_levels)
	for level_arr in level_arr_arr:
		num_levels_to_append = max_num_levels - len(level_arr)
		print('num_levels_to_append')
		print(num_levels_to_append)
		level_arr += [0]*num_levels_to_append

	print(level_arr_arr)
	return level_arr_arr

def findMaxLevelArr(level_arr_arr, max_num_nodes_in_level):
	max_level_arr = []
	level_arr_arr = np.array(level_arr_arr)
	for axis in range(max_num_nodes_in_level):
		arr = np.reshape(level_arr_arr[:,axis],[-1])
		max_num_nodes = arr[np.argmax(arr)]

		max_level_arr.append(max_num_nodes)

	return max_level_arr

def getBatches(data_json_to_be_modified):
	level_arr_arr = []
	data_arr = []
	max_num_levels_arr = []
	for data in data_json_to_be_modified:
		bfs_arr = data["dep_tree"].copy()
		bfs_arr += ['$']
		bfs(bfs_arr,0)
		
		level_arr = findMaxNodesInLevel(bfs_arr)
		max_num_levels_arr.append(len(level_arr))

		level_arr_arr.append(level_arr)
		data_arr.append(data.copy())

	max_num_levels = max_num_levels_arr[np.argmax(max_num_levels_arr)]
	level_arr_arr = makeLevelsArrEqual(level_arr_arr,max_num_levels)

	max_level_arr = findMaxLevelArr(level_arr_arr,max_num_levels)
	print('max_level_arr')
	print(max_level_arr)


	in_arr = []
	out_arr = []
	for index,data in enumerate(data_arr):
		print('in for')
		makeUniformTree(data,max_level_arr,0,dictionary)
		print('after makeUniformTree')

		input_train = makeTensor(data)
		print('after makeTensor')
 
		output = dictionary[data['def_word']]

		in_arr.append(input_train)
		out_arr.append(output)

	return in_arr, out_arr

def getWordsFromLogits(out_logits_arr, out_names):
  print('out_logits_arr')
  print(out_logits_arr)
  # print()
  # print('vec_arr shape')
  # print(vec_arr.shape)

  # print('dictEmbeddings')
  # print(dictEmbedding[0])
  # sim = np.matmul(vec_arr,np.transpose(dictEmbedding))
  # print('sim dim')
  # print(sim.shape)
  
  sorted_word_match_indices = np.argsort(out_logits_arr,axis=1)
  print('sorted_word_match_indices')
  print(sorted_word_match_indices)
  # print(sorted_word_match_indices)
  # print('best_word_match_index shape')
  # print(sorted_word_match_indices.shape)

  words_arr = []
  top3_arr = []
  for word_indices in sorted_word_match_indices:
    print('word_indices')
    print(word_indices)
    words_arr.append(out_names[word_indices[::-1][0]])
    top3_arr.append([out_names[word_index] for word_index in word_indices[::-1][:3] ])

  return words_arr, top3_arr

def savePredictions(preds, docs, answers, top3, measure=0):
  with open(config.dataNoise,'r') as f:
  	data_json_noise = json.load(f)
  
  testedJson = []
  preds_arr = preds if measure==0 else top3
  for index, pred_word in enumerate(preds_arr):
    doc = docs[index]
    answer = answers[index]
    if measure == 0:
      pred_def = data_json_noise[pred_word][0]
    elif measure == 1:
      pred_def = [data_json_noise[word][0] for word in pred_word]

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

# dictEmbedding = data_helpers.loadDictEmbedings(config.out_embedding_arr_path_parsed)

_, data_json = loadData(in_data_path)
data_json_to_be_modified = data_json.copy()
dictionary, vocabulary_size = data_helpers.loadVocabParsed()

print('vocab_size')
print(vocabulary_size)
#NOTE: might change in the future
NUM_CLASSES = config.NUM_CLASSES
print("number of classes")
print(NUM_CLASSES)

embedding_size = config.embedding_size

out_names = data_helpers.getOutNamesParsed()
out_words_dict_num = data_helpers.getOutWordsDictNum(out_names,dictionary)
answers_raw, docs_raw = data_helpers.readTestData(config.testDataPath)
docs, answers = data_helpers.process_test_data(answers_raw,docs_raw)

with tf.Session() as sess:

	W,\
	V_in,\
	V_out,\
	b,\
	W_out,\
	b_out,\
	optimizer = model_recursive_nn.build_graph(sess,
						dictionary,
						NUM_CLASSES,
						vocabulary_size,
						embedding_size,
						out_words_dict_num)
	saver = tf.train.Saver()
	model.restoreModel(config.parsed_model_checkpoint_dir, sess, saver)

	out_logits_arr = []
	for cur_example,data in enumerate(data_json_to_be_modified):
		output = out_names.index(data['def_word'])
		evaluated_logits, _ = model_recursive_nn.run_test_graph(sess,
			W,
			V_in,
			V_out,
			b,
			W_out,
			b_out,
			data,
			output,
			optimizer,
			dictionary,
			NUM_CLASSES)

		out_logits_arr.append(evaluated_logits[0])

	words, top3 = getWordsFromLogits(out_logits_arr,out_names)
	accuracy = getAccuracy(words,answers,top3,measure=config.measure)
	print('Accuracy: {}'.format(accuracy))
	savePredictions(words,docs,answers,top3,measure=config.measure)
	print()
	print('best match words')
	print(words)

			