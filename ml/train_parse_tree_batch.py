import tensorflow as tf

import sys
sys.path.append('./../')
sys.path.append('./../data_preprocess')
sys.path.append('./')

import config
import model_recursive_nn_batch
import data_helpers
import model

import json

import numpy as np

import tqdm
import math

in_data_path = config.parsedJsonInputFile

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

def makeArr(node):
	if node.get('dep_tree',-1)!=-1 and\
	 str(type(node['dep_tree']))=="<class 'list'>" and\
	 len(node['dep_tree']) == 0:
		return []
	if node.get('dep_tree',-1)!=-1 and\
	 str(type(node['dep_tree']))=="<class 'dict'>" and\
	 len(node['dep_tree']['next']) == 0:
		return [node['dep_tree']['val']]
	if node.get('next',-1)!=-1 and len(node['next']) == 0:
		return [node['val']]

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

	arr = []
	for child in node_next:
		arr += makeArr(child)
	if node_val is not None:
		arr += [node_val]

	return arr


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

def prepareData(data_json_to_be_modified):
	level_arr_arr = []
	data_arr = []
	max_num_levels_arr = []
	for data in tqdm.tqdm(data_json_to_be_modified):
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
	for index,data in tqdm.tqdm(enumerate(data_arr)):
		# print('in for')
		makeUniformTree(data,max_level_arr,0,dictionary)
		# print('after makeUniformTree')

		input_train = makeArr(data)
		# print('after makeTensor')
		# print()

		output = dictionary[data['def_word']]

		in_arr.append(input_train)
		out_arr.append(output)
		break

	return in_arr, out_arr, max_level_arr

def getBatches(in_arr,out_arr,batch_size=config.batch_size_parsed):
	batches_in = []
	batches_out = []
	num_batches = math.ceil(len(in_arr)/batch_size)
	for i in range(num_batches):
		start = batch_size*i
		end = min(start+batch_size,len(in_arr))
		batch_in = np.array(in_arr[start:end])
		batch_in = np.transpose(batch_in)

		batch_out = np.array(out_arr[start:end])

		batches_in.append(batch_in)
		batches_out.append(batch_out)

	return batches_in,batches_out


words, data_json = loadData(in_data_path)
data_json_to_be_modified = data_json.copy()
dictionary, reverse_dictionary, vocabulary_size = data_helpers.build_vocab_parsed(words)

#NOTE: might change in the future
NUM_CLASSES = len(data_json)
print("number of classes")
print(NUM_CLASSES)

embedding_size = config.embedding_size
saveEvery = 100
restore = True
num_epochs = 1000

if config.load_parsed_data:
	with open(config.parsed_x_batch_path,'r') as f:
		x_batches = json.load(f)
	with open(config.parsed_y_batch_path,'r') as f:
		y_batches = json.load(f)
	level_arr = config.level_arr
else:
	print('Preparing data...')
	in_arr,out_arr,level_arr = prepareData(data_json_to_be_modified)
	x_batches,y_batches = getBatches(in_arr,out_arr,batch_size=config.batch_size_parsed)
	# x_batches[0] = x_batches[0].tolist()
	print('Done preparing data')
	print(config.parsed_x_batch_path)
	# with open(config.parsed_x_batch_path,'w') as f:
	# 	json.dump(x_batches,f,indent=4)
	# with open(config.parsed_y_batch_path,'w') as f:
	# 	json.dump(y_batches,f,indent=4)
	num_words_in_arr = len(x_batches[0])

with tf.Session() as sess:

	input_tensor_shape = (num_words_in_arr,None)
	output_tensor_shape = (None)

	recursion_out,\
	loss,\
	train_step_op,\
	input_tensor,\
	output_tensor = model_recursive_nn_batch.build_graph(sess,
						dictionary,
						NUM_CLASSES,
						vocabulary_size,
						embedding_size,
						input_tensor_shape,
						output_tensor_shape,
						level_arr)

	saver = tf.train.Saver(max_to_keep=2)
	if restore:
		model.restoreModel(config.parsed_model_checkpoint_dir,sess,saver)

	step = 0
	for epoch in range(num_epochs):
		for index,x_batch in enumerate(x_batches):
			y_batch = y_batches[index]
			evaluated_loss = model_recursive_nn_batch.train_step(sess,
				recursion_out,
				loss,
				train_step_op,
				input_tensor,
				output_tensor,
				x_batch,
				y_batch)

			if step%saveEvery == 0:
				saver.save(sess,config.parsed_model_path_prefix,global_step=step)

			print('step: {}, loss: {}'.format(step, evaluated_loss))
			step += 1

			