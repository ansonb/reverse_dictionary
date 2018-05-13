import tensorflow as tf

import sys
sys.path.append('./../')
sys.path.append('./../data_preprocess')
sys.path.append('./')

import config
import model_recursive_nn_bucket
import data_helpers
import model

import json

import numpy as np

import tqdm
import math

import pdb

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
			num_children_to_append = level_arr[cur_level] - len(node['dep_tree'])
			for _ in range(num_children_to_append):
					node['dep_tree'].append({
						'next': []
						})
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
	# print('max_num_levels')
	# print(max_num_levels)
	for level_arr in level_arr_arr:
		num_levels_to_append = max_num_levels - len(level_arr)
		# print('num_levels_to_append')
		# print(num_levels_to_append)
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

BUCKET_GROUP_SIZE = 2
def getBucketsSize(level_arr_arr):
	# pdb.set_trace()
	group_size = BUCKET_GROUP_SIZE
	buckets_size_arr = []
	for level_arr in level_arr_arr:
		bucket_size = math.ceil(len(level_arr)/group_size)*group_size
		if bucket_size not in buckets_size_arr:
			buckets_size_arr.append(bucket_size)

	return buckets_size_arr

def findCorrespondingBucket(num_levels,bucket_size_arr):
	#assumption that bucket_size_arr is sorted
	for bucket_num,bucket_size in enumerate(bucket_size_arr):
		if num_levels<bucket_size:
			return bucket_num
	raise("num_levels must be lesser than the maximum bucket size")

def makeBucketwiseLevelArr(level_arr_arr,bucket_size_arr):
	bucket_level_arr = [[] for _ in bucket_size_arr]
	bucket_info = []
	for level_arr in level_arr_arr:
		bucket_num = findCorrespondingBucket(len(level_arr),bucket_size_arr)
		bucket_level_arr[bucket_num].append(level_arr)
		bucket_info.append(bucket_num)

	return bucket_level_arr, bucket_info


def prepareData(data_json_to_be_modified):
	level_arr_arr = []
	data_arr = []
	bucket_for_data_arr = []
	max_num_levels_arr = []

	# buckets = getBuckets(data_json_to_be_modified)

	for data in tqdm.tqdm(data_json_to_be_modified):
		bfs_arr = data["dep_tree"].copy()
		bfs_arr += ['$']
		bfs(bfs_arr,0)
		
		level_arr = findMaxNodesInLevel(bfs_arr)
		max_num_levels_arr.append(len(level_arr))

		level_arr_arr.append(level_arr)
		data_arr.append(data.copy())

	# max_num_levels = max_num_levels_arr[np.argmax(max_num_levels_arr)]

	bucket_size_arr = getBucketsSize(level_arr_arr)

	bucket_size_arr.sort()

	bucketwise_level_arr_arr, bucket_info = makeBucketwiseLevelArr(level_arr_arr,bucket_size_arr)

	bucketwise_max_level_arr = []
	print('Preparing max number of nodes in each level for each bucket...')
	for index,level_arr_arr in tqdm.tqdm(enumerate(bucketwise_level_arr_arr)):
		bucket_size = bucket_size_arr[index]
		# print('bucket: ',bucket_size)

		cur_level_arr_arr = makeLevelsArrEqual(level_arr_arr,bucket_size)
		# print('cur_level_arr_arr')
		# print(cur_level_arr_arr)

		max_level_arr = findMaxLevelArr(cur_level_arr_arr,bucket_size)
		# print('max_level_arr')
		# print(max_level_arr)

		# print('--------------------')

		bucketwise_max_level_arr.append(max_level_arr)

	#TODO: don't remove comment	
	# bucketwise_in_arr = [ [] for _ in range(len(bucket_size_arr)) ]
	# bucketwise_out_arr = [ [] for _ in range(len(bucket_size_arr)) ]
	bucketwise_in_arr = [ [] for _ in range(2) ]
	bucketwise_out_arr = [ [] for _ in range(2) ]

	print('bucketwise_in_arr')
	print(bucketwise_in_arr)
	# printed = False
	# print('bucket wise max level arr')
	# print(bucketwise_max_level_arr)
	print('Making tree uniform according to the bucket...')
	for index,data in tqdm.tqdm(enumerate(data_arr)):
		# print('in for')
		bucket_num = bucket_info[index]
		if bucket_num>1:
			continue
		makeUniformTree(data,bucketwise_max_level_arr[bucket_num],0,dictionary)
		# print('after makeUniformTree')

		input_train = makeArr(data)
		# if not printed and bucket_num==1:
		# 	print('input_train')
		# 	print(input_train)
		# 	print('data')
		# 	print(data)
		# 	printed = True
		# print('after makeTensor')
		# print()

		output = dictionary[data['def_word']]

		# print('bucketwise_in_arr[bucket_num]')
		# print(type(bucketwise_in_arr[bucket_num]))
		# print(bucketwise_in_arr[bucket_num])
		print('bucket num')
		print(bucket_num)
		print('input_train')
		print(input_train)
		print('output word')
		print(output)
		bucketwise_in_arr[bucket_num].append(input_train)
		bucketwise_out_arr[bucket_num].append(output)
		# in_arr.append(input_train)
		# out_arr.append(output)
		

	return bucketwise_in_arr, bucketwise_out_arr, bucketwise_max_level_arr

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
restore = False
num_epochs = 1000

if config.load_parsed_data:
	with open(config.parsed_x_batch_path,'r') as f:
		x_batches = json.load(f)
	with open(config.parsed_y_batch_path,'r') as f:
		y_batches = json.load(f)
	level_arr = config.level_arr
else:
	print('Preparing data...')
	bucketwise_in_arr,bucketwise_out_arr,bucketwise_max_level_arr = prepareData(data_json_to_be_modified)
	bucketwise_batches_x = []
	bucketwise_batches_y = []
	#TODO: shuffle batches
	for index,_ in enumerate(bucketwise_in_arr):
		if index>config.MAX_NUM_BUCKETS_TO_TRAIN:
			continue
		in_arr = bucketwise_in_arr[index]
		out_arr = bucketwise_out_arr[index]
		# print('===========================in arr=========================')
		# print(len(in_arr[0]))
		x_batches,y_batches = getBatches(in_arr,out_arr,batch_size=config.batch_size_parsed)
		bucketwise_batches_x.append(x_batches)
		bucketwise_batches_y.append(y_batches)
	# x_batches[0] = x_batches[0].tolist()
	print('Done preparing data')
	print(config.parsed_x_batch_path)
	# with open(config.parsed_x_batch_path,'w') as f:
	# 	json.dump(x_batches,f,indent=4)
	# with open(config.parsed_y_batch_path,'w') as f:
	# 	json.dump(y_batches,f,indent=4)
	# num_words_in_arr = len(x_batches[0])

with tf.Session() as sess:

	input_tensor_shape_arr = []
	output_tensor_shape_arr = []
	for index,max_level_arr in enumerate(bucketwise_max_level_arr):
		if index>config.MAX_NUM_BUCKETS_TO_TRAIN:
			continue
		cur_bucket_input_tensor_shape = 1
		product_till_node = []
		# print('max level arr')
		# print(max_level_arr)
		for max_num_nodes_in_level in max_level_arr:
			if len(product_till_node)==0:
				product_till_node.append(max_num_nodes_in_level)
			else:
				product_till_node.append(max_num_nodes_in_level*product_till_node[-1])
		cur_bucket_input_tensor_shape = sum(product_till_node)
		input_tensor_shape_arr.append((cur_bucket_input_tensor_shape,None))
		output_tensor_shape_arr.append((None))

	recursion_out_arr,\
	loss_arr,\
	train_step_op_arr,\
	input_tensor_arr,\
	output_tensor_arr = model_recursive_nn_bucket.build_graph(sess,
						dictionary,
						NUM_CLASSES,
						vocabulary_size,
						embedding_size,
						input_tensor_shape_arr,
						output_tensor_shape_arr,
						bucketwise_max_level_arr)

	saver = tf.train.Saver(max_to_keep=2)
	if restore:
		model.restoreModel(config.parsed_model_checkpoint_dir,sess,saver)

	step = 0
	for epoch in range(num_epochs):
		for bucket_index,_ in enumerate(bucketwise_batches_x):
			x_batches = bucketwise_batches_x[bucket_index]
			y_batches = bucketwise_batches_y[bucket_index] 
			for batch_index,x_batch in enumerate(x_batches):
				y_batch = y_batches[batch_index]
				evaluated_loss = model_recursive_nn_bucket.train_step(sess,
					recursion_out_arr[bucket_index],
					loss_arr[bucket_index],
					train_step_op_arr[bucket_index],
					input_tensor_arr[bucket_index],
					output_tensor_arr[bucket_index],
					x_batch,
					y_batch)

				if step%saveEvery == 0:
					saver.save(sess,config.parsed_model_path_prefix,global_step=step)

				print('step: {}, loss: {}'.format(step, evaluated_loss))
				step += 1

			