import tensorflow as tf
import numpy as np

import sys
sys.path.append('./../')

import config

def build_graph(sess,
  dictionary,
  NUM_CLASSES,
  vocabulary_size,
  embedding_size,
  out_words_dict_num,
  num_pos_tags):

	#TODO experiment with minval and maxval
	W = tf.Variable(tf.random_uniform([num_pos_tags,embedding_size,embedding_size], minval=-1.0, maxval=1.0), name='rnn_w_general', dtype=tf.float32)

	V_in = tf.Variable(tf.random_uniform([vocabulary_size,embedding_size], minval=-1.0, maxval=1.0), name='embedding_matrix_in', dtype=tf.float32)
	V_out = tf.Variable(tf.random_uniform([NUM_CLASSES,embedding_size], minval=-1.0, maxval=1.0), name='embedding_matrix_out', dtype=tf.float32)
	
	b = tf.Variable(np.zeros((embedding_size)), name='bias_general', dtype=tf.float32)

	W_out = tf.transpose(tf.nn.embedding_lookup(V_in,out_words_dict_num))
	# W_out = tf.Variable(tf.random_uniform([embedding_size,NUM_CLASSES], minval=-1.0, maxval=1.0), name='w_out', dtype=tf.float32)
	b_out = tf.Variable(np.zeros((NUM_CLASSES)), name='bias_out', dtype=tf.float32)

	W_weightage = tf.Variable(tf.random_uniform([embedding_size,16], minval=-1.0, maxval=1.0), name='weightage_matrix', dtype=tf.float32)
	W_fc1 = tf.Variable(tf.random_uniform([32,10], minval=-1.0, maxval=1.0), name='W_fc1', dtype=tf.float32)
	b_fc1 = tf.Variable(np.zeros((10)), name='bias_fc1', dtype=tf.float32)
	W_fc2 = tf.Variable(tf.random_uniform([10,1], minval=-1.0, maxval=1.0), name='W_fc2', dtype=tf.float32)
	b_fc2 = tf.Variable(np.zeros((1)), name='bias_fc2', dtype=tf.float32)

	# input_shape_tensor = tf.placeholder(tf.int32,shape=[1])
	# input_tensor = tf.placeholder(tf.int32,shape=input_shape)

	# output_tensor = tf.placeholder(tf.int32,shape=())
	#TODO: define a dynamic tensor with zero padding and embeddings as 0 
	# recursion_out = runRecursiveGraph(input_tensor,W,V_in,V_out,b,initialize_f(config.embedding_size),input_shape,0)

	# loss, logits = getLoss({
	# 	'weights': W_out,
	# 	'biases': b_out,
	# 	'num_classes': NUM_CLASSES,
	# 	'output_word': output_tensor,
	# 	'network_output': recursion_out,
	# 	})

	optimizer = tf.train.AdamOptimizer(1e-3)
	# gradients, variables = zip(*optimizer.compute_gradients(loss))
	# gradients = [
	#   None if gradient is None else tf.clip_by_norm(gradient, 5.0)
	#   for gradient in gradients]
	# train_step_op = optimizer.apply_gradients(zip(gradients, variables))

	print("*************Initialising variables*****************")
	for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
		print("Initialising " + v.op.name)
		sess.run(v.initializer)
	print("Uninitialised varaiables")
	print(tf.report_uninitialized_variables())

	return W,\
	V_in,\
	V_out,\
	b,\
	W_out,\
	b_out,\
	optimizer,\
	W_weightage,\
	W_fc1,\
	b_fc1,\
	W_fc2,\
	b_fc2

def runRecursiveGraph(_input, W, V_in, V_out, b, f, input_shape, cur_pos):
	# print('len(input_shape)')
	# print(len(input_shape))
	# print('cur_pos')
	# print(cur_pos)
	if len(input_shape)<=cur_pos:
		return f
	else:
		return false_fn_0(_input, W, V_in, V_out, b, f, input_shape, cur_pos)

# def runRecursiveGraph(_input, W, V_in, V_out, b, f, input_shape, cur_pos):
# 	print('len(input_shape)')
# 	print(len(input_shape))
# 	print('cur_pos')
# 	print(cur_pos)
# 	return tf.cond(tf.equal(len(input_shape),cur_pos),
# 		lambda : true_fn_0(f),
# 		lambda : false_fn_0(_input, W, V_in, V_out, b, f, input_shape, cur_pos)
# 		)

def true_fn_0(f):
	return f

def false_fn_0(_input, W, V_in, V_out, b, f, input_shape, cur_pos):
	# print('input_shape')
	# print(input_shape)
	# print('cur_pos')
	# print(cur_pos)	
	count = input_shape[cur_pos]

	while count>0:
		_input,\
		W,\
		V_in,\
		V_out,\
		b,\
		f,\
		count,\
		input_shape = loop_func(_input,\
			W,\
			V_in,\
			V_out,\
			b,\
			f,\
			count,\
			input_shape,\
			cur_pos)
		

	cond = tf.equal(tf.count_nonzero(_input),1)
	f_relu = tf.cond(cond,
		lambda : f,
		lambda : tf.nn.relu(f))

	return f_relu

# def false_fn_0(_input, W, V_in, V_out, b, f, input_shape, cur_pos):
# 	# print('input_shape')
# 	# print(input_shape)
# 	# print('cur_pos')
# 	# print(cur_pos)	
# 	count = input_shape[cur_pos]

# 	cond = lambda input_json, W, V_in, V_out, b, f, count, input_shape, cur_pos: tf.greater(count,0)
# 	_input, W, V_in, V_out, b, f, count, input_shape = tf.while_loop(cond,
# 		lambda _input, W, V_in, V_out, b, f, count, input_shape, cur_pos: loop_func(_input, W, V_in, V_out, b, f, count, input_shape, cur_pos),
# 		[_input, W, V_in, V_out, b, f, count, input_shape, cur_pos])

# 	cond2 = tf.equal(tf.count_nonzero(_input),1)
# 	f_relu = tf.cond(cond2,
# 		lambda : f,
# 		lambda : tf.nn.relu(f))

# 	return f_relu


def loop_func(_input, W, V_in, V_out, b, f_sibling, count, input_shape, cur_pos):
	count = count-1
	f_children = runRecursiveGraph(_input[count],W,V_in,V_out,b,initialize_f(config.embedding_size),input_shape,cur_pos+1)

	#TODO: check
	if cur_pos==len(input_shape)-1:
		word = _input[count]

		cond = tf.greater(word,-1)
		embedding_char_in = tf.cond(cond,
			lambda : tf.nn.embedding_lookup(V_in,word),
			lambda : zero_embedding(config.embedding_size)) 
		
		embedding_char_in = tf.reshape(embedding_char_in,(1,32))

		f_node = tf.add(tf.matmul(embedding_char_in,W),b)
		f_till_node = tf.add(f_node,f_children)
		f = tf.add(f_sibling,f_till_node)
		return _input, W, V_in, V_out, b, f, count, input_shape
	else:
		return _input, W, V_in, V_out, b, f_children, count, input_shape


# def loop_func(_input, W, V_in, V_out, b, f_sibling, count, input_shape, cur_pos):
# 	count = count-1
# 	f_children = runRecursiveGraph(_input[count],W,V_in,V_out,b,initialize_f(config.embedding_size),input_shape,cur_pos+1)

# 	#TODO: check
# 	word = _input[count]
# 	for _ in range(len(input_shape)-cur_pos):
# 		word = word[0]
# 	cond = tf.greater(word,-1)
# 	embedding_char_in = tf.cond(cond,
# 		lambda : tf.nn.embedding_lookup(V_in,word),
# 		lambda : zero_embedding(config.embedding_size)) 
	

# 	f_node = tf.add(tf.matmul(embedding_char_in,W),b)
# 	f_till_node = tf.add(f_node,f_children)
# 	f = tf.add(f_sibling,f_till_node)
# 	return _input, W, V_in, V_out, b, f, count, input_shape

def initialize_f(embedding_size):
	return tf.constant(np.zeros([config.embedding_size]),dtype=tf.float32)

def zero_embedding(embedding_size):
	return tf.constant(np.zeros([config.embedding_size]),dtype=tf.float32)

def run_train_graph(sess,
	W,
	V_in,
	V_out,
	b,
	W_out,
	b_out,
	_input,
	output,
	optimizer,
	dictionary,
	NUM_CLASSES,
	op_arr,
	epoch,
	pos_dict,
	W_weightage,
	W_fc1,
	b_fc1,
	W_fc2,
	b_fc2,
	cur_example=0):
	
	if epoch == 0:
		input_json = _input['dep_tree']

		recursion_out = runRecursiveGraph2(input_json,W,V_in,V_out,b,dictionary,pos_dict,W_weightage,W_fc1,b_fc1,W_fc2,b_fc2)

		loss, logits = getLoss({
			'weights': W_out,
			'biases': b_out,
			'num_classes': NUM_CLASSES,
			'output_word': output,
			'network_output': recursion_out,
			})

		gradients, variables = zip(*optimizer.compute_gradients(loss))
		gradients = [
		  None if gradient is None else tf.clip_by_norm(gradient, 5.0)
		  for gradient in gradients]
		train_step_op = optimizer.apply_gradients(zip(gradients, variables))

		# print('variables')
		for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
			# print(v.op.name)
			# print(sess.run(tf.is_variable_initialized(v)))
			# print()
			if not sess.run(tf.is_variable_initialized(v)):
				sess.run(v.initializer)
			# sess.run(v.initializer)
		# print('========uninitialised variable==========')
		# print(tf.report_uninitialized_variables())

		evaluated_rec_out, evaluated_loss, _ = sess.run([recursion_out,loss,train_step_op])

		op_arr.append([recursion_out,loss,train_step_op])

		output = None
		input_json = None
		recursion_out = None
		loss = None
		logits = None
		gradients = None
		variables = None
		train_step_op = None
	else:
		recursion_out = op_arr[cur_example][0]
		loss = op_arr[cur_example][1]
		train_step_op = op_arr[cur_example][2]
		evaluated_rec_out, evaluated_loss, _ = sess.run([recursion_out,loss,train_step_op])

	return evaluated_loss

def run_test_graph(sess,
	W,
	V_in,
	V_out,
	b,
	W_out,
	b_out,
	_input,
	output,
	optimizer,
	dictionary,
	NUM_CLASSES,
	pos_dict,
	W_weightage,
    W_fc1,
    b_fc1,
    W_fc2,
    b_fc2):
	
	input_json = _input['dep_tree']

	recursion_out = runRecursiveGraph2(input_json,W,V_in,V_out,b,dictionary,pos_dict,W_weightage,W_fc1,b_fc1,W_fc2,b_fc2)

	loss, logits = getLoss({
		'weights': W_out,
		'biases': b_out,
		'num_classes': NUM_CLASSES,
		'output_word': output,
		'network_output': recursion_out,
		})

	evaluated_logits, evaluated_loss, evaluated_rec_out = sess.run([logits,loss,recursion_out])

	output = None
	input_json = None
	recursion_out = None
	loss = None
	logits = None
	gradients = None
	variables = None

	return evaluated_logits, evaluated_rec_out

def getOutEmbeddings(sess,
	W_out,
	b_out,
	outputs
	):
	
	print('outputs')
	print(outputs)
	out_embedding = tf.nn.embedding_lookup(tf.transpose(W_out),outputs)

	evaluated_out_embedding = sess.run(out_embedding)

	return evaluated_out_embedding

def getWeightage(words,W_weightage,W_fc1,b_fc1,W_fc2,b_fc2,dictionary,V_in):
	out = []
	for word in words:
		word_emb = tf.nn.embedding_lookup(V_in,word)
		word_emb_reshaped = tf.reshape(word_emb,(1,32))

		fc1_out_0 = tf.matmul(word_emb_reshaped,W_fc1)
		fc1_out_1 = tf.add(fc1_out_0,b_fc1)
		fc1_out = tf.nn.tanh(fc1_out_1)

		fc2_out_0 = tf.matmul(fc1_out,W_fc2)
		fc2_out_1 = tf.add(fc2_out_0,b_fc2)
		fc2_out = tf.nn.tanh(fc2_out_1)
		fc2_out = tf.reshape(fc2_out,())

		out.append(fc2_out)

	n_h = len(out)
	out_reshaped = tf.reshape(out,[1,n_h,1,1])
	k_size = [1,n_h,1,1]
	strides = [1,1,1,1]
	padding = 'VALID'
	pooled_outputs = tf.nn.max_pool(out_reshaped,k_size,strides,padding)
	pooled_outputs_reshaped = tf.reshape(pooled_outputs,())
	
	final_weight = tf.sign(pooled_outputs_reshaped)

	# print('fc2_out')
	# print(fc2_out)

	return final_weight


def runRecursiveGraph2(input_json,W,V_in,V_out,b,dictionary,pos_dict,W_weightage,W_fc1,b_fc1,W_fc2,b_fc2):
	if len(input_json)==0:
		return initialize_f(config.embedding_size)

	f = initialize_f(config.embedding_size)

	for node in input_json:
		# print('node')
		# print(node)
		if node.get('dep_tree',-1) != -1:
			cur_word = dictionary.get(node['dep_tree']['word'],dictionary['UNK'])
			children = node['dep_tree']['next']
			cur_pos = node['dep_tree']['pos']
		else:
			cur_word = dictionary.get(node['word'],dictionary['UNK'])
			children = node['next']
			cur_pos = node['pos']

		cur_embedding = tf.nn.embedding_lookup(V_in,cur_word)
		embedding_char_in = tf.reshape(cur_embedding,(1,32))
		f_cur = tf.add(tf.matmul(embedding_char_in,W[pos_dict[cur_pos]]),b)

		f_children = runRecursiveGraph2(children,W,V_in,V_out,b,dictionary,pos_dict,W_weightage,W_fc1,b_fc1,W_fc2,b_fc2)
		f_0 = tf.nn.relu(tf.add(f_cur,f_children))

		#find out the weightage to be given to this bunch
		words = [dictionary.get(cur_word,dictionary['UNK'])] + [dictionary.get(node['word'],dictionary['UNK']) for node in children]
		w = getWeightage(words,W_weightage,W_fc1,b_fc1,W_fc2,b_fc2,dictionary,V_in)

		f = tf.add(f,f_0*w)

	# f = tf.nn.relu(f)

	return f

def train_step(sess,
	recursion_out,
	loss,
	train_step_op,
	input_tensor,
	output_tensor,
	_input,
	output):
	
	feed_dict = {
		input_tensor: _input,
		output_tensor: output
	}

	evaluated_rec_out, evaluated_loss, _ = sess.run([recursion_out,loss,train_step_op],feed_dict=feed_dict)	

	return evaluated_loss

#define the loss function
def getLoss(params):
  logits = tf.matmul(params['network_output'], params['weights'])
  # logits = tf.nn.bias_add(logits, params['biases'])
  labels_reshaped = tf.reshape(params['output_word'],[-1])
  labels_one_hot = tf.one_hot(labels_reshaped, params['num_classes'])
  loss = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=labels_one_hot,
      logits=logits)

  loss = tf.reduce_sum(loss, axis=1)

  return loss, logits
