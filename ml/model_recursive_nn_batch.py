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
  input_tensor_shape,
  output_tensor_shape,
  level_arr):

	#TODO experiment with minval and maxval
	W = tf.Variable(tf.random_uniform([embedding_size,embedding_size], minval=-1.0, maxval=1.0), name='rnn_w_general', dtype=tf.float32)

	V_in = tf.Variable(tf.random_uniform([vocabulary_size,embedding_size], minval=-1.0, maxval=1.0), name='embedding_matrix_in', dtype=tf.float32)
	V_out = tf.Variable(tf.random_uniform([NUM_CLASSES,embedding_size], minval=-1.0, maxval=1.0), name='embedding_matrix_out', dtype=tf.float32)
	
	b = tf.Variable(np.zeros((embedding_size)), name='bias_general', dtype=tf.float32)

	W_out = tf.Variable(tf.random_uniform([embedding_size,NUM_CLASSES], minval=-1.0, maxval=1.0), name='w_out', dtype=tf.float32)
	b_out = tf.Variable(np.zeros((NUM_CLASSES)), name='bias_out', dtype=tf.float32)

	# input_shape_tensor = tf.placeholder(tf.int32,shape=[1])
	input_tensor = tf.placeholder(tf.int32,shape=input_tensor_shape)

	output_tensor = tf.placeholder(tf.int32,shape=output_tensor_shape)
	
	recursion_out, _ = runRecursiveGraph2(input_tensor,W,V_in,V_out,b,0,level_arr,0)
	

	loss, logits = getLoss({
		'weights': W_out,
		'biases': b_out,
		'num_classes': NUM_CLASSES,
		'output_word': output_tensor,
		'network_output': recursion_out,
		})

	optimizer = tf.train.AdamOptimizer(1e-3)
	gradients, variables = zip(*optimizer.compute_gradients(loss))
	gradients = [
	  None if gradient is None else tf.clip_by_norm(gradient, 5.0)
	  for gradient in gradients]
	train_step_op = optimizer.apply_gradients(zip(gradients, variables))

	print("*************Initialising variables*****************")
	for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
		print("Initialising " + v.op.name)
		sess.run(v.initializer)
	print("Uninitialised varaiables")
	print(tf.report_uninitialized_variables())

	return recursion_out,\
	loss,\
	train_step_op,\
	input_tensor,\
	output_tensor

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
	optimizer,
	dictionary,
	NUM_CLASSES):

	output = dictionary[_input['def_word']]

	input_json = _input['dep_tree']

	recursion_out = runRecursiveGraph2(input_json,W,V_in,V_out,b,dictionary)

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

	return evaluated_loss

def runRecursiveGraph2(input_tensor,W,V_in,V_out,b,count,level_arr,cur_level):
	#TODO: check
	if cur_level>=len(level_arr):
		return initialize_f(config.embedding_size), count

	f = initialize_f(config.embedding_size)
	for node_num in range(level_arr[cur_level]):
		# print('node')
		# print(node)
		# if node.get('dep_tree',-1) != -1:
		# 	cur_word = dictionary.get(node['dep_tree']['word'],dictionary['UNK'])
		# 	children = node['dep_tree']['next']
		# else:
		# 	cur_word = dictionary.get(node['word'],dictionary['UNK'])
		# 	children = node['next']

		f_children, count = runRecursiveGraph2(input_tensor,W,V_in,V_out,b,count,level_arr,cur_level+1)
		f = tf.add(f,f_children)

		cur_word = input_tensor[count]
		eq_op = tf.equal(cur_word,-1)
		casted_op = tf.cast(eq_op,tf.int32)
		cur_word_1 = tf.add(cur_word,casted_op)
		cur_embedding = tf.nn.embedding_lookup(V_in,cur_word_1)
		print('cur_embedding')
		print(cur_embedding)
		#why am I doig this?
		embedding_char_in = tf.reshape(cur_embedding,(-1,32))

		inverted_op = tf.add(tf.multiply(casted_op,-1),1)
		inverted_op = tf.cast(inverted_op,tf.float32)
		embedding_char_in = tf.multiply(inverted_op,tf.transpose(embedding_char_in))

		embedding_char_in = tf.reshape(cur_embedding,(-1,32))

		f = tf.add(tf.matmul(embedding_char_in,W),b)

		count += 1

	f = tf.nn.relu(f)

	return f, count

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
  logits = tf.nn.bias_add(logits, params['biases'])
  labels_reshaped = tf.reshape(params['output_word'],[-1])
  labels_one_hot = tf.one_hot(labels_reshaped, params['num_classes'])
  loss = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=labels_one_hot,
      logits=logits)

  loss = tf.reduce_sum(loss, axis=1)

  return loss, logits
