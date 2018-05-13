import tensorflow as tf
# from tensorflow.python.ops import rnn
from tensorflow.contrib import rnn
import numpy as np

#define the loss function
def getLoss(params):
  logits = tf.matmul(params['inputs'], tf.transpose(params['weights']))
  # logits = tf.nn.bias_add(logits, params['biases'])
  labels_reshaped = tf.reshape(params['labels'],[-1])
  labels_one_hot = tf.one_hot(labels_reshaped, params['num_classes'])
  loss = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=labels_one_hot,
      logits=logits)

  loss = tf.reduce_sum(loss, axis=1)

  return loss, logits

def getAccuracy(labels,logits):
  # print("The shape of labels is {}, but the shpe of logits is {}".format(len(labels),logits.shape))
  assert(len(labels)==logits.shape[0]), "The number of labels and logits must be the same"
  # print("labels")
  # print(labels)
  # print("logits")
  # print(logits)
  labels_reshaped = np.reshape(labels,[-1])
  # print(np.argmax(logits,axis=1))
  # print(labels_reshaped)
  return np.sum(np.equal(labels_reshaped,np.argmax(logits,axis=1)))/logits.shape[0]


def lstm_cell(lstm_size,
  output_keep_prob=1.0):
  # return tf.contrib.rnn.BasicLSTMCell(lstm_size)
  encoDecoCell = tf.contrib.rnn.BasicLSTMCell(lstm_size, state_is_tuple=True)  # Or GRUCell, LSTMCell(args.hiddenSize)
  #Only for training output_keep_prob is 0.5
  encoDecoCell = tf.contrib.rnn.DropoutWrapper(encoDecoCell, input_keep_prob=1.0, output_keep_prob=output_keep_prob)  # TODO: Custom values
  return encoDecoCell


def train_step(x_batch,
  y_batch,
  input_x,
  input_y,
  loss,
  train_step_op,
  step_num,
  batch_size,
  batch_size_tensor,
  sess):
    """
    A single training step
    """
    feed_dict = {
      input_x: x_batch,
      input_y: y_batch,
      batch_size_tensor: batch_size
    }

    _, loss = sess.run(
        [train_step_op, loss],
        feed_dict)

    loss = np.sum(loss)/loss.shape[0]
    print("{}: step, {} loss".format(step_num, loss))

def eval_step(x_batch,
  y_batch,
  input_x,
  input_y,
  eval_loss,
  logits,
  step_num,
  batch_size,
  batch_size_tensor,
  sess):
    """
    eval step
    """
    feed_dict = {
      input_x: x_batch,
      input_y: y_batch,
      batch_size_tensor: batch_size
    }

    loss, logits = sess.run(
        [eval_loss,logits],
        feed_dict)
    print('shape of logits')
    print(logits.shape)
    accuracy = getAccuracy(y_batch,logits)
    print()
    print("{}: step, {} accuracy".format(step_num, accuracy))
    print()	

def test_step(x_batch,
  y_batch,
  input_x,
  input_y,
  logits,
  batch_size,
  batch_size_tensor,
  sess):
    """
    test step
    """
    print('input y')
    print(input_y)

    feed_dict = {
      input_x: x_batch,
      batch_size_tensor: batch_size
    }

    logits = sess.run(logits,feed_dict=feed_dict)
    print('shape of logits')
    print(logits.shape)
    accuracy = getAccuracy(y_batch,logits)

    return logits, accuracy

def eval_out_embedding(y_batch,
  input_y,
  batch_size,
  batch_size_tensor,
  embedded_chars_y,
  sess):
    """
    Get the output embedding
    """
    feed_dict = {
      input_y: y_batch,
      batch_size_tensor: batch_size
    }
    embeddings = sess.run(
        embedded_chars_y,
        feed_dict)
    # print("{} encoder_outputs".format(encoder_outputs))
    return embeddings

def eval_in_embedding(x_batch,
  embedding_var_inp,
  sess):
    """
    Get the output embedding
    """
    op = tf.nn.embedding_lookup(embedding_var_inp,x_batch)
    embeddings = sess.run(op)

    return embeddings

def getParVector(x_batch,
  input_x,
  encoder_outputs_x,
  batch_size,
  batch_size_tensor,
  sess):
    """
    Return the paragraph vector
    """
    feed_dict = {
      input_x: x_batch,
      batch_size_tensor: batch_size
    }
    encoder_outputs = sess.run(
        encoder_outputs_x,
        feed_dict)
    # print("{} encoder_outputs shape".format(encoder_outputs))
    return encoder_outputs[-1]

def build_graph(sess,
  seq_len,
  dictionary,
  batch_size,
  NUM_CLASSES,
  vocabulary_size,
  embedding_size,
  out_names_dict_indices,
  num_true=1):
  sequence_length = seq_len
  vocab_size = len(dictionary)
  output_keep_prob = 1.0

  # batch_size = 10
  batch_size_tensor = tf.placeholder(tf.int32, shape=(), name='batch_size')

  hiddenSize = 32

  # num_true = 1
  num_sampled = 32
  num_classes = NUM_CLASSES

  input_x = tf.placeholder(tf.int32, shape=(None, sequence_length), name="input_x")
  input_y = tf.placeholder(tf.int32, shape=(None,num_true), name="input_y")


  #This will be the weights variable for the nce loss
  embedding_var_inp = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], minval=-1.0, maxval=1.0),name="embedding_inp")
  embedding_var_out = tf.nn.embedding_lookup(embedding_var_inp,out_names_dict_indices,name="embedding_out")
  # embedding_var_out = tf.Variable(
  #       tf.random_uniform([num_classes, embedding_size], minval=-1.0, maxval=1.0),name="embedding_out")

  #This will be the bias variable for the nce loss
  bias_var = tf.Variable(tf.zeros(num_classes),name="emb_bias")

  embedded_chars_x = tf.nn.embedding_lookup(embedding_var_inp, input_x)
  embedded_chars_x = tf.transpose(embedded_chars_x,perm=[0,2,1])  
  
  #get a list of of tensors length = sequence_length
  #each tensor has shape = (batch_size x embedding_size)
  embedded_chars_x_list = [tf.reshape(tf.slice(embedded_chars_x,[0,0,i],[batch_size_tensor,embedding_size,1]),[batch_size_tensor,embedding_size]) for i in range(sequence_length)]

  embedded_chars_y = tf.nn.embedding_lookup(embedding_var_out, input_y)

  encoDecoCell1 = lstm_cell(hiddenSize, output_keep_prob=output_keep_prob)
  encoDecoCell2 = lstm_cell(hiddenSize, output_keep_prob=output_keep_prob)
  encoDecoCell = tf.contrib.rnn.MultiRNNCell([encoDecoCell1,encoDecoCell2], state_is_tuple=True)
  # sess.run(tf.global_variables_initializer())

  encoder_outputs_x, encoder_state_x = rnn.static_rnn(encoDecoCell, embedded_chars_x_list, dtype=tf.float32)

  train_loss,_ = getLoss({
    'weights': embedding_var_out,
    'biases': bias_var,
    'labels': input_y,
    'inputs': encoder_outputs_x[-1],
    'num_classes': num_classes
    })
  eval_loss,logits = getLoss({
    'weights': embedding_var_out,
    'biases': bias_var,
    'labels': input_y,
    'inputs': encoder_outputs_x[-1],
    'num_classes': num_classes
    })

  optimizer = tf.train.AdamOptimizer(1e-3)
  gradients, variables = zip(*optimizer.compute_gradients(train_loss))
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

  return input_x,\
    input_y,\
    train_loss,\
    train_step_op,\
    batch_size_tensor,\
    eval_loss,\
    logits,\
    embedded_chars_y,\
    encoder_outputs_x,\
    embedded_chars_x,\
    embedding_var_inp

def loadTestGraph(sess,
  meta_file_path,
  checkpoint_dir):
  print('meta file path')
  print(meta_file_path)
  print(checkpoint_dir)
  saver = tf.train.import_meta_graph(meta_file_path)
  restoreModel(checkpoint_dir,sess,saver)
  return saver


def restoreModel(checkpoint_dir, sess, saver):
  latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
  print('latest_checkpoint')
  print(latest_checkpoint)
  saver.restore(sess,latest_checkpoint)
  start_index_model = latest_checkpoint.rfind('-') + 1
  step = int(latest_checkpoint[start_index_model:])
  print('restoring from step: {}'.format(step))

  return step
