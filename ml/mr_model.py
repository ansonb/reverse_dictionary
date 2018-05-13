import tensorflow as tf
# from tensorflow.python.ops import rnn
from tensorflow.contrib import rnn
import numpy as np

#define the loss function
def getLoss(params):
  labels_reshaped = tf.reshape(params['labels'],[-1])
  labels_one_hot = tf.one_hot(labels_reshaped, params['num_classes'])
  loss = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=labels_one_hot,
      logits=params['network_output'])

  loss = tf.reduce_sum(loss, axis=1)

  return loss

def getAccuracy(labels,logits):
  assert(len(labels)==logits.shape[0]), "The number of labels and logits must be the same"

  labels_reshaped = np.reshape(labels,[-1])

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

    accuracy = getAccuracy(y_batch,logits)
    print()
    print("{}: step, {} accuracy".format(step_num, accuracy))
    print()	

def test_step(x_batch,
  input_x,
  logits,
  batch_size,
  batch_size_tensor,
  sess):
    """
    test step
    """

    feed_dict = {
      input_x: x_batch,
      batch_size_tensor: batch_size
    }

    evaluated_logits = sess.run(logits,feed_dict=feed_dict)

    return evaluated_logits

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
  input_x,
  batch_size,
  batch_size_tensor,
  embedded_chars_x,
  sess):
    """
    Get the output embedding
    """
    feed_dict = {
      input_x: x_batch,
      batch_size_tensor: batch_size
    }
    embeddings = sess.run(
        embedded_chars_x,
        feed_dict)
    # print("{} encoder_outputs".format(encoder_outputs))
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
  NUM_CLASSES,
  vocabulary_size,
  embedding_size):
  sequence_length = seq_len
  vocab_size = len(dictionary)
  output_keep_prob = 1.0

  batch_size_tensor = tf.placeholder(tf.int32, shape=(), name='batch_size')

  hiddenSize = 32

  # num_true = 1
  num_sampled = 32
  num_classes = NUM_CLASSES

  input_x = tf.placeholder(tf.int32, shape=(None, sequence_length), name="input_x")
  input_y = tf.placeholder(tf.int32, shape=(None), name="input_y")

  #This will be the weights variable for the nce loss
  embedding_var_inp = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], minval=-1.0, maxval=1.0),name="embedding_inp")

  #This will be the bias variable for the nce loss
  bias_var = tf.Variable(tf.zeros(num_classes),name="emb_bias")

  embedded_chars_x = tf.nn.embedding_lookup(embedding_var_inp, input_x)
  embedded_chars_x = tf.transpose(embedded_chars_x,perm=[0,2,1])  

  #get a list of of tensors length = sequence_length
  #each tensor has shape = (batch_size x embedding_size)
  embedded_chars_x_list = [tf.reshape(tf.slice(embedded_chars_x,[0,0,i],[batch_size_tensor,embedding_size,1]),[batch_size_tensor,embedding_size]) for i in range(sequence_length)]

  encoDecoCell1 = lstm_cell(hiddenSize, output_keep_prob=output_keep_prob)
  encoDecoCell2 = lstm_cell(hiddenSize, output_keep_prob=output_keep_prob)
  encoDecoCell = tf.contrib.rnn.MultiRNNCell([encoDecoCell1,encoDecoCell2], state_is_tuple=True)

  encoder_outputs_x, encoder_state_x = rnn.static_rnn(encoDecoCell, embedded_chars_x_list, dtype=tf.float32)

  lstm_output = encoder_outputs_x[-1]

  if config.mr_pretrained:
    saver = tf.train.Saver()
    restore(config.checkpoint_dir,sess,saver)  
    variables_to_not_be_initialised = ['embedding_inp','emb_bias',
      'rnn/multi_rnn_cell/cell_0/basic_lstm_cell/weights',
      'rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biases',
      'Initialising rnn/multi_rnn_cell/cell_1/basic_lstm_cell/weights',
      'Initialising rnn/multi_rnn_cell/cell_1/basic_lstm_cell/biases',
      'Initialising beta1_power',
      'Initialising beta2_power',
      'Initialising embedding_inp/Adam',
      'Initialising embedding_inp/Adam_1',
      'Initialising emb_bias/Adam',
      'Initialising emb_bias/Adam_1',
      'Initialising rnn/multi_rnn_cell/cell_0/basic_lstm_cell/weights/Adam',
      'Initialising rnn/multi_rnn_cell/cell_0/basic_lstm_cell/weights/Adam_1',
      'Initialising rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biases/Adam',
      'Initialising rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biases/Adam_1',
      'Initialising rnn/multi_rnn_cell/cell_1/basic_lstm_cell/weights/Adam',
      'Initialising rnn/multi_rnn_cell/cell_1/basic_lstm_cell/weights/Adam_1',
      'Initialising rnn/multi_rnn_cell/cell_1/basic_lstm_cell/biases/Adam',
      'Initialising rnn/multi_rnn_cell/cell_1/basic_lstm_cell/biases/Adam_1']
    if not config.mr_fine_tune:
      variables_to_not_be_trained = ['embedding_inp','emb_bias',
      'rnn/multi_rnn_cell/cell_0/basic_lstm_cell/weights',
      'rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biases',
      'Initialising rnn/multi_rnn_cell/cell_1/basic_lstm_cell/weights',
      'Initialising rnn/multi_rnn_cell/cell_1/basic_lstm_cell/biases',
      'Initialising beta1_power',
      'Initialising beta2_power',
      'Initialising embedding_inp/Adam',
      'Initialising embedding_inp/Adam_1',
      'Initialising emb_bias/Adam',
      'Initialising emb_bias/Adam_1',
      'Initialising rnn/multi_rnn_cell/cell_0/basic_lstm_cell/weights/Adam',
      'Initialising rnn/multi_rnn_cell/cell_0/basic_lstm_cell/weights/Adam_1',
      'Initialising rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biases/Adam',
      'Initialising rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biases/Adam_1',
      'Initialising rnn/multi_rnn_cell/cell_1/basic_lstm_cell/weights/Adam',
      'Initialising rnn/multi_rnn_cell/cell_1/basic_lstm_cell/weights/Adam_1',
      'Initialising rnn/multi_rnn_cell/cell_1/basic_lstm_cell/biases/Adam',
      'Initialising rnn/multi_rnn_cell/cell_1/basic_lstm_cell/biases/Adam_1']
    else:
      variables_to_not_be_trained = []
  else:
    variables_to_not_be_trained = []
    variables_to_not_be_initialised = []


  #fc1 layer
  W_fc1 = tf.Variable(
    tf.random_uniform([embedding_size, 1000], minval=-1.0, maxval=1.0),name="W_fc1")
  b_fc1 = tf.Variable(tf.zeros([1000]))
  fc1_out = tf.add( tf.matmul(lstm_output,W_fc1), b_fc1 )

  #fc2 layer
  W_fc2 = tf.Variable(
    tf.random_uniform([1000, NUM_CLASSES], minval=-1.0, maxval=1.0),name="W_fc2")
  b_fc2 = tf.Variable(tf.zeros([NUM_CLASSES]))
  fc2_out = tf.add( tf.matmul(fc1_out,W_fc2), b_fc2 )

  train_loss = getLoss({
    'network_output': fc2_out,
    'labels': input_y,
    'num_classes': NUM_CLASSES
    })
  eval_loss = getLoss({
    'network_output': fc2_out,
    'labels': input_y,
    'num_classes': NUM_CLASSES
    })

  optimizer = tf.train.AdamOptimizer(1e-3)
  gradients, variables = zip(*optimizer.compute_gradients(train_loss))
  gradients = [
      None if gradient is None else tf.clip_by_norm(gradient, 5.0)
      for gradient in gradients]
  gradients_to_train = []
  variables_to_train = []
  for index,variable in enumerate(variables):
    gradient = gradients[index]
    if variable.op.name not in variables_to_not_be_trained:
      variables_to_train.append(variable)
      gradients_to_train.append(gradient)
  print('variables to be trained')
  print(variables_to_train)
  train_step_op = optimizer.apply_gradients(zip(gradients_to_train, variables_to_train))



  print("*************Initialising variables*****************")
  for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    if v.op.name not in variables_to_not_be_initialised:
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
    encoder_outputs_x,\
    embedded_chars_x,\
    embedding_var_inp,\
    fc2_out

def restoreModel(checkpoint_dir, sess, saver):
  latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
  print('latest_checkpoint')
  print(latest_checkpoint)
  saver.restore(sess,latest_checkpoint)
  start_index_model = latest_checkpoint.rfind('-') + 1
  step = int(latest_checkpoint[start_index_model:])
  print('restoring from step: {}'.format(step))

  return step

def getSentimentsFromLogits(evaluated_logits):
  print('evaluated_logits.shape')
  print(evaluated_logits.shape)
  sentiments = np.argmax(evaluated_logits,axis=1)
  print('sentiments shape')
  print(sentiments.shape)

  return sentiments