import tensorflow as tf
import numpy as np

from config import Config
from src.criterion import Criterion
from src.rnn import Rnn

import time


class Model:
    def __init__(self, n_layers, h, v, d, is_train=True):
        self.isTrain = is_train
        self.config = Config()
        self.criterion = Criterion()
        self.optimizer = tf.train.AdamOptimizer(self.config.lr)
        self.models = []
        self.embedding_grads = []
        self.graph = tf.get_default_graph()
        self.count = 0
        self.back_count = -1
        self.input_vecs = []
        self.prediction_tensor = []
        self.accuracy_tensor = []
        self.train_op = []
        self.loss = []
        self.loss, self.prediction_tensor, self.train_op, self.accuracy_tensor = self.construct_model_from_api()

    def construct_model_from_api(self):
        self._add_placeholders()
        inputs = self._one_hot_layer()

        def rnn_cell():
            return tf.nn.rnn_cell.BasicRNNCell(num_units=self.config.hidden_dim)

        cell = tf.contrib.rnn.MultiRNNCell([rnn_cell() for _ in range(self.config.n_layers)])
        outputs, last_states = tf.nn.dynamic_rnn(
                cell=cell,
                dtype=tf.float32,
                inputs=inputs)
        return self.add_projection(last_states[-1])

    def add_projection(self, last_state):
        with tf.variable_scope('projection', reuse=tf.AUTO_REUSE):
            self.U = tf.get_variable(name='U', shape=[self.config.hidden_dim, self.config.num_classes],
                                     initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            self.B = tf.get_variable(name='B', shape=[self.config.num_classes, ],
                                     initializer=tf.zeros_initializer(), trainable=True)
        scores = self._score(last_state)
        prediction_tensor = tf.cast(tf.argmax(scores, axis=1), tf.int32, name='prediction')
        loss = self.criterion.forward(scores, self.output_placeholder)

        # grad_output = self.criterion.backward(scores, self.output_placeholder)
        # grad_output = tf.gradients(ys=scores, xs=self.models[-1].outputs[-1], grad_ys=grad_output)
        train_op = self._apply_gradients(loss)
        accuracy_tensor = tf.reduce_mean(tf.cast(tf.equal(prediction_tensor, self.output_placeholder),
                                                 tf.float32), name='accuracy')
        return loss, prediction_tensor, train_op, accuracy_tensor

    def construct_model(self, n_layers, h, v, d):
        start = time.time()
        print('model construct model ...')
        self._add_placeholders()
        self.inputs = self._one_hot_layer()

        self.models.append(Rnn(d, h, self.dropout_placeholder, 0))
        for i in range(1, n_layers):
            self.models.append(Rnn(h, h, self.dropout_placeholder, i))

        with tf.variable_scope('projection', reuse=tf.AUTO_REUSE):
            self.U = tf.get_variable(name='U', shape=[self.config.hidden_dim, self.config.num_classes],
                                     initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            self.B = tf.get_variable(name='B', shape=[self.config.num_classes, ],
                                     initializer=tf.zeros_initializer(), trainable=True)

        print('model forward starting ....')
        for i in range(self.config.seq_length):
            input_vec = tf.squeeze(self.inputs[:, i:i + 1, :], axis=1)
            self.input_vecs.append(self.forward(input_vec, tf.less(i, self.seq_length)))
        loss, prediction_tensor, train_op, accuracy_tensor = self.compute_graph_for_time_step(self.seq_length)
        self.loss = loss
        self.prediction_tensor = prediction_tensor
        self.train_op = train_op
        self.accuracy_tensor = accuracy_tensor
        print("time taken for model creation = ", time.time() - start)

    def forward(self, input_vec, pred):
        print('model forward ....')
        for rnn in self.models:
            input_vec = rnn.forward(input_vec, pred)

        return input_vec

    def backward(self, input_vec, grad_output):
        print('model backward ...')
        grad_outputs = np.empty(len(self.models), dtype=object)

        stop = self.config.seq_length - self.models[0].extracted * self.config.truncated_delta
        start = stop - self.config.truncated_delta
        ys = self.models[-1].outputs[stop]

        for i in range(len(self.models)):
            if i > 0:
                previous_layer_states = self.models[i-1].outputs[start]
            else:
                previous_layer_states = input_vec
            grad_outputs[i] = self.models[i].backward(previous_layer_states, grad_output, ys)

        for rnn in self.models:
            rnn.extracted += 1
        self.back_count -= 1

        return grad_outputs[-1][1]

    def compute_graph_for_time_step(self, num_steps):
        # output = self._batch_norm(self.models[-1].outputs[num_steps], axes=[0])
        scores = self._score(self.models[-1].outputs[-1])
        prediction_tensor = tf.cast(tf.argmax(scores, axis=1), tf.int32, name='prediction')
        loss = self.criterion.forward(scores, self.output_placeholder)

        grad_output = self.criterion.backward(scores, self.output_placeholder)
        grad_output = tf.gradients(ys=scores, xs=self.models[-1].outputs[-1], grad_ys=grad_output)
        train_op = self._apply_gradients(loss)
        accuracy_tensor = tf.reduce_mean(tf.cast(tf.equal(prediction_tensor, self.output_placeholder),
                                                 tf.float32), name='accuracy')
        return loss, prediction_tensor, train_op, accuracy_tensor

    def run_batch(self, sess: tf.Session, train_data, train_length, label_data=None):
        if self.isTrain:
            drop_out = 1.0
        else:
            drop_out = 1.0

        if label_data is None:
            feed_dict = {
                self.input_placeholder: train_data,
                self.dropout_placeholder: drop_out,
                self.seq_length: train_length
            }
            return sess.run([self.prediction_tensor], feed_dict)

        feed_dict = {
            self.input_placeholder: train_data,
            self.output_placeholder: label_data,
            self.dropout_placeholder: drop_out,
            self.seq_length: train_length
        }

        if self.isTrain:
            loss, accuracy, prediction, _ = sess.run([self.loss, self.accuracy_tensor, self.prediction_tensor,
                                                      self.train_op], feed_dict)
        else:
            loss, accuracy, prediction = sess.run([self.loss, self.accuracy_tensor, self.prediction_tensor], feed_dict)
        return loss, accuracy, prediction

    def _add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32, shape=(None, None), name="input")
        self.output_placeholder = tf.placeholder(tf.int32, shape=(None, ), name="output")
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=(), name="dropout")
        self.seq_length = tf.placeholder(tf.int32, shape=(), name="seq_length")

    def _one_hot_layer(self):
        one_hots = []
        for i in range(self.config.batch_size):
            one_hots.append([tf.one_hot(self.input_placeholder[i], self.config.vocab_size)])
        return tf.concat(one_hots, axis=0)

    def _score(self, output):
        return tf.add(tf.matmul(output, self.U), self.B, name='score')

    def _batch_norm(self, tensor, axes):
        mean, var = tf.nn.moments(tensor, axes)
        return tf.nn.batch_normalization(tensor, mean, var, tf.zeros(mean.get_shape()),
                                         tf.ones(mean.get_shape()), 1e-6)

    def _apply_gradients(self, loss):
        grads, variables = zip(*self.optimizer.compute_gradients(loss))
        grads, _ = tf.clip_by_global_norm(grads, 1e2)
        return self.optimizer.apply_gradients(zip(grads, variables))
