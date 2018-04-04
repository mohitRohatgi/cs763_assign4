import tensorflow as tf
import numpy as np

from config import Config
from src.criterion import Criterion
from src.rnn import Rnn


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
        self.construct_model(n_layers, h, v, d)

    def construct_model(self, n_layers, h, v, d):
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):

            print('model construct model ...')
            self._add_placeholders()
            self.inputs = self._one_hot_layer()

            self.models.append(Rnn(d, h, self.dropout_placeholder, 0))
            for i in range(1, n_layers):
                self.models.append(Rnn(h, h, self.dropout_placeholder, i))

            print('model forward starting ....')
            self.input_vecs = []
            for i in range(self.config.num_steps):
                input_vec = tf.squeeze(self.inputs[:, i:i + 1, :], axis=1)
                self.input_vecs.append(self.forward(input_vec))

            for i in range(n_layers):
                self.models[i].dropout(self.dropout_placeholder)

            self.output = self._batch_norm(self.models[-1].outputs[-1], axes=[0])
            # output = self.models[-1].outputs[-1]
            scores = self._score(self.output)
            self.prediction_tensor = tf.cast(tf.argmax(scores, axis=1), tf.int32, name='prediction')
            self.loss = self.criterion.forward(scores, self.output_placeholder)

            grad_output = self.criterion.backward(scores, self.output_placeholder)
            grad_output = tf.gradients(ys=scores, xs=self.models[-1].outputs[-1], grad_ys=grad_output)
            self.train_op = self._apply_gradients()
            self.accuracy_tensor = tf.reduce_mean(tf.cast(tf.equal(self.prediction_tensor, self.output_placeholder),
                                                          tf.float32), name='accuracy')
            print('construct model done ...')

    def forward(self, input_vec):
        print('model forward ....')
        for rnn in self.models:
            input_vec = rnn.forward(input_vec)

        return input_vec

    def backward(self, input_vec, grad_output):
        print('model backward ...')
        grad_outputs = np.empty(len(self.models), dtype=object)

        stop = self.config.num_steps - self.models[0].extracted * self.config.truncated_delta
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

    def run_batch(self, sess: tf.Session, train_data, label_data=None):
        if self.isTrain:
            drop_out = 1.0
        else:
            drop_out = 1.0

        if label_data is None:
            feed_dict = {
                self.input_placeholder: train_data,
                self.dropout_placeholder: drop_out
            }
            return sess.run([self.prediction_tensor], feed_dict)

        feed_dict = {
            self.input_placeholder: train_data,
            self.output_placeholder: label_data,
            self.dropout_placeholder: drop_out
        }

        if self.isTrain:
            loss, accuracy, prediction, _ = sess.run([self.loss, self.accuracy_tensor, self.prediction_tensor,
                                                      self.train_op], feed_dict)
        else:
            loss, accuracy, prediction = sess.run([self.loss, self.accuracy_tensor, self.prediction_tensor], feed_dict)
        return loss, accuracy, prediction

    def _add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.seq_length), name="input")
        self.output_placeholder = tf.placeholder(tf.int32, shape=(None, ), name="output")
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=(), name="dropout")

    def _one_hot_layer(self):
        one_hots = []
        for i in range(self.config.batch_size):
            one_hots.append(tf.expand_dims(tf.one_hot(self.input_placeholder[i], self.config.vocab_size), axis=0))
        return tf.concat(one_hots, axis=0)

    def _score(self, output):
        with tf.variable_scope('projection', reuse=tf.AUTO_REUSE):
            U = tf.get_variable(name='U', shape=[self.config.hidden_dim, self.config.num_classes])
            B = tf.get_variable(name='B', shape=[self.config.num_classes, ])
        return tf.add(tf.matmul(output, U), B, name='score')

    def _batch_norm(self, tensor, axes):
        mean, var = tf.nn.moments(tensor, axes)
        return tf.nn.batch_normalization(tensor, mean, var, tf.zeros(mean.get_shape()),
                                         tf.ones(mean.get_shape()), 1e-6)

    def _apply_gradients(self):
        grads, variables = zip(*self.optimizer.compute_gradients(self.loss))
        grads, _ = tf.clip_by_global_norm(grads, 1e2)
        return self.optimizer.apply_gradients(zip(grads, variables))
