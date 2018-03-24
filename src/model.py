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
        self._construct_model(n_layers, h, v, d)

    def forward(self, input_vec):
        for rnn in self.models:
            input_vec = rnn.forward(input_vec)
        return input_vec

    def backward(self, input_vec, grad_output):
        grad_outputs = np.empty(len(self.models), dtype=object)

        stop = self.config.num_steps - self.models[0].extracted * self.config.truncated_delta
        start = stop - self.config.truncated_delta
        embeds = tf.get_collection("embeddings")
        ys = self.models[-1].outputs[stop]
        self.embedding_grads += tf.gradients(ys=ys, xs=embeds[start:stop], grad_ys=grad_output)

        for i in range(len(self.models)):
            if i > 0:
                previous_layer_states = self.models[i-1].outputs[start]
            else:
                previous_layer_states = input_vec
            grad_outputs[i] = self.models[i].backward(previous_layer_states, grad_output, ys)

        for rnn in self.models:
            rnn.extracted += 1

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

    def predict(self, scores):
        return tf.cast(tf.round(tf.sigmoid(scores)), tf.int32, name='prediction')

    def _construct_model(self, n_layers, h, v, d):
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):

            self._add_placeholders()
            self._add_embeddings(v, d)
            self._add_projection(h)

            self.models.append(Rnn(d, h, self.dropout_placeholder, 0))
            for i in range(1, n_layers):
                self.models.append(Rnn(h, h, self.dropout_placeholder, i))

            input_vecs = []
            for i in range(self.config.num_steps):
                input_vec = tf.squeeze(self.inputs[:, i:i + 1, :], axis=1)
                input_vecs.append(self.forward(input_vec))

            for i in range(n_layers):
                self.models[i].dropout(self.dropout_placeholder)

            scores = self._score()
            self.prediction_tensor = self.predict(scores)
            self.loss = self.criterion.forward(scores, self.output_placeholder)

            grad_output = self.criterion.backward(scores, self.output_placeholder)
            grad_output = tf.gradients(ys=scores, xs=self.models[-1].outputs[-1], grad_ys=grad_output)

            for current in range(1, int(np.ceil(self.config.num_steps / float(self.config.truncated_delta))) + 1):
                index = self.config.num_steps - current * self.config.truncated_delta
                if index < -len(input_vecs):
                    index = 0
                grad_output = self.backward(input_vecs[index], grad_output)

            self.embedding_grad = tf.add_n(self.embedding_grads) / (self.config.truncated_delta * self.config.batch_size)
            self.accuracy_tensor = tf.reduce_mean(tf.cast(tf.equal(self.prediction_tensor, self.output_placeholder),
                                                          tf.float32), name='accuracy')
            self.train_op = self._apply_gradients()

    def _add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.seq_length), name="input")
        self.output_placeholder = tf.placeholder(tf.int32, shape=(None, ), name="output")
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=(), name="dropout")

    def _add_projection(self, h):
        with tf.variable_scope('projection', reuse=tf.AUTO_REUSE):
            self.U = tf.get_variable(name='U', shape=[h, self.config.num_classes],
                                     initializer=tf.contrib.layers.xavier_initializer())
            self.B = tf.get_variable(name='B', shape=[self.config.num_classes, ],
                                     initializer=tf.contrib.layers.xavier_initializer())

    def _add_embeddings(self, v, d):
        with tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE):
            self.embedding = tf.get_variable('embedding_matrix', shape=(v, d),
                                             initializer=tf.contrib.layers.xavier_initializer())
            self.inputs = self._lookup_layer(self._one_hot_layer())
            self.inputs = tf.nn.dropout(self.inputs, self.dropout_placeholder)
            self.inputs = self._batch_norm(self.inputs)

    def _one_hot_layer(self):
        one_hots = []
        for i in range(self.config.batch_size):
            one_hots.append(tf.expand_dims(tf.one_hot(self.input_placeholder[i], self.config.vocab_size), axis=0))
        return tf.concat(one_hots, axis=0)

    def _lookup_layer(self, one_hot_input):
        embeddings = tf.get_variable('embedding_matrix', [self.config.vocab_size, self.config.embed_size])
        embeddings = tf.identity(embeddings)
        self.graph.add_to_collection('embeddings', embeddings)
        inputs = []
        for i in range(self.config.batch_size):
            inputs.append(tf.expand_dims(tf.matmul(one_hot_input[i, :, :], embeddings), axis=0))
        return tf.concat(inputs, axis=0)

    def _score(self):
        score = tf.add(tf.matmul(self.models[-1].outputs[-1], self.U), self.B, name='score')
        return self._batch_norm(score)

    def _batch_norm(self, tensor):
        mean, var = tf.nn.moments(tensor, axes=[0])
        return tf.nn.batch_normalization(tensor, mean, var, tf.ones(mean.get_shape().as_list()[1:]),
                                         tf.zeros(mean.get_shape().as_list()[1:]), 1e-3)

    def _apply_gradients(self):
        grad_and_vars = []
        for model in self.models:
            grad_and_vars += model.get_gradients()

        grad_and_vars.append((self.embedding_grad, self.embedding))
        grad_and_vars += self.optimizer.compute_gradients(-self.loss, tf.trainable_variables())

        return self.optimizer.apply_gradients(grad_and_vars)
