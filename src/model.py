"""
Truncated BPTT should be introduced in this class.
Develop a structure for generic interaction of different RNN layers.
"""
import tensorflow as tf
from config import Config
from src.criterion import Criterion
from src.rnn import Rnn
import numpy as np


class Model:
    def __init__(self, n_layers, h, v, d, is_train=True):
        self.isTrain = is_train
        self.config = Config()
        self.criterion = Criterion()
        self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.seq_length))
        self.output_placeholder = tf.placeholder(tf.int32, shape=(None, ))
        self.optimizer = tf.train.AdamOptimizer(self.config.lr)
        self.models = []
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            for i in range(n_layers):
                self.models.append(Rnn(d, h, i))

            with tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE):
                embedding = tf.get_variable("embedding", shape=(v, d),
                                            initializer=tf.random_uniform_initializer(-1, 1), trainable=True)
                self.inputs = tf.nn.embedding_lookup(embedding, self.input_placeholder)

            with tf.variable_scope("projection", reuse=tf.AUTO_REUSE):
                self.U = tf.get_variable(name="U", shape=[h, self.config.num_classes],
                                         initializer=tf.contrib.layers.xavier_initializer())
                self.B = tf.get_variable(name="B", shape=[self.config.num_classes, ],
                                         initializer=tf.contrib.layers.xavier_initializer())
        self._construct_model()

    # One step forward, responsibility is take care of all the layers
    def forward(self, input_vec):
        for rnn in self.models:
            input_vec = rnn.forward(input_vec)
        return input_vec

    # adding window for truncated BPTT.
    def backward(self, input_vec, grad_output):
        grad_outputs = np.empty(len(self.models), dtype=object)
        for i in range(len(self.models)):
            if i > 0:
                previous_layer_states = self.models[i-1].get_output()
            else:
                previous_layer_states = input_vec
            grad_outputs[i] = self.models[i].backward(previous_layer_states, grad_output)

        for rnn in self.models:
            rnn.extracted += 1

        return grad_outputs[-1][0]

    def run_batch(self, sess: tf.Session, train_data, label_data):
        feed_dict = {
            self.input_placeholder: train_data,
            self.output_placeholder: label_data
        }

        if self.isTrain:
            loss, accuracy, prediction, _ = sess.run([self.loss, self.accuracy_tensor, self.prediction_tensor,
                                                      self.grad_updates], feed_dict)
        else:
            loss, accuracy, prediction = sess.run([self.loss, self.accuracy_tensor, self.prediction_tensor], feed_dict)
        return loss, accuracy, prediction

    def score(self):
        projection_output = tf.matmul(self.models[-1].outputs[-1], self.U) + self.B
        return tf.sigmoid(projection_output)

    # TODO: find roc curve and implement thresholds for prediction.
    def predict(self, scores):
        return tf.cast(tf.round(scores), tf.int32)

    # add model in this method.
    # assumption is first layer is placed at index 0.
    def _construct_model(self):
        input_vecs = []
        for i in range(self.config.num_steps):
            input_vec = tf.squeeze(self.inputs[:, i:i + 1, :], axis=1)
            input_vecs.append(self.forward(input_vec))
        scores = self.score()
        grad_output = self.criterion.backward(scores, self.output_placeholder)
        grad_output = tf.gradients(ys=scores, xs=self.models[-1].outputs[-1], grad_ys=grad_output)
        self.loss = self.criterion.forward(scores, self.output_placeholder)
        self.optimizer.minimize(loss=self.loss, var_list=[self.U, self.B])
        index = -1
        self.grad_updates = []
        for _ in range(int(np.ceil(self.config.num_steps / float(self.config.truncated_delta)))):
            index -= self.config.truncated_delta
            if index < -len(input_vecs):
                index = 0
            grad_output = self.backward(input_vecs[index], grad_output)

        for model in self.models:
            self.grad_updates.append(model.apply_gradients())

        self.prediction_tensor = self.predict(scores)
        self.accuracy_tensor = tf.reduce_mean(tf.cast(tf.equal(self.prediction_tensor, self.output_placeholder),
                                                      tf.float32))
