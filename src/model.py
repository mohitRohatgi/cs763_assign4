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
        self._construct_model(n_layers, h, v)
        self.input_placeholder = tf.placeholder(tf.float32, shape=(self.config.batch_size, self.config.seq_length, v))
        self.output_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, ))
        self.optimizer = tf.train.AdamOptimizer(self.config.lr)

    # One step forward, responsibility is take care of all the layers
    def forward(self, input_vec):
        for rnn in self.models:
            input_vec = rnn.forward(input_vec)
        return input_vec

    # adding window for truncated BPTT.
    def backward(self, input_vec, grad_output, window=None):
        grad_outputs = []
        for i in range(len(self.models)):
            if i > 0:
                previous_layer_states = self.models[i-1].get_output(window)
            else:
                previous_layer_states = input_vec
            grad_outputs.append(self.models[i].backward(previous_layer_states, grad_output))
            self.models[i].clean(window)
        grad_output = grad_outputs[-1]
        del grad_outputs
        return grad_output

    # add model in this method.
    # assumption is first layer is placed at index 0.
    def _construct_model(self, n_layers, h, v):
        self.models = []
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            for i in range(n_layers):
                self.models.append(Rnn(v, h))

            with tf.variable_scope("projection", reuse=tf.AUTO_REUSE):
                self.U = tf.get_variable(name="U", shape=[h, self.config.num_classes],
                                         initializer=tf.contrib.layers.xavier_initializer())
                self.B = tf.get_variable(name="B", shape=[self.config.num_classes, ],
                                         initializer=tf.contrib.layers.xavier_initializer())

    def run_batch(self, sess: tf.Session, criterion: Criterion, train_data, label_data):
        batch_size, seq_length, vocab_size = train_data.shape
        input_vecs = []
        # TODO: change to seq_length
        for i in range(10):
            input_vec = tf.squeeze(self.input_placeholder[:, i:i+1, :], axis=1)
            input_vecs.append(self.forward(input_vec))
        scores = self.score()
        grad_output = criterion.backward(scores, self.output_placeholder)
        loss = criterion.forward(scores, self.output_placeholder)
        self.optimizer.minimize(loss=loss, var_list=[self.U, self.B])
        index = -1
        grad_updates = []
        # TODO: change to seq_length
        for _ in range(int(np.ceil(10 / float(self.config.truncated_delta)))):
            index -= self.config.truncated_delta
            if index < -len(input_vecs):
                index = 0
            grad_output = self.backward(input_vecs[index], grad_output, self.config.truncated_delta)
            # for model in self.models:
            #     grad_updates.append(model.apply_gradients(index == 0))

        feed_dict = {
            self.input_placeholder: train_data,
            self.output_placeholder: label_data
        }

        prediction_tensor = self.predict(scores)
        accuracy_tensor = tf.reduce_mean(tf.cast(tf.equal(prediction_tensor, label_data), tf.float32))

        sess.run(tf.global_variables_initializer())
        if self.isTrain:
            loss, accuracy, prediction, _ = sess.run([loss, accuracy_tensor, prediction_tensor, grad_updates], feed_dict)
        else:
            loss, accuracy, prediction = sess.run([loss, accuracy_tensor, prediction_tensor], feed_dict)
        return loss, accuracy, prediction

    def score(self):
        projection_output = tf.matmul(self.models[-1].outputs[-1], self.U) + self.B
        return tf.sigmoid(projection_output)

    # TODO: find roc curve and implement thresholds for prediction.
    def predict(self, scores):
        return tf.round(scores)
