import tensorflow as tf
import numpy as np

from config import Config


class Rnn:
    def __init__(self, input_dimension, hidden_dimension, dropout_tensor, index=0):
        self.index = index
        with tf.variable_scope("rnn_" + str(index)):
            self.config = Config()
            self.graph = tf.get_default_graph()
            self.initial_state = tf.get_variable(shape=[self.config.batch_size, hidden_dimension], trainable=True,
                                                 initializer=tf.zeros_initializer(), validate_shape=True,
                                                 name="initial_state_" + str(index))
            self.initial_state_grad = None
            self.W = tf.get_variable(name="W_" + str(index), shape=[input_dimension + hidden_dimension, hidden_dimension],
                                     initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            self.B = tf.get_variable(name="B_" + str(index), shape=[hidden_dimension],
                                     initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            self.gradW = None
            self.gradB = None
            self.dropout_placeholder = dropout_tensor
            self.gradWs = []
            self.gradBs = []
            self.outputs = []
            self.outputs.append(self.initial_state)
            self.extracted = 0

    def forward(self, input_vec):
        with tf.variable_scope("rnn_" + str(self.index), reuse=tf.AUTO_REUSE):
            W = tf.identity(self.W)
            B = tf.identity(self.B)
            input_concat = tf.concat([input_vec, self.outputs[-1]], axis=1)
            state = tf.nn.tanh(tf.add(tf.matmul(input_concat, W), B), name="hidden_states")
            self.outputs.append(tf.nn.dropout(state, self.dropout_placeholder))
        self.graph.add_to_collection("W_" + str(self.index), W)
        self.graph.add_to_collection("B_" + str(self.index), B)

        return state

    def backward(self, input_vec, grad_output, ys):
        input_vec_grads = np.empty(2, dtype=object)
        stop = self.config.num_steps - self.extracted * self.config.truncated_delta
        start = stop - self.config.truncated_delta
        if start < 0:
            start = 0
        Ws = tf.get_collection("W_" + str(self.index))
        Bs = tf.get_collection("B_" + str(self.index))
        self.gradWs.extend(tf.gradients(ys=ys, xs=Ws[start: stop], grad_ys=grad_output))
        self.gradBs.extend(tf.gradients(ys=ys, xs=Bs[start: stop], grad_ys=grad_output))
        input_vec_grads[0] = tf.gradients(ys=ys, xs=input_vec, grad_ys=grad_output)
        input_vec_grads[1] = tf.gradients(ys=ys, xs=self.outputs[start], grad_ys=grad_output)
        if start == 0:
            self.initial_state_grad = tf.gradients(ys=ys, xs=self.initial_state, grad_ys=grad_output)
        return input_vec_grads

    def dropout(self, dropout_tensor):
        self.outputs = [tf.nn.dropout(output, dropout_tensor) for output in self.outputs]

    def get_output(self):
        if self.extracted * self.config.truncated_delta >= len(self.outputs):
            return self.outputs[1]
        return self.outputs[-self.extracted * self.config.truncated_delta - 1]

    def get_gradients(self):
        self.gradW = tf.add_n(self.gradWs) / (self.config.truncated_delta * self.config.batch_size)
        self.gradB = tf.add_n(self.gradBs) / (self.config.truncated_delta * self.config.batch_size)

        return [(self.gradW, self.W), (self.gradB, self.B),
                (tf.reshape(self.initial_state_grad, [self.config.batch_size, self.config.hidden_dim]),
                 self.initial_state)]
