import tensorflow as tf
import numpy as np

from config import Config


class Rnn:
    def __init__(self, input_dimension, hidden_dimension, dropout_tensor, index=0):
        self.index = index
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        with tf.variable_scope("rnn_" + str(index)):
            self.config = Config()
            self.graph = tf.get_default_graph()
            self.initial_state = tf.get_variable(shape=[self.config.batch_size, hidden_dimension],
                                                 name="initial_state_" + str(index))
            self.initial_state_grad = None
            self.W = tf.get_variable(name="W_" + str(index), shape=[input_dimension + hidden_dimension, hidden_dimension],
                                     initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            self.B = tf.get_variable(name="B_" + str(index), shape=[hidden_dimension],
                                     initializer=tf.zeros_initializer(), trainable=True)
            self.gradW = None
            self.gradB = None
            self.dropout_placeholder = dropout_tensor
            self.gradWs = []
            self.gradBs = []
            self.outputs = []
            self.extracted = 0
            self.count = 0
            self.back_count = -1
            self.outputs.append(self.initial_state)

    def forward(self, input_vec):
        print("rnn forward ...")
        with tf.variable_scope("rnn_" + str(self.index), reuse=tf.AUTO_REUSE):
            W = tf.get_variable("W_" + str(self.index), [self.input_dimension + self.hidden_dimension,
                                                         self.hidden_dimension])
            B = tf.get_variable("B_" + str(self.index), [self.hidden_dimension])
            input_concat = tf.concat([input_vec, self.outputs[-1]], axis=1)
            state = tf.nn.tanh(tf.add(tf.matmul(input_concat, W), B), name="hidden_states")
            self.outputs.append(tf.nn.dropout(state, self.dropout_placeholder))
            self.count += 1
        return state

    def backward(self, input_vec, grad_output, ys):
        print('rnn backward ....')
        input_vec_grads = np.empty(2, dtype=object)
        stop = self.config.num_steps - self.extracted * self.config.truncated_delta
        start = stop - self.config.truncated_delta
        if start < 0:
            start = 0
        Ws = tf.get_collection("W_" + str(self.index))
        Bs = tf.get_collection("B_" + str(self.index))
        self.gradWs.extend(tf.gradients(ys=ys, xs=Ws[self.back_count], grad_ys=grad_output))
        self.gradBs.extend(tf.gradients(ys=ys, xs=Bs[self.back_count], grad_ys=grad_output))
        input_vec_grads[0] = tf.gradients(ys=ys, xs=input_vec, grad_ys=grad_output)
        input_vec_grads[1] = tf.gradients(ys=ys, xs=self.outputs[start], grad_ys=grad_output)
        self.initial_state_grad = tf.gradients(ys=ys, xs=self.initial_state, grad_ys=grad_output)
        self.back_count -= 1
        return input_vec_grads
