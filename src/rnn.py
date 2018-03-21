import tensorflow as tf
from config import Config
from src.optimizers import AdamOptimizer
import numpy as np


# TODO: implement truncated BPTT.
# adding index for identifying the rnn layer.
class Rnn:
    def __init__(self, input_dimension, hidden_dimension, dropout_tensor, index=0):
        self.index = index
        with tf.variable_scope("rnn_" + str(index)):
            self.config = Config()
            self.graph = tf.get_default_graph()
            self.initial_state = tf.get_variable(shape=[self.config.batch_size, hidden_dimension], name="initial_state_"
                                                 + str(index), initializer=tf.zeros_initializer())
            self.initial_state_grad = None
            self.W = tf.get_variable(name="W_" + str(index), shape=[input_dimension + hidden_dimension, hidden_dimension],
                                     initializer=tf.contrib.layers.xavier_initializer())
            self.B = tf.get_variable(name="B_" + str(index), shape=[hidden_dimension],
                                     initializer=tf.contrib.layers.xavier_initializer())
            self.gradW = tf.get_variable(name="gradW", shape=[input_dimension + hidden_dimension, hidden_dimension],
                                         initializer=tf.zeros_initializer(), trainable=False)
            self.gradB = tf.get_variable(name="gradB", shape=[hidden_dimension, 1],
                                         initializer=tf.zeros_initializer(), trainable=False)
            self.dropout_placeholder = dropout_tensor
            self.gradWs = []
            self.gradBs = []
            self.outputs = []
            self.outputs.append(self.initial_state)
            self.optimizer = AdamOptimizer(1e-5)
            self.extracted = 0

    # considering forward to move forward only one time step.
    # input_vec is a tensor of shape -> None, input_dimension.
    # padding would be used for variable length inputs.
    def forward(self, input_vec):
        with tf.variable_scope("rnn_" + str(self.index), reuse=tf.AUTO_REUSE):
            W = tf.identity(self.W)
            B = tf.identity(self.B)
            input_concat = tf.concat([input_vec, self.outputs[-1]], axis=1)
            state = tf.sigmoid(tf.matmul(input_concat, W) + B, name="hidden_states")
            self.outputs.append(tf.nn.dropout(state, self.dropout_placeholder))
        tf.add_to_collection("W_" + str(self.index), W)
        tf.add_to_collection("B_" + str(self.index), B)

        return state

    # input vec would determine the window for gradient calculation.
    # grad_output is considered to be the gradient at the last layer.
    # adding a variable, window, to be used for deciding the window for truncated BPTT.
    def backward(self, input_vec, grad_output):
        input_vec_grads = np.empty(2, dtype=object)
        stop = self.config.num_steps - self.extracted * self.config.truncated_delta
        start = stop - self.config.truncated_delta
        if start < 0:
            start = 0
        Ws = tf.get_collection("W_" + str(self.index))
        Bs = tf.get_collection("B_" + str(self.index))
        xs = [Ws, Bs, input_vec]
        if start == 0:
            xs.append(self.initial_state)
        ys = self.outputs[-self.extracted * self.config.truncated_delta - 1]
        self.gradWs += tf.gradients(ys=ys, xs=Ws[start: stop], grad_ys=grad_output)
        self.gradBs += tf.gradients(ys=ys, xs=Bs[start: stop], grad_ys=grad_output)
        input_vec_grads[0] = tf.gradients(ys=ys, xs=input_vec, grad_ys=grad_output)
        input_vec_grads[1] = tf.gradients(ys=ys, xs=self.outputs[start + 1], grad_ys=grad_output)
        if start < self.config.truncated_delta:
            self.initial_state_grad = tf.gradients(ys=ys, xs=self.initial_state, grad_ys=grad_output)
        return input_vec_grads

    # TODO: change output fetching.
    def get_output(self):
        if self.extracted * self.config.truncated_delta > len(self.outputs):
            return self.outputs[1]
        return self.outputs[-self.extracted * self.config.truncated_delta]

    # TODO: check truncated BPTT.
    def apply_gradients(self):
        self.gradW = tf.add_n(self.gradWs) / (self.config.truncated_delta * self.config.batch_size)
        self.gradB = tf.add_n(self.gradBs) / (self.config.truncated_delta * self.config.batch_size)
        grads_and_vars = [(self.gradW, self.W), (self.gradB, self.B), (self.initial_state_grad, self.initial_state)]
        return self.optimizer.apply_gradients(grads_and_vars)
