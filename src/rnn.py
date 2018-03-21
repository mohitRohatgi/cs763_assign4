import tensorflow as tf
from config import Config
from src.optimizers import AdamOptimizer


# TODO: implement truncated BPTT.
# adding index for identifying the rnn layer.
class Rnn:
    def __init__(self, input_dimension, hidden_dimension, index=0):
        self.index = index
        with tf.variable_scope("rnn_" + str(index)):
            self.config = Config()
            self.graph = tf.get_default_graph()
            self.initial_state = tf.get_variable(shape=[self.config.batch_size, hidden_dimension], name="initial_state_"
                                                 + str(index), initializer=tf.random_uniform_initializer())
            self.initial_state_grad = None
            self.W = tf.get_variable(name="W_" + str(index), shape=[input_dimension + hidden_dimension, hidden_dimension],
                                     initializer=tf.contrib.layers.xavier_initializer())
            self.B = tf.get_variable(name="B_" + str(index), shape=[hidden_dimension],
                                     initializer=tf.random_uniform_initializer())
            self.gradW = tf.get_variable(name="gradW", shape=[input_dimension + hidden_dimension, hidden_dimension],
                                         initializer=tf.zeros_initializer(), trainable=False)
            self.gradB = tf.get_variable(name="gradB", shape=[hidden_dimension, 1],
                                         initializer=tf.zeros_initializer(), trainable=False)
            self.gradWs = []
            self.gradBs = []
            self.outputs = []
            self.outputs.append(self.initial_state)
            self.optimizer = AdamOptimizer(1e-5)

    # considering forward to move forward only one time step.
    # input_vec is a tensor of shape -> None, input_dimension.
    # padding would be used for variable length inputs.
    def forward(self, input_vec):
        with tf.variable_scope("rnn_" + str(self.index), reuse=tf.AUTO_REUSE):
            W = tf.identity(self.W)
            B = tf.identity(self.B)
            input_concat = tf.concat([input_vec, self.outputs[-1]], axis=1)
            state = tf.tanh(tf.matmul(input_concat, W) + B)
            self.outputs.append(state)
        tf.add_to_collection("W_" + str(self.index), W)
        tf.add_to_collection("B_" + str(self.index), B)
        return state

    # input vec would determine the window for gradient calculation.
    # grad_output is considered to be the gradient at the last layer.
    # adding a variable, window, to be used for deciding the window for truncated BPTT.
    def backward(self, input_vec, grad_output):
        stop = len(self.outputs) - 1
        start = stop - self.config.truncated_delta
        if start < 0:
            start = 0
        Ws = tf.get_collection("W_" + str(self.index))
        Bs = tf.get_collection("B_" + str(self.index))
        xs = Ws[start: stop] + Bs[start: stop]
        xs.append(input_vec)
        if start == 0:
            xs.append(self.initial_state)
        grads = tf.gradients(ys=self.outputs[-1], xs=xs, grad_ys=grad_output)
        self.gradWs += grads[0: stop - start]
        self.gradBs += grads[stop - start: 2 * (stop - start)]
        input_vec_grad = grads[-2]
        if start == 0:
            self.initial_state_grad = grads[-1]
        return input_vec_grad

    def get_output(self, window=None):
        if window is None:
            return self.outputs[1]
        else:
            return self.outputs[-window]

    # delete outputs for the next back prop.
    def clean(self, window):
        if window is None:
            del self.outputs[1:]
        else:
            if window >= len(self.outputs):
                window = -1
            del self.outputs[-window:]

    # TODO: check truncated BPTT.
    def apply_gradients(self):
        self.gradW = tf.add_n(self.gradWs) / (self.config.truncated_delta * self.config.batch_size)
        self.gradB = tf.add_n(self.gradBs) / (self.config.truncated_delta * self.config.batch_size)
        grads_and_vars = [(self.gradW, self.W), (self.gradB, self.B), (self.initial_state_grad, self.initial_state)]
        self._clear()
        return self.optimizer.apply_gradients(grads_and_vars)

    def _clear(self):
        del tf.get_collection_ref("W_" + str(self.index))[:]
        del tf.get_collection_ref("B_" + str(self.index))[:]
        self.gradWs = []
        self.gradWs = []
