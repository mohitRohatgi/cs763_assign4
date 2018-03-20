import tensorflow as tf
from config import Config
from src.optimizers import AdamOptimizer


class Rnn:
    def __init__(self, input_dimension, hidden_dimension, index=0):
        with tf.variable_scope("rnn_"+ str(index)):
            self.config = Config()
            self.initial_state = tf.get_variable(shape=[self.config.batch_size, hidden_dimension], name="initial_state",
                                                 initializer=tf.random_uniform_initializer())
            self.initial_state_grad = None
            self.W = tf.get_variable(name="W", shape=[input_dimension + hidden_dimension, hidden_dimension],
                                     initializer=tf.contrib.layers.xavier_initializer())
            self.B = tf.get_variable(name="B", shape=[hidden_dimension],
                                     initializer=tf.random_uniform_initializer())
            self.gradW = tf.get_variable(name="gradW", shape=[input_dimension + hidden_dimension, hidden_dimension],
                                         initializer=tf.zeros_initializer(), trainable=False)
            self.gradB = tf.get_variable(name="gradB", shape=[hidden_dimension, 1],
                                         initializer=tf.zeros_initializer(), trainable=False)
            self.outputs = []
            self.outputs.append(self.initial_state)
            # TODO: implement optimizer in optimizer module.
            self.optimizer = AdamOptimizer()

    # considering forward to move forward only one time step.
    # input_vec is a tensor of shape -> None, input_dimension.
    # padding would be used for variable length inputs.
    def forward(self, input_vec):
        input_concat = tf.concat([input_vec, self.outputs[-1]], axis=1)
        state = tf.tanh(tf.matmul(input_concat, self.W) + self.B)
        self.outputs.append(state)
        return state

    # input vec would determine the window for gradient calculation.
    # grad_output is considered to be the gradient at the last layer.
    # adding a variable, window, to be used for deciding the window for truncated BPTT.
    def backward(self, input_vec, grad_output):
        grads = tf.gradients(ys=self.outputs[-1], xs=[self.W, self.B, input_vec, self.initial_state],
                             grad_ys=grad_output)
        self.gradW = grads[0]
        self.gradB = grads[1]
        input_vec_grad = grads[2]
        self.initial_state_grad = grads[3]
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

    # TODO: customize gradient application for different optimizers.
    def apply_gradients(self, update_initial_state=False):
        grads_and_vars = [(self.gradW, self.W), (self.gradB, self.B)]
        if update_initial_state:
            grads_and_vars.append((self.initial_state_grad, self.initial_state))
        return self.optimizer.apply_gradients(grads_and_vars)
