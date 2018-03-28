import tensorflow as tf


class Criterion:
    def forward(self, input_vec, target):
        batch_size, num_classes = input_vec.get_shape().as_list()
        one_hot_target = tf.one_hot(target, num_classes)
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_target, logits=input_vec))

    # using optimizers instead of gradients.
    def backward(self, input_vec, target):
        loss = self.forward(input_vec, target)
        return tf.gradients(loss, [input_vec])[0]
