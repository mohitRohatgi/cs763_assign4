import tensorflow as tf


class AdamOptimizer(tf.train.Optimizer):
    def _apply_dense(self, grad, var):
        pass

    def _resource_apply_dense(self, grad, handle):
        pass

    def _resource_apply_sparse(self, grad, handle, indices):
        pass

    def _apply_sparse(self, grad, var):
        pass

    def __init__(self, use_locking=False, name="optimizer", alpha=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8):

        super().__init__(use_locking, name)
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = {}
        self.u = {}
        self.t = tf.Variable(0.0, trainable=False)

        for v in tf.trainable_variables():
            self.m[v] = tf.Variable(tf.zeros(tf.shape(v.initial_value)), trainable=False)
            self.u[v] = tf.Variable(tf.zeros(tf.shape(v.initial_value)), trainable=False)

    def apply_gradients(self, gvs, **kwargs):
        t = self.t.assign_add(1.0)
        gvs = [(tf.reshape(tf.clip_by_value(grad, -0.99, 0.99), tf.shape(var)), var) for grad, var in gvs]

        update_ops = []
        for (g, v) in gvs:
            m = self.m[v].assign(self.beta1 * self.m[v] + (1 - self.beta1) * g)
            u = self.u[v].assign(self.beta2 * self.u[v] + (1 - self.beta2) * g * g)
            m_hat = m / (1 - tf.pow(self.beta1, t))
            u_hat = u / (1 - tf.pow(self.beta2, t))

            update = -self.alpha * m_hat / (tf.sqrt(u_hat) + self.epsilon)
            update_ops.append(v.assign_add(update))

        return tf.group(*update_ops)
