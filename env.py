import tensorflow as tf


class TFBandit(object):
    def __init__(self, dist):
        self._p1 = tf.get_variable(
            'p1', shape=[], dtype=tf.float32,
            initializer=tf.constant_initializer(0),
            trainable=False, use_resource=True)
        self.dist = dist
        self.action_space = 2

    def reset(self):
        possible_arm_probs = tf.constant(
            self.dist, shape=[2, 1], dtype=tf.float32)
        idx = tf.random_uniform([1], minval=0, maxval=2, dtype=tf.int32)
        p1 = tf.gather_nd(possible_arm_probs, idx)
        p1 = tf.squeeze(p1)
        with tf.control_dependencies([self._p1.assign(p1)]):
            return tf.zeros(1)

    def step(self, action):
        logprobs = tf.log([self._p1, 1 - self._p1])
        logprobs = tf.expand_dims(logprobs, axis=0)
        sample = tf.multinomial(
            logprobs, num_samples=1, output_dtype=tf.int32)[0]
        reward = tf.equal(action, sample)
        reward = tf.to_float(reward)
        return reward

    @property
    def p1(self):
        return self._p1
