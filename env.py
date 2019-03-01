from collections import namedtuple

import tensorflow as tf
import numpy as np

EnvOutput = namedtuple('EnvOutput', 'reward')


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
            return EnvOutput(tf.zeros(1))

    def step(self, action):
        logprobs = tf.log([self._p1, 1 - self._p1])
        logprobs = tf.expand_dims(logprobs, axis=0)
        sample = tf.multinomial(
            logprobs, num_samples=1, output_dtype=tf.int32)[0]
        reward = tf.equal(action, sample)
        reward = tf.to_float(reward)
        return EnvOutput(reward)

    @property
    def p1(self):
        return self._p1

    @staticmethod
    def calculate_cumulative_regret(actions, p1, action_space):
        actions = one_hot(actions, action_space)
        opt_val = np.max([p1, 1 - p1])
        exp_reward = (actions * [p1, 1 - p1]).dot([1., 1.])
        regret = opt_val - exp_reward
        return np.cumsum(regret)


def one_hot(input_, depth):
    num_rows = input_.shape[0]
    zeros = np.zeros((num_rows, depth))
    zeros[np.arange(num_rows), input_] = 1
    return zeros
