from collections import namedtuple

import tensorflow as tf

nest = tf.contrib.framework.nest

Policy = namedtuple('Policy', 'logits action values')


def head(x, action_space):
    """Policy and value networks."""
    initializer = tf.contrib.layers.variance_scaling_initializer(
        factor=1.0, mode='FAN_IN', uniform=False)
    with tf.variable_scope('Policy', reuse=tf.AUTO_REUSE):
        logits = tf.layers.dense(x, action_space, activation=None,
                                 kernel_initializer=initializer)
        action = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
    with tf.variable_scope('Value', reuse=tf.AUTO_REUSE):
        values = tf.layers.dense(x, 1, activation=None,
                                 kernel_initializer=initializer)
    logits, action, values = nest.map_structure(
        lambda t: tf.squeeze(t), (logits, action, values))
    return Policy(logits, action, values)
