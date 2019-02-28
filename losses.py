import tensorflow as tf


def entropy(logits):
    policy = tf.nn.softmax(logits)
    log_policy = tf.nn.log_softmax(logits)
    entropy_per_timestep = - tf.reduce_sum(policy * log_policy, axis=1)
    return tf.reduce_sum(-entropy_per_timestep)


def policy_gradient(logits, actions, advantages):
    cross_entropy_per_timestep = \
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=actions)
    policy_gradient_per_timestep = \
        cross_entropy_per_timestep * tf.stop_gradient(advantages)
    return tf.reduce_sum(policy_gradient_per_timestep)


def compute_return(rewards, discounts):
    return discount(rewards, discounts, 0.)


def discount(rewards, discounts, bootstrap_value):
    reversed_rewards = tf.reverse(rewards, axis=[0])
    reversed_discounts = tf.reverse(discounts, axis=[0])
    discounted = tf.scan(
        lambda R, v: v[0] + R * v[1],
        [reversed_rewards, reversed_discounts],
        initializer=bootstrap_value,
        back_prop=False,
        parallel_iterations=1)
    discounted = tf.reverse(discounted, axis=[0])
    return tf.stop_gradient(discounted)


def advantage_loss(advantages):
    advantage_loss_per_timestep = 0.5 * tf.square(advantages)
    return tf.reduce_sum(advantage_loss_per_timestep)
