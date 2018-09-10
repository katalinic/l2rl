import os
import time
import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
flags.DEFINE_float("gamma", 0.8, "Discount rate.")
flags.DEFINE_float("beta_v", 0.05, "Value loss coefficient.")
flags.DEFINE_float("beta_e", 0.05, "Entropy loss coefficient.")
flags.DEFINE_float("rms_decay", 0.99, "RMS decay.")
flags.DEFINE_integer("hidden_size", 48, "LSTM size.")
flags.DEFINE_integer("num_actions", 2, "Number of arms.")
flags.DEFINE_integer("unroll_length", 100, "Number of trials.")
flags.DEFINE_integer("train_eps", 30000, "Number of train episodes.")
flags.DEFINE_integer("test_eps", 150, "Number of test episodes.")
flags.DEFINE_boolean("training", False, "Boolean for training. Testing if False.")
flags.DEFINE_string("train_difficulty", 'medium', "Training difficulty.")
flags.DEFINE_string("test_difficulty", 'easy', "Testing difficulty.")

difficulties = {
    'easy' : [0.1, 0.9],
    'medium' : [0.25, 0.75],
    'hard' : [0.4, 0.6]
    }


class TFBandit(object):
    def __init__(self, dist):
        self._p1 = tf.get_variable(
            'p1', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(0),
            trainable=False, use_resource=True)
        self.dist = dist

    def reset(self):
        possible_arm_probs = tf.constant(self.dist, shape=[2, 1], dtype=tf.float32)
        idx = tf.random_uniform([1], minval=0, maxval=2, dtype=tf.int32)
        p1 = tf.gather_nd(possible_arm_probs, idx)
        p1 = tf.squeeze(p1)
        return self._p1.assign(p1)

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


def base():
    prev_action = tf.zeros([1], dtype=tf.int32)
    prev_action_one_hot = tf.one_hot(prev_action, depth=FLAGS.num_actions)
    prev_reward = tf.zeros([1, 1], dtype=tf.float32)

    cell = tf.contrib.rnn.LSTMBlockCell(FLAGS.hidden_size)
    cell_init = cell.zero_state(1, dtype=tf.float32)  # Batch size hardcoded as 1.
    cell_init_c, cell_init_h = tf.unstack(cell_init)  # Nesting workaround.

    cell_input = tf.concat([prev_action_one_hot, prev_reward, cell_init_h], axis=1)

    # Dummy agent logits for initialisation.
    dummy_logits = tf.zeros([1, FLAGS.num_actions], tf.float32)
    step_init = (cell_input, cell_init_c, cell_init_h, dummy_logits)
    return cell, step_init

def torso(bandit, cell, step_init):
    def step(input_, unused_i):
        # a is one-hot.
        prev_concat, c, h, _ = input_
        a, r = prev_concat[:, :FLAGS.num_actions], prev_concat[:, FLAGS.num_actions]
        # c, h = rnn_state
        new_h, new_rnn_state = cell(prev_concat, (c, h))
        new_c, new_h = tf.unstack(new_rnn_state)
        # New reward. a is one-hot initially.
        native_a = tf.argmax(a, axis=1, output_type=tf.int32)
        new_r = bandit.step(native_a)
        new_r = tf.expand_dims(new_r, 0)

        logits_ = tf.contrib.layers.fully_connected(
            new_h, FLAGS.num_actions, activation_fn=None)
        new_a = tf.multinomial(logits_, num_samples=1, output_dtype=tf.int32)[0]
        new_a = tf.one_hot(new_a, depth=FLAGS.num_actions, axis=1)

        new_concat = tf.concat([new_a, new_r, new_h], axis=1)
        return new_concat, new_c, new_h, logits_

    with tf.control_dependencies([bandit.reset()]):
        unrolled = tf.scan(
            step,
            tf.range(FLAGS.unroll_length),
            initializer=step_init,
            parallel_iterations=1)

    actions_and_rewards, _, outputs, logits = unrolled
    outputs = tf.squeeze(outputs)
    logits = tf.squeeze(logits)
    actions_and_rewards = tf.squeeze(actions_and_rewards)
    actions_1h = actions_and_rewards[:, :FLAGS.num_actions]  # This is one-hot
    rewards = actions_and_rewards[:, FLAGS.num_actions]

    values = tf.contrib.layers.fully_connected(outputs, 1, activation_fn=None)
    values = tf.squeeze(values)

    # Subsetting.
    values = values[:-1]
    logits = logits[:-1]
    actions_1h = actions_1h[:-1]

    return logits, actions_1h, rewards, values


def optimisation(logits, actions_1h, rewards, values):
    # Discount calculation
    def discount(rewards, gamma):
        tf_gamma = tf.constant(gamma, tf.float32)
        reversed_rewards = tf.squeeze(rewards)
        reversed_rewards = tf.reverse(reversed_rewards[1:], axis=[0])
        discounted_reward = tf.scan(
            lambda R, r: r + tf_gamma * R,
            reversed_rewards,
            initializer=tf.zeros([], tf.float32),
            back_prop=False,
            parallel_iterations=1)
        discounted_reward = tf.reverse(discounted_reward, axis=[0])
        return discounted_reward

    discounted_targets = discount(rewards, FLAGS.gamma)
    advantages = discounted_targets - values

    def policy_gradient_loss(logits, actions_1h, advantages):
        actions = tf.argmax(actions_1h, axis=1, output_type=tf.int32)
        cross_entropy_per_timestep = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=actions)
        policy_gradient = tf.reduce_sum(
            cross_entropy_per_timestep * tf.stop_gradient(advantages))
        return policy_gradient

    def advantage_loss(advantages):
        advantage_loss = 0.5 * tf.reduce_sum(tf.square(advantages))
        return advantage_loss

    def entropy_loss(logits):
        policy = tf.nn.softmax(logits)
        log_policy = tf.nn.log_softmax(logits)
        entropy = -tf.reduce_sum(policy * log_policy)
        return -entropy

    loss = policy_gradient_loss(logits, actions_1h, advantages)
    loss += FLAGS.beta_v * advantage_loss(advantages)
    loss += FLAGS.beta_e * entropy_loss(logits)

    optimiser = tf.train.RMSPropOptimizer(
        learning_rate=FLAGS.learning_rate, decay=FLAGS.rms_decay)
    train_op = optimiser.minimize(loss)
    return train_op

def train(model_directory, train_difficulty):
    bandit = TFBandit(difficulties[train_difficulty])

    cell, step_init = base()
    torso_output = torso(bandit, cell, step_init)
    train_op = optimisation(*torso_output)

    # Train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    start_time = time.time()
    for _ in range(FLAGS.train_eps):
        sess.run(train_op)
    print("Training completed. Time taken: {:.2f}".format(time.time() - start_time))

    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    saver = tf.train.Saver()
    saver.save(sess, model_directory + 'model.checkpoint')

def test(model_directory, test_difficulty):
    bandit = TFBandit(difficulties[test_difficulty])

    cell, step_init = base()
    torso_output = torso(bandit, cell, step_init)
    _, actions_1h, _, _ = torso_output

    sess = tf.Session()

    saver = tf.train.Saver()
    chckpoint = tf.train.get_checkpoint_state(model_directory)
    saver.restore(sess, chckpoint.model_checkpoint_path)

    def cumulative_regret():
        episodic_regret = []
        for _ in range(FLAGS.test_eps):
            all_actions = sess.run(actions_1h)
            # p1 has to be called separately to ensure it's the value actions_1h
            # was produced on.
            p1 = sess.run(bandit.p1)
            opt_val = np.max([p1, 1 - p1])
            exp_reward = (all_actions * [p1, 1 - p1]).dot([1., 1.])
            regret = opt_val - exp_reward
            episodic_regret.append(np.cumsum(regret))
        episodic_regret = np.array(episodic_regret)
        avg_cumulative_regret = np.mean(episodic_regret, axis=0)
        print(avg_cumulative_regret[-1])
        return avg_cumulative_regret[-1]
    return cumulative_regret()


def main(_):
    model_directory = './models/{}/'.format(FLAGS.train_difficulty)
    if FLAGS.training:
        train(model_directory, FLAGS.train_difficulty)
    else:
        if not os.path.exists(model_directory):
            raise ValueError(
                'Model of difficulty {} has not been trained yet.'.format(
                    FLAGS.train_difficulty))
        test(model_directory, FLAGS.test_difficulty)

if __name__ == '__main__':
    tf.app.run()
