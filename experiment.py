import time
import tensorflow as tf
import numpy as np

unroll_length = 10
hidden_size = 2
num_actions = 2
gamma = 0.8
lr = 1e-3
beta_v = 0.05
beta_e = 0.05

class TFBandit(object):
    def __init__(self):
        self._p1 = tf.get_variable(
            'p1', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(0),
            trainable=False, use_resource=True)

    def reset(self):
        possible_arm_probs = tf.constant([0.1, 0.9], shape=[2, 1], dtype=tf.float32)
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
        # reward = tf.squeeze(reward)
        return reward

bandit = TFBandit()
bandit_reset_op = bandit.reset()

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
    return tf.stop_gradient(discounted_reward)


prev_action = tf.zeros([1], dtype=tf.int32)
prev_action_one_hot = tf.one_hot(prev_action, depth=num_actions)
prev_reward = tf.zeros([1,1], dtype=tf.float32)

cell = tf.contrib.rnn.LSTMBlockCell(hidden_size)
cell_init = cell.zero_state(1, dtype=tf.float32)  # Batch size hardcoded as 1.
cell_init_c, cell_init_h = tf.unstack(cell_init)  # Nesting workaround.

cell_input = tf.concat([prev_action_one_hot, prev_reward, cell_init_h], axis=1)

# Dummy agent logits for initialisation.
dummy_logits = tf.zeros([1, num_actions], tf.float32)

def step(input_, unused_i):
    # a is one-hot.
    prev_concat, c, h, _ = input_
    a, r = prev_concat[:, :num_actions], prev_concat[:, num_actions]
    # c, h = rnn_state
    new_h, new_rnn_state = cell(prev_concat, (c, h))
    new_c, new_h = tf.unstack(new_rnn_state)
    # New reward. a is one-hot initially.
    native_a = tf.argmax(a, axis=1, output_type=tf.int32)
    new_r = bandit.step(native_a)
    new_r = tf.expand_dims(new_r, 0)

    logits_ = tf.contrib.layers.fully_connected(new_h, num_actions, activation_fn=None)
    new_a = tf.multinomial(logits_, num_samples=1, output_dtype=tf.int32)[0]
    new_a = tf.one_hot(new_a, depth=num_actions, axis=1)

    new_concat = tf.concat([new_a, new_r, new_h], axis=1)
    return new_concat, new_c, new_h, logits_

with tf.control_dependencies([bandit_reset_op]):
    unrolled = tf.scan(
        step,
        tf.range(unroll_length),
        initializer=(cell_input, cell_init_c, cell_init_h, dummy_logits),
        parallel_iterations=1)

actions_and_rewards, _, outputs, logits = unrolled
outputs = tf.squeeze(outputs)
logits = tf.squeeze(logits)
actions_and_rewards = tf.squeeze(actions_and_rewards)
actions_taken = actions_and_rewards[:, :num_actions]  # This is one-hot
actions_taken = tf.argmax(actions_taken, axis=1, output_type=tf.int32)
rewards = actions_and_rewards[:, num_actions]
rewards = tf.squeeze(rewards)


value = tf.contrib.layers.fully_connected(outputs, 1, activation_fn=None)
value = tf.squeeze(value)

# Subsetting.
value = value[:-1]
logits = logits[:-1]
actions_taken = actions_taken[:-1]


discounted_targets = discount(rewards, gamma)
advantage = discounted_targets - value

cross_entropy_per_timestep = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=actions_taken)
policy_gradient_loss = tf.reduce_sum(
    cross_entropy_per_timestep * tf.stop_gradient(advantage))

advantage_loss = 0.5 * tf.reduce_sum(tf.square(advantage))

policy = tf.nn.softmax(logits)
log_policy = tf.nn.log_softmax(logits)
entropy = -tf.reduce_sum(policy * log_policy)

loss = policy_gradient_loss + beta_v * advantage_loss - beta_e * entropy
optimiser = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.99)
train_op = optimiser.minimize(loss)


# Train

T = 30000
sess = tf.Session()
sess.run(tf.global_variables_initializer())

start_time = time.time()
for i in range(T):
    sess.run(train_op)
    if i % 3000 == 0:
         print(sess.run(policy)[-1])
    # _, r = sess.run([train_op, rewards])
    # print (np.sum(r))
print(time.time()-start_time)
