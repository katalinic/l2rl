import tensorflow as tf

from sub_networks import head


class L2RLAgent:
    def __init__(self, rnn_size, action_space):
        self.action_space = action_space
        self._cell = tf.contrib.rnn.LSTMBlockCell(rnn_size)

    def init_state(self):
        return self._cell.zero_state(batch_size=1, dtype=tf.float32)

    def build(self, agent_state):
        policy_output = head(agent_state.h, self.action_space)
        return policy_output

    def update_state(self, env_output, agent_output, prev_agent_state):
        r = tf.expand_dims(env_output.reward, 0)
        a = tf.expand_dims(agent_output.action, 0)
        a = tf.one_hot(a, depth=self.action_space, axis=1)
        concat = tf.concat([a, r], axis=1)
        with tf.variable_scope('RNN'):
            _, rnn_state = self._cell(concat, prev_agent_state)
        return rnn_state
