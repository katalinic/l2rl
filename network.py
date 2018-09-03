import tensorflow as tf

class BanditNetwork(object):
    def __init__(
            self,
            task,
            number_of_actions,
            hidden_units=48,
            learning_rate=1e-4,
            beta_v=0.05,
            beta_e=0.05,
            number_of_episodes=2e4):
        self.task = task
        self.number_of_actions = number_of_actions
        self.hidden_units = hidden_units
        self.lr = learning_rate
        self.beta_v = beta_v
        global_step = tf.get_variable(
            "global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
            trainable=False)
        self.global_step_increment = tf.assign_add(global_step, tf.constant(1, tf.int32))
        self.beta_e = tf.train.polynomial_decay(1.0, global_step, number_of_episodes, 0, power=1.) \
            if beta_e is None else beta_e
        self._build_placeholders()
        self._build_network()
        self._build_optimisation()

    def _build_placeholders(self):
        self.prev_r_pl = tf.placeholder(tf.float32, [None, self.number_of_actions+1]) \
            if self.task == 3 else tf.placeholder(tf.float32, [None, 1])
        self.value_pl = tf.placeholder(tf.float32, [None])
        self.prev_a_pl = tf.placeholder(tf.int32, [None])
        prev_a_pl_1h = tf.one_hot(self.prev_a_pl, depth=self.number_of_actions)
        input_pl = tf.concat([prev_a_pl_1h, self.prev_r_pl], axis=1)
        self.input_pl = tf.expand_dims(input_pl, axis=0)
        self.target_a_pl = tf.placeholder(tf.int32, [None])

    def _build_network(self):
        cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_units)
        self.init_rnn_state = cell.zero_state(1, tf.float32)
        unrolled, self.rnn_state = tf.nn.dynamic_rnn(
            cell, self.input_pl, initial_state=self.init_rnn_state, dtype=tf.float32)
        unrolled = tf.reshape(unrolled, [-1, self.hidden_units])
        self.logits = tf.contrib.layers.fully_connected(unrolled, self.number_of_actions,
                                                        activation_fn=None)
        value = tf.contrib.layers.fully_connected(unrolled, 1, activation_fn=None)
        self.value = tf.squeeze(value)
        self.probs = tf.nn.softmax(self.logits)
        action = tf.multinomial(self.logits, num_samples=1, output_dtype=tf.int32)
        self.action = tf.squeeze(action)

    def _build_optimisation(self):
        adv = self.value_pl - self.value
        logp = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.target_a_pl)
        pg = tf.reduce_sum(logp * tf.stop_gradient(adv))
        sq = 0.5 * tf.reduce_sum(tf.square(adv))
        entropy = -tf.reduce_sum(self.probs * tf.log(self.probs))
        loss = pg + self.beta_v * sq - self.beta_e * entropy
        optimiser = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=0.99)
        self.optimise = optimiser.minimize(loss)
