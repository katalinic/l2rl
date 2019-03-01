import os
import time

import tensorflow as tf
import numpy as np

from rollouts import rollout
from optimisation import loss_calculation


class Worker:
    """Handles all agent-environment interaction.

    This includes rollouts, optimisation, model loading and saving,
    performance tracking.
    """
    def __init__(self, env, agent, constants=None):
        self.env = env
        self.agent = agent
        self.action_space = self.agent.action_space
        self.constants = constants

    def build_rollout(self):
        self.rollout_outputs = rollout(self.env, self.agent,
                                       self.constants.unroll_length)

    def build_loss(self):
        self.loss = loss_calculation(
            self.rollout_outputs, self.action_space, self.constants)

    def build_optimisation(self, optimiser):
        self.train_op = optimiser.minimize(self.loss)

    def train(self, sess):
        start = time.time()
        for _ in range(self.constants.train_steps):
            sess.run(self.train_op)
        print(time.time() - start)

    def evaluate(self, sess):
        print('Testing.')
        episodic_regret = []
        for _ in range(self.constants.test_eps):
            _, _, agent_outputs = sess.run(self.rollout_outputs)
            p1 = sess.run(self.env.p1)
            regret = self.env.calculate_cumulative_regret(
                agent_outputs.action, p1, self.action_space)
            episodic_regret.append(regret)
        episodic_regret = np.array(episodic_regret)
        avg_cumulative_regret = np.mean(episodic_regret, axis=0)
        print(avg_cumulative_regret[-1])

    def save_model(self, sess, saver, model_directory):
        if not os.path.exists(model_directory):
            os.mkdir(model_directory)
        saver.save(sess, model_directory + 'model.checkpoint')

    def load_model(self, sess, saver, model_directory):
        chckpoint = tf.train.get_checkpoint_state(model_directory)
        if chckpoint is not None:
            saver.restore(sess, chckpoint.model_checkpoint_path)
