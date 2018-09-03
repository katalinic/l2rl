import tensorflow as tf
import numpy as np
import time

class BanditAgent(object):
    def __init__(self, task, number_of_actions, sess, net):
        self.task = task
        self.number_of_actions = number_of_actions
        self.sess = sess
        self.net = net

    def trial_rollout(self, bandit, train):
        bandit.reset()
        a = 0
        r = 0.
        all_actions = [a]
        if train:
            all_rewards = [r]
        _rnn_state = self.sess.run(self.net.init_rnn_state)
        done = False
        while not done:
            #previous r will be task dependent (task 3 has it one-hot)
            prev_r = np.array(r).reshape(1, 1) if self.task != 3 \
                else self._one_hot(bandit.t3_indexer([r]), self.number_of_actions + 1)
            fd = {
                self.net.prev_a_pl : [a],
                self.net.prev_r_pl : prev_r,
                self.net.init_rnn_state : _rnn_state}
            a, self._probs, _rnn_state = self.sess.run(
                [self.net.action, self.net.probs, self.net.rnn_state], feed_dict=fd)
            r, done = bandit.step(a)
            all_actions.append(a)
            if train:
                all_rewards.append(r)
        if train:
            return all_actions, all_rewards
        else: return all_actions

    def run_optimisation(self, all_actions, all_rewards, gamma, bandit):
        R = 0
        discounted_reward = []
        for r in all_rewards[1:][::-1]:
            R = r + gamma * R
            discounted_reward.append(R)
        discounted_reward = discounted_reward[::-1]
        all_prev_r = np.array(all_rewards[:-1]).reshape(-1, 1) if self.task != 3 \
            else self._one_hot(bandit.t3_indexer(all_rewards[:-1]), self.number_of_actions + 1)
        fd = {
            self.net.prev_a_pl : all_actions[:-1],
            self.net.prev_r_pl : all_prev_r,
            self.net.target_a_pl : all_actions[1:],
            self.net.value_pl : discounted_reward}
        self.sess.run([self.net.global_step_increment, self.net.optimise], feed_dict=fd)

    def train(self, bandit, num_episodes, gamma, model_directory):
        init_t = time.time()
        for ep in range(num_episodes + 1):
            all_actions, all_rewards = self.trial_rollout(bandit, True)
            self.run_optimisation(all_actions, all_rewards, gamma, bandit)
            if ep % (num_episodes // 10) == 0:
                print("Training Completion: {:.2f}%, Time Taken: {:.2f}".format(
                    100*(ep / num_episodes), time.time() - init_t))
                init_t = time.time()
        saver = tf.train.Saver()
        saver.save(self.sess, model_directory + 'model.checkpoint')

    def test(self, bandit, test_eps, model_directory):
        saver = tf.train.Saver()
        chckpoint = tf.train.get_checkpoint_state(model_directory)
        saver.restore(self.sess, chckpoint.model_checkpoint_path)
        cumulative_regret = []
        for _ in range(test_eps):
            all_actions = self.trial_rollout(bandit, False)
            #cumulative regret - assumes reward for positive outcomes, no penalty for negative
            opt_val = np.max(bandit.outcome_probs*bandit.rewards)
            all_actions_1h = self._one_hot(all_actions[1:], self.number_of_actions)
            exp_reward = (all_actions_1h * bandit.outcome_probs).dot(bandit.rewards)
            regret = opt_val-exp_reward
            cumulative_regret.append(np.cumsum(regret))
        cumulative_regret = np.array(cumulative_regret)
        avg_cumulative_regret = np.mean(cumulative_regret, axis=0)
        return avg_cumulative_regret

    @staticmethod
    def _one_hot(array, depth):
        """Multi-dimensional one-hot"""
        a = np.array(array)
        x = a.flatten()
        b = np.eye(depth)[x, :depth]
        return b.reshape(a.shape + (depth,))
