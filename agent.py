import tensorflow as tf
import numpy as np
from metaRLNet import MetaRLNetwork

class BanditAgent(object):
    def __init__(self, task, sess, learning_rate, number_of_actions, number_of_episodes):
        self.task = task
        self.number_of_actions = number_of_actions
        self.sess = sess
        self.net = MetaRLNetwork(task = task, number_of_actions = number_of_actions, learning_rate = learning_rate, beta_e = None, number_of_episodes = number_of_episodes)

    def trial_rollout(self, bandit, train):
        bandit.reset()
        a = np.random.choice(self.number_of_actions)
        r = 0.
        all_actions = [a]
        if train: all_rewards = [r]
        _rnn_state = self.sess.run(self.net.init_rnn_state)
        done = False

        while not done:
            #previous r will be task dependent (task 3 has it one-hot)
            prev_r = np.array(r).reshape(1,1) if self.task != 3 else self._one_hot(bandit.t3_indexer([r]), self.number_of_actions+1)
            self._probs, _rnn_state = self.sess.run([self.net.probs, self.net.rnn_state], \
                feed_dict = {self.net.prev_a_pl : [a], self.net.prev_r_pl : prev_r, self.net.init_rnn_state : _rnn_state})
            a = np.random.choice(self.number_of_actions, p=self._probs[0])
            r, done = bandit.step(a)
            all_actions.append(a)

            if train: all_rewards.append(r)

        if train:
            return all_actions, all_rewards
        else: return all_actions

    def run_optimisation(self, all_actions, all_rewards, gamma, bandit):
        R = 0
        discounted_reward = []

        for r in all_rewards[1:][::-1]:
            R=r+gamma*R
            discounted_reward.append(R)

        all_prev_r = np.array(all_rewards[:-1]).reshape(-1,1) if self.task != 3 else self._one_hot(bandit.t3_indexer(all_rewards[:-1]), self.number_of_actions+1)
        fd = {
            self.net.prev_a_pl : all_actions[:-1],
            self.net.prev_r_pl : all_prev_r,
            self.net.target_a_pl : all_actions[1:],
            self.net.value_pl : discounted_reward[::-1]
            }

        self.sess.run([self.net.global_step_increment, self.net.optimise], feed_dict = fd)

    def train(self, bandit, num_episodes, gamma, save_progress, **kwargs):
        if save_progress:
            saver = tf.train.Saver(max_to_keep=5)
        for ep in range(num_episodes+1):
            all_actions, all_rewards = self.trial_rollout(bandit, True)
            self.run_optimisation(all_actions, all_rewards, gamma, bandit)
            if ep>0 and ep%kwargs['save_every']==0:
                saver.save(self.sess, kwargs['model_directory']+'model.checkpoint',global_step=ep)
            #print training progress
            if ep%(num_episodes//20)==0:
                if bandit.task == 3 :
                  print (all_actions[1:], self._probs[0,bandit.target_arm])
                else: print (self._probs)
                print ("Current Episode: {}, Training Completion: {}%".format(ep, 100*np.round(ep/num_episodes,2)))

    def test(self, bandit, test_eps, restore, **kwargs):
        if restore:
            saver = tf.train.Saver()
            chckpoint = tf.train.get_checkpoint_state(kwargs['model_directory'])
            saver.restore(self.sess, chckpoint.model_checkpoint_path)
        cumulative_regret = []
        for ep in range(test_eps):
            all_actions = self.trial_rollout(bandit, False)
            #cumulative regret - assumes reward for positive outcomes, no penalty for negative
            opt_val = np.max(bandit.outcome_probs*bandit.rewards)
            all_actions_1h = self._one_hot(all_actions[1:], self.number_of_actions)
            exp_reward = (all_actions_1h*bandit.outcome_probs).dot(bandit.rewards)
            regret = opt_val-exp_reward
            cumulative_regret.append(np.cumsum(regret))

        cumulative_regret = np.array(cumulative_regret)
        avg_cumulative_regret = np.mean(cumulative_regret,axis=0)
        return avg_cumulative_regret

    @staticmethod
    def _one_hot(array, depth):
      """Multi-dimensional one-hot"""
      a = np.array(array)
      x = a.flatten()
      b = np.eye(depth)[x, :depth]
      return b.reshape(a.shape + (depth,))