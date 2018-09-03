import numpy as np

class Bandit(object):
    def __init__(self, num_arms, T):
        self.k = num_arms
        self.T = T

    def reset(self):
        self.t = 0
        self.done = False

    def step(self):
        self.t += 1
        self.done = bool(self.t > self.T - 1)


class Indep(Bandit):
    def __init__(self, num_arms, T):
        super(Indep, self).__init__(num_arms, T)

    def reset(self):
        super(Indep, self).reset()
        self.outcome_probs = np.random.uniform(size=self.k)
        self.rewards = np.ones(self.k)

    def step(self, action):
        super(Indep, self).step()
        success = bool(np.random.random() < self.outcome_probs[action])
        reward = success * self.rewards[action] + (not success) * 0
        return reward, self.done


class Corr2Arm(Bandit):
    def __init__(self, T, difficulty):
        super(Corr2Arm, self).__init__(2, T)
        valid_args = ('easy', 'medium', 'hard', 'unif', 'indep')
        if difficulty not in valid_args:
            raise ValueError('Difficulty must be one of: {}'.format(valid_args))
        self.difficulty = difficulty

    def reset(self):
        super(Corr2Arm, self).reset()
        if self.difficulty == 'easy':
            p1 = np.random.choice([0.1, 0.9])
        elif self.difficulty == 'medium':
            p1 = np.random.choice([0.25, 0.75])
        elif self.difficulty == 'hard':
            p1 = np.random.choice([0.4, 0.6])
        elif self.difficulty == 'unif':
            p1 = np.random.uniform()

        if self.difficulty == 'indep':
            self.outcome_probs = np.random.uniform(size=self.k)
        else:
            self.outcome_probs = [p1, 1 - p1]
        self.rewards = np.ones(self.k)

    def step(self, action):
        super(Corr2Arm, self).step()
        success = bool(np.random.random() < self.outcome_probs[action])
        reward = success * self.rewards[action] + (not success) * 0
        return reward, self.done


class ElevenArm(Bandit):
    def __init__(self, T):
        super(ElevenArm, self).__init__(11, T)
        keys = np.append(np.arange(11) / 10, np.array(5))
        key_map = {}
        for i, k in enumerate(keys):
            key_map[k] = i
        self.key_map = key_map

    def reset(self):
        super(ElevenArm, self).reset()
        self.target_arm = np.random.choice(self.k - 1)
        self.outcome_probs = np.ones(self.k)
        self.rewards = np.ones(self.k)
        self.rewards[self.target_arm] = 5
        self.rewards[-1] = (self.target_arm + 1) / (self.k - 1)

    def step(self, action):
        super(ElevenArm, self).step()
        reward = self.rewards[action]
        return reward, self.done

    def t3_indexer(self, array):
        return [self.key_map[i] for i in array]
