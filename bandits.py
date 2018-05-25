import numpy as np

class Bandits(object):
    def __init__(self, task, **kwargs):
        self.task = task
        if kwargs: self.difficulty = kwargs['difficulty']
        self.rollout_length = 100 if task !=3 else 5
        self.number_of_actions = 2 if task != 3 else 11
        #one-hot reward section for task 3
        if task == 3:
            #keeping initial prev reward of 0 so adding an additional entry
            keys = np.append(np.arange(self.number_of_actions)/10,np.array(5))
            key_map = {}
            for i,k in enumerate(keys):
                key_map[k] = i
            self.key_map = key_map

    def reset(self):
        self.t = 0
        if self.task==1:
            self.outcome_probs = np.random.uniform(self.number_of_actions)
            self.rewards = np.ones(self.number_of_actions)
        elif self.task==2:
            if self.difficulty == 'easy': p1 = np.random.choice([0.1, 0.9])
            elif self.difficulty == 'medium': p1 = np.random.choice([0.25, 0.75])
            elif self.difficulty == 'hard': p1 = np.random.choice([0.4, 0.6])
            else: p1 = np.random.uniform()
            self.outcome_probs = [p1, 1-p1]
            self.rewards = np.ones(self.number_of_actions)
        else:
            self.target_arm = np.random.choice(self.number_of_actions-1)
            self.outcome_probs = np.ones(self.number_of_actions)
            self.rewards = np.ones(self.number_of_actions)
            self.rewards[self.target_arm] = 5
            self.rewards[-1] = (self.target_arm+1)/(self.number_of_actions-1)

    def step(self, action):
        self.t += 1
        done = True if self.t > self.rollout_length-1 else False
        if self.task != 3:
            success = bool(np.random.random() < self.outcome_probs[action])
            reward = success * self.rewards[action] + (not success) * 0
        else:
            reward = self.rewards[action]
        return reward, done

    def t3_indexer(self, array):
        return [self.key_map[i] for i in array]
