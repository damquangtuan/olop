import numpy as np

from gym.utils import seeding
from gym import Env, spaces

class SixArmsEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, nS=7, nA=6):
        
        # Defining the number of actions
        self.nA = nA
        self.nS = nS

        rew = [50.0 / 6000.0, 133.0 / 6000.0, 300.0 / 6000.0, 800.0 / 6000.0, 1660.0 / 6000.0, 6000.0 / 6000.0]
        self.p = self.compute_probabilities_arms(nS, nA)
        self.r = self.compute_rewards_arms(nS, nA, rew)
        self.mu = self.compute_mu_arms(nS)

        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)

    def compute_probabilities_arms(self, nS, nA):
        p = np.zeros((nS, nA, nS))
        # state 1
        p[0, 0, 1] = 1
        p[1, 0, 1] = p[1, 1, 1] = p[1, 2, 1] = p[1, 3, 1] = p[1, 5, 1] = 1
        p[1, 4, 0] = 1
        # state 2
        p[0, 1, 2] = 0.15
        p[0, 1, 0] = 0.85
        p[2, 0, 0] = p[2, 2, 0] = p[2, 3, 0] = p[2, 4, 0] = p[2, 5, 0] = 1
        p[2, 1, 2] = 1
        # state 3
        p[0, 2, 3] = 0.1
        p[0, 2, 0] = 0.9
        p[3, 2, 3] = 1
        p[3, 0, 0] = p[3, 1, 0] = p[3, 3, 0] = p[3, 4, 0] = p[3, 5, 0] = 1
        # state 4
        p[0, 3, 4] = 0.05
        p[0, 3, 0] = 0.95
        p[4, 3, 4] = 1
        p[4, 0, 0] = p[4, 1, 0] = p[4, 2, 0] = p[4, 4, 0] = p[4, 5, 0] = 1
        # state 5
        p[0, 4, 5] = 0.03
        p[0, 4, 0] = 0.97
        p[5, 4, 5] = 1
        p[5, 0, 0] = p[5, 1, 0] = p[5, 2, 0] = p[5, 3, 0] = p[5, 5, 0] = 1
        # state 6
        p[0, 5, 6] = 0.01
        p[0, 5, 0] = 0.99
        p[6, 5, 6] = 1
        p[6, 0, 0] = p[6, 1, 0] = p[6, 2, 0] = p[6, 3, 0] = p[6, 4, 0] = 1
        return p

    def compute_rewards_arms(self, nS, nA, rew):
        r = np.zeros((nS, nA, nS))
        r[1, 0, 1] = r[1, 1, 1] = r[1, 2, 1] = r[1, 3, 1] = r[1, 5, 1] = rew[0]
        for i in range(2, nS):
            r[i, i - 1, i] = rew[i - 1]
        return r

    def compute_mu_arms(self, nS):
        mu = np.zeros(nS)
        mu[0] = 1
        return mu

    def reset(self, state=None):
        if state is None:
            if self.mu is not None:
                self._state = np.array(
                    [np.random.choice(self.mu.size, p=self.mu)])
            else:
                self._state = np.array([np.random.choice(self.p.shape[0])])
        else:
            self._state = state

        return self._state[0]

    def step(self, action):
        p = self.p[self._state[0], action, :]
        next_state = np.array([np.random.choice(p.size, p=p)])
        absorbing = not np.any(self.p[next_state[0]])
        reward = self.r[self._state[0], action, next_state[0]]

        self._state = next_state

        return self._state[0], reward, absorbing, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        pass

    def close(self):
        pass
