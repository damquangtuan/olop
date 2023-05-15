import numpy as np


from gym.utils import seeding
from gym import Env, spaces


LEFT = 0
RIGHT = 1

class RiverSwimEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, nS=6, nA=2):

        # Defining the number of actions
        self.nA = nA
        self.nS = nS

        self.mu = self.compute_mu_river(nS=nS)
        self.r = self.compute_rewards_river(nS=nS, nA=nA, small=0.0005, large=1)
        self.p = self.compute_probabilities_river(nS=nS, nA=nA)

        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)

    def compute_probabilities_river(self, nS, nA):
        p = np.zeros((nS, nA, nS))
        for i in range(1, nS):
            p[i, 0, i - 1] = 1
            if i != nS - 1:
                p[i, 1, i - 1] = 0.1
                p[i, 1, i] = 0.6
            else:
                p[i, 1, i - 1] = 0.7
                p[i, 1, i] = 0.3
        for i in range(nS - 1):
            p[i, 1, i + 1] = 0.3
        # state 0
        p[0, 0, 0] = 1
        p[0, 1, 0] = 0.7

        return p

    def compute_rewards_river(self, nS, nA, small, large):
        r = np.zeros((nS, nA, nS))
        r[0, 0, 0] = float(small) / float(large)
        r[nS - 1, 1, nS - 1] = float(large) / float(large)
        return r

    def compute_mu_river(self, nS):
        mu = np.zeros(nS)
        mu[1] = 0.5
        mu[2] = 0.5
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
