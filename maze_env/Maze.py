

import sys

import gym
import numpy as np
import gym.spaces
import random
from copy import deepcopy


class MazeEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}
    MAX_STEPS = 100


    def __init__(self):
        super().__init__()
        # action_space, observation_space, reward_range を設定する
        #角度:足場の上を0度とする

        self.action_space = gym.spaces.Discrete(4)  
        self.observation_space = gym.spaces.Box(
            low=0,
            high=2,
            shape=(5, 5)
        )
        self.clear = 0
        self.reward_range = [-1., 1.]
        self.reset()

    def reset(self,d=0):
        self.done=False
        self.steps=0
        self.observation = np.zeros(25).reshape(5, 5)
        self.observation[0, 0] = 1
        self.player_pos = [4, 4]
        
        observation = deepcopy(self.observation)
        observation[self.player_pos[1], self.player_pos[0]] = 1
        obs = [np.ones(25).reshape(5,5) for j in range(10)]
        obs.append(observation)
        return np.array(obs).transpose(1, 2, 0)

    def step(self, action):
        if(self.steps>100):
            self.done=True
        reward = 0

        if (action == 0):
            self.player_pos = [max(self.player_pos[0] - 1, 0), self.player_pos[1]]
        elif (action == 1):
            self.player_pos = [min(self.player_pos[0] + 1, 4), self.player_pos[1]]
        elif (action == 2):
            self.player_pos = [self.player_pos[0], max(self.player_pos[1] - 1, 0)]
        elif (action == 3):
            self.player_pos = [self.player_pos[0], min(self.player_pos[1] + 1, 4)]
        
        if (self.player_pos == [0, 0]):
            reward = 1
            self.done = True    

        observation = deepcopy(self.observation)
        observation[self.player_pos[1], self.player_pos[0]] = 1
        obs = [np.ones(25).reshape(5, 5) for j in range(10)]
        obs.append(observation)
        self.steps += 1
        return np.array(obs).transpose(1, 2, 0), reward, self.done, {}

    def render(self, mode='human', close=False):
        # human の場合はコンソールに出力。ansiの場合は StringIO を返す
        observation = deepcopy(self.observation)
        observation[self.player_pos[1], self.player_pos[0]] = 1
        print(observation)
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass
