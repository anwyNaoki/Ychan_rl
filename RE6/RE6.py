

import sys

import gym
import numpy as np
import gym.spaces
import random
from copy import deepcopy
import pyautogui

class MazeEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}



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
        self.reset()

    def get_obs(self):
        img=ImageGrab.grab(bbox=(8, 50, 245, 210))
        imagearray=np.asarray(img)
        return imagearray


    def reset(self,d=0):
        return np.array(obs).transpose(1, 2, 0)

    def step(self, action):
        return np.array(obs).transpose(1, 2, 0), reward, self.done, {}

    def render(self, mode='human', close=False):
        # human の場合はコンソールに出力。ansiの場合は StringIO を返す
        
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass
