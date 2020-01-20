from learner import Learner
from model import DQNModel
import gym
import maze_env

env=gym.make('Maze-v0')
learner = Learner(env,model=DQNModel())
learner.run()