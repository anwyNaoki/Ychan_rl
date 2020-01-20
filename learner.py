import gym
from collections import deque

from model import DQNModel
from replay_buffer import PrioritizedReplayBuffer,ExperienceMemory
from policy import EpsilonGreedyPolicy
from Experience import Experience




class Learner:
    def __init__(self,env,model):
        self.env = env


        # env = gym.make('Maze-v0')
        self.buffer = PrioritizedReplayBuffer(10000, 0.7)  # ExperienceMemory(2000)
        self.model = model
        self.policy = EpsilonGreedyPolicy(self.env.action_space.n)

        self.model.set_model()


    def run(self):
        results = deque(maxlen=10)
        t_r = 0
        loss = None
        ep = 0
        step = 0

        s=self.env.reset()


        for j in range(30000):
            action = self.policy.action(self.model.predict(s))
            n_s, reward, done, _ = self.env.step(action)
            t_r += reward
            e = Experience(s, action, reward, n_s, done)
            self.buffer.add(e)

            if (j % 1000 == 0):
                self.model.check_attention(s, j)

            if (j == 32):
                self.policy.learning_start()
            if (j > 32 and j % 8 == 0):
                index, batch = self.buffer.sample(32)
                loss = self.model.update(batch)
                [self.buffer.update(index[i], batch[i], m)
                for i, m in enumerate(self.model.return_td(batch, index))]

            if(j > 32 and j % 100 == 0):
                self.model.reset_teacher()

            if (done):
                results.append(t_r)
                t_r = 0
                s = self.env.reset()
                done = False

                ep += 1
                if (ep % 10 == 0 and loss != None):
                    print(j, loss.history['loss'], ep, sum(results) / 10)
            else:
                s = n_s
