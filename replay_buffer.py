from sum_tree import SumTree
import numpy as np
from collections import deque
import random



class ReplayBuffer:
    def __init__(self):
        pass

class ExperienceMemory(ReplayBuffer):
    def __init__(self, buffer_size):
        self.capacity = buffer_size
        self.memory = deque(maxlen=buffer_size)

    def add(self, experience):
        # if len(self.memory) < self.capacity:
        #    self.memory.append(None)

        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, alpha):
        self.capacity = buffer_size
        self.tree = SumTree(buffer_size)
        self.alpha = alpha
        self.max_priority = 1
        #self.beta_initial = ??
        #self.beta_steps = ??

    def add(self, experience):
        self.tree.add(self.max_priority, experience)

    def update(self, index, experience, td_error):
        priority = (abs(td_error) + 0.0001) ** self.alpha
        self.tree.update(index, priority)
        if self.max_priority < priority:
            self.max_priority = priority

    def sample(self, batch_size):
        indexes = []
        batchs = []
        total = self.tree.total()
        section = total / batch_size
        for i in range(batch_size):
            r = section*i + np.random.random()*section
            (idx, priority, experience) = self.tree.get(r)
            indexes.append(idx)  # 後のpriority更新に使う
            batchs.append(experience)
        return (indexes, batchs)
