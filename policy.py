import numpy as np

class Policy:
    def __init__(self):
        pass


    def action(self):
        pass


class EpsilonGreedyPolicy(Policy):
    def __init__(self, action_space, first_epsilon=1.0,final_epsilon=1e-4,exploration_frame=10000):
        super().__init__()
        self._action_space = action_space
        self.epsilon = first_epsilon
        self.final_epsilon=final_epsilon
        self.epsilon_decrease=(first_epsilon-final_epsilon)/exploration_frame


    def learning_start(self):
        self.epsilon = 1.0
        
    def action(self, a):
        self.epsilon=max(self.epsilon-self.epsilon_decrease,self.final_epsilon)
        return np.argmax(a) if self.epsilon < np.random.random() else np.random.randint(self._action_space)
    
