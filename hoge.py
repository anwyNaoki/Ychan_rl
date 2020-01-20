import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential, load_model
from sklearn.preprocessing import StandardScaler
from operator import itemgetter
from collections import deque
from itertools import chain
import pandas as pd
import numpy as np
import pickle
import random
import time
import gym
import os
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164
165
166
167
168
169
170
171
172
173
174
175
176
177
178
179
180
181
182
183
184
185
186
187
188
189
190
191
192
193
194
195
196
197
198
199
200
201
202
203
204
205
206
207
208
209
210
211
212
213
214
215
216


plt.style.use('seaborn')
plt.rcParams['font.family'] = 'IPAexGothic'


class DQN(object):

    def __init__(self, env_id, agent_hist_len=1, memory_size=2000,
                 replay_start_size=32, gamma=0.99, eps=1.0, eps_min=1e-4,
                 final_expl_step=1000, mb_size=32, C=100, n_episodes=400,
                 max_steps=500):

        self.env_id = env_id
        self.env = gym.make(env_id)
        self.path = './data/' + env_id
        self.agent_hist_len = 1#agent_hist_len
        self.memory_size = memory_size
        self.replay_start_size = replay_start_size
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.final_expl_step = final_expl_step
        self.eps_decay = (eps - eps_min) / final_expl_step
        self.mb_size = mb_size
        self.C = C
        self.n_episodes = n_episodes
        self.max_steps = max_steps

        self._init_memory()
        self.scaler = StandardScaler()
        self.scaler.fit(np.array([t[0] for t in self.memory]))

    @staticmethod
    def _flatten_deque(d):
        return np.array(list(chain(*d)))

    def _get_optimal_action(self, network, agent_hist):
        agent_hist_normalized = self.scaler.transform(self._flatten_deque(agent_hist).reshape(1, -1))
        return np.argmax(network.predict(agent_hist_normalized)[0])

    def _get_action(self, agent_hist=None):
        if agent_hist is None:
            return self.env.action_space.sample()
        else:
            self.eps = max(self.eps - self.eps_decay, self.eps_min)
            if np.random.random() < self.eps:
                return self.env.action_space.sample()
            else:
                return self._get_optimal_action(self.Q, agent_hist)

    def _remember(self, agent_hist, action, reward, new_state, done):
        self.memory.append([self._flatten_deque(agent_hist), action, reward,
                            new_state if not done else None])

    def _init_memory(self):
        print('Initializing replay memory: ', end='')
        self.memory = deque(maxlen=self.memory_size)
        while True:
            state = self.env.reset()
            agent_hist = deque(maxlen=self.agent_hist_len)
            agent_hist.append(state)
            while True:
                action = self._get_action(agent_hist=None)
                new_state, reward, done, _ = self.env.step(action)
                if len(agent_hist) == self.agent_hist_len:
                    self._remember(agent_hist, action, reward, new_state, done)
                if len(self.memory) == self.replay_start_size:
                    print('done')
                    return
                if done:
                    break
                state = new_state
                agent_hist.append(state)

    def _build_network(self):
        nn = Sequential()
        nn.add(Dense(20, activation='relu',
                     input_dim=(self.agent_hist_len
                                * self.env.observation_space.shape[0])))
        nn.add(Dense(20, activation='relu'))
        nn.add(Dense(10, activation='relu'))
        nn.add(Dense(self.env.action_space.n))
        nn.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return nn

    def _clone_network(self, nn):
        clone = self._build_network()
        clone.set_weights(nn.get_weights())
        return clone

    def _get_samples(self):
        samples = random.sample(self.memory, self.mb_size)
        agent_hists = np.array([s[0] for s in samples])
        Y = self.target_Q.predict(self.scaler.transform(agent_hists))
        actions = [s[1] for s in samples]
        rewards = np.array([s[2] for s in samples])
        future_rewards = np.zeros(self.mb_size)
        new_states_idx = [i for i, s in enumerate(samples) if s[3] is not None]
        new_states = np.array([s[3]
                               for s in itemgetter(*new_states_idx)(samples)])
        new_agent_hists = np.hstack(
            [agent_hists[new_states_idx, self.env.observation_space.shape[0]:],
             new_states])
        future_rewards[new_states_idx] = np.max(
            self.target_Q.predict(self.scaler.transform(new_agent_hists)), axis=1)
        rewards += self.gamma*future_rewards
        for i, r in enumerate(Y):
            Y[i, actions[i]] = rewards[i]
        return agent_hists, Y

    def _replay(self):
        agent_hists, Y = self._get_samples()
        agent_hists_normalized = self.scaler.transform(agent_hists)
        for i in range(self.mb_size):
            self.Q.train_on_batch(agent_hists_normalized[i, :].reshape(1, -1),
                                  Y[i, :].reshape(1, -1))

    def learn(self, render=False, verbose=True):

        self.Q = self._build_network()
        self.target_Q = self._clone_network(self.Q)

        if verbose:
            print('Learning target network:')
        self.scores = []
        for episode in range(self.n_episodes):
            state = self.env.reset()
            agent_hist = deque(maxlen=self.agent_hist_len)
            agent_hist.append(state)
            score = 0
            for step in range(self.max_steps):
                if render:
                    self.env.render()
                if len(agent_hist) < self.agent_hist_len:
                    action = self._get_action(agent_hist=None)
                else:
                    action = self._get_action(agent_hist)
                new_state, reward, done, _ = self.env.step(action)
                if verbose:
                    print('episode: {:4} | step: {:3} | memory: {:6} | \
eps: {:.4f} | action: {} | reward: {: .1f} | best score: {: 6.1f} | \
mean score: {: 6.1f}'.format(
                        episode+1, step +
                        1, len(self.memory), self.eps, action, reward,
                        max(self.scores) if len(self.scores) != 0 else np.nan,
                        np.mean(self.scores) if len(self.scores) != 0 else np.nan),
                        end='\r')
                score += reward
                if len(agent_hist) == self.agent_hist_len:
                    self._remember(agent_hist, action, reward, new_state, done)
                    self._replay()
                if step % self.C == 0:
                    self.target_Q = self._clone_network(self.Q)
                if done:
                    self.scores.append(score)
                    break
                state = new_state
                agent_hist.append(state)

        self.target_Q.save(self.path + '_model.h5')
        with open(self.path + '_scores.pkl', 'wb') as f:
            pickle.dump(self.scores, f)

    def plot_training_scores(self):
        with open(self.path + '_scores.pkl', 'rb') as f:
            scores = pd.Series(pickle.load(f))
        avg_scores = scores.cumsum() / (scores.index + 1)
        plt.figure(figsize=(12, 6))
        n_scores = len(scores)
        plt.plot(range(n_scores), scores, color='gray', linewidth=1)
        plt.plot(range(n_scores), avg_scores, label='平均')
        plt.legend()
        plt.xlabel('学習エピソード')
        plt.ylabel('スコア')
        plt.title(self.env_id)
        plt.margins(0.02)
        plt.tight_layout()
        plt.show()

    def run(self, render=True):

        fname = self.path + '_model.h5'
        if os.path.exists(fname):
            self.target_Q = load_model(fname)
        else:
            print('Q-network not found. Start learning.')
            self.learn()

        state = self.env.reset()
        agent_hist = deque(maxlen=self.agent_hist_len)
        agent_hist.extend([state]*self.agent_hist_len)
        score = 0
        while True:
            if render:
                self.env.render()
            action = self._get_optimal_action(self.target_Q, agent_hist)
            new_state, reward, done, _ = self.env.step(action)
            score += reward
            if done:
                print('{} score: {}'.format(self.env_id, score))
                return
            state = new_state
            agent_hist.append(state)
            time.sleep(0.05)



dqn = DQN('CartPole-v0')
dqn.learn()
