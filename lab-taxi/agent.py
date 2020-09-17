import numpy as np
from collections import defaultdict
import random

class Agent:

    def __init__(self, nA=6, alpha=1, gamma=.9, eps_start=.003, eps_decay=.9999, eps_min=0.03):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.number_of_actions = nA
        self.Q = defaultdict(lambda: np.zeros(self.number_of_actions))
        self.epsilon = eps_start
        self.alpha = alpha
        self.gamma = gamma
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if np.random.random() <= self.epsilon:
            action = random.randrange(self.number_of_actions)
        else:
            action = np.argmax(self.Q[state])
#         self.epsilon = max(self.epsilon*self.eps_decay, self.eps_min)
        return action


    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if not done:
            action_probabilities = self.compute_action_probabilities(self.Q[next_state])
            Qsa_next = np.sum(np.dot(action_probabilities, self.Q[next_state]))
        if done:
            Qsa_next = 0
        
        self.Q[state][action] += self.alpha*(reward + self.gamma*Qsa_next - self.Q[state][action])
        
        
    def compute_action_probabilities(self, q_values):
        probabilities = np.ones(self.number_of_actions)*self.epsilon/self.number_of_actions
        probabilities[np.argmax(q_values)] += 1 - self.epsilon
        return probabilities
    