from agent import Agent
from monitor import interact
# !pip install gym==0.9.6 for 'Taxi-v2'
import gym

import numpy as np

env = gym.make('Taxi-v2')
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)