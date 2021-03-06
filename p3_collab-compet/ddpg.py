# individual network settings for each actor + critic pair
# see networkforall for details

from networkforall import Actor, Critic
from utilities import hard_update, gumbel_softmax, onehot_from_logits
from torch.optim import Adam
import torch
import numpy as np

# add OU noise for exploration
from OUNoise import OUNoise

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

class DDPGAgent:
    def __init__(self,
                 in_actor,
                 hidden_in_actor,
                 hidden_out_actor,
                 out_actor,
                 in_critic_state,
                 hidden_in_critic,
                 hidden_out_critic,
                 critic_input_action_size,
                 lr_actor=1.0e-4,
                 lr_critic=3.0e-4):
        super(DDPGAgent, self).__init__()

        self.actor = Actor(in_actor, hidden_in_actor, hidden_out_actor, out_actor).to(device)
        self.critic = Critic(in_critic_state, hidden_in_critic, hidden_out_critic, critic_input_action_size).to(device)
        self.target_actor = Actor(in_actor, hidden_in_actor, hidden_out_actor, out_actor).to(device)
        self.target_critic = Critic(in_critic_state, hidden_in_critic, hidden_out_critic, critic_input_action_size).to(device)
        self.action_size = out_actor
        self.noise = OUNoise(out_actor, scale=1.0)
        # initialize targets same as original networks one time in the initial step
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        self.noise_reduction = 1.0
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor, weight_decay=0)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=0)

    def act(self, obs, episode, number_of_episode_before_training, noise_reduction_factor, noise=0.0, grad_zero=False):
        obs = obs.to(device)
        if grad_zero:
            self.actor.eval()
            with torch.no_grad():
                # action = self.actor(obs) + self.noise_reduction*self.noise.noise()*noise
                action = self.actor(obs) + self.noise_reduction*self.noise.add_noise2()*noise
        else:
            # action = self.actor(obs) + self.noise_reduction*self.noise.noise()*noise
            action = self.actor(obs) + self.noise_reduction*self.noise.add_noise2()*noise
        self.actor.train()
        if episode >= number_of_episode_before_training:
            memory = episode - number_of_episode_before_training
            self.noise_reduction = max(noise_reduction_factor**memory, 0.1)
        return action

    def target_act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.target_actor(obs) + noise*self.noise.noise()
        return action

    def reset(self):
        self.noise.reset()

