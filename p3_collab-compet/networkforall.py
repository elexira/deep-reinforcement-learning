import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tanh
import numpy as np


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_in_dim, hidden_out_dim, output_dim):
        super(Actor, self).__init__()

        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""

        self.fc1 = nn.Linear(input_dim, hidden_in_dim)
        self.batch_norm_fc1 = nn.BatchNorm1d(hidden_in_dim)
        self.fc2 = nn.Linear(hidden_in_dim, hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim, output_dim)
        self.nonlin = F.relu  # leaky_relu
        self.nonlin_2 = F.elu  # leaky_relu

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, x):
        h1 = self.nonlin_2(self.fc1(x))
        h2 = self.nonlin_2(self.fc2(h1))
        h3 = tanh(self.fc3(h2))
        return h3


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, hidden_in_dim, hidden_out_dim, action_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.fcs1 = nn.Linear(state_size, hidden_in_dim)
        self.batch_norm_fcs1 = nn.BatchNorm1d(hidden_in_dim)
        self.fc2 = nn.Linear(hidden_in_dim+action_size, hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.batch_norm_fcs1(self.fcs1(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
