import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np


# learning rate = beta
class CriticNetwork(nn.Module):
    def __init__(self, learning_rate, input_dims, actions_num, layer1_size=256, layer2_size=256,
                 file_name='critic_network', save_path='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.n_actions = actions_num
        self.name = file_name
        self.save_path = save_path
        self.save_path = os.path.join(self.save_path, file_name + '_soft_actor_critic')

        # input is a state and action pair because critic evaluates state+action pair
        self.layer1 = nn.Linear(self.input_dims[0] + actions_num, self.layer1_size)
        self.layer2 = nn.Linear(self.layer1_size, self.layer2_size)
        # q value is a scalar output
        self.q_value = nn.Linear(self.layer2_size, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # get the graphics card if possible
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    # feed forward state action pair through critic network
    def forward(self, state, action):
        q = self.layer1(T.cat([state, action], dim=1))
        q = F.relu(q)
        q = self.layer2(q)
        q = F.relu(q)
        q = self.q_value(q)
        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.save_path)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.save_path))


class ValueNetwork(nn.Module):
    def __init(self, learning_rate, input_dims, layer1_size=256, layer2_size=256,
               file_name='value_network', save_path='tmp/sac'):
        super(ValueNetwork, self).__init()
        self.input_dims = input_dims
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.file_name = file_name
        self.save_path = save_path
        self.save_path = os.path.join(self.save_path, file_name + '_soft_actor_critic')

        self.layer1 = nn.Linear(*self.input_dims, self.layer1_size)
        self.layer2 = nn.Linear(self.layer1_size, self.layer2_size)
        # state value is also a scalar output
        self.state_value = nn.Linear(self.layer2_size, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        v = self.layer1(state)
        v = F.relu(v)
        v = self.layer2(v)
        v = F.relu(v)
        v = self.state_value(v)
        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.save_path)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.save_path))


class ActorNetwork(nn.Module):
    def __init__(self, learning_rate, input_dims, max_action, layer1_dims=256, layer2_dims=256,
                 actions_num=2, file_name='actor_network', save_path='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.max_action = max_action
        self.layer1_dims = layer1_dims
        self.layer2_dims = layer2_dims
        self.n_actions = actions_num
        self.file_name = file_name
        self.save_path = save_path
        self.save_path = os.path.join(self.save_path, file_name + '_soft_actor_critic')
        # lower_sigma_bound used as noise
        self.lower_sigma_bound = 1e-6

        self.layer1 = nn.Linear(*self.input_dims, self.layer1_dims)
        self.layer1 = nn.Linear(*self.input_dims, self.layer1_dims)
        # two outputs
        # mu is mean of the policy distribution
        self.layer3 = nn.Linear(self.layer2_dims, self.n_actions)
        # standard deviation
        self.sigma = nn.Linear(self.layer2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        probability = self.layer1(state)
        probability = F.relu(probability)
        probability = self.layer2(probability)
        probability = F.relu(probability)

        mu = self.layer3(probability)
        sigma = self.sigma(mu)

        # don't want broad distribution
        # sigma (standard deviation) defines how broad the distribution is
        # so we constrain it, but evade zero hence the lower_sigma_bound
        # clamp is faster than sigma
        sigma = T.clamp(sigma, min=self.lower_sigma_bound, max=1)

        return mu, sigma

    # Gaussian policy distribution - continuous space
    def sample_normal(self, state, reparametrize=True):
        mu, sigma = self.forward(state)
        # normal distribution of probabilities to take each action
        policy_distribution = Normal(mu, sigma)

        if reparametrize:
            actions = policy_distribution.rsample()
        else:
            actions = policy_distribution.sample()
        action = T.tanh(actions)*T.tensor(self.max_action).to(self.device)
        # for calculating the loss function
        log_probs = policy_distribution.log_prob(actions)
        # self.reparam_noise was added for situations like log(1-1)
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.save_path)

    def load_checkpoint(self):
            self.load_state_dict(T.load(self.save_path))
