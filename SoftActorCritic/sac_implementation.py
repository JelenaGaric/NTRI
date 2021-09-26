import os
import torch as T
import torch.nn.functional as F
import numpy as np
from replay_buffer import ReplayBuffer
from actor_critic_value_networks import ActorNetwork, CriticNetwork, ValueNetwork


# reward scaling depends on num of actions

class Agent():
    def __init__(self, actions_num, lr_alpha=0.0003, lr_beta=0.0003, input_dims=[8], env=None, gamma=0.99, tau=0.005,
                 max_memory_size=1000000, layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):
        self.gamma = gamma
        # for parameters of target value network
        self.tau = tau
        self.batch_size = batch_size
        self.actions_num = actions_num

        self.memory = ReplayBuffer(max_memory_size, input_dims, actions_num)

        self.actor_network = ActorNetwork(lr_alpha, input_dims, actions_num=actions_num, max_action=env.action_space.high)

        # two critics
        self.critic_network_1 = CriticNetwork(lr_beta, input_dims, actions_num, file_name='critic_network_1')
        self.critic_network_2 = CriticNetwork(lr_beta, input_dims, actions_num, file_name='critic_network_2')

        self.value_network = ValueNetwork(lr_beta, input_dims, file_name="value_network")
        self.target_value_network = ValueNetwork(lr_beta, input_dims, file_name="target_value_network")

        self.scale = reward_scale
        # setting value network parameters to target values
        self.update_network(tau=1)

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.actor_network.device)
        # feed the state forward to actor network to get the actions
        actions, _ = self.actor_network.sample_normal(state, reparameterize=False)

        # because of torch, send it to cpu and detach form the graph and turn it to numpy and take the zeroth element
        return actions.cpu().detach().numpy()[0]

    # interface function between agent and its memory
    def store_transition_to_memory(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network(self, tau=None):
        if tau is None:
            # at the beginning we want target and value networks to be the same but at any other it should
            # be slightly different
            tau = self.tau
        # create a copy of params, modify them and then upload them
        target_value_params = self.target_value_network.named_parameters()
        value_params = self.value_network.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + (1-tau)*target_value_state_dict[name].clone()
        # target copies the model network
        self.target_value_network.load_state_dict(value_state_dict)

    def learn(self):
        #if batch is not filled yet
        if self.memory.available_memory_counter < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        # transform sampled numpy arrays to tensors for torch
        new_state = T.tensor(new_state, dtype=T.float).to(self.actor_network.device)
        state = T.tensor(state, dtype=T.float).to(self.actor_network.device)
        reward = T.tensor(reward, dtype=T.float).to(self.actor_network.device)
        done = T.tensor(done).to(self.actor_network.device)
        action = T.tensor(action, dtype=T.float).to(self.actor_network.device)

        # calculate the value and target value of states and new states
        # returns view of tensor minus one dim because it's a scalar
        value = self.value_network(state).view(-1)
        new_value = self.target_value_network(new_state).view(-1)
        new_value[done] = 0.0

        # action probability values according to the new policy for value and actor networks
        actions, log_probs = self.actor_network.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        # Clipped Double Q-learning (reduce it to the minimum, avoiding overestimation):
        new_policy_q_1 = self.critic_network_1.forward(state, actions)
        new_policy_q_2 = self.critic_network_2.forward(state, actions)
        # taking the min of q values because it improves the learning
        critic_value = T.min(new_policy_q_1, new_policy_q_2)
        critic_value = critic_value.view(-1)

        # value network loss
        self.value_network.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value_network.optimizer.step()

        # actor network loss
        actions, log_probs = self.actor_network.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        new_policy_q_1 = self.critic_network_1.forward(state, actions)
        new_policy_q_2 = self.critic_network_2.forward(state, actions)
        critic_value = T.min(new_policy_q_1, new_policy_q_2)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor_network.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_network.optimizer.step()

        #critic network loss
        self.critic_network_1.optimizer.zero_grad()
        self.critic_network_2.optimizer.zero_grad()
        # new value of the state resulting from the actions the agent took
        q_target = self.scale * reward + self.gamma * new_value
        q1_old_policy = self.critic_network_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_network_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_target)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_target)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_network_1.optimizer.step()
        self.critic_network_2.optimizer.step()

        self.update_network()

    def save_models(self):
        print("Saving models...")
        self.actor_network.save_checkpoint()
        self.value_network.save_checkpoint()
        self.critic_network_1.save_checkpoint()
        self.critic_network_2.save_checkpoint()
        self.target_value_network.save_checkpoint()
        self.target_value_network.save_checkpoint()

    def load_models(self):
        print("Loading models...")
        self.actor_network.load_checkpoint()
        self.critic_network_1.load_checkpoint()
        self.critic_network_2.load_checkpoint()
        self.value_network.load_checkpoint()
        self.target_value_network.load_checkpoint()



