import torch as T
import numpy as np
from DeepQLearning.deep_q_network import DeepQNetwork


# main functionality
# gamma = discount factor
# epsilon = exploration factor
# eps_dec = decrement of epsilon through time but it will never be 0


class Agent():
    def __init__(self, gamma, epsilon, learning_rate, input_dims,
                 batch_size, actions_num, max_memory_size=100000,
                 eps_end=0.01, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.max_memory_size = max_memory_size
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.learning_rate = learning_rate
        self.action_space = [i for i in range(actions_num)]
        self.memory_counter = 0

        self.Q_eval = DeepQNetwork(self.learning_rate, actions_num=actions_num,
                                   input_dims=self.input_dims, layer1_size=256,
                                   layer2_size=256)
        self.state_memory = np.zeros((self.max_memory_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.max_memory_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.max_memory_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.max_memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.max_memory_size, dtype=bool)

    def store_transition(self, state, action, reward, new_state, done):
        memory_index = self.memory_counter % self.max_memory_size
        self.state_memory[memory_index] = state
        self.new_state_memory[memory_index] = new_state
        self.action_memory[memory_index] = action
        self.reward_memory[memory_index] = reward
        self.terminal_memory[memory_index] = done

        self.memory_counter += 1

    def choose_action(self, observation):
        # only in 1-epsilon cases will the agent explore
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        # if batch is not filled yet
        if self.memory_counter < self.batch_size:
            return

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch, new_states_batch, reward_batch, terminal_batch, action_batch = self.sample_buffer()

        self.Q_eval.optimizer.zero_grad()

        # value of the action the agent took
        q_value = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        # all q values of the next possible actions
        q_next = self.Q_eval.forward(new_states_batch)
        q_next[terminal_batch] = 0.0

        # the one we want to achieve
        # zeroth element is the value
        # we want to go in the direction of max q in the next state
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        # so the loss is between the actual q_value we took and the q_target in which direction we want to move
        loss = self.Q_eval.loss(q_target, q_value).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_end else self.eps_end

    def sample_buffer(self):
        max_memory = min(self.memory_counter, self.max_memory_size)
        batch = np.random.choice(max_memory, self.batch_size, replace=False)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_states_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        # these can be just a numpy array
        action_batch = self.action_memory[batch]

        return state_batch, new_states_batch, reward_batch, terminal_batch, action_batch
