import numpy as np

class ReplayBuffer():
    def __init__(self, max_memory_size, input_shape, actions_num):
        self.memory_size = max_memory_size
        self.available_memory_counter = 0
        self.states_memory = np.zeros((self.memory_size, *input_shape))
        # states that result after taken actions
        self.new_states_memory = np.zeros((self.memory_size, *input_shape))
        self.actions_memory = np.zeros((self.memory_size, actions_num))
        # rewards are scalars so they don't need the other dim
        self.rewards_memory = np.zeros(self.memory_size)
        # done flags from env
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.bool)

    def store_transition(self, state, action, reward, new_state, is_done):
        # replay memory if out of it
        memory_index = self.available_memory_counter % self.memory_size
        self.states_memory[memory_index] = state
        self.new_states_memory[memory_index] = new_state
        self.actions_memory[memory_index] = action
        self.rewards_memory[memory_index] = reward
        self.terminal_memory[memory_index] = is_done

        self.available_memory_counter += 1

    def sample_buffer(self, batch_size):
        max_memory = min(self.available_memory_counter, self.memory_size)

        # randomize memories
        random_memories_batch = np.random.choice(max_memory, batch_size)

        states = self.states_memory[random_memories_batch]
        new_states = self.new_states_memory[random_memories_batch]
        actions = self.actions_memory[random_memories_batch]
        rewards = self.rewards_memory[random_memories_batch]
        terminals = self.terminal_memory[random_memories_batch]

        return states, actions, rewards, new_states, terminals


