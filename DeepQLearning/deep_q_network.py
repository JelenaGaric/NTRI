import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DeepQNetwork(nn.Module):
    def __init__(self, learning_rate, input_dims, layer1_size, layer2_size,
                 actions_num):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.actions_num = actions_num

        # * operator unpacks values of a list in this case to nums
        self.layer1 = nn.Linear(*self.input_dims, self.layer1_size)
        self.layer2 = nn.Linear(self.layer1_size, self.layer2_size)
        self.layer3 = nn.Linear(self.layer2_size, self.actions_num)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device )

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        actions = self.layer3(x)

        # no activation because we want agent's raw output for actions
        return actions

