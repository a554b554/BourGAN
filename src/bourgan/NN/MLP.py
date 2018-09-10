import torch.nn as nn
import torch.nn.functional as F
import torch

class DeepMLP_G(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepMLP_G, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.leaky_relu(self.map1(x), 0.1)
        x = F.leaky_relu(self.map2(x), 0.1)
        return self.map3(x)

class DeepMLP_D(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepMLP_D, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.leaky_relu(self.map1(x), 0.1)
        x = F.leaky_relu(self.map2(x), 0.1)
        return torch.sigmoid(self.map3(x))