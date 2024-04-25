import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, state_dim)
        self.fc4 = nn.Linear(state_dim, action_dim)

    def forward(self, X:torch.tensor):
        x = self.fc1(X)
        x = self.fc2(torch.relu(x))
        x = self.fc3(torch.relu(x))
        x = self.fc4(x)
        return x