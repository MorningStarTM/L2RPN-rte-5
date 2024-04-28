import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        """
        building model for Deep Q Network

        Args:
            state_size (int): Dimension of state/observation
            actions_size (int): Dimension of action

        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 364)
        self.fc2 = nn.Linear(364, 364)
        self.fc3 = nn.Linear(364, 182)
        self.fc4 = nn.Linear(182, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        return self.fc4(x)
    
    
