import torch as T
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch

class DQN(torch.nn.Module):
    '''
    '''
    def __init__(self, state_size=8, action_size=4, hidden_size=64):
        '''
        '''
        super(DQN, self).__init__()
        self.layer1 = torch.nn.Linear(state_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, action_size)

    def forward(self, state):
        '''
        '''
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)
    
