import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class GenericNetwork(nn.Module):
    def __init__(self, HP:dict):
        """
        Args:
            HP: dict - (alpha, input_dim, fc1, fc2, n_action)
        """
        super(GenericNetwork, self).__init__()
        self.input_dims = HP['input_dim']
        self.fc1_dim = HP['fc1']
        self.fc2_dim = HP['fc2']
        self.n_actions = HP['n_action']
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.fc3 = nn.Linear(self.fc2_dim, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=H['lr'])

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        state = T.tensor(state).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
