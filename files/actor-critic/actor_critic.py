import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.nn as gnn

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
        self.optimizer = optim.Adam(self.parameters(), lr=HP['lr'])

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        state = T.tensor(state).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class GraphGenericNetwork(nn.Module):
    def __init__(self, HP: dict, n_actions):
        """
        Args:
            HP: dict - (alpha, input_dim, fc1, fc2, n_action)
        """
        super(GraphGenericNetwork, self).__init__()
        self.input_dims = HP['input_dim']
        self.fc1_dim = HP['fc1']
        self.fc2_dim = HP['fc2']
        self.node_feature = HP['node_feature']
        self.n_actions = n_actions
        self.gc1 = gnn.GCNConv(self.node_feature, 21)
        self.gc2 = gnn.GCNConv(21, 21)
        self.fc1 = nn.Linear(441, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, self.n_actions)
        self.optimizer = T.optim.Adam(self.parameters(), lr=HP['alpha'])

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, adj):
        x = self.gc1(state, adj)
        x = self.gc2(x, adj)
        x = x.view(x.size(0), -1)[0]
        #x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
    


class GraphActorCriticNetwork(nn.Module):
    def __init__(self, HP:dict):
        """
        Args:
            HP: dict - (alpha, input_dim, fc1, fc2, n_action)
        
        """
        super(GraphActorCriticNetwork, self).__init__()
        self.input_dims = HP['input_dim']
        self.node_feature = HP['node_feature']
        self.fc1_dim = HP['fc1']
        self.fc2_dim = HP['fc2']
        self.n_actions = HP['n_action']
        self.gc1 = gnn.GCNConv(self.node_feature, 21)
        self.gc2 = gnn.GCNConv(21, 21)
        self.fc1 = nn.Linear(441, self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.fc3 = nn.Linear(self.fc2_dim, 256)
        self.fc4 = nn.Linear(256, 64)
        self.pi = nn.Linear(self.fc2_dim, self.n_actions)
        self.v = nn.Linear(64, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=HP['lr'])

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)


    def forward(self, state, adj):
        #state = T.tensor(state).to(self.device)
        x = self.gc1(state, adj)
        x = self.gc2(x, adj)
        x = x.view(x.size(0), -1)[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        vx = F.relu(self.fc3(x))
        vx = F.relu(self.fc4(vx))
        pi = self.pi(x)
        v = self.v(vx)
        return (pi, v)
    


class GACAgent(object):
    """
    This agent use spearate network for actor and critic. Graph based Actor critiv network
    """
    def __init__(self, HP:dict):
        """
        Args:
            HP: dict - (alpha, beta, input_dim, gamma, laer1, layer2, n_action)
        
        """
        self.HP = HP
        self.gamma = HP['gamma']
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.actor = GraphGenericNetwork(HP, 132)#.to(self.device)
        self.critic = GraphGenericNetwork(HP, 1)#.to(self.device)
        self.losses = []

        self.log_probs = None

    def choose_action(self, observation, adj):
        probabilities = F.softmax(self.actor.forward(observation, adj), dim=-1)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        self.log_probs = action_probs.log_prob(action)
        return action.item()
    

    def learn(self, state, adj, reward, new_state, new_adj, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        critic_value_ = self.critic.forward(new_state, new_adj)
        critic_value = self.critic.forward(state, adj)
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)

        delta = reward + self.gamma * critic_value_ * (1 - int(done)) - critic_value

        actor_loss = -self.log_probs * delta
        critic_loss = delta**2
        self.losses.append(critic_loss)

        (actor_loss + critic_loss).backward()

        self.actor.optimizer.step()
        self.critic.optimizer.step()

    def save(self, filename):
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'losses': self.losses
        }
        T.save(checkpoint, filename)




class GraphBasedAgent(object):
    """
        this agent use single Graph Convolutional network for both actor and critic
    """
    def __init__(self, HP:dict):
        """
        Args:
            HP: dict - (alpha, input_dim, gamma, laer1, layer2, n_action)
        
        """
        self.gamma = HP['gamma']
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.actor_critic = GraphActorCriticNetwork(HP).to(self.device)
        self.log_probs = None
        

    def choose_action(self, state, adj):
        prob, _ = self.actor_critic.forward(state, adj)
        prob = F.softmax(prob, dim=-1)
        action_prob = T.distributions.Categorical(prob)
        action = action_prob.sample()
        log_prob = action_prob.log_prob(action)
        self.log_probs = log_prob
        return action.item()
    
    def learn(self, state, adj, reward, next_state, next_adj, done):
        self.actor_critic.optimizer.zero_grad()

        _, critic_value_ = self.actor_critic.forward(next_state, next_adj)
        _, critic_value = self.actor_critic.forward(state, adj)
        reward = T.tensor(reward, dtype=T.float).to(self.actor_critic.device)

        delta = reward + self.gamma * critic_value_ * (1 - int(done)) - critic_value

        actor_loss = -self.log_probs * delta
        critc_loss = delta ** 2

        (actor_loss + critc_loss).backward()

        self.actor_critic.optimizer.step()

    def save_model(self, pathdir):
        T.save(self.actor_critic.state_dict(), pathdir)