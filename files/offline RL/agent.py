import torch
import numpy as np
import pickle
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from network import QNetwork
import random
from torch.nn.utils import clip_grad_norm_

"""
SIMPLE Q LEARNING
CONSERVATIVE Q LEARNING

"""

class IQAgent:
    def __init__(self, env, HP:dict):
        self.env = env
        self.obs_state = self.env.observation_space.n
        self.action_space = 132
        self.hp = HP
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.losses_value = []
        self.policy = QNetwork(self.obs_state, self.action_space).to(self.device)
        self.target = QNetwork(self.obs_state, self.action_space).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.hp['lr'])
        self.loss = nn.SmoothL1Loss()
        self.file = open("data\\experience.pkl",'rb')
        self.data = pickle.load(self.file)

    def act(self, state, eps=0.):
        """
        choose action
        Args:
            state (torch.tensor)
        """
        with torch.no_grad():
            self.policy.eval()
            state = torch.tensor(state).float().unsqueeze(0).to(self.device)
            action = torch.argmax(self.policy(state))
            return action.cpu().detach().numpy()

    def step(self):
        pass

    def get_data(self):
        
        experiences = random.sample(self.data, k=self.hp['batch_size'])

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)
    

    def learn(self):
        self.policy.eval()

        states, actions, rewards, next_states, dones = self.get_data()

        # Get Q(s,a) for actions taken
        q_expected = self.policy(states).gather(1, actions)

        # Get V(s') for the new states w/ mask for final state
        
        next_state_values = torch.max(self.target(next_states).detach().max(1)[0].unsqueeze(1))

        q_targets = rewards + self.hp['gamma'] * next_state_values * (1 - dones)

        loss = F.mse_loss(q_expected, q_targets)
        self.losses_value.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.policy, self.target, self.hp['TAU'])

    
    def soft_update(self, local_model, target_model, tau):
        """
        soft update model parameters
        
        Args:
            local model (pytorch model)
            target model (pytorch model)
            tau (float)
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0-tau) * target_param.data)
            

    def save_loss(self, savedir:str):
        np.save(f"loss\\{savedir}.npy", np.array(self.losses_value))


class CQL:
    def __init__(self, HP:dict):
        self.state_dim = HP['STATE_DIM']
        self.action_dim = HP['ACTION_DIM']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tau = HP['TAU']
        self.gamma = HP['GAMMA']


        self.network = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network = QNetwork(self.state_dim, self.action_dim).to(self.device)

        self.optimizer = torch.optim.Adam(params=self.network.parameters(), lr=HP['LR'])

    
    def get_action(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.network.eval()
            with torch.no_grad():
                action_value = self.network(state)
            self.network.train()
            action = np.argmax(action_value.cpu().data.numpy(), axis=1)

        else:
            action = random.choice(np.range(self.action_dim), k=1)
        return action
    
    def cql_loss(self, q_values, current_action):
        """
        Compute the CQL loss for batch of q-values and actions                                                                                          
        """
        logsumexp = torch.logsumexp(q_values, dim=1, keepdim=True)
        q_a = q_values.gather(1, current_action)
        return (logsumexp - q_a).mean()
    
    def learn(self, experience):
        state, action, reward, next_state, done = experience
        with torch.no_grad():
            q_target_next = self.target_network(next_state).detach().max(1)[0].unsequeeze(1)
            q_targets = reward + (self.gamma * q_target_next * (1-done))

        Q_s_a = self.network(state)
        Q_expected = Q_s_a.gather(1, action)

        cql1_loss = self.cql_loss(Q_s_a, action)
        bellman_error = F.mse_loss(Q_expected, q_targets)

        q1_loss = cql1_loss + 0.5 * bellman_error

        self.optimizer.zero_grad()
        q1_loss.backward()
        clip_grad_norm_(self.network.parameters(), 1.)
        self.optimizer.step()

        self.soft_update(self.network, self.target_network)
        return q1_loss.detach().item(), cql1_loss.detach().item(), bellman_error.detach().item()
    
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
