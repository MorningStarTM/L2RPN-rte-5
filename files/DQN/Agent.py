import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from Network import QNetwork
from ReplyBuffer import ReplyBuffer
import matplotlib.pyplot as plt
import os

class Agent():
    def __init__(self, state_size, action_size, HP, device, seed):
        """
        Building Agent

        Args:
            state_size (int): Dimension of state/observation
            actions_size (int): Dimension of action
        
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.HP = HP
        self.device = device
        self.losses_value = []

        #Network
        self.qnet_local = QNetwork(self.state_size, self.action_size, seed).to(self.device)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, seed).to(self.device)
        self.optimizer = torch.optim.Adam(self.qnet_local.parameters(), lr=HP['LR'])

        #Replay Buffer
        self.memory = ReplyBuffer(self.action_size, self.HP['BUFFER_SIZE'], self.HP['BATCH_SIZE'], seed)
        self.t_step = 0

    
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.HP['UPDATE_EVERY']
        if self.t_step == 0:
            if len(self.memory) > self.HP['BATCH_SIZE']:
                experiences = self.memory.sample()
                self.learn(experiences, self.HP['GAMMA'])

    def act(self, state, eps=0.):
        """
        Function for take action
        
        Args:
            state (array): current state
            eps (float): epsilon

        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnet_local.eval()
        with torch.no_grad():
            action_values = self.qnet_local(state)
        self.qnet_local.train()


        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    
    def learn(self, experiences, gamma):
        """
        Update value parameters

        Args:
            experience (Tuple): (s, a, r, s', done)
            gamma (float): discount factor
        
        """
        states, actions, rewards, next_states, dones = experiences
        # extract next maximum estimated value from target network
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        #target value 
        q_targets = rewards + gamma * q_targets_next * (1 - dones)

        #Expected value from local network
        q_expected = self.qnet_local(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_targets)
        self.losses_value.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update the target network
        self.soft_update(self.qnet_local, self.qnetwork_target, self.HP['TAU'])

    
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
            

    def plot_loss(self):
        return self.losses_value