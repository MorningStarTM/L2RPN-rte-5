import torch 
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from Network import DeepQNetwork
import matplotlib.pyplot as plt
import os
from replay_buffer import ReplayBuffer
from Network import DQN


class DQNAgent:
    '''
    '''
    def __init__(self, state_size=8, action_size=4, hidden_size=64, 
                 learning_rate=1e-3, gamma=0.99, buffer_size=10000, batch_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.gamma = gamma

        self.batch_size = batch_size

        self.action_size = action_size

        self.q_network = DQN(state_size, action_size, hidden_size).to(self.device)
        self.target_network = DQN(state_size, action_size, hidden_size).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        self.memory = ReplayBuffer(buffer_size)

    def step(self, state, action, reward, next_state, done):
        '''
        
        '''
        self.memory.push(state, action, reward, next_state, done)
        
        if len(self.memory) > self.batch_size:
            self.update_model()

    def act(self, state, eps=0.):
        '''
        
        '''
        if random.random() > eps:  
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)  
            self.q_network.eval()  

            with torch.no_grad():
                action_values = self.q_network(state)

            self.q_network.train() 
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))  
        
    def update_model(self):
        '''
        Update the Q-network based on a batch of experiences from the replay memory.
        '''
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(np.array(dones).astype(np.uint8)).float().to(self.device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = torch.nn.MSELoss()(q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        
        loss.backward()
        
        self.optimizer.step()

    def update_target_network(self):
        '''
        Update the weights of the target network to match those of the Q-network.
        '''
        self.target_network.load_state_dict(self.q_network.state_dict())
