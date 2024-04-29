import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from Network import QNetwork
from ReplyBuffer import ReplyBuffer
import matplotlib.pyplot as plt
import os
from ReplyBuffer import ReplayBuffer

class DQNAgent:
    def __init__(self, env, HP:dict):
        self.memory = ReplayBuffer()
        self.HP = HP
        self.exploration_rate = HP['EXPLORATION_MAX']
        self.network = QNetwork()
        self.env = env
        self.losses = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def choose_action(self, observation):
        if random.random() < self.exploration_rate:
            return self.env.action_space.sample()
        
        state = torch.tensor(observation).float().detach()
        state = state.to(self.device)
        state = state.unsqueeze(0)
        q_values = self.network(state)
        return torch.argmax(q_values).item()
    
    def learn(self):
        if self.memory.mem_count < self.HP['BATCH_SIZE']:
            return
        
        states, actions, rewards, states_, dones = self.memory.sample()
        states = torch.tensor(states , dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        states_ = torch.tensor(states_, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        batch_indices = np.arange(self.HP['BATCH_SIZE'], dtype=np.int64)

        q_values = self.network(states)
        next_q_values = self.network(states_)
        
        predicted_value_of_now = q_values[batch_indices, actions]
        predicted_value_of_future = torch.max(next_q_values, dim=1)[0]
        
        q_target = rewards + self.HP['GAMMA'] * predicted_value_of_future * dones

        loss = self.network.loss(q_target, predicted_value_of_now)
        self.losses.append(loss)
        self.network.optimizer.zero_grad()
        loss.backward()
        self.network.optimizer.step()

        self.exploration_rate *= self.HP['EXPLORATION_DECAY']
        self.exploration_rate = max(self.HP['EXPLORATION_MIN'], self.exploration_rate)

    def returning_epsilon(self):
        return self.exploration_rate
    
    def save_model(self, path):
        torch.save(self.network.state_dict(), path)

    def load_model(self, path):
        self.network.load_state_dict(path)