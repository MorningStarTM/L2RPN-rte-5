from collections import namedtuple, deque
import torch
import random
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, MEM_SIZE, BATCH_SIZE):
        self.mem_count = 0
        self.MEM_SIZE = MEM_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        
        self.states = np.zeros((self.MEM_SIZE, 211),dtype=np.float32)
        self.actions = np.zeros(self.MEM_SIZE, dtype=np.int64)
        self.rewards = np.zeros(self.MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((self.MEM_SIZE, 211),dtype=np.float32)
        self.dones = np.zeros(self.MEM_SIZE, dtype=np.bool_)
    
    def add(self, state, action, reward, state_, done):
        mem_index = self.mem_count % self.MEM_SIZE
        
        self.states[mem_index]  = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] =  1 - done

        self.mem_count += 1
    
    def sample(self):
        MEM_MAX = min(self.mem_count, self.MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, self.BATCH_SIZE, replace=True)
        
        states  = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones   = self.dones[batch_indices]

        return states, actions, rewards, states_, dones