import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from GCN import GCNLayer
import os
import numpy as np


class ActorCritic(nn.Module):
    """
     This is the network tp produce policy and value
     Args:
        state_dim (int): features dimension
        action_dim (int): number of action

     Return:
        policy
        value
    """
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_pi = nn.Linear(64, action_dim)
        self.fc_v = nn.Linear(64, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        pi = torch.softmax(self.fc_pi(x), dim=0)
        v = self.fc_v(x)
        
        return pi, v
    
class Agent:
    def __init__(self, env, ckpt, alpha=0.0003, gamma=0.9):
        self.gamma = gamma
        self.env = env
        self.n_actions = self.env.action_space.shape[0]
        self.n_state = self.env.observation.dim_topo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt = os.path.join(ckpt, "model.pth")
        self.agent = ActorCritic(self.n_state, self.n_actions)
        self.optimizer = opt.Adam(self.agent.parameters(), lr=self.alpha)

    def save_model(self):
        torch.save(self.agent.state_dict(), self.ckpt)

    def load_model(self):
        self.agent.load_state_dict(torch.load(self.ckpt))

    def learn(self, num_episode):
        for episode in num_episode:
            state, _ = self.env.reset()
            done = False
            score = 0

            while done:
                state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
                probs, val = self.agent(state_tensor)
                action = np.random.choice(np.arange(len(probs.cpu().detach().numpy())), p=probs.cpu().detach().numpy())

                next_state, reward, done, _, _ = self.env.step(action)
                score += reward
                
                # Calculate the TD error and loss
                _, next_val = self.agent(torch.tensor(next_state, dtype=torch.float32).to(self.device))
                err = reward + self.gamma * (next_val * (1 - done)) - val
                actor_loss = -torch.log(probs[action]) * err
                critic_loss = torch.square(err)
                loss = actor_loss + critic_loss

                # Update the network
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Set the state to the next state
                state = next_state

            # Print the total reward for the episode
            print(f'Episode {episode}: Total reward = {score}')