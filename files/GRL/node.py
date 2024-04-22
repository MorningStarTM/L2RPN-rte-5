import torch
import numpy as np

class Node:
    def __init__(self, obs, env):
        self.env = env
        self.obs = obs
        self.node_types = ['substation', 'load', 'generator', 'line']

    def extract_substation_data(self):
        return self.obs.time_before_cooldown_sub
    
    def extract_load_data(self):
        return self.obs.load_p, self.obs.load_q, self.obs.load_v , self.obs.load_theta
    
    def extract_gen_data(self):
        return self.obs.gen_p.tolist(), self.obs.gen_q.tolist(), self.obs.gen_v.tolist(), self.obs.gen_theta.tolist()
    
    def extract_line_data(self):
        return self.obs.p_or, self.obs.q_or, self.obs.v_or, self.obs.a_or, self.obs.theta_or, self.obs.p_ex, self.obs.q_ex, self.obs.v_ex, self.obs.a_ex, self.obs.theta_ex, self.obs.rho, self.obs.line_status, self.obs.time_before_cooldown_line, self.obs.time_next_maintenance, self.obs.duration_next_maintenance

    def create_data(self):
        # Extract data for each node type
        substation_data = np.array([self.extract_substation_data()]).T
        load_data = np.array(self.extract_load_data()).T
        gen_data = np.array(self.extract_gen_data()).T
        line_data = np.array(self.extract_line_data()).T

        max_length = len(substation_data[0]) + len(load_data[0]) + len(gen_data[0]) + len(line_data[0])


        # Pad feature arrays to match the maximum length
        sub_padd = np.pad(substation_data, ((0, 0), (0, max_length - len(substation_data[0]))), mode='constant')
        load_padd = np.pad(load_data, ((0, 0), (0, max_length - len(load_data[0]))), mode='constant')
        gen_padd = np.pad(gen_data, ((0, 0), (0, max_length - len(gen_data[0]))), mode='constant')
        line_padd = np.pad(line_data, ((0, 0), (0, max_length - len(line_data[0]))), mode='constant')

        # Combine padded feature arrays into a single array
        feature_data = np.concatenate((sub_padd, load_padd, gen_padd, line_padd), axis=0)

        # Return the combined feature array
        return feature_data, self.obs.connectivity_matrix()
    
    def convert_obs(self, obs):
        obs_vect = obs.to_vect()
        obs_vect = torch.FloatTensor(obs_vect).unsqueeze(0)
        length = self.env.action_space.dim_topo

        rho_ = torch.zeros(obs_vect.size(0), length, device=self.device)
        rho_[..., self.env.action_space.line_or_pos_topo_vect] = torch.tensor(obs.rho, device=self.device)
        rho_[..., self.env.action_space.line_ex_pos_topo_vect] = torch.tensor(obs.rho, device=self.device)


        p_ = torch.zeros(obs_vect.size(0), length, device=self.device)
        p_[..., self.env.action_space.gen_pos_topo_vect] = torch.tensor(obs.gen_p, device=self.device)
        p_[..., self.env.action_space.load_pos_topo_vect] = torch.tensor(obs.load_p, device=self.device)
        p_[..., self.env.action_space.line_or_pos_topo_vect] = torch.tensor(obs.p_or, device=self.device)
        p_[..., self.env.action_space.line_ex_pos_topo_vect] = torch.tensor(obs.p_ex, device=self.device)


        danger_ = torch.zeros(obs_vect.size(0), length, device=self.device)
        danger = obs.rho >= 0.98
        danger_[..., self.env.action_space.line_or_pos_topo_vect] = torch.tensor(danger, device=self.device).float()
        danger_[..., self.env.action_space.line_ex_pos_topo_vect] = torch.tensor(danger, device=self.device).float() 

        state = torch.stack([p_, rho_, danger_], dim=2).to(self.device)

        adj = (torch.FloatTensor(obs.connectivity_matrix()) + torch.eye(int(obs.dim_topo))).to(self.device)

        return state, adj

    