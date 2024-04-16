import torch
import numpy as np

class Node:
    def __init__(self, obs):
        self.obs = obs
        self.node_types = ['substation', 'load', 'generator', 'line']

    def extract_substation_data(self):
        return self.obs.time_before_cooldown_sub
    
    def extract_load_data(self):
        return self.obs.load_p, self.obs.load_q, self.obs.load_v , self.obs.load_theta
    
    def extract_gen_data(self):
        return self.obs.gen_p, self.obs.gen_q, self.obs.gen_v, self.obs.gen_theta
    
    def extract_line_data(self):
        return self.obs.p_or, self.obs.q_or, self.obs.v_or, self.obs.a_or, self.obs.theta_or, self.obs.p_ex, self.obs.q_ex, self.obs.v_ex, self.obs.a_ex, self.obs.theta_ex, self.obs.rho, self.obs.line_status, self.obs.time_before_cooldown_line, self.obs.time_next_maintenance, self.obs.duration_next_maintenance

    def create_data(self):
        data = {'substations': np.array(self.extract_substation_data()),
                'lines': np.array(self.extract_line_data())}
        return data
