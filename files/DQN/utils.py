import torch
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

def obs_to_vect(observation):
    obs_vect = []
    for i in observation.keys():
        if type(observation[i]) != int:
            for j in observation[i]:
                obs_vect.append(j)
        else:
            obs_vect.append(observation[i])
    return np.array(obs_vect)


class Converter:
    def __init__(self, env):
        self.env = env
        self.n_powerlines = self.env.n_line
        self.n_substations = self.env.n_sub
        self.total_bus_actions = 21
        self.one_hot_encoding_act_conv, self.env_act_dict_list = self.create_one_hot_converter()

    def create_one_hot_converter(self):
        """
        Creates two 2-d np.arrays used for conversion between grid2op action to one hot encoding action vector used by a neural network
        """
        one_hot_encoding_act_conv = []
        env_act_dict_list = []
        zero_act = np.zeros((self.n_powerlines+self.total_bus_actions,1))

        ## Add do nothing action vector (all zeroes)
        one_hot_encoding_act_conv.append(zero_act)
        env_act_dict_list.append({}) ## {} is the do nothing dictonary for actions in grid2op

        ## Powerline change actions
        for idx in range(self.n_powerlines):
            one_hot_encoding_act_conv_pwline = zero_act.copy()
            one_hot_encoding_act_conv_pwline[self.total_bus_actions+idx] = 1
            one_hot_encoding_act_conv.append(one_hot_encoding_act_conv_pwline)
            env_act_dict_list.append({'change_line_status': [idx]}) ## {'change_line_status': [idx]} set an action of changing line status for lineid with id idx


        ## Bus change actions
        start_slice = 0
        for sub_station_id, nb_el in enumerate(self.env.action_space.sub_info):
            one_hot_encoding_act_conv_substation = zero_act.copy()

            possible_bus_actions = np.array(list(product('01', repeat=nb_el))).astype(int)
            for possible_bus_action in possible_bus_actions:
                if possible_bus_action.sum()>0: # Do not include no change action vector
                    one_hot_encoding_act_conv_substation[start_slice:(start_slice+nb_el)] = possible_bus_action.reshape(-1,1)
                    one_hot_encoding_act_conv.append(one_hot_encoding_act_conv_substation.copy())
                    env_act_dict_list.append({"change_bus": {"substations_id": [(sub_station_id, possible_bus_action.astype(bool))]}})
            start_slice += nb_el

        one_hot_encoding_act_conv = np.array(one_hot_encoding_act_conv).reshape(len(one_hot_encoding_act_conv),self.n_powerlines+self.total_bus_actions)

        return one_hot_encoding_act_conv,env_act_dict_list

    def convert_env_act_to_one_hot_encoding_act(self,env_act):
        """
        Converts an grid2op action (in numpy format) to a one hot encoding vector
        """
        
        one_hot_encoding_act = np.zeros(len(self.one_hot_encoding_act_conv))
        env_act = env_act.reshape(-1,)
        action_idx = (self.one_hot_encoding_act_conv[:, None] == env_act).all(-1).any(-1)
        one_hot_encoding_act[action_idx] = 1
        return one_hot_encoding_act

    def convert_one_hot_encoding_act_to_env_act(self,one_hot_encoding_act):
        """
        Converts a one hot encoding action to a grid2op action
        """
        return self.env.action_space(self.env_act_dict_list[one_hot_encoding_act.argmax().item()])

    def int_to_one_hot_encoding(self, action_nb):
        b = np.zeros((1, 132))
        b[np.arange(1), action_nb] = 1
        return b
    


def plotLearning(x, scores, epsilons, filename, lines=None):
    fig, ax = plt.subplots()

    ax.plot(x, epsilons, color="C0", label="Epsilon")
    ax.set_xlabel("Game")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x')
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2 = ax.twinx()
    ax2.plot(x, running_avg, color="C1", label="Score")
    ax2.set_ylabel('Score', color="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            ax.axvline(x=line)

    plt.savefig(filename)
