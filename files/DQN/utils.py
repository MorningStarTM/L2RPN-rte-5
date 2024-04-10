import torch
import numpy as np

def obs_to_vect(observation):
    obs_vect = []
    for i in observation.keys():
        if type(observation[i]) != int:
            for j in observation[i]:
                obs_vect.append(j)
        else:
            obs_vect.append(observation[i])
    return np.array(obs_vect)