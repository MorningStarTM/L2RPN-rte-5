import numpy as np
from collections import deque, namedtuple
from stable_baselines3 import PPO
from grid2op.Agent import RandomAgent
from utils.converter import Converter
import pickle

class Generator:
    def __init__(self, env, Experience):
        self.env = env
        self.obs_shape = env.observation_space.n
        self.action_shape = env.action_space.n
        self.state = np.zeros(self.obs_shape)
        self.experience = Experience
        self.data = []
        self.con = Converter(self.env)
        self.one_hot_encoding_act_conv,self.env_act_dict_list = self.con.create_one_hot_converter()

    def save(self, state, action, reward, next_state, done):
        self.data.append(self.experience(state, action, reward, next_state, done))
    
    def pickle_save(self, save_dir:str):
        with open(save_dir, 'wb') as f:
            pickle.dump(self.data, f)

        print("Experiences saved to", save_dir)

    def collect_data(self, episode_count):
        reward = 0
        done = False
        total_reward = 0
        agent = RandomAgent(self.env.action_space)
        for i in range(episode_count):
            obs = self.env.reset()
            while True:
                action = agent.act(obs, reward, done)
                next_state, reward, done, info = self.env.step(action)
                print(f"episode : {i}")
                self.save(obs.to_vect(), self.con.convert_env_act_to_one_hot_encoding_act(self.one_hot_encoding_act_conv,env_act=action.to_vect()), reward, next_state.to_vect(), done)
                obs = next_state

                if done:
                    # in this case the episode is over
                    break
        self.pickle_save("data\\experience.pkl")
        print("done")
                