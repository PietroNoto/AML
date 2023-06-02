import gym
from env.custom_hopper import *
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from os.path import exists
import numpy as np


class Model:


    def __init__(self, train_env_name: str, test_env_name: str):
        
        self.train_env_name = train_env_name
        self.test_env_name = test_env_name
        self.train_env = Monitor(gym.make(train_env_name))
        self.test_env = Monitor(gym.make(test_env_name))


    def enable_udr(self):
        self.train_env.enable_udr()


    def train(self, timesteps = 50000, **hyperparams):

        if self.train_env_name == "CustomHopper-source-v0":
            arch_name = "SAC_source_env"
        elif self.train_env_name == "CustomHopper-target-v0":
            arch_name = "SAC_target_env"
        if exists(arch_name + '.zip'):
            self.model = SAC.load(arch_name)
        else:
            self.model = SAC(MlpPolicy, self.train_env, verbose = 1, **hyperparams)
            self.model.learn(total_timesteps = timesteps, log_interval = 20)
            self.model.save(arch_name)



    def test(self, n_eval = 50):

        if self.train_env_name == "CustomHopper-source-v0":
            arch_name = "SAC_source_env"
        elif self.train_env_name == "CustomHopper-target-v0":
            arch_name = "SAC_target_env"
        if exists(arch_name + '.zip'):
            self.model = SAC.load(arch_name)
            self.test_env.reset()
            self.mean_reward, self.std_reward = evaluate_policy(self.model, self.test_env, n_eval_episodes = n_eval, deterministic = True)
            print(f"mean_reward={self.mean_reward:.2f} +/- {self.std_reward:.2f}")

            

if __name__ == '__main__':

    
    #Source-source
    ss_model = Model("CustomHopper-source-v0", "CustomHopper-source-v0")
    ss_model.enable_udr()
    ss_model.train(50000, learning_rate = 0.003)
    ss_model.test(50)
    

    #Source-target
    st_model = Model("CustomHopper-source-v0", "CustomHopper-target-v0")
    st_model.train(50000, learning_rate = 0.003)
    st_model.test(50)

    #Target-target
    tt_model = Model("CustomHopper-target-v0", "CustomHopper-target-v0")
    tt_model.train(50000, learning_rate = 0.003)
    tt_model.test(50)

    #Re-train source environment using UDR
    ss_model.enable_udr()
    
