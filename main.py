import gym
from CNN import VisionWrapper
from env.custom_hopper import *
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.sac.policies import MlpPolicy, CnnPolicy
from stable_baselines3.common.monitor import Monitor
from os.path import exists


class Model:

    def __init__(self, train_env_name: str, test_env_name: str):
        
        self.train_env_name = train_env_name
        self.test_env_name = test_env_name
        self.train_env = Monitor(gym.make(train_env_name))
        self.test_env = Monitor(gym.make(test_env_name))


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


    def train_udr(self, timesteps = 50000, n_distr = 0, lower = 1, upper = 5, **hyperparams):

        if n_distr < 0:
            return
        else:
            arch_name = "SAC_source_env_udr"
            if exists(arch_name + '.zip'):
                self.model = SAC.load(arch_name)
            else:
                self.model = SAC(MlpPolicy, self.train_env, verbose = 1, **hyperparams)
                if n_distr == 0:
                    self.train_env.enable_infinite_udr(lower, upper)
                else:
                    self.train_env.enable_finite_udr(n_distr)
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

    n_test_eps = 50
    n_timesteps = 50000
    lr = 0.003
    n_distr = 3
    
    #Source-source
    ss_model = Model("CustomHopper-source-v0", "CustomHopper-source-v0")
    #ss_model.train(n_timesteps, learning_rate = lr)
    #ss_model.test(n_test_eps)
    
    #Source-target
    st_model = Model("CustomHopper-source-v0", "CustomHopper-target-v0")
    #st_model.train(n_timesteps, learning_rate = lr)
    #st_model.test(n_test_eps)

    #Target-target
    tt_model = Model("CustomHopper-target-v0", "CustomHopper-target-v0")
    #tt_model.train(n_timesteps, learning_rate = lr)
    #tt_model.test(n_test_eps)

    #Source-source using UDR
    ss_udr = Model("CustomHopper-source-v0", "CustomHopper-source-v0")
    #ss_udr.train_udr(n_timesteps, n_distr, learning_rate = lr)
    #ss_udr.train_udr(n_timesteps, 0, 1, 5, learning_rate = lr)
    #ss_udr.test(n_test_eps)

    #Source-target using UDR
    st_udr = Model("CustomHopper-source-v0", "CustomHopper-target-v0")
    #st_udr.train_udr(n_timesteps, n_distr, learning_rate = lr)
    st_udr.train_udr(n_timesteps, 0, 1, 5, learning_rate = lr)
    #st_udr.test(n_test_eps)
    
    #Source-source using CNN
    env = VisionWrapper(gym.make("CustomHopper-source-v0"))
    #vis_ss_model = SAC(CnnPolicy, env, verbose = 1, buffer_size=10000)
    #vis_ss_model.learn(n_timesteps, log_interval=20, progress_bar=True)
    #vis_ss_model.save("SAC_CNN_source_env")
    #mean_reward, std_reward = evaluate_policy(vis_ss_model, env, n_eval_episodes = n_test_eps, deterministic = True)
    #print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}") 
    
