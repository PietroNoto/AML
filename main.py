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
        self.arch_name = None


    def train(self, arch_name = None, timesteps = 50000, **hyperparams):
        
        if arch_name is None:                       #it means that I don't have any trained models so I want to train a new one
            arch_name = "SAC_"
            if self.train_env_name == "CustomHopper-source-v0":
                arch_name += "s_"
            elif self.train_env_name == "CustomHopper-target-v0":
                arch_name += "t_"
            arch_name = arch_name + str(hyperparams["learning_rate"]) + '_' + str(timesteps)
            self.model = SAC(MlpPolicy, self.train_env, verbose = 1, **hyperparams)
            self.model.learn(total_timesteps = timesteps, log_interval = 20)
            self.model.save(arch_name)   
        elif exists(arch_name):
            self.model = SAC.load(arch_name)
        self.arch_name = arch_name


    def train_dummy(self):

        self.model = SAC(CnnPolicy, self.train_env, verbose = 1)
        self.model.learn(500)


    def train_udr(self, arch_name = None, timesteps = 50000, n_distr = 3, **hyperparams):

        if n_distr < 0:
            return
        elif arch_name is None: 
            arch_name = "SAC_"
            if self.train_env_name == "CustomHopper-source-v0":
                arch_name += "s_"
            elif self.train_env_name == "CustomHopper-target-v0":
                arch_name += "t_"
            arch_name = arch_name + str(hyperparams["learning_rate"]) + '_' + str(timesteps) + "_UDR"
            self.model = SAC(MlpPolicy, self.train_env, verbose = 1, **hyperparams)
            self.train_env.enable_udr()
            self.model.learn(total_timesteps = timesteps, log_interval = 20)
            self.model.save(arch_name) 
        elif exists(arch_name):
            self.model = SAC.load(arch_name)
        self.arch_name = arch_name    
        

    def test(self, n_eval = 50):

        if exists(self.arch_name):
            self.model = SAC.load(self.arch_name)
            self.test_env.reset()
            self.mean_reward, self.std_reward = evaluate_policy(self.model, self.test_env, n_eval_episodes = n_eval, deterministic = True)
            print(f"mean_reward={self.mean_reward:.2f} +/- {self.std_reward:.2f}") 

if __name__ == '__main__':

    n_test_eps = 50
    n_timesteps = 50
    lr = 0.003
    n_distr = 3

    #dummy = Model("CustomHopper-source-v0", "CustomHopper-source-v0")
    #dummy.train_dummy()
    
    #Source-source
    ss_model = Model("CustomHopper-source-v0", "CustomHopper-source-v0")
    ss_model.train(None, n_timesteps, learning_rate = lr)
    ss_model.test(n_test_eps)
    
    #Source-target
    st_model = Model("CustomHopper-source-v0", "CustomHopper-target-v0")
    st_model.train("SAC_s_0.003_50", n_timesteps, learning_rate = lr)
    st_model.test(n_test_eps)

    #Target-target
    tt_model = Model("CustomHopper-target-v0", "CustomHopper-target-v0")
    #tt_model.train(n_timesteps, learning_rate = lr)
    #tt_model.test(n_test_eps)

    #Source-source using UDR
    ss_udr = Model("CustomHopper-source-v0", "CustomHopper-source-v0")
    ss_udr.train_udr(None, n_timesteps, n_distr, learning_rate = lr)
    ss_udr.test(n_test_eps)

    #Source-target using UDR
    st_udr = Model("CustomHopper-source-v0", "CustomHopper-target-v0")
    #st_udr.train_udr(n_timesteps, n_distr, learning_rate = lr)
    st_udr.train_udr("SAC_s_0.003_50_UDR", n_timesteps, n_distr, learning_rate = lr)
    st_udr.test(n_test_eps)

    #Source-source using CNN
    env = VisionWrapper(gym.make("CustomHopper-source-v0"))
    #vis_ss_model = SAC(CnnPolicy, env, verbose = 1, buffer_size=10000)
    #vis_ss_model.learn(n_timesteps, log_interval=20, progress_bar=True)
    #vis_ss_model.save("SAC_CNN_source_env")
    #mean_reward, std_reward = evaluate_policy(vis_ss_model, env, n_eval_episodes = n_test_eps, deterministic = True)
    #print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}") 

    
