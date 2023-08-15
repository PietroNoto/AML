import gym
from CNN import VisionWrapper
from env.custom_hopper import *
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.sac.policies import MlpPolicy, CnnPolicy
from stable_baselines3.common.monitor import Monitor
from os.path import exists


#plotting
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common import results_plotter
import os
import matplotlib.pyplot as plt


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve',filename="train_plot"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    
    #y = moving_average(y, window=50)
    
    # Truncate x
    #x = x[len(x) - len(y):]

    fig,ax = plt.subplots()  #title
    ax.plot(x, y)
    ax.set_xlabel('Number of Timesteps')
    ax.set_ylabel('Rewards')
    ax.set_title(title + " Smoothed")
    #plt.show()
    fig.savefig(filename+".png")




class Model:

<<<<<<< HEAD
    def __init__(self, train_env_name: str, test_env_name: str,log_folder:str):
=======
    def __init__(self, train_env_name: str, test_env_name: str):
>>>>>>> 5ed3b2d (Added CNN module and vision wrapper)
        
        self.log_folder = log_folder
        os.makedirs(self.log_folder, exist_ok=True)

        self.train_env_name = train_env_name
        self.test_env_name = test_env_name
        self.train_env = Monitor(gym.make(train_env_name),self.log_folder)
        self.test_env = Monitor(gym.make(test_env_name),self.log_folder)
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
            self.train_env.reset()
            self.model.learn(total_timesteps = timesteps, log_interval = 10)
            self.model.save(arch_name)   
        elif exists(arch_name):
            self.model = SAC.load(arch_name)
        self.arch_name = arch_name


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

        if exists(self.arch_name):
            self.model = SAC.load(self.arch_name)
            self.test_env.reset()
            self.mean_reward, self.std_reward = evaluate_policy(self.model, self.test_env, n_eval_episodes = n_eval, deterministic = True)
            print(f"mean_reward={self.mean_reward:.2f} +/- {self.std_reward:.2f}") 
<<<<<<< HEAD


    def plot_results(self):
        e = "s" if self.test_env_name == "CustomHopper-source-v0" else "t"
        filename = self.arch_name[:5] + e + self.arch_name[5:]
        plot_results(log_folder=self.log_folder,filename = filename)


=======
>>>>>>> 5ed3b2d (Added CNN module and vision wrapper)

if __name__ == '__main__':

    n_test_eps = 50
    n_timesteps = 50000
    lr = 0.03
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
    print("Source-target with UDR:")
    t_log_dir = "log_st_" + str(lr) + "_" + str(n_timesteps)
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
    
