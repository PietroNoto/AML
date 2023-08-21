#learning rate scheduling
from typing import Callable, Union

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    # Force conversion to float
    initial_value_ = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value_

    return func


#plotting
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common import results_plotter
import pandas
import matplotlib.pyplot as plt
import os


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'same')


def plot_results(log_folder,train_env,udr_prefix):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param train_env: (str) train environment: "source" or "target"

    monitor.csv output: 
        r=sum of all the rewards of the episode 
        l=episode len 
        t=episode wall clock time
    train log (train_log/progress.csv):
    """
    plot_folder=os.path.join(log_folder,"plots")
    if not os.path.isdir(plot_folder): os.mkdir(plot_folder)

    x, y = ts2xy(load_results(os.path.join(log_folder,udr_prefix+train_env+"_monitor_log")), 'timesteps') #legge tutti i file *monitor.csv, deve distinguere source e target!
    
    #y = moving_average(y, window=50)
    # Truncate x
    #x = x[len(x) - len(y):]

    fig,ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel('timesteps')
    ax.set_ylabel('rewards')
    ax.set_title("learning curve")
    fig.savefig(os.path.join(plot_folder,udr_prefix+train_env+"monitor_learning_curve.svg"))

    log_path=os.path.join(log_folder,"train_logs",udr_prefix+train_env+'_train_log.csv' )

    df = pandas.read_csv(log_path)

    fig,ax = plt.subplots()
    ax.plot(df["time/total_timesteps"],df["rollout/ep_rew_mean"])
    ax.set_xlabel('timesteps')   #potremmo usare anche gli episodi, sotto
    ax.set_ylabel('mean episode rewards')
    ax.set_title("learning curve")
    fig.savefig(os.path.join(plot_folder,udr_prefix+train_env+"_learning_curve.svg"))

    fig,ax = plt.subplots()
    ax.plot(df["time/episodes"],df["rollout/ep_rew_mean"])
    ax.set_xlabel('episodes')        #viene quasi uguale
    ax.set_ylabel('mean episode rewards')
    ax.set_title("episode learning curve")
    fig.savefig(os.path.join(plot_folder,udr_prefix+train_env+"_episode_learning_curve.svg"))

    fig,ax = plt.subplots()
    ax.plot(df["time/episodes"],df["rollout/ep_len_mean"])
    ax.set_xlabel('episodes')
    ax.set_ylabel('mean episode length') #time/time_elapsed
    ax.set_title("episodes length")
    fig.savefig(os.path.join(plot_folder,udr_prefix+train_env+"_episode_len.svg"))

    if "train/learning_rate" in df:
        fig,ax = plt.subplots()
        ax.plot(df["time/total_timesteps"],df["train/learning_rate"])
        ax.set_title("learning rate schedule")
        ax.set_xlabel('timesteps')
        ax.set_ylabel('learning rate')
        fig.savefig(os.path.join(plot_folder,udr_prefix+train_env+"_lr_schedule.svg"))

    if "train/actor_loss" in df:
        fig,ax = plt.subplots()
        ax.plot(df["time/total_timesteps"],df["train/actor_loss"], label="actor loss")
        ax.plot(df["time/total_timesteps"],df["train/critic_loss"], label="critic loss")
        ax.set_title("actor and critic loss")
        ax.set_xlabel('timesteps')
        ax.set_ylabel('SAC losses')
        ax.legend()
        fig.savefig(os.path.join(plot_folder,udr_prefix+train_env+"_sac_losses.svg"))

    ss_log_path=os.path.join(log_folder,"train_logs" ,udr_prefix+'source_train_log.csv')
    
    ss_monitor_dir=os.path.join(log_folder,udr_prefix+"source_monitor_log")
    st_log_path=os.path.join(log_folder,"train_logs" ,udr_prefix+'st_eval_log.csv' )
    tt_monitor_dir=os.path.join(log_folder,udr_prefix+"target_monitor_log")
    
    tt_log_path=os.path.join(log_folder,"train_logs" ,udr_prefix+'target_train_log.csv' )

    if os.path.exists(ss_log_path) and os.path.exists(st_log_path)\
         and os.path.exists(tt_log_path):  #ss_monitor_dir  tt_monitor_dir
        
        ss_df = pandas.read_csv(ss_log_path)
        st_df = pandas.read_csv(st_log_path)
        tt_df = pandas.read_csv(tt_log_path)

        ssx, ssy = ts2xy(load_results(ss_monitor_dir), 'timesteps')
        ttx, tty = ts2xy(load_results(tt_monitor_dir), 'timesteps')
        window=10

        fig,ax = plt.subplots()
        #ax.plot(ssx,ssy,label="source-source")  #ss_df["time/total_timesteps"]  ss_df["rollout/ep_rew_mean"]
        ax.plot(ss_df["time/total_timesteps"],ss_df["rollout/ep_rew_mean"],label="source-source")

        ax.plot(st_df["time/total_timesteps"],st_df["eval/mean_reward"],label="source-target")

        #ax.plot(ttx,tty,label="target-target") #tt_df["time/total_timesteps"] tt_df["rollout/ep_rew_mean"]
        ax.plot(tt_df["time/total_timesteps"],tt_df["rollout/ep_rew_mean"],label="target-target")

        ax.set_xlabel('timesteps')
        ax.set_ylabel('mean episode rewards')
        ax.set_title("learning curve")
        ax.legend()
        fig.savefig(os.path.join(plot_folder,"combined_learning_curve.svg"))

        fig,ax = plt.subplots()
        ax.plot(ssx,moving_average(ssy,window),label="source-source")  #ss_df["time/total_timesteps"]  ss_df["rollout/ep_rew_mean"]
        #ax.plot(ss_df["time/total_timesteps"],ss_df["rollout/ep_rew_mean"],label="source-source")

        ax.plot(st_df["time/total_timesteps"],st_df["eval/mean_reward"],label="source-target")

        ax.plot(ttx,moving_average(tty,window),label="target-target") #tt_df["time/total_timesteps"] tt_df["rollout/ep_rew_mean"]
        #ax.plot(tt_df["time/total_timesteps"],tt_df["rollout/ep_rew_mean"],label="target-target")

        ax.set_xlabel('timesteps')
        ax.set_ylabel('mean episode rewards')
        ax.set_title("learning curve")
        ax.legend()
        fig.savefig(os.path.join(plot_folder,"combined_monitor_learning_curve.svg"))

#monitoring callbacks
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
#import gym #solo per i controlli sul tipo

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    from: https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#callbacks-monitoring-training
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

class EvalOnTargetCallback(BaseCallback):
    """
    from: https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#callbacks-monitoring-training
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param eval_env: evaluation environment (the target one)
    :param check_freq:
    :param log_file: Path of the log file
    :param n_eval_eps: number of evaluation episodes
    """
    def __init__(self,eval_env, check_freq: int, log_file: str,n_eval_eps:int): #: Union[gym.Env, VecEnv]
        super(EvalOnTargetCallback, self).__init__()
        self.check_freq = check_freq
        self.log_file = log_file
        self.eval_env=eval_env
        self.n_eval_eps=n_eval_eps
    
    def _init_callback(self) -> None:
        # create log file
        self.fp=open(self.log_file,'w')
        self.fp.write("time/total_timesteps,eval/mean_reward,eval/mean_ep_length")
    def _on_step(self) -> bool:
        if self.check_freq > 0 and self.n_calls % self.check_freq == 0:
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_eps,
                render=False,
                deterministic=True,
                return_episode_rewards=False,
            )
            self.fp.write(f"\n{self.num_timesteps},{episode_rewards},{episode_lengths}")

    def _on_training_end(self) -> None:
        self.fp.close()

from stable_baselines3.common.utils import set_random_seed
import gym
from CNN import VisionWrapper

#vectorized environment
def make_env(env_id: str, rank: int, seed: int = 0,use_udr=None,vision=False):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make(env_id) #, render_mode="human"
        if use_udr=="infinite":
            env.enable_infinite_udr(2,6)
        elif use_udr=="finite":
            env.enable_finite_udr()
        if vision:
            env=VisionWrapper(env)
        else:
            env.reset()
        return env
    set_random_seed(seed+rank)
    return _init
