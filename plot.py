import matplotlib.pyplot as plt
import os
import numpy as np
from util import moving_average
import pandas
from stable_baselines3.common.results_plotter import load_results, ts2xy

import argparse

def plot2mln():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-dir", type=list, help='folders to plot', default=["prova_vec_env"])
    args=parser.parse_args()

    #for curr_dir in args.plot_dir:
        
    ss_log_path=os.path.join("prova_vec_env_8","train_logs" ,'source_train_log.csv' )
    st_log_path=os.path.join("prova_vec_env_8","train_logs" ,'st_eval_log.csv' )
    tt_log_path=os.path.join("prova_vec_env_8tris","train_logs" ,'target_train_log.csv' )

    ss_monitor_dir=os.path.join("prova_vec_env_8","source_monitor_log")
    tt_monitor_dir=os.path.join("prova_vec_env_8tris","target_monitor_log")

    window_st=10
    window=20

    if os.path.exists(ss_log_path) and os.path.exists(st_log_path)\
         and os.path.exists(tt_log_path):  #ss_monitor_dir  tt_monitor_dir
        
        ss_df = pandas.read_csv(ss_log_path)
        st_df = pandas.read_csv(st_log_path)
        tt_df = pandas.read_csv(tt_log_path)

        #ssx, ssy = ts2xy(load_results(ss_monitor_dir), 'timesteps')
        #ttx, tty = ts2xy(load_results(tt_monitor_dir), 'timesteps')

        fig,ax = plt.subplots()
        #ax.plot(ssx,ssy,label="source-source")  #ss_df["time/total_timesteps"]  ss_df["rollout/ep_rew_mean"]
        ax.plot(ss_df["time/total_timesteps"],ss_df["rollout/ep_rew_mean"],label="source-source")

        ax.plot(st_df["time/total_timesteps"],moving_average(st_df["eval/mean_reward"],window_st),label="source-target")

        #ax.plot(ttx,tty,label="target-target") #tt_df["time/total_timesteps"] tt_df["rollout/ep_rew_mean"]
        ax.plot(tt_df["time/total_timesteps"],tt_df["rollout/ep_rew_mean"],label="target-target")

        ax.set_xlabel('timesteps')
        ax.set_ylabel('mean episode rewards')
        ax.set_title("learning curve")
        ax.legend()
        plt.show()
        #fig.savefig(os.path.join(plot_folder,"combined_learning_curve.svg"))

        ssx, ssy = ts2xy(load_results(ss_monitor_dir), 'timesteps')
        ttx, tty = ts2xy(load_results(tt_monitor_dir), 'timesteps')

        fig,ax = plt.subplots()
        ax.plot(ssx,moving_average(ssy,window),label="source-source")  #ss_df["time/total_timesteps"]  ss_df["rollout/ep_rew_mean"]

        ax.plot(st_df["time/total_timesteps"],moving_average(st_df["eval/mean_reward"],window_st),label="source-target")

        ax.plot(ttx,moving_average(tty,window),label="target-target") #tt_df["time/total_timesteps"] tt_df["rollout/ep_rew_mean"]

        ax.set_xlabel('timesteps')
        ax.set_ylabel('mean episode rewards')
        ax.set_title("learning curve")
        ax.legend()
        plt.show()
        #fig.savefig(os.path.join(plot_folder,"combined_monitor_learning_curve.svg"))


if __name__ == '__main__':

    ss_log_path_1=os.path.join("","train_logs" ,'source_train_log.csv' )
    st_log_path_1=os.path.join("","train_logs" ,'st_eval_log.csv' )
    tt_log_path_1=os.path.join("","train_logs",'target_train_log.csv' )

    ss_df = pandas.read_csv(ss_log_path_1)
    st_df = pandas.read_csv(st_log_path_1)
    tt_df = pandas.read_csv(tt_log_path_1)

    fig,ax = plt.subplots()
    ax.plot(ss_df["time/total_timesteps"],ss_df["rollout/ep_rew_mean"],label="source-source")

    ax.plot(st_df["time/total_timesteps"],moving_average(st_df["eval/mean_reward"],10),label="source-target")

    ax.plot(tt_df["time/total_timesteps"],tt_df["rollout/ep_rew_mean"],label="target-target")

    ax.set_xlabel('timesteps')
    ax.set_ylabel('mean episode rewards')
    ax.set_title("learning curve")
    ax.legend()
    plt.show()
