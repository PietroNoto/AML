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

    hyp_opt_dir="hyp_opt_lr_3"

    lrs=[]
    ss_rew=[]
    st_rew=[]
    tt_rew=[]

    for curr_dir in os.listdir(hyp_opt_dir):
        if os.path.isdir(os.path.join(hyp_opt_dir,curr_dir)):
            lrs.append(float(curr_dir[3:]))
            print(curr_dir)
            with open(os.path.join(hyp_opt_dir,curr_dir,"test_results.txt")) as res_file:
                lines=res_file.readlines()
                print(lines)
                ss_curr_rew,ss_curr_std=lines[0].split('=')[1].split('+/-')
                ss_rew.append(float(ss_curr_rew))
                st_curr_rew,st_curr_std=lines[1].split('=')[1].split('+/-')
                st_rew.append(float(st_curr_rew))
                #tt_curr_rew,tt_curr_std=lines[2].split('=')[1].split('+/-')
                #tt_rew.append(float(tt_curr_rew))
    sort_val=np.array([list(e) for e in zip(lrs,ss_rew,st_rew)])
    sort_val=sort_val[sort_val[:,0].argsort()]
    #sort_val=np.sort([list(e) for e in zip(lrs,ss_rew,st_rew)],axis=0) #,tt_rew
    print(sort_val)

    fig,ax = plt.subplots()
    ax.plot(sort_val[:,0],sort_val[:,1])
    ax.set_xscale('log')
    ax.set_xlabel('learning rates')
    ax.set_ylabel('source mean reward')
    ax.set_title("learning rate optimization")
    plt.show()





