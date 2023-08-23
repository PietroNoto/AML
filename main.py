import gym
from CNN import VisionWrapper
#from env.custom_hopper import *
#from stable_baselines3 import SAC
#from stable_baselines3.common.evaluation import evaluate_policy
#from stable_baselines3.sac.policies import MlpPolicy, CnnPolicy
from os.path import exists

import argparse
import os

#logging
#from stable_baselines3.common.monitor import Monitor
#from stable_baselines3.common.logger i
# mport configure
from model import Model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, help='Path where the model is saved', default="prova_vec_env")
    parser.add_argument("--test-eps", type=int, help='Number of test episodes', default=50)
    parser.add_argument("--lr", type=float, help='starting learning rate', default=7.3e-4)
    parser.add_argument("--timesteps", type=int, help='number of max timesteps', default=1_000) #circa 23s ogni 1000 timesteps, 38m per 100_000,~ 6h per 1mln
    parser.add_argument("--use-udr", type=str,choices=["only","both","no"], help='use udr', default="no")
    parser.add_argument("--n-distr", type=int, help='number of udr distributions', default=3)
    parser.add_argument("--source-env", type=str, help='source environment name', default="CustomHopper-source-v0")
    parser.add_argument("--target-env", type=str, help='target environment name', default="CustomHopper-target-v0")
    parser.add_argument("--checkpoint", type=str, help='path of a checkpoint file', default=None)
    parser.add_argument("--buffer-size", type=int, help='buffer size', default=1_000_000)
    parser.add_argument("--lr-scheduling", type=str, help='learning rate scheduling', default="constant",
                        choices=["constant","linear"]) #aggiungere "cosine"

    parser.add_argument("--use-vision", type=bool, help='change observation space to pixels', default=False)

    args=parser.parse_args()
    
    n_test_eps = args.test_eps #50
    n_timesteps = args.timesteps #50_000
    lr = args.lr #7.3e-4
    n_distr = args.n_distr #3
    source_env_name="CustomHopper-source-v0"
    target_env_name="CustomHopper-target-v0"
    use_udr=args.use_udr #False

    #args:  train->plots: output-dir/  n_timesteps  lr  batch_size  checkpoint_file  policy [MLP, CNN]? <- argomenti
    #                       checkpoints/
    #                           checkpoint_1.pt ...   #serve callback, per allenamenti lunghi
    #                       train_logs/progress.csv    #rinomina, per source e target
    #                       plots/                    #distinguere source e plot anche qui
    #                            source_train_plot.svg
    #                            target_train_plot.svg
    #       test -> spostare in un altro script che carica modello e testa

    output_dir = args.output_dir
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    logs_dir = os.path.join(output_dir, 'train_logs')

    #se la cartella é già esistente si blocca, perderemmo i file di log, altrimenti crea la struttura
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        os.mkdir(checkpoints_dir)
        os.mkdir(logs_dir)
    else:
        if len(os.listdir(output_dir)) > 0:
            raise FileExistsError('Output dir contains files')
        else:
            os.mkdir(checkpoints_dir)

    #stampa file con gli iperparametri
    with open(os.path.join(output_dir,'params.txt'), 'w') as param_file:
        param_str="lr: "+str(lr)+\
            "\nscheduling: "+args.lr_scheduling +\
            "\ntotal steps: "+str(n_timesteps)+\
            "\nsource env: "+source_env_name+\
            "\ntarget env: "+target_env_name+\
            "\nuse udr: "+str(use_udr)+\
            "\nuse vision: "+str(args.use_vision)+\
            "\ntest episodes: "+str(n_test_eps)+\
            "\nbuffer size: "+str(args.buffer_size)
        param_file.write(param_str)

    #file con i risultati dell'esperimento
    test_fp=open(os.path.join(output_dir,'test_results.txt'), 'w')

    if args.use_udr !="only":

        print("Source-source:")
        s_model = Model(source_env_name, target_env_name,output_dir,vision=args.use_vision)
        if args.checkpoint!=None:
            s_model.load_model(args.checkpoint)
        s_model.train(timesteps=n_timesteps, learning_rate = lr,lr_schedule=args.lr_scheduling,buffer_size=args.buffer_size)
        s_model.plot_results()
        ss_mean_rew,ss_std_rew=s_model.test(test_env_name=source_env_name,n_eval=n_test_eps)
        test_fp.write("Source-source: "+f"mean_reward={ss_mean_rew:.2f} +/- {ss_std_rew:.2f}")
        
        print("Source-target") #testa modello già allenato
        st_mean_rew,st_std_rew=s_model.test(target_env_name,n_test_eps)
        test_fp.write("\nSource-target: "+f"mean_reward={st_mean_rew:.2f} +/- {st_std_rew:.2f}")
        
        print("Target-target:")
        tt_model = Model(target_env_name, target_env_name, output_dir,vision=args.use_vision)
        if args.checkpoint!=None: #non l'ho ancora testato, serve ad allenare a partire da un checkpoint
            tt_model.load_model(args.checkpoint.replace("source","target"))
        tt_model.train(timesteps=n_timesteps, learning_rate = lr,lr_schedule=args.lr_scheduling,buffer_size=args.buffer_size)
        tt_model.plot_results()
        tt_mean_rew,tt_std_rew=tt_model.test(test_env_name=target_env_name,n_eval=n_test_eps)
        test_fp.write("\ntarget-target: "+f"mean_reward={tt_mean_rew:.2f} +/- {tt_std_rew:.2f}")
    
    if args.use_udr != "no":
    
        #Source-source using UDR
        print("Source-source with UDR:")
        s_udr = Model(source_env_name, target_env_name,output_dir,use_udr="infinite")
        
        if args.checkpoint!=None: #non l'ho ancora testato, serve ad allenare a partire da un checkpoint
            s_udr.load_model(args.checkpoint)

        s_udr.train(timesteps=n_timesteps,
                    learning_rate = lr,
                    lr_schedule=args.lr_scheduling,
                    buffer_size=args.buffer_size,
                    use_udr=True)
        
        s_udr.plot_results()
        ss_mean_rew,ss_std_rew=s_udr.test(test_env_name=source_env_name,n_eval=n_test_eps)
        test_fp.write("Source-source: "+f"mean_reward={ss_mean_rew:.2f} +/- {ss_std_rew:.2f}")

        #Source-target using UDR
        print("Source-target with UDR:")
        st_mean_rew,st_std_rew=s_udr.test(test_env_name=target_env_name,n_eval=n_test_eps)
        test_fp.write("\nSource-target: "+f"mean_reward={st_mean_rew:.2f} +/- {st_std_rew:.2f}")

        print("Target-target with UDR:")
        t_udr = Model(target_env_name, target_env_name,output_dir,use_udr="infinite")

        if args.checkpoint!=None: #non l'ho ancora testato, serve ad allenare a partire da un checkpoint
            t_udr.load_model(args.checkpoint.replace("source","target"))

        t_udr.train(timesteps=n_timesteps,
                    learning_rate = lr,
                    lr_schedule=args.lr_scheduling,
                    buffer_size=args.buffer_size,
                    use_udr=True)
        t_udr.plot_results()
        tt_mean_rew,tt_std_rew=t_udr.test(test_env_name=target_env_name,n_eval=n_test_eps)
        test_fp.write("\nTarget-target: "+f"mean_reward={tt_mean_rew:.2f} +/- {tt_std_rew:.2f}")

    test_fp.close()

    #Source-source vision without udr
    """
    vis_s_model = Model(source_env_name, target_env_name,output_dir,use_udr="", vision=True)
    vis_s_model.train(timesteps=n_timesteps, learning_rate = lr,
                    lr_schedule=args.lr_scheduling,buffer_size=args.buffer_size)
    vis_s_model.plot_results()
    vss_mean_rew,vss_std_rew=vis_s_model.test(test_env_name=source_env_name,n_eval=n_test_eps)
    test_fp.write("Source-source: "+f"mean_reward={ss_mean_rew:.2f} +/- {ss_std_rew:.2f}")
    """
    #Source-source using CNN
    #env = VisionWrapper(gym.make("CustomHopper-source-v0"))
    #vis_ss_model = SAC(CnnPolicy, env, verbose = 1, buffer_size=10000)
    #vis_ss_model.learn(n_timesteps, log_interval=20, progress_bar=True)
    #vis_ss_model.save("SAC_CNN_source_env")
    #mean_reward, std_reward = evaluate_policy(vis_ss_model, env, n_eval_episodes = n_test_eps, deterministic = True)
    #print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}") 
    
    
