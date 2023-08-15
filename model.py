import os

import gym
from env.custom_hopper import *
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.sac.policies import MlpPolicy, CnnPolicy

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback,CheckpointCallback

from util import linear_schedule,plot_results,EvalOnTargetCallback,make_env

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

class Model:

    def __init__(self, train_env_name: str, test_env_name: str,output_dir:str):
        
        self.output_dir = output_dir
        self.log_dir=os.path.join(self.output_dir,"train_logs")
        self.checkpoint_dir=os.path.join(self.output_dir,"checkpoints")
        #self.src_flag=src_flag  # "source" oppure "target"
        #os.makedirs(self.output_dir, exist_ok=True) #creata prima

        self.train_env_name = train_env_name
        self.test_env_name = test_env_name
        self.src_flag="source" if self.train_env_name == "CustomHopper-source-v0" else "target"

        num_cpu=os.cpu_count()//2
        print("using ",num_cpu," cpu")
        #self.train_env=VecMonitor(SubprocVecEnv([make_env(train_env_name, i) for i in range(num_cpu)]),os.path.join(self.output_dir,self.src_flag+"_monitor_log","monitor.csv"))

        self.train_env=make_vec_env(train_env_name,n_envs=num_cpu,monitor_dir=os.path.join(self.output_dir,self.src_flag+"_monitor_log"))

        #self.train_env = Monitor(gym.make(train_env_name),os.path.join(self.output_dir,self.src_flag+"_monitor_log","monitor.csv"))
        self.test_env = Monitor(gym.make(test_env_name))

        self.arch_name = None
        self.model=None

    #funzione load model, spostare funzioni da train
    def load_model(self,checkpoint):
        self.model = SAC.load(checkpoint)

    def train(self, timesteps = 50000, learning_rate=0.001,lr_schedule="constant",**hyperparams):
        #aggiungere use_udr per abilitare o no udr
        if self.model!=None:
            print("Model already loaded!") #da implementare allenamento da checkpoint
            return
        
        arch_name = "SAC_"
        arch_name+="s_" if self.src_flag=="source" else "t_"
        """
        if self.train_env_name == "CustomHopper-source-v0":
            arch_name += "s_"
        elif self.train_env_name == "CustomHopper-target-v0":
            arch_name += "t_"
        """
        arch_name = arch_name +"lr_"+ str(learning_rate) + '_steps_' + str(timesteps)

        #NON IN USO: serve a valutare su target_env se alleniamo su source e salvare il miglior modello
        #salva in file evaluations.npz che potremmo leggere per fare un altra linea del grafico
        #purtroppo sminchia l'output del custom logger, dovremmo risolvere
        """
        eval_callback = EvalCallback(self.test_env,
                            best_model_save_path=self.checkpoint_dir,
                            log_path=os.path.join(self.output_dir,"callback_logs"),
                            eval_freq=10,
                            deterministic=True, render=False)
        """
        eval_target_callback=EvalOnTargetCallback(self.test_env,check_freq=10,
                                log_file=os.path.join(self.log_dir,"st_eval_log.csv"),
                                n_eval_eps=5)
        #callback che salva checkpoint
        checkpoint_callback = CheckpointCallback(
                            save_freq=1000,
                            save_path=self.checkpoint_dir,
                            name_prefix=self.src_flag,
                            save_replay_buffer=True,
                            save_vecnormalize=True,
                            )
        if lr_schedule=="linear":
            lr_schedule=linear_schedule(learning_rate) #eventualmente aggiungere costante e coseno
        else:
            lr_schedule=learning_rate
        
        self.model = SAC(MlpPolicy, self.train_env, verbose = 1,learning_rate=lr_schedule, **hyperparams) #,tensorboard_log=self.output_dir
        
        custom_logger = configure(self.log_dir, ["csv"])
        self.model.set_logger(custom_logger)

        self.train_env.reset()

        self.model.learn(total_timesteps = timesteps, log_interval = 10,callback=[checkpoint_callback,eval_target_callback]) #,progress_bar=True #mi sminchia l'output
        #rinomina i file di log, usa flag source target del model
        custom_logger.close()
        os.rename(os.path.join(self.log_dir,"progress.csv"),os.path.join(self.log_dir,self.src_flag+"_train_log.csv"))
        #os.rename(os.path.join(self.output_dir,"monitor.csv"),os.path.join(self.output_dir,self.src_flag+"_monitor.csv"))

        self.model.save(os.path.join(self.checkpoint_dir, arch_name))
        #elif exists(arch_name):
        #    self.model = SAC.load(arch_name)
        #self.arch_name = arch_name

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

            self.model = SAC(MlpPolicy, self.train_env, verbose = 1,device='cuda', **hyperparams)
            self.train_env.enable_udr()
            #serve train_env reset ?
            self.model.learn(total_timesteps = timesteps, log_interval = 20)
            self.model.save(arch_name) 
        elif exists(arch_name):
            self.model = SAC.load(arch_name)
        self.arch_name = arch_name    
        

    def test(self,test_env_name="CustomHopper-target-v0", n_eval = 50):

        #if exists(self.arch_name):
            #self.model = SAC.load(self.arch_name)
        if self.model!=None:
            test_env=Monitor(gym.make(test_env_name))
            #self.test_env.reset()
            test_env.reset()
            mean_reward, std_reward = evaluate_policy(self.model, test_env, n_eval_episodes = n_eval, deterministic = True)
            print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            return mean_reward, std_reward
        else:
            print("Model not loaded!")
            return -1,-1

    def plot_results(self):
        #e = "s" if self.test_env_name == "CustomHopper-source-v0" else "t"
        #filename = self.arch_name[:5] + e + self.arch_name[5:]
        #filename="source_train_plot" if self.train_env_name == "CustomHopper-source-v0" else "target_train_plot"
        #train_env="source" if self.train_env_name == "CustomHopper-source-v0" else "target_train_plot"
        plot_results(log_folder=self.output_dir,train_env=self.src_flag)
