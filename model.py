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

from util import linear_schedule, plot_results, EvalOnTargetCallback, make_env

from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv,VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from wrappers import *

class Model:

    def __init__(self, train_env_name: str,
                 test_env_name: str,
                 output_dir:str,
                 use_vec_env: bool,
                 use_udr: bool,
                 udr_type: str,
                 udr_lb: float,
                 udr_ub: float,
                 udr_ndistr: int,
                 udr_range: float,
                 use_vision: bool,
                 width: int,
                 height: int,
                 grayscale: bool,
                 pose_est: bool,
                 pose_config: str,
                 pose_checkpoint: str):
        
        self.train_env_name = train_env_name
        self.test_env_name = test_env_name
        self.output_dir = output_dir
        self.src_flag = "source" if self.train_env_name == "CustomHopper-source-v0" else "target"
        self.use_vec_env = use_vec_env
        self.use_vision = use_vision
        self.use_udr = use_udr
        self.udr_prefix = "udr_" if self.use_udr else ""
        self.udr_type = udr_type
        self.udr_lb = udr_lb
        self.udr_ub = udr_ub
        self.udr_ndistr = udr_ndistr
        self.udr_range = udr_range
        self.use_vision = use_vision
        self.width = width
        self.height = height
        self.gray = grayscale
        self.use_pose_est = pose_est
        self.pose_config = pose_config
        self.pose_checkpoint = pose_checkpoint
        
        self.log_dir=os.path.join(self.output_dir,"train_logs")
        self.checkpoint_dir=os.path.join(self.output_dir,"checkpoints")
        
        num_cpu = os.cpu_count()
        
        self.train_env = self.make_env(self.train_env_name)
        self.test_env = self.make_env(self.test_env_name, True)
        
        if self.use_vec_env:
            print("Using",num_cpu,"CPUs")
        
        self.arch_name = None
        self.model = None

    #funzione load model, spostare funzioni da train
    def load_model(self,checkpoint):
        self.model = SAC.load(checkpoint,env=self.train_env) #,env=self.train_env
        #self.model.set_env(self.train_env)


    def train(self, timesteps = 50000, learning_rate=0.001, lr_schedule="constant", buffer_size=1_000_000, use_udr=False,**hyperparams):
        #aggiungere use_udr per abilitare o no udr
        if self.model != None:
            print("Model already loaded!") #da implementare allenamento da checkpoint
            #return
        
        arch_name = "SAC_"
        arch_name += "s_" if self.src_flag == "source" else "t_"
        arch_name = arch_name + "lr_" + str(learning_rate) + '_steps_' + str(timesteps)
        if self.use_udr:
            arch_name += "_UDR"

        #callback che salva checkpoint
        checkpoint_callback = CheckpointCallback(
                            save_freq=1000,  #salva ogni 20_000, se usiamo 20 vec env
                            save_path=self.checkpoint_dir,
                            name_prefix=self.src_flag,
                            save_replay_buffer=False,
                            save_vecnormalize=True,
                            )
        callbacks=[]
        if self.src_flag == "source":
            eval_target_callback = EvalOnTargetCallback(self.test_env,check_freq=100,
                                    log_file=os.path.join(self.log_dir,self.udr_prefix+"st_eval_log.csv"),
                                    n_eval_eps=10)
            callbacks.append(eval_target_callback)
        callbacks.append(checkpoint_callback)

        lr_schedule = linear_schedule(learning_rate) if lr_schedule == "linear" else learning_rate
        
        if self.model == None:
            policy = CnnPolicy if (self.use_vision and not self.use_pose_est) else MlpPolicy
            self.model = SAC(policy, self.train_env, verbose = 1,
                                learning_rate = lr_schedule,
                                learning_starts = 100,
                                buffer_size = buffer_size,
                                target_entropy = -3.0,
                                **hyperparams)
        
        custom_logger = configure(self.log_dir, ["csv"])
        self.model.set_logger(custom_logger)

        #self.train_env.reset()

        self.model.learn(total_timesteps = timesteps, log_interval = 10,callback=callbacks, reset_num_timesteps=False,progress_bar=True)
        #rinomina i file di log, usa flag source target del model
        custom_logger.close()
        os.rename(os.path.join(self.log_dir,"progress.csv"),os.path.join(self.log_dir,self.udr_prefix+self.src_flag+"_train_log.csv"))
        #os.rename(os.path.join(self.output_dir,"monitor.csv"),os.path.join(self.output_dir,self.src_flag+"_monitor.csv"))

        self.model.save(os.path.join(self.checkpoint_dir, arch_name))
        self.train_env.close()
        self.test_env.close()
        #elif exists(arch_name):
        #self.model = SAC.load(arch_name)
        #self.arch_name = arch_name  

    def test(self, n_eval = 50):
        if self.model!=None:
            #self.test_env.reset()
            mean_reward, std_reward = evaluate_policy(self.model, self.test_env, n_eval_episodes = n_eval, deterministic = True)
            print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            self.test_env.close()
            return mean_reward, std_reward
        else:
            print("Model not loaded!")
            return -1,-1

    def plot_results(self):
        plot_results(log_folder=self.output_dir,train_env=self.src_flag,udr_prefix=self.udr_prefix)

        #NON IN USO: serve a valutare su target_env se alleniamo su source e salvare il miglior modello
        #salva in file evaluations.npz che potremmo leggere per fare un altra linea del grafico
        #purtroppo rompe l'output del custom logger, dovremmo risolvere
        """
        eval_callback = EvalCallback(self.test_env,
                            best_model_save_path=self.checkpoint_dir,
                            log_path=os.path.join(self.output_dir,"callback_logs"),
                            eval_freq=10,
                            deterministic=True, render=False)
        """

    def make_env(self, base, test=False):
        num_cpu = os.cpu_count() or 1
        env_kwargs = dict(use_udr = self.use_udr and not test,
                          udr_type = self.udr_type,
                          udr_lb = self.udr_lb,
                          udr_ub = self.udr_ub,
                          n_distr = self.udr_ndistr,
                          udr_range = self.udr_range,
                          use_vision = self.use_vision and not (test and self.use_pose_est),
                          w = self.width,
                          h = self.height,
                          gray = self.gray,
                          use_pose_est = self.use_pose_est and not test,
                          pose_config = self.pose_config,
                          pose_checkpoint = self.pose_checkpoint)
        monitor_path = os.path.join(self.output_dir,self.udr_prefix+self.src_flag+"_monitor_log","monitor.csv") if not test else None

        if self.use_vec_env:
            env_array = [make_env(base,i,**env_kwargs) for i in range(num_cpu)]
        else:
            env = make_env(base,0,**env_kwargs)
        if self.use_vision:
            if self.use_pose_est:
                if self.use_vec_env:
                    return VecMonitor(
                        SubprocVecEnv(env_array),
                        monitor_path
                    )
                else:
                    return VecMonitor(
                        DummyVecEnv([env]),
                        monitor_path
                    )
            else:
                if self.use_vec_env:
                    return VecMonitor(
                        VecFrameStack(
                            SubprocVecEnv(env_array),
                            n_stack=2
                        ),
                        monitor_path
                    )
                else:
                    return VecMonitor(
                        VecFrameStack(
                            DummyVecEnv([env]),
                            n_stack=2
                        ),
                        monitor_path
                    )
        else:
            if self.use_vec_env:
                return VecMonitor(
                    SubprocVecEnv(env_array),
                    monitor_path)
            else:
                return Monitor(env(), monitor_path)

    def change_env(self, scope, base):
        if scope == "train":
            self.src_flag = "source" if base == "CustomHopper-source-v0" else "target"
            self.train_env_name = base
            self.train_env = self.make_env(base)
        else:
            self.test_env_name = base
            self.test_env = self.make_env(base, True)