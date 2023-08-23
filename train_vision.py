from env.custom_hopper import *
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import CnnPolicy
from stable_baselines3.common.callbacks import BaseCallback

from train_udr import UDRCallback
from env.vision_wrapper import VisionWrapper

def main():
    env_name = 'CustomHopper-source-v0'
    env = VisionWrapper(gym.make(env_name))
    model = SAC(CnnPolicy, env, verbose=0, buffer_size=100, batch_size=64)
    udr_callback = UDRCallback()
    #model.learn(total_timesteps=50000, log_interval=10, callback=udr_callback, progress_bar=True)
    model.learn(total_timesteps=50000, log_interval=10, progress_bar=True)
    model.save("sac_vision_source_hopper")
    #allenamento 50_000: 3 ore e 4 minuti

if __name__ == "__main__":
    main()