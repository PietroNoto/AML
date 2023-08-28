import gym
from model import Model

import numpy as np
import matplotlib.pyplot as plt

from estimator import *

if __name__ == '__main__':
    checkpoint="trained_2M/source_2000000_steps.zip"
    source_env_name="CustomHopper-source-v0"
    target_env_name="CustomHopper-target-v0"
    
    episode_length = 500

    rend_env=gym.make(source_env_name) #DummyVecEnv([lambda:
    model = Model(source_env_name, target_env_name,output_dir="",vision=False)
    model.load_model(checkpoint)
    
    obs = rend_env.reset()
    img=rend_env.render("rgb_array",width=224, height=224)
    obs_history = np.ndarray((11, episode_length))
    reconstr_history = np.ndarray((11, episode_length))
    analyzer = ImgAnalyzer(5, "vision/hopper_config_extra.py", "/mnt/d/Download/epoch_100.pth")
    #test = MMPoseInferencer(
    #        pose2d= "vision/hopper_config_extra.py",
    #        pose2d_weights = "/mnt/d/Download/epoch_150.pth",
    #        det_model="whole_image",
    #        det_cat_ids=[1]
    #    )
    for i in range(episode_length):
        action, _states = model.model.predict(obs, deterministic=True)
        obs, rewards, dones, info = rend_env.step(action)
        obs_history[:, i] = np.array(obs)
        img=rend_env.render("rgb_array",width=224, height=224) #"rgb_array",
        pred = analyzer.produce_obs(img)
        reconstr_history[:, i] = np.array(pred)
    rend_env.close()

    #generator = test(img[...,::-1], show=True)
    #res = next(generator)

    labels = ["Z of top",
              "Angle of top",
              "Angle of thigh joint",
              "Angle of leg joint",
              "Angle of foot joint",
              "X velocity of top",
              "Z velocity of top",
              "Ang. vel. of top",
              "Ang. vel. of thigh hinge",
              "Ang. vel. of leg hinge",
              "Ang. vel. of foot hinge"]
    fig = plt.figure()
    fig.set_tight_layout(True)
    for i in range(11):
        gt = obs_history[i,:]
        est = reconstr_history[i, :]
        ax = fig.add_subplot(3, 4, i+1)
        ax.set_title(labels[i])
        ax.plot(range(episode_length), est)
        ax.plot(range(episode_length), gt)
    plt.show()