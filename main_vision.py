import gym
from CNN import VisionWrapper
from model import Model

from mmpose.apis import MMPoseInferencer
from math import atan2, pi

import numpy as np
import matplotlib.pyplot as plt

class ImgAnalyzer:
    def __init__(self, n_points, mmpose_config, checkpoint):
        self._inferencer = MMPoseInferencer(
            pose2d= mmpose_config,
            pose2d_weights = checkpoint,
            det_model="whole_image",
            det_cat_ids=[1]
        )
        self._n_points = n_points
        self._coldstart = True
        self._angles = np.zeros(n_points - 1)
        self._old_angles = np.zeros(n_points - 1)
        self._ang_vels = np.zeros(n_points - 1)
        self._top_coords = (0, 0)
        self._old_top_coords = (0, 0)
        self._top_vel = (0, 0)
        self._top_angle = 0
        self._old_top_angle = 0
        self._top_ang_vel = 0
    
    def compute_top(self, coords):
        return (coords[0], ((224 - coords[1]-27)/131 * 1.2475 * 6 - 5.9))
 
    def points_to_obs(self, points):
        parts = np.ndarray(self._n_points - 1)
        angles = np.ndarray(self._n_points - 2)
        
        for i in range(self._n_points-1):
            parts[i] = atan2(points[2*i+3]-points[2*i+1], points[2*i+2]-points[2*i]) - pi/2
        parts[i] += pi/2           
        for j in range(self._n_points-2):
            angles[j] = parts[j] - parts[j+1]
        if angles[j] > pi/3:
            angles[j] -= pi
        elif angles[j] < -pi/3:
            angles[j] += pi
        if (self._coldstart):
            self._old_angles = angles
            self._old_top_coords = self.compute_top((points[0], points[1]))
            self._old_top_angle = parts[0]
            self._coldstart = False
        self._angles = angles
        self._ang_vels = ((self._angles - self._old_angles)/(0.008)).clip(-10, 10)
        self._old_angles = self._angles
        self._top_coords = self.compute_top((points[0], points[1]))
        self._top_vel = (self._top_coords[0] - self._old_top_coords[0], self._top_coords[1] - self._old_top_coords[1])
        self._top_angle = parts[0]
        self._top_ang_vel = ((self._top_angle - self._old_top_angle)/(0.008 * 10)-13).clip(-10, 10)
        self._old_top_coords = self._top_coords
        return [self._top_coords[1],
                self._top_angle,
                *self._angles,
                *self._top_vel,
                self._top_ang_vel,
                *self._ang_vels]
    
    def produce_obs(self, img):
        generator = self._inferencer(img[...,::-1])
        res = next(generator)
        points = np.ravel(res["predictions"][0][0]["keypoints"])
        return self.points_to_obs(points)

if __name__ == '__main__':
    checkpoint="trained_2M/source_2000000_steps.zip"
    source_env_name="CustomHopper-source-v0"
    target_env_name="CustomHopper-target-v0"
    
    episode_length = 1000

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

    labels = ["z of top",
              "angle of top",
              "angle of thigh joint",
              "angle of leg joint",
              "angle of foot joint",
              "x velocity of top",
              "z velocity of top",
              "ang vel of top",
              "ang vel of thigh hinge",
              "ang vel of leg hinge",
              "ang vel of foot hinge"]
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