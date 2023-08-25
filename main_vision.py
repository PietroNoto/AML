import gym
from CNN import VisionWrapper
from model import Model

from mmpose.apis import MMPoseInferencer
from math import atan, atan2, pi

import numpy as np
from PIL import Image

class PointTranslator:
    def __init__(self, n_points):
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
        return (0, ((224 - coords[1]-27)/131 * 1.2475))
 
    def points_to_obs(self, points):
        parts = np.ndarray(self._n_points - 1)
        angles = np.ndarray(self._n_points - 2)
        
        for i in range(self._n_points-1):
            parts[i] = atan2(points[2*i+3]-points[2*i+1], points[2*i+2]-points[2*i]) - pi/2
            
        print(parts)
            
        for j in range(self._n_points-3):
            angles[j] = parts[j] - parts[j+1]
        if parts[j+1] < 0:
            angles[j] = parts[j] - parts[j+1] + pi
        else:
            angles[j] = parts[j] - parts[j+1]
        if (self._coldstart):
            self._old_angles = angles
            self._old_top_coords = self.compute_top((points[0], points[1]))
            self._old_top_angle = parts[0]
            self._coldstart = False
        self._angles = angles
        self._ang_vels = self._angles - self._old_angles
        self._old_angles = self._angles
        self._top_coords = self.compute_top((points[0], points[1]))
        self._top_vel = (self._top_coords[0] - self._old_top_coords[0], self._top_coords[1] - self._old_top_coords[1])
        self._top_angle = parts[0]
        self._top_ang_vel = self._top_angle - self._old_top_angle
        self._old_top_coords = self._top_coords
        return [self._top_coords[1],
                self._top_angle,
                *self._angles,
                *self._top_vel,
                self._top_ang_vel,
                *self._ang_vels]
    

if __name__ == '__main__':

    inferencer = MMPoseInferencer(
        pose2d= "vision/hopper_config_extra.py",
        pose2d_weights ="work_dirs/hopper_config_extra/epoch_30.pth",
        det_model="whole_image",
        det_cat_ids=[1]
    )
    
    checkpoint="trained_2M/source_2000000_steps.zip"
    source_env_name="CustomHopper-source-v0"
    target_env_name="CustomHopper-target-v0"
    n_test_eps=50
    use_vision=False

    rend_env=gym.make(source_env_name) #DummyVecEnv([lambda:
    model = Model(source_env_name, target_env_name,output_dir="",vision=False)
    model.load_model(checkpoint)
    
    obs = rend_env.reset()
    img=rend_env.render("rgb_array",width=224, height=224)
    
    translator = PointTranslator(5)
    for i in range(75):
    
        action, _states = model.model.predict(obs, deterministic=True)
        obs, rewards, dones, info = rend_env.step(action)

        #rend_env.render()
        img=rend_env.render("rgb_array",width=224, height=224) #"rgb_array",
        img = img[...,::-1]
        generator = inferencer(img)
        res = next(generator)
        points = np.ravel(res["predictions"][0][0]["keypoints"])
        pred = translator.points_to_obs(points)
        if (i == 74):
            Image.fromarray(img).save("env.png")
            print([round(o, 2) for o in obs])
            print([round(p, 2) for p in pred])
            print([round(obs[i] - pred[i], 2) for i in range(len(obs))])
    rend_env.close()