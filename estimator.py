from mmpose.apis import MMPoseInferencer
from math import atan2, pi
import numpy as np
from gym import Wrapper, spaces

class ImgAnalyzer:
    def __init__(self, n_points, mmpose_config, checkpoint):
        self._inferencer = MMPoseInferencer(
            pose2d = mmpose_config,
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
        return (coords[0], (224 - coords[1]-27)/131 * 1.2475)
 
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
        self._ang_vels = ((self._angles - self._old_angles)/(0.008)).clip(-15, 10)
        self._old_angles = self._angles
        self._top_coords = self.compute_top((points[0], points[1]))
        self._top_vel = (self._top_coords[0] - self._old_top_coords[0], self._top_coords[1] - self._old_top_coords[1])
        self._top_angle = parts[0]
        self._top_ang_vel = (self._top_angle - self._old_top_angle)/0.008
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
    
class PoseWrapper(Wrapper):

    def __init__(self, env, config, checkpoint):
        super().__init__(env)
        self.estimator = ImgAnalyzer(5, config, checkpoint)
        self.env.reset()
        self.width=224
        self.height=224

    def reset(self):
        super().reset()
        self.env.reset()
        img = self.env.render("rgb_array", width=self.width, height=self.height)
        obs = self.estimator.produce_obs(img)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        img = self.env.render("rgb_array", width=self.width, height=self.height)
        obs = self.estimator.produce_obs(img)
        return obs, reward, done, info 
