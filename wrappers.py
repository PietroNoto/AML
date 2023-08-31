from gym import Wrapper, spaces
from gym.wrappers.pixel_observation import PixelObservationWrapper
import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer
from math import atan2, pi

class VisionWrapper(Wrapper):

    def __init__(self, env, w, h, gray):
        super().__init__(env)
        self.env.reset()
        self.width=w
        self.height=h
        self.grayscale=gray
        dummy_obs = self.env.render("rgb_array", width=self.width, height=self.height)
        if (self.grayscale):
            dummy_obs=cv2.cvtColor(dummy_obs, cv2.COLOR_RGB2GRAY)
            dummy_obs=np.expand_dims(dummy_obs, axis=-1)
        self._observation_space = spaces.Box(low=0, high=255, shape=dummy_obs.shape, dtype=dummy_obs.dtype)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs = self.env.render("rgb_array", width=self.width, height=self.height)
        if (self.grayscale):
            obs=cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs=np.expand_dims(obs, axis=-1)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.env.render("rgb_array", width=self.width, height=self.height)
        if (self.grayscale):
            obs=cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs=np.expand_dims(obs, axis=-1)
        return obs, reward, done, info

"""
class VisionWrapper(PixelObservationWrapper):
    def __init__(self, env, w, h, gray):
        super().__init__(env, True, dict(width=w, height=h))
        self.grayscale = gray
        shape = (w, h, 3) if not gray else (w, h, 1)
        self._observation_space = spaces.Box(low=0, high=255, shape=shape, dtype=np.int8)
    
    def observation(self, observation):
        obs = super().observation(observation)["pixels"]
        if (self.grayscale):
            gr=cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs=np.expand_dims(gr, axis=-1)
        return obs
"""        

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
    
class PoseWrapper(PixelObservationWrapper):
    
    def __init__(self, env, config, checkpoint):
        super().__init__(env, True, dict(width=224, height=224))
        self.estimator = ImgAnalyzer(5, config, checkpoint)
        self.env.reset()
        self._observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,))
    
    def observation(self, observation):
        return self.estimator.produce_obs(super().observation(observation)['pixels'])