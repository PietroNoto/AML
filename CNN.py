from gym import Wrapper, spaces
import cv2
import numpy as np

class VisionWrapper(Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.env.reset()
        self.width=224
        self.height=224
        dummy_obs = self.env.render("rgb_array", width=self.width, height=self.height)
        dummy_obs=cv2.cvtColor(dummy_obs, cv2.COLOR_RGB2GRAY)
        dummy_obs=np.expand_dims(dummy_obs, axis=-1)
        self._observation_space = spaces.Box(low=0, high=255, shape=dummy_obs.shape, dtype=dummy_obs.dtype)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs = self.env.render("rgb_array", width=self.width, height=self.height)
        obs=cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs=np.expand_dims(obs, axis=-1)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.env.render("rgb_array", width=self.width, height=self.height)
        obs=cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs=np.expand_dims(obs, axis=-1)
        return obs, reward, done, info

#cose da provare
# - dimensioni dell'immagine [224,84,...]
# - normalizzazione [0-1]
# - grayscale ?
# - stacking, first last dimension