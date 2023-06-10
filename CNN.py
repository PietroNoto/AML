from gym import Wrapper, spaces

class VisionWrapper(Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.env.reset()
        dummy_obs = self.env.render("rgb_array", width=224, height=224)
        self._observation_space = spaces.Box(low=0, high=255, shape=dummy_obs.shape, dtype=dummy_obs.dtype)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs = self.env.render("rgb_array", width=224, height=224)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.env.render("rgb_array", width=224, height=224)
        return obs, reward, done, info