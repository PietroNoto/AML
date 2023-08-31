"""Implementation of the Hopper environment supporting
domain randomization optimization."""

import csv
from enum import Enum
import pdb
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv
from scipy.stats import truncnorm

class Udr(Enum):
    No = 0
    Finite = 1
    Infinite = 2

class CustomHopper(MujocoEnv, utils.EzPickle):

    def __init__(self, domain=None):

        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)

        self.original_masses = np.copy(self.sim.model.body_mass[1:])    # Default link masses
        self.useDomainRand = Udr.No

        if domain == 'source':  # Source environment has an imprecise torso mass (-1kg shift)
            self.sim.model.body_mass[1] -= 1.0

    def enable_finite_udr(self, lower_bound = 1, upper_bound = 5, n_distr = 3):
        self.useDomainRand = Udr.Finite
        self.random_masses = {k : np.random.uniform(lower_bound, upper_bound, n_distr) for k in range(3)}

    def enable_infinite_udr(self, range):
        self.useDomainRand = Udr.Infinite
        self.udr_var = range

    def disable_udr(self):
        self.useDomainRand = Udr.No
        self.sim.model.body_mass[1:] = self.original_masses

    def isUDR(self):
        return self.useDomainRand

    def set_random_parameters(self):
        self.set_parameters(self.sample_parameters())

    def sample_parameters(self):
        torso_mass = self.sim.model.body_mass[1]
        
        if self.useDomainRand == Udr.Finite:
            pos = np.random.randint(0, len(self.random_masses[0]))
            masses = [torso_mass] + [self.random_masses[k][pos] for k in self.random_masses]
        elif self.useDomainRand == Udr.Infinite:
            masses = [torso_mass] + [np.random.uniform(m-self.udr_var, m+self.udr_var) for m in self.original_masses[1:]]

        return masses

    def get_parameters(self):
        """Get value of mass for each part"""
        masses = np.array( self.sim.model.body_mass[1:] )
        return masses


    def set_parameters(self, task):
        """Set each part's mass to a new value"""
        self.sim.model.body_mass[1:] = task


    def step(self, a):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        """Get current state"""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

    def reset_model(self):
        """Reset the environment to a random initial state"""
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)

        if self.useDomainRand is not Udr.No:
            self.set_random_parameters()

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20



"""
    Registered environments
"""
gym.envs.register(
        id="CustomHopper-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
)

gym.envs.register(
        id="CustomHopper-source-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.envs.register(
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)