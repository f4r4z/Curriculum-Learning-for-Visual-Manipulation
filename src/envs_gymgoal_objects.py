from typing import List, Optional, Union
from collections import OrderedDict
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict

from stable_baselines3.common.vec_env import VecEnv
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv

class AgentViewGymGoalStoveEnv(gym.Env):
    """ Sparse reward environment with image observations
    """
    def __init__(self, **kwargs):
        self._env = OffScreenRenderEnv(**kwargs)
        self.obj_of_interest = self._env.obj_of_interest[0]  # hardcoded for now
        obs_shape = self._env.env._get_observations()["agentview_image"].shape
        achieved_goal = self.get_achieved_goal()
        goal_shape = achieved_goal.shape

        self.desired_goal = np.full(goal_shape, 2.1)  # this is hardcoded and only works for stove task
        print("desired goal ", self.desired_goal)

        self.observation_space = Dict({
            "observation": Box(low=0, high=255, shape=obs_shape, dtype="uint8"),
            "desired_goal": Box(low=-np.inf, high=np.inf, shape=goal_shape, dtype="float32"),
            "achieved_goal": Box(low=-np.inf, high=np.inf, shape=goal_shape, dtype="float32")
        })
        self.action_space = Box(low=-1, high=1, shape=(7,), dtype="float32")

        self.step_count = 0

    def get_achieved_goal(self):
        qposs = []
        for joint in self._env.env.get_object(self.obj_of_interest).joints:
            qpos_addr = self._env.env.sim.model.get_joint_qpos_addr(joint)
            qpos = self._env.sim.data.qpos[qpos_addr]
            qposs.append(qpos)
        return np.array(qposs)
    
    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        success = self._env.check_success()
        reward = 10.0 * success
        self.step_count += 1
        truncated = self.step_count >= 250
        done = success or truncated
        info["agentview_image"] = obs["agentview_image"]
        return \
            {   
                "observation": obs["agentview_image"],
                "desired_goal": self.desired_goal,
                "achieved_goal": self.get_achieved_goal()
            }, reward, done, truncated, info
    
    def reset(self, seed=None):
        obs = self._env.reset()
        self.step_count = 0
        return \
            {   
                "observation": obs["agentview_image"],
                "desired_goal": self.desired_goal,
                "achieved_goal": self.get_achieved_goal()
            }, {}
    
    def seed(self, seed=None):
        return self._env.seed(seed)
    
    def compute_reward(
        self, achieved_goal, desired_goal, _info = None
    ) -> np.float32:
        on_tolerance = 2.1 - 0.5
        on_tolerance_array = np.full(achieved_goal.shape, on_tolerance)
        return (np.abs(desired_goal - achieved_goal) < on_tolerance_array) * 10.0
    
class AgentViewGymGoalWhiteCabinetEnv(gym.Env):
    """ Sparse reward environment with image observations
    """
    def __init__(self, **kwargs):
        self._env = OffScreenRenderEnv(**kwargs)
        self.obj_of_interest = self._env.obj_of_interest[0]  # hardcoded for now
        obs_shape = self._env.env._get_observations()["agentview_image"].shape
        achieved_goal = self.get_achieved_goal()
        goal_shape = achieved_goal.shape

        self.desired_goal = np.full(goal_shape, 0.005)  # this is hardcoded and only works for stove task
        print("desired goal ", self.desired_goal)

        self.observation_space = Dict({
            "observation": Box(low=0, high=255, shape=obs_shape, dtype="uint8"),
            "desired_goal": Box(low=-np.inf, high=np.inf, shape=goal_shape, dtype="float32"),
            "achieved_goal": Box(low=-np.inf, high=np.inf, shape=goal_shape, dtype="float32")
        })
        self.action_space = Box(low=-1, high=1, shape=(7,), dtype="float32")

        self.step_count = 0

    def get_achieved_goal(self):
        qposs = []
        for joint in self._env.env.get_object(self.obj_of_interest).joints:
            qpos_addr = self._env.env.sim.model.get_joint_qpos_addr(joint)
            qpos = self._env.sim.data.qpos[qpos_addr]
            qposs.append(qpos)
        return np.array(qposs)
    
    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        success = self._env.check_success()
        reward = 10.0 * success
        self.step_count += 1
        truncated = self.step_count >= 250
        done = success or truncated
        info["agentview_image"] = obs["agentview_image"]
        return \
            {   
                "observation": obs["agentview_image"],
                "desired_goal": self.desired_goal,
                "achieved_goal": self.get_achieved_goal()
            }, reward, done, truncated, info
    
    def reset(self, seed=None):
        obs = self._env.reset()
        self.step_count = 0
        return \
            {   
                "observation": obs["agentview_image"],
                "desired_goal": self.desired_goal,
                "achieved_goal": self.get_achieved_goal()
            }, {}
    
    def seed(self, seed=None):
        return self._env.seed(seed)
    
    def compute_reward(
        self, achieved_goal, desired_goal, _info = None
    ) -> np.float32:
        on_tolerance = 0.005 - 0.0
        on_tolerance_array = np.full(achieved_goal.shape, on_tolerance)
        return (np.abs(desired_goal - achieved_goal) < on_tolerance_array) * 10.0