from typing import List, Optional, Union
from collections import OrderedDict
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from stable_baselines3.common.vec_env import VecEnv
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv


class AgentViewEnv(OffScreenRenderEnv):
    """ Sparse reward environment with only the agent view as observation.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        agent_view_shape = self.env._get_observations()["agentview_image"].shape
        self.observation_space = Box(low=0, high=255, shape=agent_view_shape, dtype="uint8")
        self.action_space = Box(low=-1, high=1, shape=(7,), dtype="float32")
        self.step_count = 0

    def step(self, action):
        obs, reward, done, info = super().step(action)
        print(obs)
        success = self.check_success()
        reward = 10.0 * success
        self.step_count += 1
        truncated = self.step_count > 250
        done = success or truncated
        return obs["agentview_image"], reward, done, truncated, info

    def reset(self):
        obs = super().reset()
        self.step_count = 0
        return obs["agentview_image"]
    
class AgentEnv(OffScreenRenderEnv):
    """ Sparse reward environment with regular observation
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        agent_view_shape = self.env._get_observations()["agentview_image"].shape
        self.observation_space = Box(low=0, high=255, shape=agent_view_shape, dtype="uint8")
        self.action_space = Box(low=-1, high=1, shape=(7,), dtype="float32")
        self.step_count = 0

    def step(self, action):
        obs, reward, done, info = super().step(action)
        print(obs)
        success = self.check_success()
        reward = 10.0 * success
        self.step_count += 1
        truncated = self.step_count > 250
        done = success or truncated
        return obs, reward, done, truncated, info

    def reset(self):
        obs = super().reset()
        self.step_count = 0
        return obs


class AgentViewGoalEnv(OffScreenRenderEnv):
    """ Sparse reward environment with only the agent view as observation for HER.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # desired goal to close microwave
        close_threshold = 0.0 # taken from articulated_objects.py Microwave object default close ranges
        self.desired_goal = np.array([close_threshold])

        agent_view_shape = self.env._get_observations()["agentview_image"].shape
        self.observation_space = self._make_observation_space(agent_view_shape, self.desired_goal.shape)
        print(self.desired_goal.shape, self.observation_space)

        self.action_space = Box(low=-1, high=1, shape=(7,), dtype="float32")
        self.step_count = 0

    def _get_obs(self, obs):
        """
        Helper to create the observation
        """
        # Since object is microwave, we will only have one joint
        object = self.obj_of_interest # sometimes this is a list?
        object = "microwave_1"
        for joint in self.env.get_object(object).joints:
            qpos_addr = self.env.sim.model.get_joint_qpos_addr(joint)
            qpos = self.env.sim.data.qpos[qpos_addr]

        return OrderedDict(
            [
                ("observation", obs['agentview_image']),
                ("achieved_goal", np.array([qpos])),
                ("desired_goal", self.desired_goal),
            ]
        )
    
    def _make_observation_space(self, agent_view_shape, goal_shape):
        return gym.spaces.Dict(
            {
                "observation": Box(low=0, high=255, shape=agent_view_shape, dtype="uint8"),
                "achieved_goal": Box(low=-np.inf, high=np.inf, shape=goal_shape, dtype=np.float16),
                "desired_goal": Box(low=-np.inf, high=np.inf, shape=goal_shape, dtype=np.float16),
            }
        )

    def step(self, action):
        obs, reward, done, info = super().step(action)

        observation = self._get_obs(obs)
        success = self.check_success()
        reward = self.compute_reward(observation['achieved_goal'], observation['desired_goal'], _info = None)
        self.step_count += 1
        truncated = self.step_count > 250
        done = success or truncated
        info = {"is_success": done, "agentview_image": observation.get("observation")}
        return observation, reward, done, truncated, info

    def compute_reward(
            self, achieved_goal, desired_goal, _info = None
    ) -> np.float32:
        close_ranges = [-0.005, 0.0]
        close_range = np.array([abs(close_ranges[1] - close_ranges[0])])

        if desired_goal - close_range < achieved_goal < desired_goal + close_range:
            return 10.0
        else:
            return 0.0

    def reset(self):
        obs = super().reset()
        self.step_count = 0
        return self._get_obs(obs)
    

class LowDimensionalObsEnv(OffScreenRenderEnv):
    """ Sparse reward environment with all the low-dimensional states
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        obs = self.env._get_observations()
        low_dim_obs = self.get_low_dim_obs(obs)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=low_dim_obs.shape, dtype="float32")
        self.action_space = Box(low=-1, high=1, shape=(7,), dtype="float32")
        self.step_count = 0
    
    def get_low_dim_obs(self, obs):
        return np.concatenate([
            obs[k] for k in obs.keys() if not k.endswith("image")
        ], axis = -1)
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        success = self.check_success()
        reward = 10.0 * success
        self.step_count += 1
        truncated = self.step_count > 250
        done = success or truncated
        return self.get_low_dim_obs(obs), reward, done, truncated, info
    
    def reset(self):
        obs = super().reset()
        self.step_count = 0
        return self.get_low_dim_obs(obs)
    
class LowDimensionalObsGymEnv(gym.Env):
    """ Sparse reward environment with all the low-dimensional states
    """
    def __init__(self, **kwargs):
        self.env = OffScreenRenderEnv(**kwargs)
        obs = self.env.env._get_observations()
        low_dim_obs = self.get_low_dim_obs(obs)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=low_dim_obs.shape, dtype="float32")
        self.action_space = Box(low=-1, high=1, shape=(7,), dtype="float32")
        self.step_count = 0
    
    def get_low_dim_obs(self, obs):
        return np.concatenate([
            obs[k] for k in obs.keys() if not k.endswith("image")
        ], axis = -1)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        success = self.env.check_success()
        reward = 10.0 * success
        self.step_count += 1
        truncated = self.step_count >= 250
        done = success or truncated
        return self.get_low_dim_obs(obs), reward, done, truncated, info
    
    def reset(self, seed=None):
        obs = self.env.reset()
        self.step_count = 0
        return self.get_low_dim_obs(obs), {}
    
    def seed(self, seed=None):
        return self.env.seed(seed)
    

class GymVecEnvs(VecEnv):
    """ Vectorized environment for gymnasium environments
    """
    def __init__(self, envs: SubprocVectorEnv):
        self.envs = envs
        num_envs = len(envs)
        super().__init__(num_envs, envs.observation_space[0], envs.action_space[0])
        self.rewards = np.zeros(num_envs)

    def reset(self):
        return self.envs.reset()
    
    def step(self, actions):
        obs, rewards, dones, _, infos = self.envs.step(actions)
        id, *_ = np.where(dones)
        if len(id) > 0:
            obs_new = self.envs.reset(id=id)
            obs[id] = obs_new
        self.rewards = rewards
        return obs, rewards, dones, infos
    
    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        obs, rewards, dones, _, infos = self.envs.step(self.actions)
        id, *_ = np.where(dones)
        if len(id) > 0:
            obs_new = self.envs.reset(id=id)
            obs[id] = obs_new
        self.rewards = rewards
        return obs, rewards, dones, infos

    def close(self) -> None:
        return self.envs.close()
    
    def env_is_wrapped(self, wrapper_class, indices):
        return False
    
    def get_attr(self, attr_name, indices = None):
        return self.envs.get_env_attr(attr_name)
    
    def set_attr(self, attr_name, value, indices):
        self.envs.set_env_attr(attr_name, value, indices)

    def env_method(self, method_name: str, *method_args, indices: Optional[Union[int, List[int], np.ndarray]] = None, **method_kwargs):
        return super().env_method(method_name, *method_args, indices=indices, **method_kwargs)


class GymGoalEnv(gym.Env):
    '''
    Gym wrapper for agent goal environment (HER)
    '''
    def __init__(self, env: AgentEnv):
        super().__init__()

        self.env = env
        self.action_space = self.env.action_space

        close_threshold = 0.0 # taken from articulated_objects.py Microwave object default close ranges
        self.desired_goal = np.array([close_threshold])

        agent_view_shape = self.env.env._get_observations()["agentview_image"].shape
        self.observation_space = self._make_observation_space(agent_view_shape, self.desired_goal.shape)

    def _get_obs(self, obs):
        """
        Helper to create the observation
        """
        # Since object is microwave, we will only have one joint
        object = self.env.obj_of_interest # sometimes this is a list?
        object = "microwave_1"
        for joint in self.env.env.get_object(object).joints:
            qpos_addr = self.env.sim.model.get_joint_qpos_addr(joint)
            qpos = self.env.sim.data.qpos[qpos_addr]
        return OrderedDict(
            [
                ("observation", obs['agentview_image']),
                ("achieved_goal", np.array([qpos])),
                ("desired_goal", self.desired_goal),
            ]
        )
    
    def _make_observation_space(self, agent_view_shape, goal_shape):
        return gym.spaces.Dict(
            {
                "observation": Box(low=0, high=255, shape=agent_view_shape, dtype="uint8"),
                "achieved_goal": Box(low=-np.inf, high=np.inf, shape=goal_shape, dtype="float64"),
                "desired_goal": Box(low=-np.inf, high=np.inf, shape=goal_shape, dtype="float64"),
            }
        )

    def reset(self, seed=None, info={}):
        obs = self.env.reset()
        return self._get_obs(obs), info
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        observation = self._get_obs(obs)
        # reward = self.compute_reward(observation['achieved_goal'], observation['desired_goal'], _info = None)
        reward = float(self.compute_reward(observation["achieved_goal"], observation["desired_goal"], None).item())
        info = {"is_success": done, "agentview_image": observation.get("observation")}
        return observation, reward, done, truncated, info

    def close(self) -> None:
        return self.env.close()

    def compute_reward(self, achieved_goal, desired_goal, _info=None):
        close_ranges = [-0.005, 0.0]
        close_range = np.array([abs(close_ranges[1] - close_ranges[0])])
        
        batch_size = 1
        if len(achieved_goal) > 1 and len(desired_goal) > 1:
            # batch rewards
            batch_size = len(achieved_goal)
            achieved_goal = np.reshape(achieved_goal, batch_size)
            desired_goal = np.reshape(desired_goal, batch_size)

        if (desired_goal - close_range).all() < achieved_goal.all() < (desired_goal + close_range).all():
            return np.full(batch_size, 10.0)
        else:
            return np.full(batch_size, 0.0)
        
    def env_is_wrapped(self, wrapper_class, indices):
        return False
    
    def get_attr(self, attr_name, indices = None):
        return self.env.get_env_attr(attr_name)
    
    def set_attr(self, attr_name, value, indices):
        self.env.set_env_attr(attr_name, value, indices)

    def env_method(self, method_name: str, *method_args, indices: Optional[Union[int, List[int], np.ndarray]] = None, **method_kwargs):
        return super().env_method(method_name, *method_args, indices=indices, **method_kwargs)
