from typing import List, Optional, Union, Tuple
import typing
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict


from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.objects.articulated_objects import Microwave, SlideCabinet, Window, Faucet, BasinFaucet, ShortCabinet, ShortFridge, WoodenCabinet, WhiteCabinet, FlatStove

from src.dense_reward import DenseReward
import datetime
import os
import h5py

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

import src.patch

class MapObjects():
    '''
    maps object name to its libero articulated objects
    '''
    def __init__(self, obj_name, instruction):
        self.articulated_object = None
        self.instruction = instruction
        if "microwave" in obj_name:
            self.articulated_object = Microwave()
        elif "slide_cabinet" in obj_name:
            self.articulated_object = SlideCabinet()
        elif "window" in obj_name:
            self.articulated_object = Window()
        elif "faucet" in obj_name:
            self.articulated_object = Faucet()
        elif "basin_facet" in obj_name:
            self.articulated_object = BasinFaucet()
        elif "short_cabinet" in obj_name:
            self.articulated_object = ShortCabinet()
        elif "short_fridge" in obj_name:
            self.articulated_object = ShortFridge()
        elif "wooden_cabinet" in obj_name:
            self.articulated_object = WoodenCabinet()
        elif "white_cabinet" in obj_name:
            self.articulated_object = WhiteCabinet()
        elif "flat_stove" in obj_name:
            self.articulated_object = FlatStove()

    def define_goal(self):
        if "open" in self.instruction:
            goal_ranges = self.articulated_object.object_properties["articulation"]["default_open_ranges"]
            goal_value = min(goal_ranges) if min(goal_ranges) < max(self.articulated_object.object_properties["articulation"]["default_close_ranges"]) else max(goal_ranges)
        elif "close" in self.instruction:
            goal_ranges = self.articulated_object.object_properties["articulation"]["default_close_ranges"]
            goal_value = min(goal_ranges) if min(goal_ranges) < max(self.articulated_object.object_properties["articulation"]["default_open_ranges"]) else max(goal_ranges)
        elif "turn on" in self.instruction:
            goal_ranges = self.articulated_object.object_properties["articulation"]["default_turnon_ranges"]
            goal_value = min(goal_ranges) if min(goal_ranges) < max(self.articulated_object.object_properties["articulation"]["default_turnoff_ranges"]) else max(goal_ranges)
        elif "turn off" in self.instruction:
            goal_ranges = self.articulated_object.object_properties["articulation"]["default_turnoff_ranges"]
            goal_value = min(goal_ranges) if min(goal_ranges) < max(self.articulated_object.object_properties["articulation"]["default_turnon_ranges"]) else max(goal_ranges)
        else:
            # this should error out
            goal_ranges = None
            goal_value = None
        
        return goal_value, goal_ranges
    
class LowDimensionalObsGymEnv(gym.Env):
    """ Sparse or dense reward environment with all the low-dimensional states
    """
    def __init__(
        self, 
        is_shaping_reward: bool, 
        sparse_reward: float, 
        reward_geoms: Optional[List[str]] = None, 
        dense_reward_multiplier: float = 1.0, 
        steps_per_episode=250, 
        setup_demo=None,
        verbose=1, # 
        **kwargs
    ):
        """
        Args:
            is_shaping_reward (bool): whether to use shaping/dense rewards
            sparse_reward (float): the amount of sparse reward on success. If zero, will not check for success
            reward_geoms (list[str] | None): optional list of geoms to use for certain predicates (e.g. reach, grasp)
            dense_reward_multiplier (float): multiplier applied to the dense reward
            steps_per_episode (int): truncate the episode if the number of steps exceeds this
            setup_demo (str): path to a demo directory (containing a demo.hdf5) to run before each episode
            verbose (int): verbosity of print output
                - 0: no prints
                - 1: a few permanent lines
                - 2: prints on every episode
                - 3: multiple prints throughout each episode
                - 4: prints on every step
        """

        assert is_shaping_reward or sparse_reward > 0, "Must use at least one of shaping or sparse rewards"

        self.env = OffScreenRenderEnv(**kwargs)
        obs = self.env.env._get_observations()
        low_dim_obs = self.get_low_dim_obs(obs)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=low_dim_obs.shape, dtype="float32")
        self.action_space = Box(low=-1, high=1, shape=(7,), dtype="float32")
        self.step_count = 0
        self.goal_states = self.env.env.parsed_problem["goal_state"]
        self.dense_reward_multiplier = dense_reward_multiplier
        self.step_count_tracker = 0
        self.images = []
        self.sparse_reward = sparse_reward
        self.steps_per_episode = steps_per_episode
        self.verbose = verbose

        # for multi-goal tasks
        self.current_goal_index = 0

        # for now, we will focus on objects with one goal state
        if is_shaping_reward:
            self.shaping_reward: typing.Dict[Tuple, DenseReward] = {}
            if self.verbose >= 1: print("using dense reward")
            if self.verbose >= 1: print("goal_states:")
            for goal_state in self.goal_states:
                state_tuple = tuple(goal_state)
                if self.verbose >= 1: print(goal_state)
                # reward geoms will be set through dense reward
                self.shaping_reward[state_tuple] = DenseReward(self.env.env, goal_state, reward_geoms=reward_geoms, verbose=self.verbose)
        else:
            self.env.env.reward_geoms = reward_geoms
            self.shaping_reward = {}

        if self.verbose >= 1:
            if is_shaping_reward:
                print("using shaping rewards:", self.shaping_reward)
            if self.sparse_reward > 0:
                print("using sparse rewards:", self.goal_states)
        
        # setup actions from demo
        if setup_demo is None:
            self.setup_actions = np.array([])
        else:
            hdf5_path = os.path.join(setup_demo, "demo.hdf5")
            f = h5py.File(hdf5_path, "r")
            self.setup_actions = f['data/demo_1/actions'][:]
            if self.verbose >= 1: print("loaded setup actions with length", len(self.setup_actions))

    
    def get_low_dim_obs(self, obs):
        return np.concatenate([
            obs[k] for k in obs.keys() if not k.endswith("image")
        ], axis = -1)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # grip_pos = obs['robot0_gripper_qpos'][0]
        # print("gripper is", "closed" if grip_pos < 0.01 else "open" if grip_pos > 0.03 else "-")

        # sparse completion reward
        if self.sparse_reward:
            success = self.env.check_success()
        else:
            success = False
        reward = 0.0
        if success:
            reward = self.sparse_reward
        elif len(self.shaping_reward) == 0:
            # if not using dense reward, only check sparse predicate
            state = self.goal_states[self.current_goal_index] # complete multiple goals in order
            state_tuple = tuple(state)
            result = self.env.env._eval_predicate(state) # FIXME: would this be an extra call to the predicates, since check_success() was called earlier?
            if result:
                reward += self.sparse_reward / 10.0
                if self.current_goal_index + 1 < len(self.goal_states):
                    self.current_goal_index += 1
                    if self.verbose >= 3: print("current goal index: ", self.current_goal_index)
        elif len(self.shaping_reward) > 0:
            # when using dense reward, check sparse predicate and add dense reward
            state = self.goal_states[self.current_goal_index] # complete multiple goals in order
            state_tuple = tuple(state)
            result = self.env.env._eval_predicate(state)
            if result:
                if self.verbose >= 3: print(f"achieved {state_tuple}")
                reward += self.sparse_reward / 10.0
                if self.current_goal_index + 1 < len(self.goal_states):
                    self.current_goal_index += 1
                    if self.verbose >= 3: print("current goal index: ", self.current_goal_index)
            else:
                dense_reward_object = self.shaping_reward[state_tuple]
                if self.current_goal_index == len(self.goal_states) - 1:
                    reward += self.dense_reward_multiplier * dense_reward_object.dense_reward(step_count=self.step_count)
                else:
                    reward += dense_reward_object.dense_reward(step_count=self.step_count)

        # small reward for a task remaining in complete mode
        if len(self.goal_states) > 1:
            for state in self.goal_states:
                if self.env.env._eval_predicate(state):        
                    if self.verbose >= 4: print("small reward for state: ", state)
                    reward += self.sparse_reward / 10000.0

        # logistics
        if self.verbose >= 4: print(f"reward at step {self.step_count}: {reward}")
        self.step_count += 1
        truncated = self.step_count >= self.steps_per_episode
        done = success or truncated
        if done and self.verbose >= 2:
            print("done. success:", success)
        info["agentview_image"] = obs["agentview_image"]
        info["is_success"] = success

        return self.get_low_dim_obs(obs), reward, done, truncated, info
    
    def reset(self, seed=None):
        obs = self.env.reset()
        
        if len(self.setup_actions) > 0:
            if self.verbose >= 2: print("running setup actions")
            for action in self.setup_actions:
                obs, _, _, _ = self.env.step(action)

        self.step_count = 0
        self.current_goal_index = 0
        return self.get_low_dim_obs(obs), {}
    
    def seed(self, seed=None):
        return self.env.seed(seed)

    def get_bodies_and_geoms(self):
        if "ketchup_1" in self.env.obj_of_interest:
            body_main = "ketchup_1_main"
            geom_names = [
                "ketchup_1_main", 
            ]
        else:
            body_main = "wooden_cabinet_1_cabinet_bottom"
            geom_names = [
                "wooden_cabinet_1_g40",
                "wooden_cabinet_1_g41",
                "wooden_cabinet_1_g42"
            ]

        geom_names = []
        for i in range(self.env.sim.model.ngeom):
            geom_name = self.env.sim.model.geom_id2name(i)
            # print(f"Geom ID {i}: {geom_name}")
            geom_pos = self.env.sim.model.geom_pos[i]
            geom_size = self.env.sim.model.geom_size[i]
            # print(f"Geom Position: {geom_pos}, Size: {geom_size}")
            for obj in self.env.obj_of_interest:
                if geom_name and obj in geom_name:
                    geom_names.append(geom_name)
        body_mains = [obj + "_main" for obj in self.env.obj_of_interest]

        return body_mains, geom_names

    def current_joint_position(self):
        qposs = []
        for joint in self.env.env.get_object(self.env.obj_of_interest[0]).joints:
            qpos_addr = self.env.env.sim.model.get_joint_qpos_addr(joint)
            qpos = self.env.sim.data.qpos[qpos_addr]
            qposs.append(qpos)
        return np.array(qposs)

    
class LowDimensionalObsGymGoalEnv(gym.Env):
    """ Sparse reward environment with all the low-dimensional states with HER
    """
    def __init__(self, verbose=1, **kwargs):
        self._env = OffScreenRenderEnv(**kwargs)
        self.obj_of_interest = self._env.obj_of_interest[0]  # hardcoded for now
        self.instruction = self._env.language_instruction
        obs = self._env.env._get_observations()
        low_dim_obs = self.get_low_dim_obs(obs)
        achieved_goal = self.get_achieved_goal()
        goal_shape = achieved_goal.shape

        goal_value, self.goal_ranges = MapObjects(self.obj_of_interest, self.instruction).define_goal()
        
        print(f"desired goal value for task {self.instruction} with object {self.obj_of_interest} is {goal_value} with tolerance {max(self.goal_ranges) - min(self.goal_ranges)}")
        self.desired_goal = np.full(goal_shape, goal_value)

        # if "flat_stove" in self.obj_of_interest:
        #     print("HER for flat stove")
        #     self.desired_goal = np.full(goal_shape, -0.005)
        # elif "microwave" in self.obj_of_interest or "white_cabinet" in self.obj_of_interest:
        #     print("HER for microwave or white cabinet")
        #     self.desired_goal = np.zeros_like(achieved_goal)  # this is hardcoded and only works for microwave task

        self.observation_space = Dict({
            "observation": Box(low=-np.inf, high=np.inf, shape=low_dim_obs.shape, dtype="float32"),
            "desired_goal": Box(low=-np.inf, high=np.inf, shape=goal_shape, dtype="float32"),
            "achieved_goal": Box(low=-np.inf, high=np.inf, shape=goal_shape, dtype="float32")
        })
        self.action_space = Box(low=-1, high=1, shape=(7,), dtype="float32")
        self.step_count = 0
        self.episode_count = 0

        # logging
        self.images = []
        self.step_count_tracker = 0

    def get_low_dim_obs(self, obs):
        return np.concatenate([
            obs[k] for k in obs.keys() if not k.endswith("image")
        ], axis = -1)

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
        # truncated = False # added in order not to truncate
        done = success or truncated

        # always truncate first episode for learning starts so HER can sample
        # if self.episode_count == 0 and self.step_count >= 250:
        #     truncated = True
        #     done = True
        if done:
            self.episode_count += 1

        info["agentview_image"] = obs["agentview_image"]
        info["is_success"] = success

        return \
            {   
                "observation": self.get_low_dim_obs(obs),
                "desired_goal": self.desired_goal,
                "achieved_goal": self.get_achieved_goal()
            }, reward, done, truncated, info
    
    def reset(self, seed=None):
        obs = self._env.reset()
        self.episode_count = 0
        self.step_count = 0
        return \
            {   
                "observation": self.get_low_dim_obs(obs),
                "desired_goal": self.desired_goal,
                "achieved_goal": self.get_achieved_goal()
            }, {}
    
    def seed(self, seed=None):
        return self._env.seed(seed)
    
    def compute_reward(
        self, achieved_goal, desired_goal, _info = None
    ) -> np.float32:
        # batch instance
        if achieved_goal.ndim > 1:
            tolerance = max(self.goal_ranges) - min(self.goal_ranges)
            return (np.linalg.norm(achieved_goal - desired_goal, axis=1) < tolerance) * 10.0
        else:
            tolerance = max(self.goal_ranges) - min(self.goal_ranges)
            return (np.linalg.norm(achieved_goal - desired_goal, axis=0) < tolerance) * 10.0

    

class AgentViewGymEnv(gym.Env):
    """ Sparse reward environment with image observations
    """
    def __init__(self, verbose=1, **kwargs):
        self._env = OffScreenRenderEnv(**kwargs)
        obs_shape = self._env.env._get_observations()["agentview_image"].shape

        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype="uint8")
        self.action_space = Box(low=-1, high=1, shape=(7,), dtype="float32")

        self.custom_attr = {"total_reward": 0, "reward": 0} # custom attributes for tensorboard logging
        self.step_count = 0
    
    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        success = self._env.check_success()
        reward = 10.0 * success
        self.step_count += 1
        truncated = self.step_count >= 250
        done = success or truncated
        info["agentview_image"] = obs["agentview_image"]
        info["is_success"] = success
        self.custom_attr["reward"] = reward
        self.custom_attr["total_reward"] = self.custom_attr["total_reward"] + reward
        return obs["agentview_image"], reward, done, truncated, info
    
    def reset(self, seed=None):
        obs = self._env.reset()
        self.step_count = 0
        self.custom_attr = {"total_reward": 0, "reward": 0}
        return obs["agentview_image"], {}
    
    def seed(self, seed=None):
        return self._env.seed(seed)
    
class AgentViewSimpleGymEnv(gym.Env):
    """ Sparse reward environment with image observations but much simpler tasks [For testing only]
    """
    def __init__(self, verbose=1, **kwargs):
        self._env = OffScreenRenderEnv(**kwargs)
        obs_shape = self._env.env._get_observations()["agentview_image"].shape

        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype="uint8")
        self.action_space = Box(low=-1, high=1, shape=(7,), dtype="float32")

        self.custom_attr = {"total_reward": 0, "reward": 0} # custom attributes for tensorboard logging
        self.step_count = 0
    
    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        reward = self.compute_reward(self.get_achieved_goal())
        success = 0 if reward == 0 else 1
        
        self.step_count += 1
        truncated = self.step_count >= 250
        done = success or truncated
        info["agentview_image"] = obs["agentview_image"]
        self.custom_attr["reward"] = reward
        self.custom_attr["total_reward"] = self.custom_attr["total_reward"] + reward
        return obs["agentview_image"], reward, done, truncated, info

    def get_achieved_goal(self):
        qposs = []
        self.obj_of_interest = self._env.obj_of_interest[0]
        for joint in self._env.env.get_object(self.obj_of_interest).joints:
            qpos_addr = self._env.env.sim.model.get_joint_qpos_addr(joint)
            qpos = self._env.sim.data.qpos[qpos_addr]
            qposs.append(qpos)
        return np.array(qposs)
    
    def compute_reward(
        self, achieved_goal, _info = None
    ) -> np.float32:
        desired_goal = np.zeros_like(achieved_goal)
        close_tolerance = 1.000
        close_tolerance_array = np.full(achieved_goal.shape, close_tolerance)
        return (np.abs(achieved_goal - desired_goal) < close_tolerance_array) * 10.0
    
    def reset(self, seed=None):
        obs = self._env.reset()
        self.step_count = 0
        self.custom_attr = {"total_reward": 0, "reward": 0}
        return obs["agentview_image"], {}
    
    def seed(self, seed=None):
        return self._env.seed(seed)


class AgentViewGymGoalEnv(gym.Env):
    """ Sparse reward environment with image observations
    """
    def __init__(self, verbose=1, **kwargs):
        self._env = OffScreenRenderEnv(**kwargs)
        self.obj_of_interest = self._env.obj_of_interest[0]  # hardcoded for now
        obs_shape = self._env.env._get_observations()["agentview_image"].shape
        achieved_goal = self.get_achieved_goal()
        goal_shape = achieved_goal.shape
        self.desired_goal = np.zeros_like(achieved_goal)  # this is hardcoded and only works for microwave task

        self.observation_space = Dict({
            "observation": Box(low=0, high=255, shape=obs_shape, dtype="uint8"),
            "desired_goal": Box(low=-np.inf, high=np.inf, shape=goal_shape, dtype="float32"),
            "achieved_goal": Box(low=-np.inf, high=np.inf, shape=goal_shape, dtype="float32")
        })
        self.action_space = Box(low=-1, high=1, shape=(7,), dtype="float32")

        self.custom_attr = {"total_reward": 0, "reward": 0} # custom attributes for tensorboard logging
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
        info["is_success"] = success
        self.custom_attr["reward"] = reward
        self.custom_attr["total_reward"] += self.custom_attr["total_reward"]
        return \
            {   
                "observation": obs["agentview_image"],
                "desired_goal": self.desired_goal,
                "achieved_goal": self.get_achieved_goal()
            }, reward, done, truncated, info
    
    def reset(self, seed=None):
        obs = self._env.reset()
        self.step_count = 0
        self.custom_attr = {"total_reward": 0, "reward": 0}
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
        close_tolerance = 0.005 # hard coded for microwave task
        # close_tolerance_array = np.full(achieved_goal.shape, close_tolerance)
        # return (np.abs(achieved_goal - desired_goal) < close_tolerance_array) * 10.0
        return (np.linalg.norm(achieved_goal - desired_goal, axis=1) < close_tolerance) * 10.0