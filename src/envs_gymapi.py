from typing import List, Optional, Union
from collections import OrderedDict
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict

import torch
from stable_baselines3.common.vec_env import VecEnv
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from libero.libero.envs.objects import OBJECTS_DICT
from libero.libero.envs.objects.articulated_objects import Microwave, SlideCabinet, Window, Faucet, BasinFaucet, ShortCabinet, ShortFridge, WoodenCabinet, WhiteCabinet, FlatStove

from src.rnd import RNDNetworkLowDim
import datetime

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

import imageio
from IPython.display import HTML

import src.patch

def obs_to_video(images, filename):
    """
    converts a list of images to video and writes the file
    """
    video_writer = imageio.get_writer(filename, fps=60)
    for image in images:
        video_writer.append_data(image[::-1])
    video_writer.close()
    HTML("""
        <video width="640" height="480" controls>
            <source src="output.mp4" type="video/mp4">
        </video>
        <script>
            var video = document.getElementsByTagName('video')[0];
            video.playbackRate = 2.0; // Increase the playback speed to 2x
            </script>    
    """)

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
    """ Sparse reward environment with all the low-dimensional states
    """
    def __init__(self, **kwargs):
        self.env = OffScreenRenderEnv(**kwargs)
        obs = self.env.env._get_observations()
        low_dim_obs = self.get_low_dim_obs(obs)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=low_dim_obs.shape, dtype="float32")
        self.action_space = Box(low=-1, high=1, shape=(7,), dtype="float32")
        self.step_count = 0
        
        self.step_count_tracker = 0
        self.images = []

        # reward initials
        self.body_main, self.geom_names = self.get_bodies_and_geoms()
        try:
            self.initial_joint_position = self.current_joint_position()
        except:
            print("Failed to get initial joint position")
        self.initial_height = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(self.body_main)][2]
    
    def get_low_dim_obs(self, obs):
        return np.concatenate([
            obs[k] for k in obs.keys() if not k.endswith("image")
        ], axis = -1)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # sparse completion reward
        success = self.env.check_success()
        
        # define which rewards to use (temporary)
        reaching = False
        contact = False
        grasp = False
        height = False
        open_ = False

        goal_state = self.env.env.parsed_problem["goal_state"]
        for state in goal_state:
            if "reach" in state:
                print(f"{state} reward: ", reward)
                shaping_reward = self.reaching_reward(self.body_main)
                reward += shaping_reward
            if "open" in state:
                print(f"{state} reward: ", reward)
                shaping_reward = self.open_reward()
                reward += shaping_reward
            if "up" in state:
                print(f"{state} reward: ", reward)
                shaping_reward = self.lift_reward(self.body_main)
                reward += shaping_reward

        # reward = 0.0
        if success:
            reward = 100.0 * success
        """
        else:
            # get body and geom names
            body_main, geom_names = self.get_bodies_and_geoms()

            # reaching
            if reaching:
                reaching_reward = self.reaching_reward(body_main)
                print("reach", reaching_reward)
                reward += reaching_reward

            # contact
            if contact:
                contact_reward = self.contact_reward(geom_names)
                print("contact", contact_reward)
                reward += contact_reward
                # experimental
                if contact_reward:
                    success = True

            # grasp
            if grasp:
                grasp_reward = self.grasp_reward(geom_names)
                print("grasp", grasp_reward)
                reward += grasp_reward
                # experimental
                if grasp_reward:
                    success = True

            # lift
            if height:
                height_reward = self.height_reward(body_main)
                print("height", height_reward)
                reward += height_reward

            # open
            if open_:
                open_reward = self.open_reward()
                print("open", open_reward)
                reward += open_reward
        """
        print("reward", reward)
        self.step_count += 1
        truncated = self.step_count >= 250
        done = success or truncated
        print("done", done)
        info["agentview_image"] = obs["agentview_image"]
        info["is_success"] = success

        return self.get_low_dim_obs(obs), reward, done, truncated, info
    
    def reset(self, seed=None):
        obs = self.env.reset()
        self.step_count = 0
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

        return body_main, geom_names

    def reaching_reward(self, body_main):
        object_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(body_main)]
        gripper_site_pos = self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id]
        dist = np.linalg.norm(gripper_site_pos - object_pos)
        reaching_reward = 1 - np.tanh(10.0 * dist)
        return reaching_reward

    def contact_reward(self, geom_names):
        # for i in range(self.env.sim.model.ngeom):
        #     geom_name = self.env.sim.model.geom_id2name(i)
        #     print(f"Geom ID {i}: {geom_name}")
        #     geom_pos = self.env.sim.model.geom_pos[i]
        #     geom_size = self.env.sim.model.geom_size[i]
        #     print(f"Geom Position: {geom_pos}, Size: {geom_size}")

        gripper_geoms = ["gripper0_finger1_pad_collision",
            "gripper0_finger2_pad_collision"]

        # Check for contact between gripper and object
        if self.env.env.check_contact(gripper_geoms, geom_names):
            reward = 10.0  # Reward for touching the object
        else:
            reward = 0.0  # No reward if not touching

        return reward

    def grasp_reward(self, geom_names):
        if self.env.env._check_grasp(gripper=self.env.robots[0].gripper, object_geoms=geom_names):
            return 50.0
        else:
            return 0.0

    def lift_reward(self, body_main):
        current_height = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(body_main)][2]
        height =  current_height - self.initial_height
        self.initial_height = current_height
        return height

    def current_joint_position(self):
        qposs = []
        for joint in self.env.env.get_object(self.env.obj_of_interest[0]).joints:
            qpos_addr = self.env.env.sim.model.get_joint_qpos_addr(joint)
            qpos = self.env.sim.data.qpos[qpos_addr]
            qposs.append(qpos)
        return np.array(qposs)

    def open_reward(self):
        displacement = np.linalg.norm(self.current_joint_position() - self.initial_joint_position)
        reward = displacement * 10
        return reward
        
        goal_value, goal_ranges = MapObjects(self.env.obj_of_interest[0], self.env.language_instruction).define_goal()
        joint_displacement = np.linalg.norm(self.current_joint_position() - np.mean(goal_ranges))
        open_reward = 1 - np.tanh(10.0 * joint_displacement)
        return open_reward * 10.0

    
class LowDimensionalObsGymGoalEnv(gym.Env):
    """ Sparse reward environment with all the low-dimensional states with HER
    """
    def __init__(self, **kwargs):
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
    def __init__(self, **kwargs):
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
    def __init__(self, **kwargs):
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
    def __init__(self, **kwargs):
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