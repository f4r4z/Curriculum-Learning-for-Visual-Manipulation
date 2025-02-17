from src.extract_xml import locate_libero_xml, find_geoms_for_site, find_body_main
from src.libero_utils import get_list_of_geom_names_for_site, split_object_name, get_body_for_site, check_contact_excluding_gripper
import robosuite.utils.transform_utils as T
from libero.libero.envs.bddl_base_domain import BDDLBaseDomain
import numpy as np
from libero.libero.envs.base_object import OBJECTS_DICT
from typing import List

class DenseReward:
    # dense reward for a specific goal goal_state
    def __init__(self, env: BDDLBaseDomain, goal_state: List[str], reward_geoms=None, verbose=1):
        self.env = env
        self.verbose = verbose
        self.object_names = []
        self.object_states = []
        self.object_geoms = []
        self.object_bodies = []
        if len(goal_state) == 3:
            # Checking binary logical predicates
            self.predicate_fn_name = goal_state[0]
            self.object_names.extend([goal_state[1], goal_state[2]])
            self.object_states.extend([self.env.object_states_dict[self.object_names[0]], self.env.object_states_dict[self.object_names[1]]])
        elif len(goal_state) == 2:
            # Checking unary logical predicates
            self.predicate_fn_name = goal_state[0]
            self.object_names.append(goal_state[1])
            self.object_states.append(self.env.object_states_dict[self.object_names[0]])


            # only for open
            if self.predicate_fn_name == 'open' or self.predicate_fn_name == 'close':
                self.initial_joint_position = self.current_joint_position()
                self.prior_displacement = 0.0

            if self.predicate_fn_name == 'close':
                close_ranges = self.env.object_states_dict[self.object_names[0]].query_dict[self.object_names[0]].object_properties["articulation"]["default_close_ranges"]
                self.close_joint_position = np.array([np.mean(close_ranges)])


        for index, obj in enumerate(self.object_states):
            if obj.object_state_type == "site":
                self.object_geoms.append(get_list_of_geom_names_for_site(obj.object_name, obj.parent_name, obj.env))
                self.object_bodies.append(get_body_for_site(obj.object_name, obj.parent_name))
            else:
                self.object_geoms.append(self.env.get_object(obj.object_name).contact_geoms)
                self.object_bodies.append(goal_state[index+1] + "_main")

        # adding geoms for one object predicates
        if len(self.object_states) == 1:
            self.env.reward_geoms = reward_geoms
        else:
            self.env.reward_geoms = None

        # for up reward
        self.prior_object_height = 0
        self.prior_orientation = self.env.sim.data.body_xquat[self.env.sim.model.body_name2id(self.object_bodies[0])]
        self.prior_position = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(self.object_bodies[0])]
        '''
        # information to print out
        print(goal_state)
        print(self.object_names)
        print(self.object_states)
        print(self.object_geoms)
        print(self.object_bodies)
        '''

    def dense_reward(self, step_count=0):
        if self.predicate_fn_name == "reach":
            if self.verbose >= 4: print("reach")
            # penalty = (0.5 * self.orientation_penalty(self.object_bodies[0])) + (0.5 * self.displacement_penalty(self.object_bodies[0]))
            return self.reach(self.object_bodies[0]) # - penalty
        if self.predicate_fn_name == "open":
            if self.verbose >= 4: print("open")
            return self.open(step_count)
        if self.predicate_fn_name == "close":
            if self.verbose >= 4: print("close")
            return self.close()
        if self.predicate_fn_name == "lift":
            if self.verbose >= 4: print("lift")
            return self.lift(self.object_bodies[0], step_count)
        if self.predicate_fn_name == "on":
            if self.verbose >= 4: print("on")
            return self.on()
        if self.predicate_fn_name == "align":
            if self.verbose >= 4: print("align")
            return self.align()
        if self.predicate_fn_name == "in":
            if self.verbose >= 4: print("in")
            return self.inside()
        if self.predicate_fn_name == "placein":
            if self.verbose >= 4: print("place in")
            return self.place_inside()
        if self.predicate_fn_name == "reset":
            print("reset")
            return self.reset_qpos()
        if self.verbose >= 4: print(f"no dense reward for {self.predicate_fn_name}")
        return 0.0

    def reset_qpos(self):
        robot_joints = self.env.robots[0]._joint_positions
        robot_initial_joints = self.env.robots[0].init_qpos
        norm = np.linalg.norm(robot_joints - robot_initial_joints)
        reward = (1 - np.tanh(10.0 * norm)) / 10.0
        return reward

    def get_object_width(self, body_main):
        # Get the geom ID of the object
        geom_id = self.env.sim.model.body_name2id(body_main)
        print("id", geom_id)
        # Extract the size of the geom (half extents)
        print("wtf", dir(self.env.sim.model))
        geom_size = self.env.sim.model.geom_size[geom_id]  # [x_half, y_half, z_half]
        print("size", geom_size)
        # Object width is typically the x-dimension (full width is `2 * x_half`)
        object_width = 2 * geom_size[0]  # Multiply by 2 to get full width
        print("width", object_width)

        return object_width
        
    def reach(self, body_main):
        if len(self.object_states) > 1:
            raise Exception("reach only accepts 1 object")
        object_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(body_main)]
        # reach specific geoms
        if self.env.reward_geoms:
            # calculate the average of geoms position
            geom_pos = 0.0
            for geom in self.env.reward_geoms:
                try:
                    geom_id = self.env.sim.model.geom_name2id(geom)
                    geom_pos += self.env.sim.data.geom_xpos[geom_id]
                except:
                    # in case it is a body and not geom
                    geom_pos += self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(geom)]
            object_pos = geom_pos / len(self.env.reward_geoms)
    
        gripper_site_pos = self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id]
        dist = np.linalg.norm(gripper_site_pos - object_pos)
        reaching_reward = (1 - np.tanh(10.0 * dist)) / 10.0

        return reaching_reward

    def open(self, step_count):
        if len(self.object_states) > 1:
            raise Exception("open only accepts 1 object")
        
        if step_count == 0:
            self.prior_displacement = 0.0

        current_joint_position = self.current_joint_position()
        displacement = np.linalg.norm(current_joint_position - self.initial_joint_position)
        # only reward if it's higher than prior
        if displacement > self.prior_displacement:
            reward = displacement
            self.prior_displacement = displacement
        else:
            reward = 0.0

        return reward

    def grasp(self, body_main):
        reward = 0.0

        # Get relevant state info
        gripper_pos = self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id]
        object_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(body_main)]
        gripper_finger_pos = env.get_finger_positions()
        object_grasped = env.check_grasp()  # Check if MuJoCo recognizes a stable grasp

        # 1. Encourage the gripper to open before grasping
        finger_distance = np.linalg.norm(gripper_finger_pos[0] - gripper_finger_pos[1])
        desired_opening = self.get_object_width(body_main)  # Adjust this based on object size
        reward += 2.0 * max(0, finger_distance - desired_opening)  # Reward for keeping fingers open initially

        # 2. Encourage proper gripper-object alignment
        gripper_to_object_dist = np.linalg.norm(gripper_pos - object_pos)
        alignment_threshold = 0.02  # Adjust for precision
        reward += 3.0 * max(0, alignment_threshold - gripper_to_object_dist)

        # 3. Encourage closing fingers **only when object is within grasp**
        if gripper_to_object_dist < alignment_threshold:  # If well-aligned
            reward += 5.0 * (desired_opening - finger_distance)  # Reward closing fingers appropriately

        # 4. Give a big reward if the object is stably grasped
        if object_grasped:
            reward += 10.0  # High reward for a successful grasp

        return reward

    def close(self):
        if len(self.object_states) > 1:
            raise Exception("open only accepts 1 object")

        current_joint_position = self.current_joint_position()
        displacement = np.linalg.norm(current_joint_position - self.close_joint_position)
        
        reward = (1 - np.tanh(displacement))

        return reward

    def lift(self, body_main, step_count):
        grasp = self.object_states[0].check_grasp()
        gripper_height = self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id][2]
        # object_height = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(body_main)][2]
        if step_count == 0:
            self.prior_object_height = gripper_height
        reward = ((gripper_height - self.prior_object_height) + (gripper_height / 10.0)) * (grasp) if gripper_height > self.prior_object_height else grasp * 0.01
        self.prior_object_height = gripper_height

        return reward

    def on(self):
        '''
        other_object on top of this_object
        '''
        if len(self.object_states) < 2:
            raise Exception("on accepts 2 objects")
        this_object = self.env.get_object(self.object_states[1].object_name)
        try:
            this_object_position = self.env.sim.data.body_xpos[
                self.env.obj_body_id[self.object_states[1].object_name]
            ]
        except:
            this_object_position = self.env.sim.data.get_site_xpos(self.object_states[1].object_name)

        other_object = self.env.get_object(self.object_states[0].object_name)
        other_object_position = self.env.sim.data.body_xpos[
            self.env.obj_body_id[self.object_states[0].object_name]
        ]

        distance = np.linalg.norm(this_object_position - other_object_position)
        reward = 1 - np.tanh(10.0 * distance)
        grasp = self.object_states[0].check_grasp()

        return grasp * reward

    def inside(self):
        '''
        other_object in this_object
        '''
        this_object = self.env.get_object(self.object_states[1].object_name)
        try:
            this_object_position = self.env.sim.data.body_xpos[
                self.env.obj_body_id[self.object_states[1].object_name]
            ]
        except:
            this_object_position = self.env.sim.data.get_site_xpos(self.object_states[1].object_name)

        gripper_site_pos = self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id]
        dist = np.linalg.norm(gripper_site_pos - this_object_position)
        reaching_reward = 1 - np.tanh(10.0 * dist)
        grasp = self.object_states[0].check_grasp()

        return grasp * reaching_reward

    def place_inside(self):
        '''
        other_object in this_object with no contact with gripper
        '''
        this_object = self.env.get_object(self.object_states[1].object_name)
        try:
            this_object_position = self.env.sim.data.body_xpos[
                self.env.obj_body_id[self.object_states[1].object_name]
            ]
        except:
            this_object_position = self.env.sim.data.get_site_xpos(self.object_states[1].object_name)

        other_object = self.env.get_object(self.object_states[0].object_name)
        other_object_position = self.env.sim.data.body_xpos[
            self.env.obj_body_id[self.object_states[0].object_name]
        ]
        dist = np.linalg.norm(other_object_position - this_object_position)
        reaching_reward = 1 - np.tanh(10.0 * dist)
        grasp = self.object_states[0].check_grasp()

        if self.object_states[1].check_contain(self.object_states[0]) and self.object_states[0].check_gripper_contact():
            return 0.0

        return grasp * reaching_reward
        
    def align(self):
        '''
        other_object align this_object (same xy coordinates)
        '''
        if len(self.object_states) < 2:
            raise Exception("align accepts 2 objects")
        # this_object = self.env.object_sites_dict[self.object_states[1].object_name]
        try:
            this_object_position = self.env.sim.data.body_xpos[
                self.env.obj_body_id[self.object_states[1].object_name]
            ]
        except:
            this_object_position = self.env.sim.data.get_site_xpos(self.object_states[1].object_name)

        # this_object_mat = self.env.sim.data.get_site_xmat(self.object_states[1].object_name)

        other_object = self.env.get_object(self.object_states[0].object_name)
        other_object_position = self.env.sim.data.body_xpos[
            self.env.obj_body_id[self.object_states[0].object_name]
        ]

        # total_size = np.abs(this_object_mat @ this_object.size)
        # ub = this_object_position + total_size
        # lb = this_object_position - total_size
        # lb[2] -= 0.01

        distance = np.linalg.norm(other_object_position[:2] - this_object_position[:2])
        reward = (1 - np.tanh(10 * distance)) / 10.0

        grasp = self.object_states[0].check_grasp()

        return grasp * reward

    def current_joint_position(self):
        qposs = []
        for joint in self.env.get_object(self.object_names[0]).joints:
            qpos_addr = self.env.sim.model.get_joint_qpos_addr(joint)
            qpos = self.env.sim.data.qpos[qpos_addr]
            qposs.append(qpos)
        return np.array(qposs)
