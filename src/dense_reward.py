from src.extract_xml import locate_libero_xml, find_geoms_for_site, find_body_main
from src.patch import get_list_of_geom_names_for_site, split_object_name
import numpy as np

def get_body_for_site(object_name, parent_name):
    target_object_name, target_site_name = split_object_name(object_name, parent_name)
    path = locate_libero_xml(target_object_name)
    body_main = find_body_main(path, target_site_name)
    return parent_name + '_' + body_main

class DenseReward:
    # dense reward for a specific goal goal_state
    def __init__(self, env, goal_state):
        self.env = env
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

        for index, obj in enumerate(self.object_states):
            if obj.object_state_type == "site":
                self.object_geoms.append(get_list_of_geom_names_for_site(obj.object_name, obj.parent_name, obj.env))
                self.object_bodies.append(get_body_for_site(obj.object_name, obj.parent_name))
            else:
                self.object_geoms.append(self.env.get_object(obj.object_name).contact_geoms)
                self.object_bodies.append(goal_state[index+1] + "_main")
        
        # for up reward
        self.prior_object_height = 0
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
            print("reach")
            return self.reach(self.object_bodies[0])
        if self.predicate_fn_name == "open":
            print("open")
            return self.open()
        if self.predicate_fn_name == "up":
            print("up")
            return self.up(self.object_bodies[0])
        if self.predicate_fn_name == "on":
            print("on")
            return self.on()
        if self.predicate_fn_name == "in":
            print("in")
            return self.contain()
        
        print("no dense reward")
        return 0.0

    def reach(self, body_main):
        if len(self.object_states) > 1:
            raise Exception("reach only accepts 1 object")
        object_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(body_main)]
        gripper_site_pos = self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id]
        dist = np.linalg.norm(gripper_site_pos - object_pos)
        reaching_reward = 1 - np.tanh(10.0 * dist)
        return reaching_reward

    def open(self):
        if len(self.object_states) > 1:
            raise Exception("open only accepts 1 object")
        displacement = np.linalg.norm(self.current_joint_position() - self.initial_joint_position)
        reward = displacement * 10
        return reward

        """
        goal_value, goal_ranges = MapObjects(self.env.obj_of_interest[0], self.env.language_instruction).define_goal()
        joint_displacement = np.linalg.norm(self.current_joint_position() - np.mean(goal_ranges))
        open_reward = 1 - np.tanh(10.0 * joint_displacement)
        return open_reward * 10.0
        """

    def up(self, body_main):
        grasp = self.object_states[0].check_grasp()
        object_height = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(body_main)][2]
        reward = grasp * object_height if object_height > self.prior_object_height else 0
        self.prior_object_height = object_height

        return reward

    '''
    # og
    def up(self, body_main, step_count):
        if len(self.object_states) > 1:
            raise Exception("up only accepts 1 object")
        if step_count == 0:
            self.initial_height = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(body_main)][2]
        current_height = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(body_main)][2]
        height =  current_height - self.initial_height
        grasp = self.object_states[0].check_grasp()
        gripper_height = self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id][2]
        reward = grasp * gripper_height if gripper_height > self.prior_gripper_height else 0
        self.prior_gripper_height = gripper_height

        return reward
        return height * 10.0 + grasp

    # 1, reward only if object goes higher
    def up(self, body_main, step_count):
        grasp = self.object_states[0].check_grasp()
        gripper_height = self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id][2]
        reward = grasp * gripper_height if gripper_height > self.prior_gripper_height else 0
        self.prior_gripper_height = gripper_height

        return reward

    # 2, velocity
    def up(self, body_main, step_count):
        # Check if the gripper is grasping the object
        grasp = self.object_states[0].check_grasp()

        # Get the current vertical velocity of the object (z-axis)
        object_body_id = self.env.sim.model.body_name2id(body_main)
        vertical_velocity = self.env.sim.data.cvel[object_body_id][2]  # z-axis velocity

        # Reward for vertical lifting
        return grasp * max(0, vertical_velocity)

    # 3, target height
    def up(self, body_main, step_count, target_height=1.0):
        # Current height of the object
        current_height = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(body_main)][2]

        # Grasp reward
        grasp = self.object_states[0].check_grasp()

        # Target height distance reward (penalizes horizontal movement)
        height_difference = target_height - current_height
        distance_reward = -abs(height_difference)

        # Combined reward: grasp + height + distance-to-target
        return grasp + (distance_reward*5.0)
    '''

    def on(self):
        '''
        other_object on top of this_object
        '''
        if len(self.object_states) < 2:
            raise Exception("on accepts 2 objects")
        this_object = self.env.get_object(self.object_states[1].object_name)
        this_object_position = self.env.sim.data.body_xpos[
            self.env.obj_body_id[self.object_states[1].object_name]
        ]
        other_object = self.env.get_object(self.object_states[0].object_name)
        other_object_position = self.env.sim.data.body_xpos[
            self.env.obj_body_id[self.object_states[0].object_name]
        ]

        xy_dist = np.linalg.norm(this_object_position[:2] - other_object_position[:2])
        z_dist = np.linalg.norm(this_object_position[2] - other_object_position[2])
        reach_xy = 1 - np.tanh(10.0 * xy_dist)
        reach_z = 1 - np.tanh(10 * z_dist)

        return reach_xy + reach_z
        
    def contain(self):
        '''
        other_object inside this_object
        '''
        if len(self.object_states) < 2:
            raise Exception("contain accepts 2 objects")
        this_object = self.env.object_sites_dict[self.object_states[1].object_name]
        this_object_position = self.env.sim.data.get_site_xpos(self.object_states[1].object_name)
        this_object_mat = self.env.sim.data.get_site_xmat(self.object_states[1].object_name)

        other_object = self.env.get_object(self.object_states[0].object_name)
        other_object_position = self.env.sim.data.body_xpos[
            self.env.obj_body_id[self.object_states[0].object_name]
        ]

        total_size = np.abs(this_object_mat @ this_object.size)
        ub = this_object_position + total_size
        lb = this_object_position - total_size
        lb[2] -= 0.01

        reward = np.linalg.norm(other_object_position - lb + ub - other_object_position)
        return reward

    def current_joint_position(self):
        qposs = []
        for joint in self.env.get_object(self.object_names[0]).joints:
            qpos_addr = self.env.sim.model.get_joint_qpos_addr(joint)
            qpos = self.env.sim.data.qpos[qpos_addr]
            qposs.append(qpos)
        return np.array(qposs)