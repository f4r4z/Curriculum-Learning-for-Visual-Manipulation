from libero.libero.envs.predicates import *
from libero.libero.envs.object_states import BaseObjectState, ObjectState
from libero.libero.envs.bddl_base_domain import BDDLBaseDomain
from libero.libero.envs.objects import SiteObject
from src.extract_xml import locate_libero_xml, find_geoms_for_site, find_body_main
import numpy as np
import re

def split_object_name(object_name, parent_name):
    # Extract the prefix and the remaining part
    match = re.match(r"(.*)_\d+", parent_name)
    if match:
        prefix = match.group(1)  # Everything before '_<digit>'
        # Remove the parent_name from full_string to get the suffix
        suffix = object_name[len(parent_name) + 1:]  # +1 for the underscore
        return prefix, suffix
    else:
        raise ValueError(f"parent name '{parent_name}' is not in the expected format.")

def get_body_for_site(object_name, parent_name):
    target_object_name, target_site_name = split_object_name(object_name, parent_name)
    path = locate_libero_xml(target_object_name)
    body_main = find_body_main(path, target_site_name)
    return parent_name + '_' + body_main

def get_list_of_geom_names_for_site(object_name, parent_name, env):
    list_of_geom_names = []
    target_object_name, target_site_name = split_object_name(object_name, parent_name)
    path = locate_libero_xml(target_object_name)
    site_geom_positions = find_geoms_for_site(path, target_site_name)
    for geom_name in env.sim.model.geom_names:
        if geom_name is None:
            continue
        if parent_name in geom_name:  # Filter for object-specific geoms
            geom_id = env.sim.model.geom_name2id(geom_name)
            geom_position = env.sim.model.geom_pos[geom_id]
            for site_geom_pos in site_geom_positions:
                if (site_geom_pos == geom_position).all():
                    list_of_geom_names.append(geom_name)
    return list_of_geom_names

def check_gripper_contact(self):
    gripper_geoms = self.env.robots[0].gripper # or gripper_geoms = ["gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision"]
    # if specific geoms mentioned
    if self.env.reward_geoms:
        return self.env.check_contact(gripper_geoms, self.env.reward_geoms)

    # object could be an object (articulated, hop) or a site
    # sites do not have a way to dynamically get geoms/check contact
    if self.object_state_type == "site":
        list_of_geom_names = get_list_of_geom_names_for_site(self.object_name, self.parent_name, self.env)
        return self.env.check_contact(gripper_geoms, list_of_geom_names)
    else:
        target_object_geoms = self.env.get_object(self.object_name).contact_geoms
        return self.env.check_contact(gripper_geoms, target_object_geoms)

def check_gripper_contain(self):
    gripper_site_pos = self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id]
    if self.object_state_type == "site":
        this_object = self.env.object_sites_dict[self.object_name]
        this_object_position = self.env.sim.data.get_site_xpos(self.object_name)
        this_object_mat = self.env.sim.data.get_site_xmat(self.object_name)

        return this_object.in_box(
            this_object_position, this_object_mat, gripper_site_pos
        )
    else:
        object_1 = self.env.get_object(self.object_name)
        object_1_position = self.env.sim.data.body_xpos[
            self.env.obj_body_id[self.object_name]
        ]

        return object_1.in_box(object_1_position, gripper_site_pos)

def check_grasp(self):
    gripper_geoms = self.env.robots[0].gripper # or  gripper_geoms = ["gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision"]
    
    # if specific geoms mentioned
    if self.env.reward_geoms:
        return self.env._check_grasp(gripper=gripper_geoms, object_geoms=self.env.reward_geoms)

    if self.object_state_type == "site":
        list_of_geom_names = get_list_of_geom_names_for_site(self.object_name, self.parent_name, self.env)
        return self.env._check_grasp(gripper=gripper_geoms, object_geoms=list_of_geom_names)
    else:
        target_object_geoms = self.env.get_object(self.object_name).contact_geoms # .contact_geoms is not really necessary, but added for readibility
        return self.env._check_grasp(gripper=gripper_geoms, object_geoms=target_object_geoms)

def reach(self):
    if self.object_state_type == "site":
        body_main = get_body_for_site(self.object_name, self.parent_name)
    else:
        body_main = self.object_name + "_main"
    object_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(body_main)]

    # reach average of specific geoms
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

    if dist < 0.05:
        return True
    else:
        return False

def align(self, arg1):
    """
    other object align with this object
    """
    if self.object_state_type == "site":
        this_object_position = self.env.sim.data.get_site_xpos(self.object_name)
    else:
        this_object_position = self.env.sim.data.body_xpos[
            self.env.obj_body_id[self.object_name]
        ]
    
    other_object_position = self.env.sim.data.body_xpos[
       self.env.obj_body_id[arg1.object_name]
    ]

    dist = np.linalg.norm(other_object_position[:2] - this_object_position[:2])
    
    if dist < 0.05:
        return True
    else:
        return False

def check_contact_excluding_gripper(sim, object_name, gripper_geoms=["gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision"]):
    '''
    returns True if object_name is in contact with another object excluding the gripper
    '''
    # Iterate over all MuJoCo contacts
    for i in range(sim.data.ncon):
        # Get geom IDs of the two contacting geoms
        contact = sim.data.contact[i]
        geom1 = contact.geom1
        geom2 = contact.geom2
        # Get geom names
        geom1_name = sim.model.geom_id2name(geom1)
        geom2_name = sim.model.geom_id2name(geom2)

        # Check if the object is involved in the contact
        if (geom1_name is not None and object_name in geom1_name) or (geom2_name is not None and object_name in geom2_name):
            # Ensure the other contact geom is not the gripper
            other_geom = geom1_name if (geom2_name is not None and object_name in geom2_name) else geom2_name
            if other_geom is None or "gripper" not in other_geom:                
                return True
    return False


def lift(self):
    '''
    no contact with another object
    higher than 1.5? 1.25? times (or absolute) the other interest object
    '''
    # get objects contacts and height
    self.env.get_object(self.object_name).contact_geoms

    try:
        this_object_height = self.env._obs_cache[self.object_name + "_pos"][2] # or self.env.sim.data.body_xpos[self.env.obj_body_id[self.object_name]][2]
    except KeyError:
        self.env.sim.data.body_xpos[self.env.obj_body_id[self.object_name]][2]

    # get other object name
    for obj in self.env.obj_of_interest:
        if self.object_name not in obj:
            other_object_name = obj
    
    # get other object height
    try:
        other_object_height = self.env._obs_cache[other_object_name + "_pos"][2] # or self.env.sim.data.body_xpos[self.env.obj_body_id[other_object_name]][2]
    except KeyError:
        other_object_height = self.env.sim.data.body_xpos[self.env.obj_body_id[other_object_name]][2]

    # print("This object height", this_object_height, "Other object height", other_object_height, "in contact with another object", check_contact_excluding_gripper(self.env.sim, self.object_name))

    if this_object_height > (other_object_height + 0.45) and not check_contact_excluding_gripper(self.env.sim, self.object_name):
        return True

    return False

def reset_qpos(self):
    """
    resets robot to original qpos
    """
    robot_joints = self.env.robots[0]._joint_positions
    robot_initial_joints = self.env.robots[0].init_qpos
    norm = np.linalg.norm(robot_joints - robot_initial_joints)

    if norm < 0.03:
        return True
    else:
        return False


class Contact(UnaryAtomic):
    def __call__(self, arg):
        return arg.check_gripper_contact()

class Grasp(UnaryAtomic):
    def __call__(self, arg):
        return arg.check_grasp()

class Reach(UnaryAtomic):
    def __call__(self, arg):
        return arg.reach()

class Lift(UnaryAtomic):
    def __call__(self, arg):
        return arg.lift()

class Align(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return arg2.align(arg1)

class PlaceIn(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return arg2.check_contact(arg1) and arg2.check_contain(arg1) and (not arg1.check_gripper_contact())

class Reset(UnaryAtomic):
    def __call__(self, arg):
        return arg.reset_qpos()

class GripperOut(UnaryAtomic):
    def __call__(self, arg1):
        return not arg1.check_gripper_contain()

VALIDATE_PREDICATE_FN_DICT["contact"] = Contact()
VALIDATE_PREDICATE_FN_DICT["grasp"] = Grasp()
VALIDATE_PREDICATE_FN_DICT["reach"] = Reach()
VALIDATE_PREDICATE_FN_DICT["lift"] = Lift()
VALIDATE_PREDICATE_FN_DICT["align"] = Align()
VALIDATE_PREDICATE_FN_DICT["placein"] = PlaceIn()
VALIDATE_PREDICATE_FN_DICT["reset"] = Reset()
VALIDATE_PREDICATE_FN_DICT["gripperout"] = GripperOut()

BaseObjectState.check_gripper_contact = check_gripper_contact
BaseObjectState.check_grasp = check_grasp
BaseObjectState.reach = reach
BaseObjectState.lift = lift
BaseObjectState.align = align
BaseObjectState.reset_qpos = reset_qpos
BaseObjectState.check_gripper_contain = check_gripper_contain

# BDDLBaseDomain.reward = reward