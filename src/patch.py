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
    # object could be an object (articulated, hop) or a site
    # sites do not have a way to dynamically get geoms/check contact
    target_object_geoms = self.env.get_object(self.object_name).contact_geoms
    gripper_geoms = ["gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision"]
    if self.object_state_type == "site":
        list_of_geom_names = get_list_of_geom_names_for_site(self.object_name, self.parent_name, self.env)
        return self.env.check_contact(gripper_geoms, list_of_geom_names)
    else:
        return self.env.check_contact(gripper_geoms, target_object_geoms)

def check_grasp(self):
    target_object_geoms = self.env.get_object(self.object_name).contact_geoms # .contact_geoms is not really necessary, but added for readibility
    gripper_geoms = ["gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision"] # or gripper_geoms = self.env.robots[0].gripper
    if self.object_state_type == "site":
        list_of_geom_names = get_list_of_geom_names_for_site(self.object_name, self.parent_name, self.env)
        return self.env._check_grasp(gripper=gripper_geoms, object_geoms=list_of_geom_names)
    else:
        return self.env._check_grasp(gripper=gripper_geoms, object_geoms=target_object_geoms)

def reach(self, body_main="ketchup_1_main"):
    # object_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(body_main)]
    # gripper_site_pos = self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id]

    # dist = np.linalg.norm(gripper_site_pos - object_pos)
    # reaching_reward = 1 - np.tanh(10.0 * dist)
    # return dist < 0.8
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
    this_object_height = self.env._obs_cache[self.object_name + "_pos"][2] # or self.env.sim.data.body_xpos[self.env.obj_body_id[self.object_name]][2]

    # get other object name
    for obj in self.env.obj_of_interest:
        if self.object_name not in obj:
            other_object_name = obj
    
    # get other object height
    other_object_height = self.env._obs_cache[other_object_name + "_pos"][2] # or self.env.sim.data.body_xpos[self.env.obj_body_id[other_object_name]][2]

    # print("This object height", this_object_height, "Other object height", other_object_height, "in contact with another object", check_contact_excluding_gripper(self.env.sim, self.object_name))

    if this_object_height > (other_object_height + 0.45) and not check_contact_excluding_gripper(self.env.sim, self.object_name):
        return True

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

VALIDATE_PREDICATE_FN_DICT["contact"] = Contact()
VALIDATE_PREDICATE_FN_DICT["grasp"] = Grasp()
VALIDATE_PREDICATE_FN_DICT["reach"] = Reach()
VALIDATE_PREDICATE_FN_DICT["lift"] = Lift()

BaseObjectState.check_gripper_contact = check_gripper_contact
BaseObjectState.check_grasp = check_grasp
BaseObjectState.reach = reach
BaseObjectState.lift = lift

# BDDLBaseDomain.reward = reward