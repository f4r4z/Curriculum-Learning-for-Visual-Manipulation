from libero.libero.envs.object_states import BaseObjectState, ObjectState, SiteObjectState
from libero.libero.envs.objects import SiteObject
from src.libero_utils import get_body_for_site, get_list_of_geom_names_for_site, check_contact_excluding_gripper, get_site_bounding_box, compute_bounding_box_from_geoms
from .utils import patch

import numpy as np


@patch(BaseObjectState)
def get_geoms(self: BaseObjectState):
    if isinstance(self, ObjectState):
        obj = self.env.get_object(self.object_name)
        return obj.contact_geoms
    elif isinstance(self, SiteObjectState):
        site_obj: SiteObject = self.env.get_object(self.object_name)
        if not hasattr(site_obj, "geoms"): # cache the geoms because this operation is expensive
            site_obj.geoms = get_list_of_geom_names_for_site(self.object_name, self.parent_name, self.env)
        return site_obj.geoms
    else:
        raise NotImplementedError
    

@patch(BaseObjectState)
def compute_bounding_box(self: BaseObjectState):
    """
    If site, return the site's bounding box. Otherwise return the bounding box of the geoms.
    """
    # return min_bounds, max_bounds
    if isinstance(self, SiteObjectState):
        return get_site_bounding_box(self.env.sim, self.object_name)
    else:
        return compute_bounding_box_from_geoms(self.env.sim, self.get_geoms())
    

@patch(BaseObjectState)
def get_position(self: BaseObjectState) -> np.ndarray:
    return self.get_geom_state()['pos']


@patch(BaseObjectState)
def check_gripper_contact(self: BaseObjectState):
    gripper_geoms = self.env.robots[0].gripper # or gripper_geoms = ["gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision"]
    # if specific geoms mentioned
    if self.env.reward_geoms:
        return self.env._check_grasp(gripper=gripper_geoms, object_geoms=self.env.reward_geoms)

    # object could be an object (articulated, hop) or a site
    # sites do not have a way to dynamically get geoms/check contact
    if self.object_state_type == "site":
        list_of_geom_names = get_list_of_geom_names_for_site(self.object_name, self.parent_name, self.env)
        return self.env.check_contact(gripper_geoms, list_of_geom_names)
    else:
        target_object_geoms = self.env.get_object(self.object_name).contact_geoms
        return self.env.check_contact(gripper_geoms, target_object_geoms)


@patch(BaseObjectState)
def check_grasp(self: BaseObjectState):
    gripper_geoms = self.env.robots[0].gripper # or  gripper_geoms = ["gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision"]
    
    # if specific geoms mentioned
    if self.env.reward_geoms:
        return self.env._check_grasp(gripper=gripper_geoms, object_geoms=self.env.reward_geoms)
    
    target_object_geoms = self.get_geoms()
    return self.env._check_grasp(gripper=gripper_geoms, object_geoms=target_object_geoms)


@patch(BaseObjectState)
def reach(self: BaseObjectState):
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


@patch(BaseObjectState)
def align(self: BaseObjectState, other: BaseObjectState):
    """
    other object align with this object
    """
    this_object_position = self.get_position()
    other_object_position = other.get_position()

    dist = np.linalg.norm(other_object_position[:2] - this_object_position[:2])
    
    if dist < 0.05:
        return True
    else:
        return False


@patch(BaseObjectState)
def lift(self: BaseObjectState):
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


@patch(BaseObjectState)
def reset_qpos(self: BaseObjectState):
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
