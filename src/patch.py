from libero.libero.envs.predicates import *
from libero.libero.envs.object_states import BaseObjectState, ObjectState
from libero.libero.envs.bddl_base_domain import BDDLBaseDomain
import numpy as np


def reward(self, action=None):
    """
    Reward function for the task.

    Sparse un-normalized reward:

        - a discrete reward of 1.0 is provided if the task succeeds.

    Args:
        action (np.array): [NOT USED]

    Returns:
        float: reward value
    """
    reward = 0.0

    # define our shaping rewards
    # goal_state = self.parsed_problem["goal_state"]
    # for state in goal_state:
    #     if "reach" in state:
    #         return eval_predicate_fn(state[0], self.object_states_dict[state[1]])

    # sparse completion reward
    if self._check_success():
        reward = 1.0

    # Scale reward if requested
    if self.reward_scale is not None:
        reward *= self.reward_scale / 1.0

    return reward

def is_in_contact(self, other):
    gripper_geoms = ["gripper0_finger1_pad_collision",
    "gripper0_finger2_pad_collision"]
    geom_names = [
        "wooden_cabinet_1_g40",
        "wooden_cabinet_1_g41",
        "wooden_cabinet_1_g42"
    ]
    # Check for contact between gripper and object
    return self.env.check_contact(gripper_geoms, geom_names)

def reach(self, body_main="ketchup_1_main"):
    # object_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(body_main)]
    # gripper_site_pos = self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id]

    # dist = np.linalg.norm(gripper_site_pos - object_pos)
    # reaching_reward = 1 - np.tanh(10.0 * dist)
    # return dist < 0.8
    return False

class Contact(UnaryAtomic):
    def __call__(self, arg):
        return arg.is_in_contact(arg)

class Reach(UnaryAtomic):
    def __call__(self, arg):
        return arg.reach()

VALIDATE_PREDICATE_FN_DICT["contact"] = Contact()
VALIDATE_PREDICATE_FN_DICT["reach"] = Reach()

BaseObjectState.is_in_contact = is_in_contact
BaseObjectState.reach = reach

# BDDLBaseDomain.reward = reward