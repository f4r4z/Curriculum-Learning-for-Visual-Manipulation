from libero.libero.envs.predicates import register_predicate_fn
from libero.libero.envs.predicates.base_predicates import *
from libero.libero.envs.object_states import BaseObjectState, SiteObjectState
from libero.libero.envs.objects import ArticulatedObject
import numpy as np

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

def check_grasp(self, other):
    gripper_geoms = ["gripper0_finger1_pad_collision",
    "gripper0_finger2_pad_collision"]
    geom_names = [
        "ketchup_1_main", 
    ]
    return self.env._check_grasp(gripper=self.env.robots[0].gripper, object_geoms=geom_names)


class Contact(UnaryAtomic):
    def __call__(self, arg):
        return arg.is_in_contact(arg)

class Grasp(UnaryAtomic):
    def __call__(self, arg):
        return arg.check_grasp(arg)



# VALIDATE_PREDICATE_FN_DICT["contact"] = Contact()
# VALIDATE_PREDICATE_FN_DICT["grasp"] = Grasp()
# VALIDATE_PREDICATE_FN_DICT["reach"] = Reach()

# BaseObjectState.is_in_contact = is_in_contact
# BaseObjectState.check_grasp = check_grasp
# BaseObjectState.reach = reach

# BDDLBaseDomain.reward = reward


@register_predicate_fn
class Reach(MultiarayAtomic):
    def __call__(self, *args):
        assert len(args) >= 1
        if len(args) == 1:
            return self.reach(args[0])
        else:
            goal_distance = float(args[1])
            return self.reach(args[0], goal_distance)
        

    def reach(self, object_state: BaseObjectState, goal_distance: float = None):
        env = object_state.env
        grip_site_pos = env.sim.data.get_site_xpos("gripper0_grip_site") # This is the position between the 2 claws

        object_pos = object_state.get_geom_state()['pos']
        dist = np.linalg.norm(grip_site_pos - object_pos)

        # Check whether object has been reached (without caring about goal_distance)
        if isinstance(object_state, SiteObjectState):
            object_mat = env.sim.data.get_site_xmat(object_state.object_name)
            object_site = env.get_object_site(object_state.object_name)
            if object_site.in_box(object_pos, object_mat, grip_site_pos):
                return True
        elif dist < 0.02: # if not a site, just use distance. There should be an in_box for regular objects, but for now we just have this
            return True
        
        # If object has not been reached, check if goal distance is reached
        return dist < goal_distance



@register_predicate_fn
class PartialOpen(MultiarayAtomic):
    def __call__(self, *args):
        assert len(args) >= 2
        open_amount = float(args[1])
        return self.is_partial_open(args[0], open_amount)
    
    def is_articulated_object_partial_open(self, object: ArticulatedObject, qpos, open_amount):
        "Checks whether the object is open by the given open_amount in range [0, 1]"

        default_open_ranges = object.object_properties["articulation"]["default_open_ranges"]
        default_close_ranges = object.object_properties["articulation"]["default_close_ranges"]

        # if no ranges provided, just assume it is closed
        if len(default_open_ranges) == 0 or len(default_close_ranges) == 0:
            return False
        
        # the ranges provide a leeway. depending on which is smaller, we choose the innermost side of the leeway
        if default_open_ranges[0] < default_close_ranges[0]:
            fully_open = max(default_open_ranges)
            fully_closed = min(default_close_ranges)
        else:
            fully_open = min(default_open_ranges)
            fully_closed = max(default_close_ranges)

        # to count as partial open, qpos must be on the fully open side of the threshold
        threshold = open_amount * fully_open + (1-open_amount) * fully_closed
        if fully_open < fully_closed:
            return qpos < threshold
        else:
            return qpos > threshold

    def is_partial_open(self, object_state: BaseObjectState, open_amount: float):
        "Checks whether any joint is open by open_amounts"
        env = object_state.env
        for joint in env.get_object_site(object_state.object_name).joints:
            qpos = env.sim.data.get_joint_qpos(joint)
            object = env.get_object(object_state.parent_name)

            assert isinstance(object, ArticulatedObject), (
                f"{object_state.object_name}'s parent, {object_state.parent_name} "
                "is not an articulated object. PartialOpen can only be used with articulated objects"
            )
            if self.is_articulated_object_partial_open(object, qpos, open_amount):
                return True
            
        return False



@register_predicate_fn
class PartialClose(MultiarayAtomic):
    def __call__(self, *args):
        assert len(args) >= 2
        close_amount = float(args[1])
        # partial close is just the opposite of partial open, but with the parameter flipped
        return not PartialOpen()(args[0], 1-close_amount, *args[2:])