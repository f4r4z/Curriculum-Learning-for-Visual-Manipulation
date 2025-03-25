from robosuite.utils.binding_utils import MjSim
from src.extract_xml import locate_libero_xml, find_geoms_for_site, find_body_main
from libero.libero.envs.bddl_base_domain import BDDLBaseDomain
import re
import numpy as np
from typing import List, Tuple


def split_object_name(object_name: str, parent_name: str):
    # Extract the prefix and the remaining part
    match = re.match(r"(.*)_\d+", parent_name)
    if match:
        prefix = match.group(1)  # Everything before '_<digit>'
        # Remove the parent_name from full_string to get the suffix
        suffix = object_name[len(parent_name) + 1:]  # +1 for the underscore
        return prefix, suffix
    else:
        raise ValueError(f"parent name '{parent_name}' is not in the expected format.")


def get_body_for_site(object_name: str, parent_name: str) -> str:
    target_object_name, target_site_name = split_object_name(object_name, parent_name)
    path = locate_libero_xml(target_object_name)
    body_main = find_body_main(path, target_site_name)
    return parent_name + '_' + body_main


def get_list_of_geom_names_for_site(object_name: str, parent_name: str, env: BDDLBaseDomain):
    list_of_geom_names: List[str] = []
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


def convert_coords(points: np.ndarray):
    """
    Stuff like size is in a different coordinate system (x,z,y) from the simulation (x,y,z)
    here x is forward-back, y is left-right, z is elevation
    this function converts from the former to simulation coords
    """
    return np.array([points[0], points[2], points[1]])
    # return np.array([points[2], points[0], points[1]])


def box_bounds(size: np.ndarray, position: np.ndarray, rotation: np.ndarray):
    unit_cube = np.array([
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [1, -1, -1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, 1],
        [-1, -1, -1],
    ])
    points = convert_coords(size) * unit_cube
    points = points @ rotation + position
    # points = (size/2 * unit_cube) @ rotation + position
    return points.min(axis=0), points.max(axis=0)

def get_site_bounding_box(sim: MjSim, site: str):
    site_id = sim.model.site_name2id(site)
    site_pos: np.ndarray = sim.data.get_site_xpos(site)
    site_rot: np.ndarray = sim.data.get_site_xmat(site)
    site_type: int = sim.model.site_type[site_id]
    if site_type == 6:
        return box_bounds(sim.model.site_size[site_id], site_pos, site_rot)
    else:
        raise Exception(f"computing bounds for site type {site_type} is not supported")

def get_geom_bounding_box(sim: MjSim, geom: str):
    geom_id = sim.model.geom_name2id(geom)
    geom_pos: np.ndarray = sim.data.get_geom_xpos(geom)
    geom_type: int = sim.model.geom_type[geom_id]
    if geom_type == 6: # box
        return box_bounds(sim.model.geom_size[geom_id], geom_pos, sim.data.get_geom_xmat(geom))
    else:
        radius = sim.model.geom_rbound[geom_id]
        print(f"geom_type_{geom_type}", geom_pos, radius)
        return geom_pos - radius, geom_pos + radius


def compute_bounding_box_from_geoms(sim: MjSim, geoms: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    min_bounds, max_bounds = zip(*[get_geom_bounding_box(sim, geom) for geom in geoms])
    return np.array(min_bounds).min(axis=0), np.array(max_bounds).max(axis=0)
