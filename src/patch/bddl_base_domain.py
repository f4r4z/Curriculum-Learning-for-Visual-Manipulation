from libero.libero.envs.bddl_base_domain import BDDLBaseDomain
from .utils import patch
import numpy as np

@patch(BDDLBaseDomain)
def get_gripper_site_pos(self: BDDLBaseDomain, robot=0) -> np.ndarray:
	return self.sim.data.site_xpos[self.robots[robot].eef_site_id]