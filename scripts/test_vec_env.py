import os
# add parent path to sys
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import robosuite as suite

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs import SubprocVectorEnv, DummyVectorEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.envs import LowDimensionalObsGymEnv


bddl_file_base = get_libero_path("bddl_files")
task_name = "libero_90/KITCHEN_SCENE6_close_the_microwave.bddl"

env_args = {
    "bddl_file_name": os.path.join(bddl_file_base, task_name),
    "camera_heights": 128,
    "camera_widths": 128,
}

def env_func():
    return LowDimensionalObsGymEnv(**env_args)

if __name__ == "__main__":
    env_num = 32
    env = SubprocVecEnv(
        [env_func for _ in range(env_num)]
    )

    env.reset()
    N = 1000
    start_time = time.time()

    for i in range(N):
        print(f"Step {i}/{N}")
        action = np.random.uniform(-1, 1, size=(env_num, 7))
        obs, rewards, dones, info = env.step(action)

    print("Frames per second: ", N * env_num / (time.time() - start_time))