"""
Right now this file removes redundant non-actions from a given demonstration
"""

# TODO: Old structure / demos are deprecated, need to update!

import time
import json
import os
import h5py
import argparse
import random
import numpy as np

import robosuite
# from robosuite.utils.mjcf_utils import postprocess_model_xml
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite import load_controller_config

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *
from termcolor import colored
from libero.lifelong.datasets import get_dataset

from create_demonstration import gather_demonstrations_as_hdf5

# patches libero
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.patch

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--directory",
		type=str,
		default="demonstration_data",
	)
	parser.add_argument(
		"--robots",
		nargs="+",
		type=list,
		default=["Panda"],
		help="Which robot(s) to use in the env",
	)
	parser.add_argument(
		"--camera",
		type=str,
		default="agentview",
		help="Which camera to use for collecting demos",
	)
	parser.add_argument(
		"--controller",
		type=str,
		default="OSC_POSE",
		help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'",
	)
	parser.add_argument(
		"--config",
		type=str,
		default="single-arm-opposed",
		help="Specified environment configuration if necessary",
	)
	parser.add_argument("--bddl-file", type=str, default=None)
	parser.add_argument("--folder", type=str, required=True)
	args = parser.parse_args()

	demo_path = args.folder
	hdf5_path = os.path.join(demo_path, "demo.hdf5")
	f = h5py.File(hdf5_path, "r")
	# env_name = f["data"].attrs["env"]

	print("attributes:", dict(f.attrs))
	print("demo:", dict(f))
	print("data:", dict(f['data']))
	print("data/demo_1:", dict(f['data/demo_1']))
	print("data/demo_1/actions:", f['data/demo_1/actions'][0])
	# for i in range(len(f['data/demo_1/actions'])):
	# 	print(f['data/demo_1/actions'][i])

	# env = robosuite.make(
	#     env_name,
	#     has_renderer=True,
	#     ignore_done=True,
	#     use_camera_obs=False,
	#     gripper_visualization=True,
	#     reward_shaping=True,
	#     control_freq=100,
	# )

    # Get controller config
	controller_config = load_controller_config(default_controller=args.controller)
	
	# Create argument configuration
	config = {
		"robots": args.robots,
		"controller_configs": controller_config,
		# "controller_configs": load_controller_config(default_controller="OSC_POSE"),
	}

	if args.bddl_file == None:
		args.bddl_file = os.path.join(args.folder, "task.bddl")
	assert os.path.exists(args.bddl_file)
	problem_info = BDDLUtils.get_problem_info(args.bddl_file)
	# Check if we're using a multi-armed environment and use env_configuration argument if so

	# Create environment
	problem_name = problem_info["problem_name"]
	domain_name = problem_info["domain_name"]
	language_instruction = problem_info["language_instruction"]
	text = colored(language_instruction, "red", attrs=["bold"])
	print("Goal of the following task: ", text)

	if "TwoArm" in problem_name:
		config["env_configuration"] = args.config
	print(language_instruction)
	env = TASK_MAPPING[problem_name](
		bddl_file_name=args.bddl_file,
		**config,
		has_renderer=True,
		has_offscreen_renderer=False,
		render_camera=args.camera,
		ignore_done=True,
		use_camera_obs=False,
		reward_shaping=True,
		control_freq=20,
	)

    # Wrap this with visualization wrapper
	env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
	env_info = json.dumps(config)

	# wrap the environment with data collection wrapper
	tmp_directory = "demonstration_data/tmp/{}_ln_{}/{}".format(
		problem_name,
		language_instruction.replace(" ", "_").strip('""'),
		str(time.time()).replace(".", "_"),
	)
	env = DataCollectionWrapper(env, tmp_directory)

	# make a new timestamped directory
	t1, t2 = str(time.time()).split(".")
	new_dir = os.path.join(
		args.directory,
		f"{domain_name}_ln_{problem_name}_{t1}_{t2}_"
		+ language_instruction.replace(" ", "_").strip('""'),
	)
	os.makedirs(new_dir)

	# list of all demonstrations episodes
	demos = list(f["data"].keys())

	print(f"{len(demos)} total episodes in demo:")
	total = 0
	for i, ep in enumerate(demos):
		states = f[f"data/{ep}/states"]
		print(f"ep {i} has {len(states)} samples")
		total += len(states)
	print(f"Total {total} samples")



	for ep in demos:
		print(f"Recreating episode {ep}... (press ESC to quit)")

		env.reset()
		env.sim.reset()
		env.viewer.set_camera(0)

		# load the flattened mujoco states
		states: h5py.Dataset = f["data/{}/states".format(ep)]#.value

		# load the initial state
		env.sim.set_state_from_flattened(states[0])
		env.sim.forward()

		env.render()

		# load the actions and play them back open-loop
		actions = f[f'data/{ep}/actions']

		no_action_count = 0
		for j, action in enumerate(actions):
			# print(action[:-1])
			if all(map(lambda x: x == 0, action[:-1])):
				no_action_count += 1
			else:
				no_action_count = 0
			if no_action_count > 8: # skip when there is pause of more than 5 steps
				continue
			env.step(action)
			env.render()

		env.close()
		
		gather_demonstrations_as_hdf5(
			tmp_directory, new_dir, problem_info, env_info, args, []
		)

	f.close()