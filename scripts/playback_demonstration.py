"""
A convenience script to playback random demonstrations from
a set of demonstrations stored in a hdf5 file.

Arguments:
	--folder (str): Path to demonstrations
	--use_actions (optional): If this flag is provided, the actions are played back 
		through the MuJoCo simulator, instead of loading the simulator states
		one by one.

Example:
	If there is a demo.hdf5 and task.bddl in demonstrations/open_the_top_drawer_of_the_cabinet:
	$ python playback_demonstrations.py --folder demonstrations/open_the_top_drawer_of_the_cabinet/
"""

# TODO: Old structure / demos are deprecated, need to update!

import os
import h5py
import argparse
import random
import numpy as np

import robosuite
# from robosuite.utils.mjcf_utils import postprocess_model_xml

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *
from termcolor import colored
from libero.lifelong.datasets import get_dataset

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
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
		"--config",
		type=str,
		default="single-arm-opposed",
		help="Specified environment configuration if necessary",
	)
	parser.add_argument("--bddl-file", type=str, default=None)
	parser.add_argument("--folder", type=str, default=None)
	parser.add_argument(
		"--use-actions", 
		action='store_true',
	)
	args = parser.parse_args()

	demo_path = args.folder
	hdf5_path = os.path.join(demo_path, "demo.hdf5")
	f = h5py.File(hdf5_path, "r")
	# env_name = f["data"].attrs["env"]

	# env = robosuite.make(
	#     env_name,
	#     has_renderer=True,
	#     ignore_done=True,
	#     use_camera_obs=False,
	#     gripper_visualization=True,
	#     reward_shaping=True,
	#     control_freq=100,
	# )
	
	# Create argument configuration
	config = {
		"robots": args.robots,
		# "controller_configs": controller_config,
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

	# list of all demonstrations episodes
	demos = list(f["data"].keys())

	print(f"{len(demos)} total episodes in demo:")
	total = 0
	for i, ep in enumerate(demos):
		states = f[f"data/{ep}/states"]
		print(f"ep {i} has {len(states)} samples")
		total += len(states)
	print(f"Total {total} samples")

	while True:
		print("Playing back random episode... (press ESC to quit)")

		# # select an episode randomly
		ep = random.choice(demos)

		# read the model xml, using the metadata stored in the attribute for this episode
		model_file = f["data/{}".format(ep)].attrs["model_file"]
		# model_path = os.path.join(demo_path, "models", model_file)
		# with open(model_path, "r") as model_f:
		#     model_xml = model_f.read()

		env.reset()
		# # xml = postprocess_model_xml(model_xml)
		# xml = model_xml
		# env.reset_from_xml_string(xml)
		env.sim.reset()
		env.viewer.set_camera(0)

		# load the flattened mujoco states
		states: h5py.Dataset = f["data/{}/states".format(ep)]#.value
		# states.values()
		# Datas
		# exit()
		#.value

		if args.use_actions:

			# load the initial state
			env.sim.set_state_from_flattened(states[0])
			env.sim.forward()

			# load the actions and play them back open-loop
			jvels = f["data/{}/joint_velocities".format(ep)].value
			grip_acts = f["data/{}/gripper_actuations".format(ep)].value
			actions = np.concatenate([jvels, grip_acts], axis=1)
			num_actions = actions.shape[0]

			for j, action in enumerate(actions):
				env.step(action)
				env.render()

				if j < num_actions - 1:
					# ensure that the actions deterministically lead to the same recorded states
					state_playback = env.sim.get_state().flatten()
					assert(np.all(np.equal(states[j + 1], state_playback)))

		else:

			# force the sequence of internal mujoco states one by one
			for state in states:
				env.sim.set_state_from_flattened(state)
				env.sim.forward()
				env.render()

	f.close()