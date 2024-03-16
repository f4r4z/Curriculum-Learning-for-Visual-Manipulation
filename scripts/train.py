# add parent path to sys
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
from dataclasses import dataclass
import tyro
import imageio
from IPython.display import HTML
import wandb
import torch
import numpy as np

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO

from src.envs import LowDimensionalObsGymEnv

@dataclass
class Args:
    # User specific arguments
    seed: int = None
    """random seed for reproducibility"""
    video_path: str = "videos/output.mp4"
    """file path of the video output file"""
    save_path: str = "logs"
    """file path of the model output file"""
    load_path: str = None # "models/checkpoints/"
    """directory path of the models checkpoints"""
    wandb_project: str = "cl_manipulation"
    """wandb project name"""
    wandb_entity: str = "<YOUR_WANDB_ENTITY>"
    """wandb entity name (username)"""
    wandb: bool = False
    """if toggled, model will log to wandb otherwise it would not"""

    # Environment specific arguments
    bddl_file_name: str = "libero_90/KITCHEN_SCENE6_close_the_microwave.bddl"

    # Algorithm specific arguments
    train: bool = True
    """if toggled, model will train otherwise it would not"""
    eval: bool = False
    """if toggled, model will load and evaluate a model otherwise it would not"""
    total_timesteps: int = 250000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    n_steps: int = 512
    """number of steps to run for each environment per update"""
    num_envs: int = 16
    """number of LIBERO environments"""
    save_freq: int = 10000
    "save frequency of model checkpoint during training"
    ent_coef: float = 0.01
    """entropy coefficient for the loss calculation"""

def obs_to_video(images, filename):
    """
    converts a list of images to video and writes the file
    """
    video_writer = imageio.get_writer(filename, fps=60)
    for image in images:
        video_writer.append_data(image[::-1])
    video_writer.close()
    HTML("""
        <video width="640" height="480" controls>
            <source src="output.mp4" type="video/mp4">
        </video>
        <script>
            var video = document.getElementsByTagName('video')[0];
            video.playbackRate = 2.0; // Increase the playback speed to 2x
            </script>    
    """)

if __name__ == "__main__":
    args = tyro.cli(Args)

    bddl_file_base = get_libero_path("bddl_files")
    task_name = "libero_90/KITCHEN_SCENE6_close_the_microwave.bddl"
    env_args = {
        "bddl_file_name": os.path.join(bddl_file_base, task_name),
        "camera_heights": 128,
        "camera_widths": 128,
    }

    print("Setting up environment")

    envs = SubprocVecEnv(
        [lambda: Monitor(LowDimensionalObsGymEnv(**env_args)) for _ in range(args.num_envs)]
    )

    # Seeding everything
    if args.seed is not None:
        envs.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.train:
        print("Start training")
        run_name = os.path.join(task_name, "ppo", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        save_path = os.path.join(args.save_path, run_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if args.wandb:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                dir=save_path,
                config=vars(args),
                sync_tensorboard=True,
                name=run_name
            )
        call_back = CheckpointCallback(save_freq=args.save_freq, save_path=save_path, name_prefix="model")
        model = PPO(
            "MlpPolicy",
            envs,
            verbose=1,
            policy_kwargs=dict(net_arch=[128, 128]),
            tensorboard_log=save_path,
            n_steps=args.n_steps,
            ent_coef=args.ent_coef,
            seed=args.seed
        )
        model.learn(total_timesteps=args.total_timesteps, log_interval=1, callback=call_back)
        model.save(save_path)

        del model

    if args.eval:
        assert args.load_path is not None, "Please provide a model checkpoint path"
        print("Start evaluation")
        model = PPO.load(args.load_path)

        obs = envs.reset()

        # second environment for visualization
        off_env = OffScreenRenderEnv(**env_args)
        off_env.reset()
        images = []

        rews = 0

        for i in range(500):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = envs.step(action)

            rews += rewards[0]
            if dones[0]:
                print("Env Reset, Episode Reward: ", rews)

            # for visualization
            off_obs, _, _, _, = off_env.step(action[0])
            images.append(off_obs["agentview_image"])
            
            # if done, stop
            if dones[0]:
                break

        obs_to_video(images, f"{args.video_path}")
        off_env.close()
        envs.close()
