import os
# add parent path to sys
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback
from libero.libero.envs import SubprocVectorEnv, OffScreenRenderEnv, DummyVectorEnv
from libero.libero import get_libero_path

from src.envs import LowDimensionalObsEnv, AgentViewEnv, GymVecEnvs
from src.callbacks import TensorboardCallback

from IPython.display import display, HTML
from PIL import Image
import imageio
from dataclasses import dataclass

import tyro

@dataclass
class Args:
    video_filename: str = "output.mp4"
    """filename of the video output file"""
    model_filename: str = "close_the_microwave"
    # Algorithm specific arguments
    train: bool = False
    """if toggled, model will train otherwise it would not"""
    eval: bool = True
    """if toggled, model will load and evaluate a model otherwise it would not"""
    total_timesteps: int = 250000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """number of LIBERO environments"""
    save_freq: int = 10000
    "save frequency of model checkpoint during training"

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

    print("setting up environment")

    '''
    envs = SubprocVectorEnv(
        [lambda: LowDimensionalObsEnv(**env_args) for _ in range(args.num_envs)]
    )
    '''
    
    envs = DummyVectorEnv(
        [lambda: AgentViewEnv(**env_args) for _ in range(args.num_envs)]
    )
    envs = GymVecEnvs(envs)
    # import ipdb; ipdb.set_trace()

    # Create the agent with two layer of 128 units
    if args.train:
        print("training")
        checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path='models/model_checkpoints/', name_prefix="pulisic_ppo_agentview_model")
        tensorboard_callback = TensorboardCallback()
        model = PPO("CnnPolicy", envs, verbose=1, policy_kwargs=dict(net_arch=[128, 128]), tensorboard_log="./logs")
        model.learn(total_timesteps=args.total_timesteps, log_interval=1, progress_bar=True, callback=[checkpoint_callback, tensorboard_callback])
        model.save(f"models/{args.model_filename}")

        del model

    if args.eval:
        print("loading model")
        model = PPO.load(f"models/{args.model_filename}")

        obs = envs.reset()

        # second environment for visualization
        off_env = OffScreenRenderEnv(**env_args)
        off_env.reset()
        images = []

        for i in range(500):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = envs.step(action)

            # for visualization
            off_obs, _, _, _, = off_env.step(action[0])
            images.append(off_obs["agentview_image"])
            


        obs_to_video(images, f"videos/{args.video_filename}")
        off_env.close()
        envs.close()