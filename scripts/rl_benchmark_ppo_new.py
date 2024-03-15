import os
# add parent path to sys
import sys
import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from libero.libero.envs import SubprocVectorEnv, OffScreenRenderEnv, DummyVectorEnv
from libero.libero import get_libero_path

from src.envs import LowDimensionalObsEnv, GymVecEnvs

from IPython.display import display, HTML
from PIL import Image
import imageio
from dataclasses import dataclass

import tyro

@dataclass
class Args:
    video_path: str = "videos/output.mp4"
    """file path of the video output file"""
    save_path: str = "logs"
    """file path of the model output file"""
    load_path: str = None # "models/checkpoints/"
    """directory path of the models checkpoints"""
    # Algorithm specific arguments
    train: bool = True
    """if toggled, model will train otherwise it would not"""
    eval: bool = False
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
        [lambda: Monitor(LowDimensionalObsEnv(**env_args)) for _ in range(args.num_envs)]
    )
    envs = GymVecEnvs(envs)
    # import ipdb; ipdb.set_trace()

    # Create the agent with two layer of 128 units
    if args.train:
        save_path = os.path.join(args.save_path, task_name, "ppo", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print("training")
        # checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path=save_path, name_prefix="pulisic_ppo_model")
        model = PPO("MlpPolicy", envs, verbose=1, policy_kwargs=dict(net_arch=[128, 128]), tensorboard_log=save_path)
        model.learn(total_timesteps=args.total_timesteps, log_interval=1)  # , callback=checkpoint_callback)
        model.save(save_path)

        del model

    if args.eval:
        print("loading model")
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