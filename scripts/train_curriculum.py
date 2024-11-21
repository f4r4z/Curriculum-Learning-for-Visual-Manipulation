# add parent path to sys
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
START_TIME = datetime.datetime.now()
START_TIME_STR = START_TIME.strftime("%Y%m%d-%H%M%S")

from dataclasses import dataclass
import tyro
import imageio
from IPython.display import HTML
import wandb
import torch
import numpy as np
from typing import List, Tuple, Optional

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3 import PPO, SAC
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.logger import configure

from src.her_replay_buffer_modified import HerReplayBufferModified
from src.envs_gymapi import LowDimensionalObsGymEnv, LowDimensionalObsGymGoalEnv, AgentViewGymEnv, AgentViewGymGoalEnv
from src.networks import CustomCNN, CustomCombinedExtractor, CustomCombinedExtractor2, CustomCombinedPatchExtractor
from src.callbacks import TensorboardCallback, RLeXploreWithOffPolicyRL, RLeXploreWithOnPolicyRL, VideoWriter, StopTrainingOnSuccessRateThreshold, StopTrainingOnSuccessRateReached

from rllte.xplore.reward import RND, Disagreement, E3B, Fabric, ICM, NGU, PseudoCounts, RE3, RIDE

import inspect

@dataclass
class Args:
    # User specific arguments
    seed: Optional[int] = None
    """random seed for reproducibility"""
    video_path: str = "videos/output.mp4"
    """file path of the video output file"""
    save_path: str = "event_logs"
    """file path of the model output file"""
    load_path: Optional[str] = None # "models/checkpoints/"
    """directory path of the models checkpoints"""
    model_path: Optional[str] = None
    """path to existing model if loading a model"""
    wandb_project: str = "cl_manipulation"
    """wandb project name"""
    wandb_entity: str = "<YOUR_WANDB_ENTITY>"
    """wandb entity name (username)"""
    wandb: bool = False
    """if toggled, model will log to wandb otherwise it would not"""

    # Environment specific arguments
    curriculum_file: Optional[str] = None
    """The path to a python file containing functions that generate BDDL files"""
    visual_observation: bool = False
    """if toggled, the environment will return visual observation otherwise it would not"""
    setup_demo_path: Optional[str] = None
    """If passed in, runs the actions in the given demonstration before every episode to setup the scene"""

    # Algorithm specific arguments
    alg: str = "ppo"
    """algorithm to use for training: ppo, sac"""
    her: bool = False
    """if toggled, SAC will use HER otherwise it would not"""
    exploration_alg: Optional[str] = None
    """algorithm for exploration techniques: rnd, e3b, disagreement, re3, ride, icm"""
    total_timesteps: int = 250000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-3 
    """the learning rate of the optimizer"""
    n_steps: int = 512
    """number of steps to run for each environment per update"""
    num_envs: int = 1
    """number of LIBERO environments"""
    multiprocessing_start_method: Optional[str] = None
    """The start method for starting processes if num_envs > 1. Can be 'fork', 'spawn', or 'forkserver'. 'forkserver' is default"""
    ent_coef: float = 0.0
    """entropy coefficient for the loss calculation"""
    clip_range: float = 0.2
    """Clipping parameter, it can be a function of the current progress remaining (from 1 to 0)."""
    truncate: bool = True
    """if toggled, algorithm with truncate after 250 steps"""
    progress_bar: bool = True
    """if toggled, progress bar will be shown"""
    device: str = ""
    """device to use for training"""

    n_eval_episodes: int = 10
    """number of episodes to run for evaluation and success rate checking. Set to 0 to not eval and always train to total_timesteps"""
    success_rate_threshold: float = 0.7
    """success rate to reach before moving on to the next subtask of the curriculum"""

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

def load_bddls(curriculum_file: str):
    assert curriculum_file is not None
    assert os.path.exists(curriculum_file)
    with open(curriculum_file, 'r') as f:
        curriculum_file_str = f.read()
    
    namespace = {}
    exec(curriculum_file_str, namespace)

    bddls: List[Tuple[str, str]] = []
    for k, func in namespace.items():
        if not inspect.isfunction(func) or inspect.isbuiltin(func): # ignore non-functions and builtins
            continue
        if k.startswith("__") and k.endswith("__"): # ignore functions named __name__
            continue
        if len(inspect.signature(func).parameters) == 0:
            bddl = func()
            if type(bddl) is str:
                bddls.append((func.__name__, bddl))
            elif type(bddl) is list:
                for i, s in enumerate(bddl):
                    assert type(s) is str
                    bddls.append((f"{func.__name__}_{i}", s))
    return bddls

def create_envs(bddl_str: str, tmp_dir = ".", num_envs_override: int = None):
    bddl_path = os.path.join(tmp_dir, f"tmp_bddl_{START_TIME_STR}.bddl")
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    with open(bddl_path, 'w') as f:
        f.write(bddl_str)
    
    env_args = {
        "bddl_file_name": bddl_path,
        "camera_heights": 128,
        "camera_widths": 128,
    }

    if not args.truncate:
        env_args["horizon"] = args.total_timesteps
    
    if args.visual_observation:
        if args.her:
            env_fn = lambda: Monitor(AgentViewGymGoalEnv(**env_args), info_keywords=["is_success"])
        else:
            env_fn = lambda: Monitor(AgentViewGymEnv(**env_args), info_keywords=["is_success"])
    else:
        if args.her:
            env_fn = lambda: Monitor(LowDimensionalObsGymGoalEnv(**env_args), info_keywords=["is_success"])
        else:
            env_fn = lambda: Monitor(LowDimensionalObsGymEnv(setup_demo=args.setup_demo_path, **env_args), info_keywords=["is_success"])
    
    num_envs = args.num_envs if num_envs_override is None else num_envs_override
    if num_envs > 1:
        envs = SubprocVecEnv([env_fn for _ in range(num_envs)], start_method=args.multiprocessing_start_method)
    else:
        envs = DummyVecEnv([env_fn])
    
    os.remove(bddl_path)

    """ ## Speed test
    import time
    envs.reset()
    N = 1000
    start_time = time.time()

    for i in range(N):
        print(f"Step {i}/{N}")
        action = np.random.uniform(-1, 1, size=(args.num_envs, 7))
        obs, rewards, dones, info = envs.step(action)
        if i % 250 == 0:
            envs.reset()

    print("Frames per second: ", N * args.num_envs / (time.time() - start_time)) 

    import ipdb; ipdb.set_trace() """

    return envs


if __name__ == "__main__":
    args = tyro.cli(Args)

    # create run name and save path
    task_name = os.path.splitext(os.path.basename(args.curriculum_file))[0]
    alg_str = args.alg
    if args.her:
        alg_str = f"her_{args.alg}"
    if args.exploration_alg is not None:
        alg_str = f"{args.exploration_alg}_{alg_str}"
    alg_str = f"{alg_str}_seed_{args.seed}"
    run_name = os.path.join(task_name, alg_str, START_TIME_STR)
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

    bddls = load_bddls(args.curriculum_file)
    assert len(bddls) > 0

    print("Setting up model")

    # Create temporary env using first bddls because model requires an env to initialize
    envs = create_envs(bddls[0][1], tmp_dir=os.path.join(save_path, "tmp"))

    # Seeding everything
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.visual_observation:
        policy_kwargs = dict(
            features_extractor_class=CustomCombinedPatchExtractor if args.her else CustomCNN,
            features_extractor_kwargs=dict(features_dim=256),
        )
        policy_class = "MultiInputPolicy" if args.her else "CnnPolicy"
    else:
        policy_kwargs = dict(net_arch=[128, 128])
        policy_class = "MultiInputPolicy" if args.her else "MlpPolicy"
        
    algorithm = None
    if args.alg == "ppo":
        algorithm = PPO
        model = PPO(
            policy_class,
            envs,
            verbose=1,
            policy_kwargs=policy_kwargs,
            learning_rate=args.learning_rate,
            tensorboard_log=os.path.join(save_path, "tensorboard"),
            n_steps=args.n_steps,
            ent_coef=args.ent_coef,
            # clip_range=args.clip_range,
            seed=args.seed
        )
        log_interval = 1
    elif args.alg == "sac":
        algorithm = SAC
        if args.her:
            model = SAC(
                policy_class,
                envs,
                verbose=1,
                policy_kwargs=policy_kwargs,
                tensorboard_log=os.path.join(save_path, "tensorboard"),
                seed=args.seed,
                learning_rate=args.learning_rate,
                learning_starts=1000*args.num_envs,
                batch_size=256,
                train_freq=(1, "step"),
                gradient_steps=-1,
                replay_buffer_class=HerReplayBufferModified,
                replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future',)
            )
        else:
            model = SAC(
                policy_class,
                envs,
                verbose=1,
                policy_kwargs=policy_kwargs,
                tensorboard_log=os.path.join(save_path, "tensorboard"),
                seed=args.seed,
                learning_rate=args.learning_rate,
                learning_starts=1000*args.num_envs,
                batch_size=256,
                train_freq=(1, "step"),
                gradient_steps=-1,
            )
        log_interval = 2
    else:
        raise ValueError(f"Algorithm {args.alg} is not in supported list [ppo, sac]")

    if args.model_path:
        print("loading model from ", args.model_path)
        model = algorithm.load(f"{args.model_path}", env=envs)
        model.learning_rate = args.learning_rate
        model.ent_coef = args.ent_coef
        # model.clip_range = args.clip_range
        model.n_steps = args.n_steps
        model.tensorboard_log = os.path.join(save_path, "tensorboard")
        # new_logger = configure(os.path.join(save_path, "tensorboard"), ["stdout", "tensorboard"])
        # model.set_logger(new_logger)
    
    # get device
    if not args.device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("Start training")
    for i, (subtask_name, bddl) in enumerate(bddls):
        print(f"Starting subtask {i} ({subtask_name}) at step {model.num_timesteps}")

        envs = create_envs(bddl)
        if args.seed is not None:
            envs.seed(args.seed)

        model.set_env(envs)

        '''
        callback list
        '''
        callbacks = []

        # checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=log_interval*32,
            save_path=os.path.join(save_path, "checkpoints"),
            name_prefix="model"
        )
        callbacks.append(checkpoint_callback)

        # log videos
        callbacks.append(VideoWriter(n_steps=5000 * args.num_envs))

        # eval callback
        # Stop training when the model reaches the reward threshold
        if i < len(bddls)-1:
            callbacks.append(StopTrainingOnSuccessRateThreshold(threshold=args.success_rate_threshold, min_count=log_interval*args.num_envs))
            # callbacks.append(StopTrainingOnSuccessRateReached(threshold=args.success_rate_threshold))

        # eval_envs = create_envs(bddl, num_envs_override=1)
        # if i < len(bddls)-1:
        #     callback_on_best = StopTrainingOnSuccessRateThreshold(threshold=args.success_rate_threshold, n_times=3, verbose=1)
        #     eval_callback = EvalCallback(eval_envs, callback_after_eval=callback_on_best, n_eval_episodes=args.n_eval_episodes, eval_freq=args.n_steps, verbose=1)
        # else:
        #     eval_callback = EvalCallback(eval_envs, n_eval_episodes=args.n_eval_episodes, eval_freq=args.n_steps, verbose=1)
        # callbacks.append(eval_callback)

        # exploration technique callbacks
        if args.exploration_alg is not None:
            if args.exploration_alg == "rnd":
                irs = RND(envs, device=device, rwd_norm_type='none', obs_norm_type='none')
            elif args.exploration_alg == "disagreement":
                irs = Disagreement(envs, device=device, rwd_norm_type='none', obs_norm_type='none')
            elif args.exploration_alg == "e3b":
                irs = E3B(envs, device=device, rwd_norm_type='none', obs_norm_type='none')
            elif args.exploration_alg == "icm":
                irs = ICM(envs, device=device, rwd_norm_type='none', obs_norm_type='none')
            elif args.exploration_alg == "pseudocounts":
                irs = PseudoCounts(envs, device=device, rwd_norm_type='rms', obs_norm_type='none')
            elif args.exploration_alg == "ride":
                irs = RIDE(envs, device=device, rwd_norm_type='rms', obs_norm_type='none')
            else:
                print("You have typed an invalid exploration technique. --help for more information.")
                exit(1)
            # on policy vs off policy
            if args.alg == 'ppo':
                callbacks.append(RLeXploreWithOnPolicyRL(irs))
            else:
                callbacks.append(RLeXploreWithOffPolicyRL(irs))
        
        '''train'''
        
        # reset these buffers so the training stats for the previous subtask doesn't leak into this subtask
        model.ep_info_buffer = None
        model.ep_success_buffer = None

        model.learn(
            total_timesteps=args.total_timesteps,
            tb_log_name=f"{i}_{subtask_name}",
            log_interval=log_interval,
            callback=callbacks,
            reset_num_timesteps=False,
            progress_bar=False
        )

        wandb.save(os.path.join(save_path, "tensorboard"), base_path=save_path, policy='now')
        model.save(os.path.join(save_path, "models", f"{i}_{subtask_name}"))

        envs.close()
        # eval_envs.close()

    del model