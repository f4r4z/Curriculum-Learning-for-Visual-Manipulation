from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv, DummyVectorEnv, SubprocVectorEnv
import os

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.nn import functional as F

import numpy as np
import time
# import tyro
import random

def obs_to_tensor(obs):
    '''
    converts a libero observation to tensor
    '''
    list_of_tensors = [torch.tensor(value) for value in obs.values()]
    max_size = 0
    for i in list_of_tensors:
        max_size = max(max_size, i.shape[0])
    list_of_padded = []
    for i in list_of_tensors:
        list_of_padded.append(F.pad(i, (0, max_size - i.shape[0])))
    final_list = []
    image_tensor = torch.randn(128,128,3)
    for i in list_of_padded:
        if len(i.shape) == 1:
            i = torch.unsqueeze(i, 1)
            i = torch.unsqueeze(i, 2)
            i = F.pad(i, (0, image_tensor.shape[-1] - i.shape[-1]))
        final_list.append(i)
    obs_tensor = torch.cat(final_list, dim=1)

    return obs_tensor

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    '''
    taken from cleanrl
    '''
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape[-1]).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape[-1]).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.shape[0]), std=0.01),
        )
        # added 
        self.double()

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
# TODO: create python argument from these
action_dim = 7
num_envs = 1
num_steps = 20
num_minibatches = 5
seed = 10
env_id = 0
exp_name = 0
total_timesteps = 20
learning_rate = 1e-4
torch_deterministic = True
cuda = False
anneal_lr = False
gamma = 1e-3
gae_lambda = 1e-2
update_epochs = 20
clip_coef = 1e-2
norm_adv = False
clip_vloss = False
ent_coef = 1e-2
vf_coef = 1e-2
max_grad_norm = 1e-2
target_kl = 1e-2

# using demo task for now. TODO: implement with custom tasks
benchmark_name = 'libero_10'
task_id = 0
bddl_file = 'bddl/task_0.bddl' # manually added, later will be automatically generated for custom tasks

# args = tyro.cli(Args)
batch_size = int(num_envs * num_steps)
minibatch_size = int(batch_size // num_minibatches)
num_iterations = total_timesteps // batch_size
run_name = f"{env_id}__{exp_name}__{seed}__{int(time.time())}"

writer = SummaryWriter(f"runs/{run_name}")
# writer.add_text(
#     "hyperparameters",
#     "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
# )

# TRY NOT TO MODIFY: seeding
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = torch_deterministic

device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

# env setup
# envs = gym.vector.SyncVectorEnv(
#     [make_env(env_id, i, args.capture_video, run_name) for i in range(num_envs)],
# )
# assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
benchmark_instance = benchmark.get_benchmark_dict()[benchmark_name]()
env_args = {
    "bddl_file_name": bddl_file,
    "camera_heights": 128,
    "camera_widths": 128,
}

os.makedirs("benchmark_tasks", exist_ok=True)

# these two lines are only necessary for retrieving task suites, custom tasks wouldn't need these
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict[benchmark_name]()

task = benchmark_instance.get_task(task_id)
init_states = benchmark_instance.get_task_init_states(task_id)

# TODO: create the possiblity to add  multiple envs
# envs = DummyVectorEnv(
#             [lambda: OffScreenRenderEnv(**env_args) for _ in range(num_envs)]
# )
envs = OffScreenRenderEnv(**env_args)

envs.seed(seed)
envs.reset()
init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
init_state_id = 0
next_obs = envs.set_init_state(init_states[init_state_id])

next_obs = obs_to_tensor(next_obs)
# print(next_obs.shape)
envs.single_observation_space = next_obs
envs.single_action_space = torch.randn(action_dim)

agent = Agent(envs).to(device)
optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

# ALGO Logic: Storage setup
obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
logprobs = torch.zeros((num_steps, num_envs)).to(device)
rewards = torch.zeros((num_steps, num_envs)).to(device)
dones = torch.zeros((num_steps, num_envs)).to(device)
values = torch.zeros((num_steps, num_envs)).to(device)

# TRY NOT TO MODIFY: start the game
global_step = 0
start_time = time.time()
envs.seed(seed)
# next_obs, _ = envs.reset(seed=seed)
# next_obs = torch.Tensor(next_obs).to(device)
next_done = torch.zeros(num_envs).to(device)

for iteration in range(1, num_iterations + 1):
    # Annealing the rate if instructed to do so.
    if anneal_lr:
        frac = 1.0 - (iteration - 1.0) / num_iterations
        lrnow = frac * learning_rate
        optimizer.param_groups[0]["lr"] = lrnow

    for step in range(0, num_steps):
        global_step += num_envs
        obs[step] = next_obs
        dones[step] = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            print(value.shape, action.shape, logprob.shape) # TODO: the action shape is 3-dimensional with action_dim as the last dimension. This causes issue in the next line
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        # TRY NOT TO MODIFY: execute the game and log data.
        # next_obs, reward, terminations, infos = envs.step(action.cpu().numpy())
        next_obs, reward, terminations, infos = envs.step(action.cpu().numpy())
        next_obs = obs_to_tensor(next_obs) # convert to tensor

        next_done = terminations
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

    # bootstrap value if not done
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values

    # flatten the batch
    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    # Optimizing the policy and value network
    b_inds = np.arange(batch_size)
    clipfracs = []
    for epoch in range(update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]
            if norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

        if target_kl is not None and approx_kl > target_kl:
            break

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    writer.add_scalar("losses/explained_variance", explained_var, global_step)
    print("SPS:", int(global_step / (time.time() - start_time)))
    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

envs.close()
writer.close()