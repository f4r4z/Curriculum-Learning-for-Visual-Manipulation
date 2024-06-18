import copy
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.her.goal_selection_strategy import KEY_TO_GOAL_STRATEGY, GoalSelectionStrategy
from stable_baselines3.her import HerReplayBuffer


class HerReplayBufferModified(HerReplayBuffer):
    """
    Modified version of Hindsight Experience Replay (HER) buffer.
    """
    env: Optional[VecEnv]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        env: VecEnv,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        n_sampled_goal: int = 4,
        goal_selection_strategy: Union[GoalSelectionStrategy, str] = "future",
        copy_info_dict: bool = False,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            env,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
            n_sampled_goal,
            goal_selection_strategy,
            copy_info_dict,
        )

    def _get_virtual_samples(
        self,
        batch_indices: np.ndarray,
        env_indices: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:
        """
        Get the samples, sample new desired goals and compute new rewards.

        :param batch_indices: Indices of the transitions
        :param env_indices: Indices of the envrionments
        :param env: associated gym VecEnv to normalize the
            observations/rewards when sampling, defaults to None
        :return: Samples, with new desired goals and new rewards
        """
        # Get infos and obs
        obs = {key: obs[batch_indices, env_indices, :] for key, obs in self.observations.items()}
        next_obs = {key: obs[batch_indices, env_indices, :] for key, obs in self.next_observations.items()}
        if self.copy_info_dict:
            # The copy may cause a slow down
            infos = copy.deepcopy(self.infos[batch_indices, env_indices])
        else:
            infos = [{} for _ in range(len(batch_indices))]
        # Sample and set new goals
        new_goals = self._sample_goals(batch_indices, env_indices)
        obs["desired_goal"] = new_goals
        # The desired goal for the next observation must be the same as the previous one
        next_obs["desired_goal"] = new_goals

        assert (
            self.env is not None
        ), "You must initialize HerReplayBuffer with a VecEnv so it can compute rewards for virtual transitions"
        # Compute new reward
        rewards = self.env.env_method(
            "compute_reward",
            # the new state depends on the previous state and action
            # s_{t+1} = f(s_t, a_t)
            # so the next achieved_goal depends also on the previous state and action
            # because we are in a GoalEnv:
            # r_t = reward(s_t, a_t) = reward(next_achieved_goal, desired_goal)
            # therefore we have to use next_obs["achieved_goal"] and not obs["achieved_goal"]
            next_obs["achieved_goal"],
            # here we use the new desired goal
            obs["desired_goal"],
            infos,
            # we use the method of the first environment assuming that all environments are identical.
            indices=[0],
        )
        rewards = rewards[0].astype(np.float32)  # env_method returns a list containing one element
        obs = self._normalize_obs(obs, env)  # type: ignore[assignment]
        next_obs = self._normalize_obs(next_obs, env)  # type: ignore[assignment]

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs.items()}

        new_dones = self.to_torch(
            np.logical_or(self.dones[batch_indices, env_indices] * (1 - self.timeouts[batch_indices, env_indices]), rewards > 0)
        ).reshape(-1, 1)

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_indices, env_indices]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=new_dones,
            rewards=self.to_torch(self._normalize_reward(rewards.reshape(-1, 1), env)),  # type: ignore[attr-defined]
        )
