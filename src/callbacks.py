from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm
import numpy as np
import torch as th

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        for index, custom_attributes in enumerate(self.training_env.get_attr('custom_attr')):
            self.logger.record(f'total reward for env {index}', custom_attributes.get('total_reward'))
            self.logger.record(f'reward for env {index}', custom_attributes.get('reward'))
        return True

class DebugCallback(BaseCallback):
    """
    A custom callback for logging debugging criteria
    """
    def __init__(self, verbose=0):
        super(DebugCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        pass
    
class RLeXploreWithOnPolicyRL(BaseCallback):
    """
    A custom callback for combining RLeXplore and on-policy algorithms from SB3.
    """
    def __init__(self, irs, verbose=0):
        super(RLeXploreWithOnPolicyRL, self).__init__(verbose)
        self.irs = irs
        self.buffer = None

    def init_callback(self, model: BaseAlgorithm) -> None:
        super().init_callback(model)
        self.buffer = self.model.rollout_buffer

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        observations = self.locals["obs_tensor"]
        device = observations.device
        actions = th.as_tensor(self.locals["actions"], device=device)
        rewards = th.as_tensor(self.locals["rewards"], device=device)
        dones = th.as_tensor(self.locals["dones"], device=device)
        next_observations = th.as_tensor(self.locals["new_obs"], device=device)

        # ===================== watch the interaction ===================== #
        self.irs.watch(observations, actions, rewards, dones, dones, next_observations)
        # ===================== watch the interaction ===================== #
        return True

    def _on_rollout_end(self) -> None:
        # ===================== compute the intrinsic rewards ===================== #
        # prepare the data samples
        obs = th.as_tensor(self.buffer.observations)
        # get the new observations
        new_obs = obs.clone()
        new_obs[:-1] = obs[1:]
        new_obs[-1] = th.as_tensor(self.locals["new_obs"])
        actions = th.as_tensor(self.buffer.actions)
        rewards = th.as_tensor(self.buffer.rewards)
        dones = th.as_tensor(self.buffer.episode_starts)
        print(obs.shape, actions.shape, rewards.shape, dones.shape, obs.shape)
        # compute the intrinsic rewards
        intrinsic_rewards = self.irs.compute(
            samples=dict(observations=obs, actions=actions, 
                         rewards=rewards, terminateds=dones, 
                         truncateds=dones, next_observations=new_obs),
            sync=True)
        # add the intrinsic rewards to the buffer
        self.buffer.advantages += intrinsic_rewards.cpu().numpy()
        self.buffer.returns += intrinsic_rewards.cpu().numpy()
        # ===================== compute the intrinsic rewards ===================== #

class RLeXploreWithOffPolicyRL(BaseCallback):
    """
    A custom callback for combining RLeXplore and off-policy algorithms from SB3. 
    """
    def __init__(self, irs, verbose=0):
        super(RLeXploreWithOffPolicyRL, self).__init__(verbose)
        self.irs = irs
        self.buffer = None

    def init_callback(self, model: BaseAlgorithm) -> None:
        super().init_callback(model)
        self.buffer = self.model.replay_buffer
        

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        device = self.irs.device
        obs = th.as_tensor(self.locals['self']._last_obs, device=device)
        actions = th.as_tensor(self.locals["actions"], device=device)
        rewards = th.as_tensor(self.locals["rewards"], device=device)
        dones = th.as_tensor(self.locals["dones"], device=device)
        next_obs = th.as_tensor(self.locals["new_obs"], device=device)
        # ===================== watch the interaction ===================== #
        self.irs.watch(obs, actions, rewards, dones, dones, next_obs)
        # ===================== watch the interaction ===================== #
        # ===================== compute the intrinsic rewards ===================== #
        intrinsic_rewards = self.irs.compute(samples={'observations':obs.unsqueeze(0).float(), 
                                            'actions':actions.unsqueeze(0), 
                                            'rewards':rewards.unsqueeze(0),
                                            'terminateds':dones.unsqueeze(0),
                                            'truncateds':dones.unsqueeze(0),
                                            'next_observations':next_obs.unsqueeze(0).float()}, 
                                            sync=False)
        # ===================== compute the intrinsic rewards ===================== #

        try:
            # add the intrinsic rewards to the original rewards
            self.locals['rewards'] += intrinsic_rewards.cpu().numpy().squeeze()
            # update the intrinsic reward module
            replay_data = self.buffer.sample(batch_size=self.irs.batch_size)
            self.irs.update(samples={'observations': th.as_tensor(replay_data.observations).unsqueeze(1).to(device), # (n_steps, n_envs, *obs_shape)
                                     'actions': th.as_tensor(replay_data.actions).unsqueeze(1).to(device),
                                     'rewards': th.as_tensor(replay_data.rewards).to(device),
                                     'terminateds': th.as_tensor(replay_data.dones).to(device),
                                     'truncateds': th.as_tensor(replay_data.dones).to(device),
                                     'next_observations': th.as_tensor(replay_data.next_observations).unsqueeze(1).to(device)
                                     })
        except:
            pass

        return True

    def _on_rollout_end(self) -> None:
        pass