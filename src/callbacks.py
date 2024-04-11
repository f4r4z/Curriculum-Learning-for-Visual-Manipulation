from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

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