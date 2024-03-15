from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.f = open("reward_logs.txt", "w")

    def _on_step(self) -> bool:              
        self.logger.record('average reward', np.mean(self.training_env.rewards))
        # np.savetxt(self.f, self.training_env.rewards, fmt='%d')
        self.f.write(f'{str(np.mean(self.training_env.rewards))}\n')
        return True