import os
import logging
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

# Your existing imports like queue, redis, pickle, etc. assumed here

class TrainLoggingCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq  # Save and log every `save_freq` timesteps
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info(torch.cuda.memory_summary())
            model_file = os.path.join(self.save_path, f"pacman_{self.num_timesteps}.zip")
            self.model.save(model_file)
            logging.info(f"Model saved at timestep {self.num_timesteps} to {model_file}")
        return True

# Optional: Wandb callback (requires wandb setup)
# import wandb
# class WandbCallback(BaseCallback):
#     def _on_step(self) -> bool:
#         infos = self.locals.get('infos', [])
#         for info in infos:
#             if 'episode' in info:
#                 wandb.log({
#                     "reward": info['episode']['r'],
#                     "length": info['episode']['l'],
#                     "timesteps": self.num_timesteps
#                 })
#         return True


class DataCollector:
    def __init__(self, env, episodes, frames_to_skip, save_locally, enable_rmq):
        self.env = env
        self.episodes = episodes
        self.frames_to_skip = frames_to_skip
        self.save_locally = save_locally
        self.enable_rmq = enable_rmq

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Initialize the SAC agent with your env and policy
        self.agent = SAC('CnnPolicy', env, verbose=1)

    def train(self):
        if self.enable_rmq:
            self._setup_rabbitmq()  # Your existing RMQ setup method

        max_steps_per_episode = 1000  # Adjust based on your env
        total_timesteps = self.episodes * max_steps_per_episode

        save_path = "./models"
        os.makedirs(save_path, exist_ok=True)
        callback = TrainLoggingCallback(save_freq=1000, save_path=save_path)

        logging.info(f"Starting training for {total_timesteps} timesteps")

        self.agent.learn(total_timesteps=total_timesteps, callback=callback)

        logging.info("Training Complete")

        self.env.close()

        if self.enable_rmq:
            self._close_rabbitmq()  # Your existing RMQ close method

    # Placeholder for RMQ setup/close methods if you have them
    def _setup_rabbitmq(self):
        pass
    def _close_rabbitmq(self):
        pass


# Example usage:
if __name__ == "__main__":
    import gym
    env = gym.make("Pendulum-v1")  # Or your custom Pacman env
    collector = DataCollector(env=env, episodes=10, frames_to_skip=4, save_locally=False, enable_rmq=False)
    collector.train()
