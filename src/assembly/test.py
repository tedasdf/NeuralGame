
# class DataCollector:
#     def __init__(self, env, episodes, frames_to_skip, save_locally, enable_rmq):
#         self.env = env
#         self.episodes = episodes
#         self.frames_to_skip = frames_to_skip
#         self.save_locally = save_locally
#         self.enable_rmq = enable_rmq

#         logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#         # Initialize the SAC agent with your env and policy
#         self.agent = SAC('MlpPolicy', env, verbose=1)

#     def train(self):
#         if self.enable_rmq:
#             self._setup_rabbitmq()  # Your existing RMQ setup method

#         max_steps_per_episode = 1000  # Adjust based on your env
#         total_timesteps = self.episodes * max_steps_per_episode

#         save_path = "./models"
#         os.makedirs(save_path, exist_ok=True)
#         callback = TrainLoggingCallback(save_freq=1000, save_path=save_path)

#         logging.info(f"Starting training for {total_timesteps} timesteps")

#         self.agent.learn(total_timesteps=total_timesteps, callback=callback)

#         logging.info("Training Complete")

#         self.env.close()

#         if self.enable_rmq:
#             self._close_rabbitmq()  # Your existing RMQ close method

#     # Placeholder for RMQ setup/close methods if you have them
#     def _setup_rabbitmq(self):
#         pass
#     def _close_rabbitmq(self):
#         pass
