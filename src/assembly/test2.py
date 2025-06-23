import os
import sys
import pickle
import json
import logging
import torch
import wandb
from stable_baselines3 import SAC
import pika
import redis
import zstandard as zstd
from PIL import Image
from redis.retry import Retry
from redis.backoff import ExponentialBackoff
from stable_baselines3.common.callbacks import BaseCallback
from typing import Any, List
from pydantic import BaseModel


class DataRecord(BaseModel):
    episode: int
    frames: List[Any]
    actions: List[int]
    batch_id: int
    is_last_batch: bool


class DataCollectorCallback(BaseCallback):
    def __init__(self, save_path="./models", enable_rmq=True, save_locally=False, frames_to_skip=4, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.enable_rmq = enable_rmq
        self.save_locally = save_locally
        self.frames_to_skip = frames_to_skip

        os.makedirs(save_path, exist_ok=True)
        self.connection = None
        self.channel = None
        self.redis_client = None
        self.episode_keys_buffer = []
        self.frames_buffer = []
        self.actions_buffer = []
        self.max_batch_size = 500 * 1024 * 1024  # 500 MB

    def _setup_communication(self):
        """Set up RabbitMQ and Redis connections."""
        if not self.enable_rmq:
            logging.info("RMQ and Redis setup skipped because enable_rmq=False")
            return

        try:
            credentials = pika.PlainCredentials('robot', 'robot_pass')
            parameters = pika.ConnectionParameters(
                host='localhost',
                port=5672,
                virtual_host='/',
                credentials=credentials
            )
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            self.channel.queue_declare(queue='HF_upload_queue')
            logging.info("RabbitMQ connection and queue declared successfully.")
        except Exception as e:
            logging.error(f"Failed to setup RabbitMQ: {e}")

        try:
            self.redis_client = redis.StrictRedis(
                host='redis',
                port=6379,
                db=0,
                decode_responses=False,
                password=os.getenv('REDIS_PASSWORD', None),
                health_check_interval=30,
                socket_keepalive=True,
                retry=Retry(ExponentialBackoff(cap=10, base=1), 25),
                retry_on_error=[ConnectionError, TimeoutError]
            )
            self.redis_client.ping()
            logging.info("Redis client initialized and connection tested successfully.")
        except Exception as e:
            logging.error(f"Failed to setup Redis client: {e}")

    def _publish_keys_to_queue(self):
        if self.enable_rmq:
            message = json.dumps(self.episode_keys_buffer)
            self.channel.basic_publish(exchange='', routing_key='HF_upload_queue', body=message)
            logging.info(f"Published keys to RabbitMQ queue: {self.episode_keys_buffer}")
            self.episode_keys_buffer.clear()

    def _close_rabbitmq(self):
        if self.connection:
            self.connection.close()

    def _save_data_to_redis(self, episode):
        """Save current buffer to Redis with compression."""
        logging.info("Saving data to Redis")
        key = f"episode_{episode}"
        data = {'episode': episode, 'frames': self.frames_buffer, 'actions': self.actions_buffer}
        serialized_data = pickle.dumps(data)

        cctx = zstd.ZstdCompressor()
        compressed_data = cctx.compress(serialized_data)

        logging.info(f"Original size: {sys.getsizeof(serialized_data)} bytes, Compressed size: {len(compressed_data)} bytes")

        self.redis_client.set(key, compressed_data)
        self.episode_keys_buffer.append(key)
        self.clear_buffers()

        if len(self.episode_keys_buffer) >= 20:
            self._publish_keys_to_queue()

    def _get_buffer_size(self):
        """Estimate current buffer size in bytes."""
        frame_size = sum(frame.nbytes for frame in self.frames_buffer)
        action_size = sum(sys.getsizeof(action) for action in self.actions_buffer)
        return frame_size + action_size

    def _save_frames_locally(self, episode):
        """Save frames as PNG locally."""
        episode_dir = f"episode_{episode}_frs{self.frames_to_skip}"
        os.makedirs(episode_dir, exist_ok=True)

        for idx, frame in enumerate(self.frames_buffer):
            action = self.actions_buffer[idx]
            filename = os.path.join(episode_dir, f"{idx:05d}_action_{action}.png")
            Image.fromarray(frame).save(filename)

    def clear_buffers(self):
        """Clear the in-memory frame and action buffers."""
        self.frames_buffer.clear()
        self.actions_buffer.clear()

    def _on_training_start(self):
        self._setup_communication()

    def _on_training_end(self):
        self._close_rabbitmq()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        actions = self.locals.get("actions", None)

        for i, info in enumerate(infos):
            frame = info.get("frame")
            action = actions[i] if actions is not None else None
            done = info.get("done", False)
            episode_idx = info.get("episode_idx", -1)

            if frame is not None and action is not None:
                self.frames_buffer.append(frame)
                self.actions_buffer.append(action)

            if self.enable_rmq:
                buffer_size = self._get_buffer_size()
                logging.info(f"Current buffer size: {buffer_size} bytes")
                if buffer_size >= self.max_batch_size:
                    logging.warning("Buffer exceeded 500MB, saving to Redis")
                    self._save_data_to_redis(episode_idx)

            if done:
                logging.info(f"Episode {episode_idx} done, saving remaining buffer immediately.")
                if self.save_locally:
                    self._save_frames_locally(episode_idx)
                if self.enable_rmq:
                    self._save_data_to_redis(episode_idx)
                self.clear_buffers()

        return True


class TrainLoggingCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
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


class WandbCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                wandb.log({
                    "reward": info['episode']['r'],
                    "length": info['episode']['l'],
                    "timesteps": self.num_timesteps
                })
        return True


# Example usage
if __name__ == "__main__":
    import gymnasium as gym
    from stable_baselines3 import SAC

    env = gym.make("Pendulum-v1")

    data_collector_cb = DataCollectorCallback( save_path="./models", enable_rmq=True, save_locally=True)
    train_logger_cb = TrainLoggingCallback(save_freq=5000, save_path="./models")

    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000, callback=[data_collector_cb, train_logger_cb])
