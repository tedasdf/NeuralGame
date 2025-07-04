from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from gym.spaces import Dict as DictSpace
import torch.nn as nn
import torch
import gym

class MultiCamFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2),  # (3, 64, 64) -> (16, 31, 31)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2),  # (32, 15, 15)
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Compute output size dynamically
        with torch.no_grad():
            dummy_img = torch.zeros(1, 3, 64, 64)
            n_flat = self.cnn(dummy_img).shape[1]

        self.total_cnn_out = n_flat * 3  # 3 cameras

        self.final_mlp = nn.Sequential(
            nn.Linear(self.total_cnn_out + 3, features_dim),  # +3 for robot_state
            nn.ReLU()
        )

    def forward(self, observations: dict) -> torch.Tensor:
        cam1 = self.cnn(observations["camera1"].float() / 255.0)
        cam2 = self.cnn(observations["camera2"].float() / 255.0)
        cam3 = self.cnn(observations["armcamera"].float() / 255.0)
        robot_state = observations["robot_state"]

        features = torch.cat([cam1, cam2, cam3, robot_state], dim=-1)
        return self.final_mlp(features)
