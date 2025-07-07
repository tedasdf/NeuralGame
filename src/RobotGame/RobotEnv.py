import gymnasium as gym
from gymnasium import spaces
import serial
import numpy as np
import time
from Robot import Robot
from game import Game
# import pyzed.sl as sl



class RobotEnv(gym.Env):
    
    def __init__(
            self,
            robot_model : Robot,
            game_model : Game,
            num_colors,
            num_lights
        ):
        H = 720
        W = 1280 * 2
        super(RobotEnv, self).__init__()
        # Define action and observation space
        # the movemnet of the robot
        self.robot_model = robot_model
        self.game = game_model
        self.num_actions = self.robot_model.num_actions
        self.action_space = spaces.Discrete(self.num_actions)

        # Observation space ( 0 : no light , 1 : red , 2 : green , 3 : blue)
        self.num_colors = num_colors 
        self.num_lights = num_lights
     
        self.observation_space = spaces.Dict({
            "camera1": spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8),
            "camera2": spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8),
            "armcamera": spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8),
            "robot_state": spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        })

        self.max_steps = 1000  # Maximum number of steps per episode

        # Camera
        # self.zed = sl.Camera()
        # devices = self.zed.get_device_list()
        # if not devices:
        #     raise ValueError("No ZED camera found.")
        # self.zed_serial = devices[:].serial_number
        # self._init_camera()


        self.reward = 0.0
        self.steps = None
        

    
    # def _init_camera(self):
    #     # Initialize the ZED camera
    #     self.zed1 = sl.Camera()
    #     init_params = sl.InitParameters()
    #     init_params.camera_resolution = sl.RESOLUTION.HD720
    #     init_params.camera_fps = 30

    #     input_type = sl.InputType()
    #     input_type.set_from_serial_number(self.zed_serial)

    #     init_params.input = input_type

    #     status = self.zed1.open(init_params)
    #     if status != sl.ERROR_CODE.SUCCESS:
    #         print(f"Error opening camera: {status}")
    #         exit(1)
    #     print("Camera opened successfully.")


    def _compute_reward(self, player_sequence, target_sequence):
        reward = 0.0
        terminated = False

        # Check partial correctness: count how many initial elements match
        correct_steps = 0
        for p, t in zip(player_sequence, target_sequence):
            if p == t:
                correct_steps += 1
            else:
                # Mistake found, terminate immediately
                terminated = True
                break

        # Reward proportional to how many correct steps matched before failure
        reward += correct_steps * 1.0  # e.g., 1 point per correct step

        # Reward based on how close robot is to the target button
        dist = np.linalg.norm(np.array(self.robot_model.position) - np.array(self.game.target_position[target_sequence[len(player_sequence)]]))
        max_dist = 10.0  # max meaningful distance, tune this
        proximity_reward = max(0, (max_dist - dist) / max_dist)  # normalized between 0 and 1

        reward += proximity_reward

        # Optionally terminate if max steps/time reached elsewhere

        return reward, terminated

    # required function for stable baselines 
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

       
        # send the current sequence to hardware
        self.current_sequence = [0, 1, 2 ,3]
        self.steps = 0
        self.reward = 0.0
        if  self.game.connected & self.game.active == False:
            game.reset(self.current_sequence)
        self.robot_position = self.robot_model.reset()
        

        return self._get_obs(), {}

    # required function for stable baselines
    def _get_obs(self):
        combined = None
        # if self.zed1.grab() == sl.ERROR_CODE.SUCCESS:
        #     left_image = sl.Mat()
        #     right_image = sl.Mat()
        #     self.zed1.retrieve_image(left_image, sl.VIEW.LEFT)
        #     self.zed1.retrieve_image(right_image, sl.VIEW.RIGHT)

        #     left_image_np = left_image.get_data()[:, :, :3]  # Drop alpha if needed
        #     right_image_np = right_image.get_data()[:, :, :3]
        #     combined = np.hstack((left_image_np, right_image_np)).astype(np.uint8)

        return {
            "position": np.array(self.robot_model.state, dtype=np.int32),
            "zed_image": combined  # ✅ include image
        }

    # required function for stable baselines
    def step(self, action):
        self.steps += 1

        arm_obs , valid  = self.robot_model.step(action)        

        # Reward , terminated logic 
        sequence = self.game_model.sequence()
        player_sequence = self.game_model.player_sequence
        
        # reward function 
        reward, terminated = self._compute_reward(player_sequence, sequence)

        terminated = terminated or (self.steps >= self.max_steps)


        info = {
            "robot_position": self.robot_position,
            "expected_sequence": self.game.sequence,
            "player_"
            "steps": self.steps
        }

        return self._get_obs(), reward, terminated, info # apply action and return (obs, reward, done, info)
    
    # Optional function for rendering the environment
    def render(self, mode="human"):

        print(f"Robot at {self.robot_position}")
   

if __name__ == "__main__":
    from game import Game
    from Robot import Robot
    
    # Your ESP32 WebSocket port
    url = f"ws://192.168.0.100:81"

    game = Game(url, sequence_length=4)

    robot_model = Robot(L1=7.14, L2=7, L3=8.34 , x_offset=0.9185 , y_offset=0, z_offset=0, serial_port='COM3', test_mode=False)
    
    # # Example usage - replace ... with actual robot_model and game_model instances
    env = RobotEnv(
        robot_model= robot_model,
        game_model= game,
        num_colors=4,
        num_lights=4,
    )

    obs, info = env.reset()
    # action = env.action_space.sample()
    # print("Initial Action:", action)
    # obs, reward, terminated, truncated, info = env.step(action)

    # print("Observation keys:", obs.keys())
    # print("Reward:", reward)
    # print("Terminated:", terminated)
    # print("Truncated:", truncated)
    # print("Info:", info)
