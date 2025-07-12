
import cv2
import numpy as np
import requests
import time
import serial
import modern_robotics as mr
import math
import matplotlib.pyplot as plt
import os
####
#  Robot class for controlling the robot and capturing images
#  interact with arudnio to send arudino signal for the movement
#  init angle 105,35,0

class Robot:
    def __init__(self,
                 L1, L2, L3, 
                 x_offset , y_offset , z_offset,
                 serial_port='COM3',
                 init_thetas=np.deg2rad(np.array([100, 40, 0])),
                 test_mode=True):
        self.test_mode = test_mode

        
        # connect to camera
        if not self.test_mode:
            # ceonnect to arduino
            self.ser = serial.Serial(serial_port, 9600, timeout=1, write_timeout=1)

        # self.init_position = init_position
        self.test_mode = test_mode
        self.num_actions = 7

        # robot parameters
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z_offset = z_offset
        self.init_thetas = init_thetas
        self.theta_range = np.radians(np.array([[0, 0, 0], [180, 180, 180]]))
        self.scale_factor = 0.4

        self.state = np.zeros((4, 4), dtype=np.float32)  # Homogeneous transformation matrix
        self.thetas = self.init_thetas  # Joint angles in radians
        self.next_step = None
        
        self._init_robot_param(L1, L2, L3)  # Example link lengths


        # state of robot

    def _init_robot_param(self, L1, L2 ,L3):

        # Define joint axes and locations
        w1 = np.array([0, 0, 1])
        q1 = np.array([0, 0, 0])
        v1 = -np.cross(w1, q1)

        w2 = np.array([0, -1, 0])
        q2 = np.array([self.x_offset, 0, L1])
        v2 = -np.cross(w2, q2)

        w3 = np.array([0, 1, 0])
        q3 = np.array([L2+self.x_offset, 0, L1])
        v3 = -np.cross(w3, q3)

        # Screw axes (6x3 matrix)
        S1 = np.hstack((w1, v1))  # shape (6,)
        S2 = np.hstack((w2, v2))
        S3 = np.hstack((w3, v3))
        self.Slist = np.column_stack((S1, S2, S3))  # shape (6,3)

        # Store home configuration
        self.M = np.array([
            [1, 0, 0, L2+self.x_offset],
            [0, 1, 0, 0],
            [0, 0, 1, L1-L3],
            [0, 0, 0, 1]
        ])
        print(self.Slist)
        self.return_init_pos()

    def return_init_pos(self):
        success = False
        if not self.test_mode:
           success = self.read_state_from_robot()
        else:
            success = True
        if not success:
            raise RuntimeError("Failed to read state from robot.")
        
        thetas_signal = np.rad2deg(self.thetas)
        thetas_signal = np.round(thetas_signal).astype(int)
        print(f"Sending signal: {thetas_signal}")
        if not self.test_mode:
            self.ser.write(f"{thetas_signal[0]},{thetas_signal[1]},{thetas_signal[2]}\n".encode())
        self.thetas = self.init_thetas
        self.state = self.forward_kinematics(thetas=self.thetas)
        print(f"Robot initialized at position: {self.state}")


    def read_state_from_robot(self) -> bool:
        self.ser.write(b"0\n")  # Request current angles from Arduino
        lines = []
        start_time = time.time()
        while len(lines) < 3 and time.time() - start_time < 2.0:  # Timeout after 2s
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode().strip()
                if line:
                    lines.append(line)
        if len(lines) == 3:
            try:
                theta1 = int(lines[0])
                theta2 = int(lines[1])
                theta3 = int(lines[2])
                self.theta = np.deg2rad([theta1, theta2, theta3])
                return True
            except ValueError:
                print("Error: Could not parse angles from Arduino.")
        else:
            print("Timeout or incomplete data from Arduino.")
        # Fallback: estimate or return dummy values
        return False  # ← make sure this exists and is valid


    def get_neghbour_positions(self, delta_deg, step=0):
        """
        Generate 3D neighbor positions around current joint angles.
        Only include neighbors with total joint delta >= min_deg.
        """
        neighbor_points = []
        theta_ns = []
        print("GET NEGIHBOUR POSITION")
        for d1_deg in range(-delta_deg, delta_deg + 1):
            for d2_deg in range(-delta_deg, delta_deg + 1):
                for d3_deg in range(-delta_deg, delta_deg + 1):
                    if d1_deg == 0 and d2_deg == 0 and d3_deg == 0:
                        continue  # skip center
                    
                    # Convert to radians
                    d1, d2, d3 = np.radians(d1_deg), np.radians(d2_deg), np.radians(d3_deg)
                    theta_n = [self.thetas[0] + d1, self.thetas[1] + d2, self.thetas[2] + d3]
                    theta_ns.append(np.rad2deg(theta_n))
                    # FK
                    T_n = mr.FKinSpace(self.M, self.Slist, theta_n)
                    x_n, y_n, z_n = T_n[0, 3], T_n[1, 3], T_n[2, 3]
                    # if np.linalg.norm(pos_n - self.state[0:3,3]) < min_distance:
                    #     continue
                    neighbor_points.append((x_n, y_n, z_n))
        
    
        return np.array(neighbor_points) , np.array(theta_ns)

    def forward_kinematics(self, thetas):
        return mr.FKinSpace(self.M, self.Slist, thetas)    


    # def inverse_kinematics(self, T_target, initial_guess=None, epsilon = 10000 , max_iter= 0.1):
    #     if initial_guess is None:
    #         initial_guess = np.zeros(3)
        
    #     thetas, success = mr.IKinSpace(self.Slist, self.M, T_target, initial_guess, epsilon, max_iter)

    #     if not success:
    #         print("Warning: IK did not converge")

    #     return thetas,success

    def closest_distance_node(self, points,corresponding_theta, proposed_pos):
        # Compute Euclidean distances
        distances = np.linalg.norm(points - proposed_pos, axis=1)

        # Find closest index
        closest_idx = np.argmin(distances)
        closest_point = points[closest_idx]
        closest_distance = distances[closest_idx]
        closest_theta = corresponding_theta[closest_idx]
        
        return closest_point, closest_theta, closest_distance

    def to_transformed_signal(self,thetas_list, previous_thetas):
        """
        Given a list of theta arrays and a reference (previous_thetas),
        returns the transformed theta (with rounding and +90 on theta3) 
        that is closest to previous_thetas.
        """
        previous_thetas = np.array(previous_thetas)
        min_dist = float('inf')
        best_thetas = None
        print(thetas_list)
        for thetas in thetas_list:
            # Step 1: Transform each theta
            thetas_rounded = np.round(thetas).astype(int)
            thetas_rounded[2] += 90
            
            thetas_rounded[2] *= -1
            # Step 2: Compare to previous_thetas
            dist = np.linalg.norm(thetas_rounded - previous_thetas)

            if dist < min_dist:
                min_dist = dist
                best_thetas = thetas_rounded

        return best_thetas

    def velocity_sensitivity(self):
        self.J = mr.JacobianSpace(self.Slist, self.thetas)
        print("J",self.J)
        dtheta = np.array([0.01745 * 10, 0.01745*10, 0.01745*10])  # rad/s

        V = self.J @ dtheta
        omega = V[0:3]  # angular velocity of end-effector (rad/s)
        v = V[3:6]  
        return omega, v 
    
    def final_solution(self, action_idx):
        
        direction_map = {
            0: np.array([+1, 0, 0]),
            1: np.array([0, +1, 0]),
            2: np.array([0, 0, +1]),
            3: np.array([-1, 0, 0]),
            4: np.array([0, -1, 0]),
            5: np.array([0, 0, -1]),
            6: np.array([0, 0, 0])
        }

        omega , v = self.velocity_sensitivity()
        print("v:",v)
        proposed_translation = np.multiply(direction_map[action_idx] , v)
        proposed_pos = self.state[0:3, 3] + proposed_translation
        print(proposed_pos)
        try_thetas , success =self.inverse_kinematics_3d( proposed_pos[0], proposed_pos[1], proposed_pos[2])
        if not success:
            return success
        print("result from inverse kinematics:"  ,try_thetas)

        thetas_signal = self.to_transformed_signal(try_thetas, self.thetas)

        print("Thetas signal " ,thetas_signal)
        if not self.test_mode:
            self.ser.write(f"{thetas_signal[0]},{thetas_signal[1]},{thetas_signal[2]}\n".encode())

        self.read_state_from_robot()
        self.state = self.forward_kinematics(self.thetas)
        return success
    

    def retry(self, action_idx):
        points, corresponding_theta = self.get_neghbour_positions(delta_deg=2)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # xs, ys, zs = points[:,0], points[:,1], points[:,2]
        # ax.scatter(xs, ys, zs, c='blue')
        # ax.scatter(self.state[0,3], self.state[1,3], self.state[2,3], c='green', label="current pos")
        # ax.set_title("±1° Joint Neighbor Workspace Region")


        direction_map = {
            0: np.array([+1, 0, 0]),
            1: np.array([0, +1, 0]),
            2: np.array([0, 0, +1]),
            3: np.array([-1, 0, 0]),
            4: np.array([0, -1, 0]),
            5: np.array([0, 0, -1]),
            6: np.array([0, 0, 0])
        }

        proposed_translation = direction_map[action_idx] * self.scale_factor
        proposed_pos = self.state[0:3, 3] + proposed_translation
        print(f"proposed position : {proposed_pos}")
        # ax.scatter(proposed_pos[0], proposed_pos[1], proposed_pos[2],
        #        c='red', s=50, label="Proposed Translation")
    
        closest_node , closest_theta, distance = self.closest_distance_node(points,corresponding_theta, proposed_pos)
        # ax.scatter(closest_node[0], closest_node[1], closest_node[2],
        #        c='magenta', s=70, label="Closest Node", marker='^')
        # # Final touches
        # ax.set_title("±1° Joint Neighbor Workspace Region")
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        # # Set custom 3D axis limits
        
        # ax.legend()
        # ax.grid(True)

       
        new_node = self.state.copy()
        new_node[0:3,3] = closest_node
        # print(f"New node : {new_node}")
        print(f"actual theta : {closest_theta}")
        # thetas, success = self.inverse_kinematics(new_node, initial_guess=self.thetas)
        try_thetas , success = self.inverse_kinematics_3d( proposed_pos[0], proposed_pos[1], proposed_pos[2])
        thetas_signal = self.to_transformed_signal(try_thetas, self.thetas)


        # if not success:
        #     raise RuntimeError()

        print(f"China your ass another inverse_kinematics : {try_thetas}")
        print(f"China get the signal across : {thetas_signal}")
        thetas = np.deg2rad(closest_theta)
        success = True
        print(f"distance : {distance}")
        print(f"theta moving (in radian):{thetas}")
        thetas_signal = thetas_signal
        print(f"signal theta moving (in degree): {thetas_signal}")
        if not self.test_mode:
            self.ser.write(f"{thetas_signal[0]},{thetas_signal[1]},{thetas_signal[2]}\n".encode())
        print(f"Succes: {success}")

        # update
        self.thetas = thetas
        self.state = self.forward_kinematics(thetas)
        self.next_step = self.get_neghbour_positions(delta_deg=1)
        print("=========================================================")

        return success

    def step(self, action ):
        # capture image 
        # neural network 
        # move
        # new state
        success = self.move(action)
        obs = None
        if not self.test_mode:
            obs = self.camera_cap()
            if obs is not None:
                cv2.imshow("Robot Camera", obs)
                cv2.waitKey(1)

        return obs, success 
    
    def inverse_kinematics_3d(self, x, y, z):
        """
        Calculates inverse kinematics for a 3DOF arm:
        - Joint 1 rotates around Z-axis
        - Joints 2 & 3 act in YZ plane
        Returns the solution closest to previous_thetas if provided.
        """
        x_offset , y_offset , z_offset = self.x_offset, self.y_offset , self.z_offset
        L1,L2,L3 = self.L1, self.L2, self.L3
        # 1. θ1 = base rotation to reach (x, y)
        theta1 = np.arctan2(y, x)  # rotation around Z to reach y-axis

        # 2. Project the target onto the YZ plane by rotating into frame
        y_proj = np.sqrt(x**2 + y**2) - x_offset
        z_eff = z - L1  # offset for base height
        
        # 3. Compute r for planar arm (in YZ)

        r = np.sqrt(y_proj**2 + z_eff**2)

        # Check reachability
        if r > (L2 + L3) or r < abs(L2 - L3):
            return (None, None) ,False

        # 4. Two possible θ3 (elbow angle)
        cos_theta3 = (r**2 - L2**2 - L3**2) / (2 * L2 * L3)
        cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
        theta3_up =  np.arccos(cos_theta3)
        theta3_down = -theta3_up

        # 5. Corresponding θ2 (shoulder angle)
        k1_up = L2 + L3 * np.cos(theta3_up)
        k2_up = L3 * np.sin(theta3_up)
        theta2_up = np.arctan2(z_eff, y_proj) - np.arctan2(k2_up, k1_up)

        k1_down = L2 + L3 * np.cos(theta3_down)
        k2_down = L3 * np.sin(theta3_down)
        theta2_down = np.arctan2(z_eff, y_proj) - np.arctan2(k2_down, k1_down)

        # Pack solutions
        sol_up = np.array([theta1, theta2_up, theta3_up])
        sol_down = np.array([theta1, theta2_down, theta3_down])

        sol_up_deg = np.rad2deg(sol_up)
        sol_down_deg = np.rad2deg(sol_down)


        return (sol_up_deg, sol_down_deg) , True

 

    # def capture_led_sequence(self, duration=4.0, interval=0.8, save_dir="led_captures"):
    #     """
    #     Captures camera frames for a duration to catch each LED lighting up.

    #     :param duration: Total capture time in seconds (default 4.0s)
    #     :param interval: Time between frames (default 0.2s = 200 ms)
    #     :param save_dir: Directory to save the captured images
    #     """
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)

    #     start_time = time.time()
    #     count = 0

    #     while (time.time() - start_time) < duration:
    #         frame = self.camera_cap()
    #         if frame is not None:
    #             filename = os.path.join(save_dir, f"frame_{count:03d}.jpg")
    #             cv2.imwrite(filename, frame)
    #             print(f"Saved: {filename}")
    #             count += 1
    #         else:
    #             print("No frame captured.")
    #         time.sleep(interval)
        
    #     print(f"Captured {count} frames over {duration:.1f}s.")



    def reset(self):
        # Initialize the robot's position to the initial position
        self.return_init_pos()
        # self.capture_led_sequence()

    # def capture_led_sequence(self, duration=4.0, interval=0.1, save_dir="led_captures"):
    #     """
    #     Captures camera frames for a duration to catch each LED lighting up.

    #     :param duration: Total capture time in seconds (default 4.0s)
    #     :param interval: Time between frames (default 0.2s = 200 ms)
    #     :param save_dir: Directory to save the captured images
    #     """


    #     start_time = time.time()
    #     frames = []
    #     while (time.time() - start_time) < duration:
            
    #         frame_each ,frame_time  = self.camera_cap()
    #         print(f"this frame takes")
    #         if frame_each is not None:
    #             frames.append(frame_each)
    #             print(f"Capture")
    #         else:
    #             print("No frame captured.")
            
    #         time.sleep(interval-frame_time)

    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)

    #     for idx,frame in enumerate(frames):
    #         filename = os.path.join(save_dir, f"frame_{idx:03d}.jpg")
    #         cv2.imwrite(filename, frame)
    #         print(f"Saved: {filename}")
    
    #     print(f"Captured {len(frames)} frames over {duration:.1f}s.")


    # def camera_cap(self):
    #     try:
    #         start_time = time.time()
    #         ret, frame = self.cap.read()
    #         time_takes = time.time() - start_time

    #         if ret and frame is not None:
    #             return frame, time_takes
    #         else:
    #             print("Failed to capture frame from VideoCapture.")
    #             return None, 0
    #     except Exception as e:
    #         print(f"Exception during image capture: {e}")
    #         return None, 0


    
    # def move(self, action_idx):

    #     # depend on the action _idx find the idrection and proposed state =
    #     # use IK to find the theta
    #     # send theta to ardunio 
    #     # ipdate the next step 
    #     # update other 
    #     direction_map = {
    #         0: np.array([+1, 0, 0]),
    #         1: np.array([0, +1, 0]),
    #         2: np.array([0, 0, +1]),
    #         3: np.array([-1, 0, 0]),
    #         4: np.array([0, -1, 0]),
    #         5: np.array([0, 0, -1]),
    #         6: np.array([0, 0, 0])
    #     }
    #     print("===============================================")
    #     print(f"current position :{self.state}")
    #     print(f"current theta : {self.thetas} ")
    #     # Save current state in case we need to revert
    #     proposed_state  = self.state.copy()
    #     # Apply scaled movement
    #     proposed_state[0:3,3] += direction_map[action_idx] * self.scale_factor
    #     print(f"proposed position :{proposed_state}")

    #     if self.next_step is not None and len(self.next_step) > 0:
    #         current_pos = proposed_state[0:3, 3]
    #         next_points = np.array(self.next_step)

    #         # Find the closest reachable next step
    #         distances = np.linalg.norm(next_points - current_pos, axis=1)
    #         closest_idx = np.argmin(distances)
    #         closest_point = next_points[closest_idx]

    #         # Set the desired state to this point
    #         self.state[0:3, 3] = closest_point
    #         print(f"Moving to closest reachable point from next_step: {closest_point}")

    #         plt.figure()
    #         plt.scatter(current_pos[0],current_pos[1] , current_pos[2] , c='green', s=50, label="Current Pose")
    #         plt.scatter(next_points[:, 0], next_points[:, 1], next_points[:,2], c='red', s=20, label="Neighbors")
    #         plt.title("Workspace with Neighbors Around Pose Near (0,0)")
    #         plt.xlabel("X")
    #         plt.ylabel("Y")
    #         plt.axis("equal")
    #         plt.grid(True)
    #         plt.legend()
    #         plt.show()


    #     else:
    #         # Fall back to naive direction movement
    #         self.state[0:3, 3] = proposed_state[0:3, 3]
    #         self.next_step = self.get_neghbour_positions(delta_deg=1)
    #         print("No next_step list; moving by direction_map.")
    #         return
        
    #     # Compute inverse kinematics
    #     thetas, success = self.inverse_kinematics(self.state, initial_guess=self.theta)
    #     print(f"Inverse kinematics result: {thetas}, success: {success}")
 
    #     if not success:
    #         print(f"IK did not converge for position {self.state}. Reverting to previous state.")
    #         return False
   
    #     thetas = np.rad2deg(thetas)  # Convert to degrees for Arduino
    #     thetas = np.round(thetas).astype(int)

    #     if not self.test_mode:
            
    #         print(f"Send theta: {thetas}")
    #         self.ser.write(f"{thetas[0]},{thetas[1]},{thetas[2]}\n".encode())
    #         self.next_step = self.get_neghbour_positions(delta_deg=1)

    #     # Estimate forward kinematics based on thetas
    #     self.estimated_state = self.forward_kinematics(thetas)

    #     if self.test_mode:
    #         # Update state as the position extracted from FK result
    #         self.state = self.estimated_state
    #     else:
    #         # Read actual robot state from Arduino and update
    #         time.sleep(0.1)  # Allow time for Arduino to process
    #         self.read_state_from_robot()
    #         self.actual_state = self.forward_kinematics(self.thetas)
    #         self.state = self.actual_state
    #         print(f"Current state: {self.state}")

    #     time.sleep(0.3)  # Simulate delay
    #     return success


    # def plot_robot(self, ax):
    #     ax.clear()

    #     # Global axes
    #     origin = np.array([0, 0, 0])
    #     axis_len = 5
    #     ax.quiver(*origin, axis_len, 0, 0, color='r', arrow_length_ratio=0.1, label='X')
    #     ax.quiver(*origin, 0, axis_len, 0, color='g', arrow_length_ratio=0.1, label='Y')
    #     ax.quiver(*origin, 0, 0, axis_len, color='b', arrow_length_ratio=0.1, label='Z')

    #     # Current end-effector position
    #     ee_pos = self.state[0:3, 3]
    #     ax.scatter(*ee_pos, c='green', s=80, label='End-Effector')

    #     # Optional: path history
    #     if hasattr(self, "path") and len(self.path) > 1:
    #         path_arr = np.array(self.path)
    #         ax.plot(path_arr[:, 0], path_arr[:, 1], path_arr[:, 2], 'm-', label='Path')

    #     ax.set_title("Live 3D Robot Workspace")
    #     ax.set_xlabel("X")
    #     ax.set_ylabel("Y")
    #     ax.set_zlabel("Z")
    #     ax.set_xlim(-20, 20)
    #     ax.set_ylim(-20, 20)
    #     ax.set_zlim(0, 20)
    #     ax.legend()
    #     ax.grid(True)
    #     plt.draw()
    #     plt.pause(0.001)

if __name__ == "__main__":

    robot = Robot(L1=7.14, L2=7, L3=8.34 , x_offset=0.9185 , y_offset=0, z_offset=0, serial_port='COM3', test_mode=False)
    import keyboard
    import time
    import matplotlib.pyplot as plt

    try:
        print("Use arrow keys to move, W/S for Z, Space to stop, ESC to exit.")
        while True:
            if keyboard.is_pressed("right"):
                action = 0
            elif keyboard.is_pressed("up"):
                action = 1
            elif keyboard.is_pressed("w"):
                action = 2
            elif keyboard.is_pressed("left"):
                action = 3
            elif keyboard.is_pressed("down"):
                action = 4
            elif keyboard.is_pressed("s"):
                action = 5
            elif keyboard.is_pressed("space"):
                action = 6
            elif keyboard.is_pressed("esc"):
                print("\nExiting...")
                break
            else:
                time.sleep(0.05)
                continue

            success = robot.retry(action)
            if success:
                print(f"Action {action} executed successfully.")
                print("New position:", robot.state)
               
            else:
                print(f"Action {action} failed.")

            # robot.plot_robot(ax)
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nInterrupted. Exiting...")

   
    # print("done ")
    # current_pos = np.array(robot.state[0:3,3])           # Ensure shape (3,)
    # next_points = np.array(robot.next_step)           # Shape (N, 3)
    # print(current_pos)
    # print(next_points)
    # # Compute distances to each point
    # distances = np.linalg.norm(next_points - current_pos, axis=1)

    # # Get index of closest point
    # closest_idx = np.argmin(distances)
    # closest_point = next_points[closest_idx]

    # print("Closest point:", closest_point)
    # print("Distance:", distances[closest_idx])

    # import numpy as np
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D

    # # def draw_dotted_box(ax, x_range, y_range, z_range, linestyle=':', color='gray'):
    # #     # Corners of the box
    # #     x = [x_range[0], x_range[1]]
    # #     y = [y_range[0], y_range[1]]
    # #     z = [z_range[0], z_range[1]]

    # #     # List all 8 corners
    # #     corners = np.array([
    # #         [x[0], y[0], z[0]],
    # #         [x[1], y[0], z[0]],
    # #         [x[1], y[1], z[0]],
    # #         [x[0], y[1], z[0]],
    # #         [x[0], y[0], z[1]],
    # #         [x[1], y[0], z[1]],
    # #         [x[1], y[1], z[1]],
    # #         [x[0], y[1], z[1]],
    # #     ])

    # #     # Edges defined by pairs of corner indices
    # #     edges = [
    # #         (0,1), (1,2), (2,3), (3,0),  # Bottom square
    # #         (4,5), (5,6), (6,7), (7,4),  # Top square
    # #         (0,4), (1,5), (2,6), (3,7)   # Vertical edges
    # #     ]

    # #     for edge in edges:
    # #         p1 = corners[edge[0]]
    # #         p2 = corners[edge[1]]
    # #         ax.plot(
    # #             [p1[0], p2[0]],
    # #             [p1[1], p2[1]],
    # #             [p1[2], p2[2]],
    # #             linestyle=linestyle,
    # #             color=color
    # #         )
    
    # # # robot = Robot(L1 = 6 , L2 = 7, L3 = 7, serial_port='COM3', test_mode=False)
    
    # # plt.close('all')
    # # fig = plt.figure()
    # # ax = fig.add_subplot(111, projection='3d')

    # # # Define workspace ranges
    # # x_range = (-15, 15)
    # # y_range = (0, 15)
    # # z_range = (0, 15)

    # # draw_dotted_box(ax, x_range, y_range, z_range)

    # # ax.set_xlabel('X axis')
    # # ax.set_ylabel('Y axis')
    # # ax.set_zlabel('Z axis')

    # # ax.set_xlim(x_range)
    # # ax.set_ylim(y_range)
    # # ax.set_zlim(z_range)
    # # ax.set_title('Workspace Boundary Box')

    # # pos = robot.state[0:3, 3]
    # # point_plot = ax.scatter(pos[0], pos[1], pos[2], color='red', s=100)

    # # plt.ion()
    # # plt.show()

   