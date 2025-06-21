import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting toolkit
import numpy as np
import modern_robotics as mr
import pygame


def forward_kinematics(S1 , S2 , S3 , theta1 ,theta2, theta3, M):
    # Forward kinematics
    T1 = mr.MatrixExp6(S1 * theta1)
    T2 = mr.MatrixExp6(S2 * theta2)
    T3 = mr.MatrixExp6(S3 * theta3)
    return T2 @ T3 @ M



if __name__ == "__main__":

    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Robot Arm Simulation")

    # Example usage
    theta1 , theta2, theta3 = 0, 0, 0
    L1, L2 , L3  = 0.5, 0.2 , 0.1

    # forward kinematics

    w1 = np.array([0, 0, 1])
    q1 = np.array([0, 0, 0])
    v1 = -np.cross(w1, q1)

    w2 = np.array([1, 0, 0])
    q2 = np.array([0, 0, L1])
    v2 = -np.cross(w2, q2)

    w3 = np.array([1, 0, 0])
    q3 = np.array([0, L2, L1])
    v3 = -np.cross(w3, q3)

    # Screw axes in se(3)
    S1 = mr.VecTose3(np.hstack((w1, v1)))
    S2 = mr.VecTose3(np.hstack((w2, v2)))
    S3 = mr.VecTose3(np.hstack((w3, v3)))

    M = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, L2+L3],
        [0, 0, 1, L1],
        [0, 0, 0, 1]
    ])


    # Create data
    T = forward_kinematics(S1 ,S2 , S3, theta1 ,theta2, theta3, M)    
    x, y, z = T[0, 3], T[1, 3], T[2, 3]

    # Create a new figure
    fig = plt.figure()

    # Add 3D subplot
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D line
    ax.plot3D([x], [y], [z], 'bo')  # or use ax.plot(x, y, z)

    # Set axis labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Show the plot
    plt.ion()
    plt.tight_layout()
    plt.show()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            x += 0.01
        if keys[pygame.K_DOWN]:
            x -= 0.01
        if keys[pygame.K_RIGHT]:
            y += 0.01
        if keys[pygame.K_LEFT]:
            y -= 0.01
        if keys[pygame.K_w]:
            z += 0.01
        if keys[pygame.K_s]:
            z -= 0.01

        ax.clear()
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(0, 0.9)
        ax.set_title("End-Effector Position (x, y, z)")
        ax.plot3D([x], [y], [z], 'bo')

        plt.pause(0.01)  # Allows real-time plot update

        # try:
        #     theta2, theta3 = inverse_kinematics_2d(y, z, L1, L2, L3)
        #     T = forward_kinematics(theta2, theta3)
        #     _, y_fk, z_fk = T[0, 3], T[1, 3], T[2, 3]

        #     y_hist.append(y_fk)
        #     z_hist.append(z_fk)
        #     theta2_hist.append(np.degrees(theta2))
        #     theta3_hist.append(np.degrees(theta3))

        #     # Update position plot
        #     ax_pos.clear()
        #     ax_pos.set_xlim(-0.5, 0.5)
        #     ax_pos.set_ylim(0, 0.9)
        #     ax_pos.set_title("End-Effector Position (y, z)")
        #     ax_pos.plot(y_hist[-10:], z_hist[-10:], 'bo-')  # Trajectory
        #     ax_pos.plot(y_fk, z_fk, 'ro')

        #     # Update theta plot
        #     ax_theta.clear()
        #     ax_theta.set_xlim(-180, 180)
        #     ax_theta.set_ylim(-180, 180)
        #     ax_theta.set_title("Theta2 and Theta3 Over Time (deg)")
        #     ax_theta.plot(np.degrees(theta2), np.degrees(theta3), 'ro')  # Current position
        #     ax_theta.plot(theta2_hist[-10:], theta3_hist[-10:], 'bo-')
        #     ax_theta.legend()

        #     plt.draw()
        #     plt.pause(0.1)

        #     print(f"Current position: (y={y:.2f}, z={z:.2f})")
        #     print(f"Angles: theta2={np.degrees(theta2):.2f}, theta3={np.degrees(theta3):.2f}")
        # except ValueError:
        #     print("Target out of reach")
        #     print(f"Current position: (y={y:.2f}, z={z:.2f})")
