import matplotlib.pyplot as plt
import pygame
import numpy as np
import modern_robotics as mr


# Initial point
x, y = [0], [5]
L1, L2 , L3  = 0.5, 0.2 , 0.1

# forward kinematics
w2 = np.array([1, 0, 0])
q2 = np.array([0, 0, L1])
v2 = -np.cross(w2, q2)

w3 = np.array([1, 0, 0])
q3 = np.array([0, L2, L1])
v3 = -np.cross(w3, q3)

# Screw axes in se(3)
S2 = mr.VecTose3(np.hstack((w2, v2)))
S3 = mr.VecTose3(np.hstack((w3, v3)))


M = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, L2+L3],
    [0, 0, 1, L1],
    [0, 0, 0, 1]
])

def forward_kinematics(theta2, theta3):
    T2 = mr.MatrixExp6(S2 * theta2)
    T3 = mr.MatrixExp6(S3 * theta3)
    return T2 @ T3 @ M



def inverse_kinematics_2d(y, z, L1, L2, L3):
    z_eff = z - L1  # remove base height offset
    r = np.sqrt(y**2 + z_eff**2)

    # Check reachability
    if r > (L2 + L3) or r < abs(L2 - L3):
        raise ValueError("Target is out of reach")

    # Elbow angle (theta3)
    cos_theta3 = (r**2 - L2**2 - L3**2) / (2 * L2 * L3)
    theta3 = np.arccos(np.clip(cos_theta3, -1.0, 1.0))

    # Shoulder angle (theta2)
    k1 = L2 + L3 * np.cos(theta3)
    k2 = L3 * np.sin(theta3)
    theta2 = np.arctan2(z_eff, y) - np.arctan2(k2, k1)

    return theta2, theta3






if __name__ == "__main__":
    
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Robot Arm Simulation")

    theta2, theta3 = 0, 0

    # Set up figure and axis
    fig, (ax_pos, ax_theta) = plt.subplots(1, 2, figsize=(10, 5))

    plt.ion()
    plt.tight_layout()
    plt.show()

    T = forward_kinematics(theta2, theta3)    
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    y_hist, z_hist = [y], [z]
    theta2_hist, theta3_hist = [], []
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            z += 0.01
        if keys[pygame.K_DOWN]:
            z -= 0.01
        if keys[pygame.K_RIGHT]:
            y += 0.01
        if keys[pygame.K_LEFT]:
            y -= 0.01

        try:
            theta2, theta3 = inverse_kinematics_2d(y, z, L1, L2, L3)
            T = forward_kinematics(theta2, theta3)
            _, y_fk, z_fk = T[0, 3], T[1, 3], T[2, 3]

            y_hist.append(y_fk)
            z_hist.append(z_fk)
            theta2_hist.append(np.degrees(theta2))
            theta3_hist.append(np.degrees(theta3))

            # Update position plot
            ax_pos.clear()
            ax_pos.set_xlim(-0.5, 0.5)
            ax_pos.set_ylim(0, 0.9)
            ax_pos.set_title("End-Effector Position (y, z)")
            ax_pos.plot(y_hist[-10:], z_hist[-10:], 'bo-')  # Trajectory
            ax_pos.plot(y_fk, z_fk, 'ro')

            # Update theta plot
            ax_theta.clear()
            ax_theta.set_xlim(-180, 180)
            ax_theta.set_ylim(-180, 180)
            ax_theta.set_title("Theta2 and Theta3 Over Time (deg)")
            ax_theta.plot(np.degrees(theta2), np.degrees(theta3), 'ro')  # Current position
            ax_theta.plot(theta2_hist[-10:], theta3_hist[-10:], 'bo-')
            ax_theta.legend()

            plt.draw()
            plt.pause(0.1)

            print(f"Current position: (y={y:.2f}, z={z:.2f})")
            print(f"Angles: theta2={np.degrees(theta2):.2f}, theta3={np.degrees(theta3):.2f}")
        except ValueError:
            print("Target out of reach")
            print(f"Current position: (y={y:.2f}, z={z:.2f})")



# point, = ax.plot(x, y, 'ro')  # red point

# theta2 , theta3 = 0, 0

# def on_key(event):
    
#     dx, dy = 0, 0
#     if event.key == 'up':
#         dy = 0.1
#     elif event.key == 'down':
#         dy = -0.1
#     elif event.key == 'left':
#         dx = -0.1
#     elif event.key == 'right':
#         dx = 0.1

#     # Update point
#     x[0] += dx
#     y[0] += dy
#     point.set_data(x, y)
#     ax.set_title(f"Point: ({x[0]:.2f}, {y[0]:.2f})")
#     fig.canvas.draw_idle()






# # Connect key press event
# fig.canvas.mpl_connect('key_press_event', on_key)
# ax.set_title("Use arrow keys to move the point")
# plt.show()
