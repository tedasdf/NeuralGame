import matplotlib.pyplot as plt
import numpy as np
import modern_robotics as mr
import pygame

def forward_kinematics(Slist, thetas, M):
    return mr.FKinSpace(M, Slist, thetas)

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Robot Arm Simulation")

    # Link lengths
    L1, L2, L3 = 15, 2, 10

    # Define screw axes as you had
    w1 = np.array([0, 0, 1])
    q1 = np.array([0, 0, 0])
    v1 = -np.cross(w1, q1)

    w2 = np.array([1, 0, 0])
    q2 = np.array([0, 0, L1])
    v2 = -np.cross(w2, q2)

    w3 = np.array([1, 0, 0])
    q3 = np.array([0, L2, L1])
    v3 = -np.cross(w3, q3)

    # Slist shape (6,3)
    Slist = np.column_stack((np.hstack((w1, v1)),
                            np.hstack((w2, v2)),
                            np.hstack((w3, v3))))

    # Home configuration matrix
    M = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, L2 + L3],
        [0, 0, 1, L1],
        [0, 0, 0, 1]
    ])

    # Initial joint angles
    thetas = np.array([0.0, 0.0, 0.0])
    
    # Initial FK
    T = forward_kinematics(Slist, thetas, M)
    pos = T[0:3, 3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()
    plt.show()

    running = True
    delta_pos = np.array([0.0, 0.0, 0.0])
    step_size = 0.01

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        # Update desired end-effector position increments
        if keys[pygame.K_RIGHT]:
            delta_pos[0] += step_size  # +X
        if keys[pygame.K_LEFT]:
            delta_pos[0] -= step_size  # -X
        if keys[pygame.K_UP]:
            delta_pos[1] += step_size  # +Y
        if keys[pygame.K_DOWN]:
            delta_pos[1] -= step_size  # -Y
        if keys[pygame.K_w]:
            delta_pos[2] += step_size  # +Z
        if keys[pygame.K_s]:
            delta_pos[2] -= step_size  # -Z

        # Build target transform matrix with updated position
        T_target = np.copy(T)
        T_target[0:3, 3] += delta_pos

        # Use inverse kinematics to get joint angles for new position
        thetas_guess = thetas.copy()
        thetas_new, success = mr.IKinSpace(Slist, M, T_target, thetas_guess, eomg=1e-3, ev=1e-3)

        if success:
            thetas = thetas_new
            T = forward_kinematics(Slist, thetas, M)
            pos = T[0:3, 3]
            delta_pos[:] = 0  # Reset delta after successful move
            print(pos)
        else:
            # If IK failed, ignore delta_pos and do nothing
            delta_pos[:] = 0
            print("IK did not converge for target position.")

        ax.clear()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(0, 1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("End-Effector Position")

        ax.scatter(pos[0], pos[1], pos[2], c='r', s=50)
        plt.pause(0.01)

    pygame.quit()
    plt.close()
