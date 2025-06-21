import numpy as np
from scipy.linalg import expm

M = np.array([])  # Placeholder for the home position of the end effector


w_list = np.array([])
v_list = np.array([])



def make_screw_axis_matrices(w_list, v_list):
    """
    Create a list of screw axis matrices from lists of angular and linear velocity vectors.

    Parameters:
    - w_list: list of 3-element lists or arrays (angular velocity vectors)
    - v_list: list of 3-element lists or arrays (linear velocity vectors)

    Returns:
    - List of 4x4 numpy arrays representing screw axis matrices
    """
    screw_axes = []
    for w, v in zip(w_list, v_list):
        w = np.array(w)
        v = np.array(v)
        S = np.array([
            [0,     -w[2],  w[1],  v[0]],
            [w[2],   0,    -w[0],  v[1]],
            [-w[1],  w[0],  0,     v[2]],
            [0,      0,     0,     0]
        ])
        screw_axes.append(S)
    return screw_axes


def forward_kinematics(S, theta):
    """
    Calculate the forward kinematics of a robot arm using the product of exponentials formula.

    Parameters:
    S (list): List of screws (twists) for each joint.
    theta (list): List of joint angles.
    M (numpy.ndarray): Home position of the end effector.

    Returns:
    numpy.ndarray: The transformation matrix representing the end effector's pose.
    """


    T = np.eye(4)  # Initialize the transformation matrix as identity
    for i in range(len(S)):
        T = T @ expm(S[i] * theta[i])  # Update the transformation matrix

    return T @ M  # Return the final transformation matrix





