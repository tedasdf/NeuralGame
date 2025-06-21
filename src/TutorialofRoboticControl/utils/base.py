import numpy as np


def make_screw_axis_matrix(w,v):
    """Create a screw axis matrix from a 3D vector."""
    return np.array([
        [0, -w[2], w[1], v[0]],
        [w[2], 0, -w[0], v[1]],
        [-w[1], w[0], 0, v[2]],
        [0, 0, 0, 0]
    ])

def make_B_matrix(M, S):
    """Create the B matrix from the transformation matrix M and screw axis S."""    
    R = M[:3, :3]
    t = M[:3, 3]

    adj = np.block([
        [R.T, np.zeros((3, 3))],
        [(-R.T @ t) @ R.T, R.T]
    ])

    return adj @ S

def make_exp_matrix(S, theta):
    """Create the exponential matrix from a screw axis and angle."""
    size = S.shape[0]
    return np.eye(size) + np.sin(theta) * S + (1 - np.cos(theta)) * S @ S


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation
    about the given axis ('x', 'y', or 'z') by theta radians.
    """
    c, s = np.cos(theta), np.sin(theta)

    if axis == 'x':
        return np.array([
            [1, 0,  0],
            [0, c, -s],
            [0, s,  c]
        ])
    
    elif axis == 'y':
        return np.array([
            [ c, 0, s],
            [ 0, 1, 0],
            [-s, 0, c]
        ])
    
    elif axis == 'z':
        return np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
    
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")