"""
Method to retrieve 12 split spline patch surface triangulated bounds coefficients. 
"""
import numpy as np


def PS12tri_bounds_coeffs() -> np.ndarray:
    """
    Returns np.ndarray of shape (12,3,3), initialized to pre-defined numbers.
    """
    bound_coeffs: np.ndarray = np.zeros(shape=(12, 3, 3))

    bound_coeffs[0][0][0] = -0.1e1 / 0.8e1
    bound_coeffs[0][0][1] = 0.1e1 / 0.4e1
    bound_coeffs[0][0][2] = 0.1e1 / 0.4e1
    bound_coeffs[0][1][0] = 0
    bound_coeffs[0][1][1] = -0.1e1 / 0.12e2
    bound_coeffs[0][1][2] = 0.1e1 / 0.12e2
    bound_coeffs[0][2][0] = 0.1e1 / 0.6e1
    bound_coeffs[0][2][1] = -0.1e1 / 0.6e1
    bound_coeffs[0][2][2] = -0.1e1 / 0.3e1
    bound_coeffs[1][0][0] = 0.1e1 / 0.6e1
    bound_coeffs[1][0][1] = -0.1e1 / 0.3e1
    bound_coeffs[1][0][2] = -0.1e1 / 0.6e1
    bound_coeffs[1][1][0] = 0
    bound_coeffs[1][1][1] = 0.1e1 / 0.12e2
    bound_coeffs[1][1][2] = -0.1e1 / 0.12e2
    bound_coeffs[1][2][0] = -0.1e1 / 0.8e1
    bound_coeffs[1][2][1] = 0.1e1 / 0.4e1
    bound_coeffs[1][2][2] = 0.1e1 / 0.4e1
    bound_coeffs[2][0][0] = 0.1e1 / 0.8e1
    bound_coeffs[2][0][1] = -0.1e1 / 0.4e1
    bound_coeffs[2][0][2] = 0
    bound_coeffs[2][1][0] = 0.1e1 / 0.12e2
    bound_coeffs[2][1][1] = -0.1e1 / 0.12e2
    bound_coeffs[2][1][2] = -0.1e1 / 0.6e1
    bound_coeffs[2][2][0] = -0.1e1 / 0.6e1
    bound_coeffs[2][2][1] = 0.1e1 / 0.3e1
    bound_coeffs[2][2][2] = 0.1e1 / 0.6e1
    bound_coeffs[3][0][0] = 0
    bound_coeffs[3][0][1] = 0.1e1 / 0.6e1
    bound_coeffs[3][0][2] = -0.1e1 / 0.6e1
    bound_coeffs[3][1][0] = -0.1e1 / 0.12e2
    bound_coeffs[3][1][1] = 0.1e1 / 0.12e2
    bound_coeffs[3][1][2] = 0.1e1 / 0.6e1
    bound_coeffs[3][2][0] = 0.1e1 / 0.8e1
    bound_coeffs[3][2][1] = -0.1e1 / 0.4e1
    bound_coeffs[3][2][2] = 0
    bound_coeffs[4][0][0] = 0.1e1 / 0.8e1
    bound_coeffs[4][0][1] = 0
    bound_coeffs[4][0][2] = -0.1e1 / 0.4e1
    bound_coeffs[4][1][0] = -0.1e1 / 0.12e2
    bound_coeffs[4][1][1] = 0.1e1 / 0.6e1
    bound_coeffs[4][1][2] = 0.1e1 / 0.12e2
    bound_coeffs[4][2][0] = 0
    bound_coeffs[4][2][1] = -0.1e1 / 0.6e1
    bound_coeffs[4][2][2] = 0.1e1 / 0.6e1
    bound_coeffs[5][0][0] = -0.1e1 / 0.6e1
    bound_coeffs[5][0][1] = 0.1e1 / 0.6e1
    bound_coeffs[5][0][2] = 0.1e1 / 0.3e1
    bound_coeffs[5][1][0] = 0.1e1 / 0.12e2
    bound_coeffs[5][1][1] = -0.1e1 / 0.6e1
    bound_coeffs[5][1][2] = -0.1e1 / 0.12e2
    bound_coeffs[5][2][0] = 0.1e1 / 0.8e1
    bound_coeffs[5][2][1] = 0
    bound_coeffs[5][2][2] = -0.1e1 / 0.4e1
    bound_coeffs[6][0][0] = 0.1e1 / 0.2e1
    bound_coeffs[6][0][1] = -0.1e1 / 0.2e1
    bound_coeffs[6][0][2] = -0.1e1 / 0.2e1
    bound_coeffs[6][1][0] = -0.1e1 / 0.8e1
    bound_coeffs[6][1][1] = 0.1e1 / 0.4e1
    bound_coeffs[6][1][2] = 0
    bound_coeffs[6][2][0] = -0.1e1 / 0.4e1
    bound_coeffs[6][2][1] = 0.1e1 / 0.4e1
    bound_coeffs[6][2][2] = 0.1e1 / 0.2e1
    bound_coeffs[7][0][0] = -0.1e1 / 0.4e1
    bound_coeffs[7][0][1] = 0.1e1 / 0.2e1
    bound_coeffs[7][0][2] = 0.1e1 / 0.4e1
    bound_coeffs[7][1][0] = -0.1e1 / 0.8e1
    bound_coeffs[7][1][1] = 0
    bound_coeffs[7][1][2] = 0.1e1 / 0.4e1
    bound_coeffs[7][2][0] = 0.1e1 / 0.2e1
    bound_coeffs[7][2][1] = -0.1e1 / 0.2e1
    bound_coeffs[7][2][2] = -0.1e1 / 0.2e1
    bound_coeffs[8][0][0] = 0
    bound_coeffs[8][0][1] = 0.1e1 / 0.2e1
    bound_coeffs[8][0][2] = 0
    bound_coeffs[8][1][0] = -0.1e1 / 0.8e1
    bound_coeffs[8][1][1] = 0
    bound_coeffs[8][1][2] = 0.1e1 / 0.4e1
    bound_coeffs[8][2][0] = 0.1e1 / 0.4e1
    bound_coeffs[8][2][1] = -0.1e1 / 0.2e1
    bound_coeffs[8][2][2] = -0.1e1 / 0.4e1
    bound_coeffs[9][0][0] = 0
    bound_coeffs[9][0][1] = -0.1e1 / 0.4e1
    bound_coeffs[9][0][2] = 0.1e1 / 0.4e1
    bound_coeffs[9][1][0] = 0.1e1 / 0.8e1
    bound_coeffs[9][1][1] = -0.1e1 / 0.4e1
    bound_coeffs[9][1][2] = -0.1e1 / 0.4e1
    bound_coeffs[9][2][0] = 0
    bound_coeffs[9][2][1] = 0.1e1 / 0.2e1
    bound_coeffs[9][2][2] = 0
    bound_coeffs[10][0][0] = 0
    bound_coeffs[10][0][1] = 0
    bound_coeffs[10][0][2] = 0.1e1 / 0.2e1
    bound_coeffs[10][1][0] = 0.1e1 / 0.8e1
    bound_coeffs[10][1][1] = -0.1e1 / 0.4e1
    bound_coeffs[10][1][2] = -0.1e1 / 0.4e1
    bound_coeffs[10][2][0] = 0
    bound_coeffs[10][2][1] = 0.1e1 / 0.4e1
    bound_coeffs[10][2][2] = -0.1e1 / 0.4e1
    bound_coeffs[11][0][0] = 0.1e1 / 0.4e1
    bound_coeffs[11][0][1] = -0.1e1 / 0.4e1
    bound_coeffs[11][0][2] = -0.1e1 / 0.2e1
    bound_coeffs[11][1][0] = -0.1e1 / 0.8e1
    bound_coeffs[11][1][1] = 0.1e1 / 0.4e1
    bound_coeffs[11][1][2] = 0
    bound_coeffs[11][2][0] = 0
    bound_coeffs[11][2][1] = 0
    bound_coeffs[11][2][2] = 0.1e1 / 0.2e1

    # Redundant check
    assert bound_coeffs.shape == (12, 3, 3)
    return bound_coeffs
