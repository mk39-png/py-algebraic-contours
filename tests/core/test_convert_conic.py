
import math

import numpy as np

from pyalgcon.core.common import (Matrix2x2f, Vector2f,
                                  float_equal)
from pyalgcon.core.convert_conic import \
    compute_symmetric_matrix_eigen_decomposition


def test_parametrize_contour_identity():
    """
    The eigen decomposition of the identity is trivial
    """

    A: Matrix2x2f = np.array([[1, 0],
                             [0, 1]])
    eigenvalues: Vector2f
    rotation: Matrix2x2f
    eigenvalues, rotation = compute_symmetric_matrix_eigen_decomposition(A)

    assert float_equal(eigenvalues[0], 1.0)
    assert float_equal(eigenvalues[1], 1.0)
    assert float_equal(rotation[0, 0], 1.0)
    assert float_equal(rotation[0, 1], 0.0)
    assert float_equal(rotation[1, 0], 0.0)
    assert float_equal(rotation[1, 1], 1.0)


def test_parametrize_contour_diagonal():
    """
    The eigen decomposition of a diagonal matrix is trivial
    """
    A: Matrix2x2f = np.array([[5, 0],
                             [0, 0.5]])
    assert A.shape == (2, 2)
    eigenvalues: Vector2f
    rotation: Matrix2x2f
    eigenvalues, rotation = compute_symmetric_matrix_eigen_decomposition(A)

    assert float_equal(eigenvalues[0], 5.0)
    assert float_equal(eigenvalues[1], 0.5)
    #    Not generally unique
    assert float_equal(rotation[0, 0], 1.0)
    assert float_equal(rotation[0, 1], 0.0)
    assert float_equal(rotation[1, 0], 0.0)
    assert float_equal(rotation[1, 1], 1.0)


def test_parametrize_contour_orthogonal() -> None:
    """
    "Singular values (5, 0.5), rotation angle 0.1"
    """
    D: Matrix2x2f
    U: Matrix2x2f
    A: Matrix2x2f
    theta: float = 0.1

    D = np.array([[5, 0],
                 [0, 0.5]], dtype=np.float64)
    U = np.array([[math.cos(theta), -math.sin(theta)],
                 [math.sin(theta), math.cos(theta)]], dtype=np.float64)
    A = U.T @ D @ U
    assert D.shape == (2, 2)
    assert A.shape == (2, 2)
    eigenvalues: Vector2f
    rotation: Matrix2x2f
    eigenvalues, rotation = compute_symmetric_matrix_eigen_decomposition(A)

    print("Expected rotation is %s", U)
    print("Computed rotation is %s", rotation)

    assert float_equal(eigenvalues[0], 5.0)
    assert float_equal(eigenvalues[1], 0.5)

    # Not generally unique
    # assert float_equal(rotation[0, 0], U[0, 0])
    # assert float_equal(rotation[0, 1], U[0, 1])
    # assert float_equal(rotation[1, 0], U[1, 0])
    # assert float_equal(rotation[1, 1], U[1, 1])


def test_parametrize_contour_orthogonal_rotation_angle_1() -> None:
    """
    Singular values (-5, -0.5), rotation angle 1
    """
    D: Matrix2x2f
    U: Matrix2x2f
    A: Matrix2x2f
    theta: float = 0.1
    D = np.array([[-5, 0],
                  [0, -0.5]])
    theta = 1
    U = np.array([[math.cos(theta), -math.sin(theta)],
                 [math.sin(theta), math.cos(theta)]])
    A = U.T @ D @ U
    eigenvalues: Vector2f
    rotation: Matrix2x2f
    eigenvalues, rotation = compute_symmetric_matrix_eigen_decomposition(A)

    print("Expected rotation is %s", U)
    print("Computed rotation is %s", rotation)

    assert float_equal(eigenvalues[0], -0.5)
    assert float_equal(eigenvalues[1], -5.0)

    # Not generally unique
    # assert float_equal(rotation[0, 0], U[0, 1])
    # assert float_equal(rotation[1, 0], U[1, 1])
    # assert float_equal(rotation[0, 1], -U[1, 1])
    # assert float_equal(rotation[1, 1], U[0, 1])
