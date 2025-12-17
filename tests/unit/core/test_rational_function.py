"""
Tests to find that the derivative of a rational function can be found.
"""

import pathlib

import numpy as np

from pyalgcon.core.common import (Matrix3x2f, SpatialVector1d, Vector3f,
                                  compare_eigen_numpy_matrix,
                                  deserialize_eigen_matrix_csv_to_numpy,
                                  float_equal)
from pyalgcon.core.rational_function import RationalFunction
from pyalgcon.utils.rational_function_testing_utils import \
    deserialize_rational_functions_from_file

# def test_deserialize_rational_functions():
#     folderpath = "spot_control\\contour_network\\compute_cusps\\compute_spline_surface_cusps\\"
#     filename = "contour_segments.json"

#     rational_functions: list[RationalFunction] = deserialize_rational_functions(folderpath+filename)


def test_evaluate_normalized_coordinate(testing_fileinfo) -> None:
    """
    Testing with values from contour_network compute_segment_quantitative_invisibility
    """
    # Initialize parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = (base_data_folderpath / "core" / "rational_function" /
                              "evaluate_normalized_coordinate")

    # Retrieve parameters
    for i in range(198):
        spatial_curve: RationalFunction = deserialize_rational_functions_from_file(
            filepath / "spatial_curve" / f"{i}.csv")[0]
        sample_parameter: float = deserialize_eigen_matrix_csv_to_numpy(
            filepath / "sample_parameter" / f"{i}.csv").item()

        # Execute method
        sample_point: SpatialVector1d = spatial_curve.evaluate_normalized_coordinate(
            sample_parameter)

        # Compare results
        compare_eigen_numpy_matrix(filepath / "sample_point" / f"{i}.csv",
                                   sample_point)


def test_zero_function() -> None:
    """
    From original C++ code
    """
    # TODO: change up from_real_line so that it works with (n,) shape rather than whatever funky thing is here.
    # P_coeffs = np.array([0, 0]).reshape(2, 1)
    # Q_coeffs = np.array([1, 0]).reshape(2, 1)
    # Meanwhile, denominator is ALWAYS going to be a vector.
    P_coeffs = np.array([[0], [0]], dtype=np.float64)
    Q_coeffs = np.array([1, 0], dtype=np.float64)
    F = RationalFunction(1, 1, P_coeffs, Q_coeffs)

    # TODO: problem is the below since the denom and numerator are NOT (n,) shaped....
    # F_derivative = RationalFunction.from_zero_function(2, 1)
    # print("\n")
    # print(F)

    F_derivative: RationalFunction = F.compute_derivative()

    assert float_equal(F_derivative(-1.0)[0], 0.0)
    assert float_equal(F_derivative(0.0)[0], 0.0)
    assert float_equal(F_derivative(1.0)[0], 0.0)


def test_constant_function() -> None:
    """
    From original C++ code
    """
    P_coeffs = np.array([[1], [0]], dtype=np.float64)
    Q_coeffs = np.array([1, 0], dtype=np.float64)
    F = RationalFunction(1, 1, P_coeffs, Q_coeffs)
    # F_derivative = RationalFunction.from_zero_function(2, 1)
    print(F)

    F_derivative: RationalFunction = F.compute_derivative()

    assert float_equal(F_derivative(-1.0)[0], 0.0)
    assert float_equal(F_derivative(0.0)[0],  0.0)
    assert float_equal(F_derivative(1.0)[0],  0.0)


def test_linear_function() -> None:
    """
    From original C++ code
    """

    P_coeffs = np.array([-1, 2], dtype=np.float64).reshape(2, 1)
    Q_coeffs = np.array([1, 0], dtype=np.float64)  # .reshape(2, 1)
    F = RationalFunction(1, 1, P_coeffs, Q_coeffs)

    # TODO: fix "from_zero_function"
    # F_derivative = RationalFunction.from_zero_function(2, 1)
    # F.compute_derivative(F_derivative)
    F_derivative: RationalFunction = F.compute_derivative()

    assert float_equal(F_derivative(-1.0)[0], 2.0)
    assert float_equal(F_derivative(0.0)[0],  2.0)
    assert float_equal(F_derivative(1.0)[0],  2.0)


def test_quadratic_function() -> None:
    """
    From original C++ code
    """

    # P_coeffs = np.array([1, -2, 1])
    # Q_coeffs = np.array([1, 0])
    # NOTE: P_coeffs and Q_coeffs must have the same degree (i.e. same number of rows)
    P_coeffs: np.ndarray = np.array([[1], [-2], [1]], dtype=np.float64)
    Q_coeffs: np.ndarray = np.array([1, 0, 0], dtype=np.float64)

    F = RationalFunction(2, 1, P_coeffs, Q_coeffs)
    # F_derivative = RationalFunction.from_zero_function(4, 1)
    F_derivative: RationalFunction = F.compute_derivative()

    assert float_equal(F_derivative(-1.0)[0], -4.0)
    assert float_equal(F_derivative(0.0)[0],  -2.0)
    assert float_equal(F_derivative(1.0)[0],  0.0)


def test_inverse_monomial_function() -> None:
    """
    From original C++ code
    """

    P_coeffs = np.array([[1], [0], [0]], dtype=np.float64)
    Q_coeffs = np.array([0, 0, 1], dtype=np.float64)
    F = RationalFunction(2, 1, P_coeffs, Q_coeffs)
    # F_derivative = RationalFunction.from_zero_function(4, 1)
    F_derivative: RationalFunction = F.compute_derivative()

    assert float_equal(F_derivative(-1.0)[0], 2.0)
    assert float_equal(F_derivative(1.0)[0], -2.0)
    assert float_equal(F_derivative(2.0)[0], -0.25)


def test_inverse_quadratic_function() -> None:
    """
    From original C++ code
    """
    P_coeffs = np.array([[1], [0], [0]], dtype=np.float64)
    Q_coeffs = np.array([1, 0, 1], dtype=np.float64)
    F = RationalFunction(2, 1, P_coeffs, Q_coeffs)
    # F_derivative = RationalFunction.from_zero_function(4, 1)
    F_derivative: RationalFunction = F.compute_derivative()

    # -2t / (1 + t ^ 2) ^ 2
    assert float_equal(F_derivative(-1.0)[0], 0.5)
    assert float_equal(F_derivative(0.0)[0], 0.0)
    assert float_equal(F_derivative(1.0)[0], -0.5)
    assert float_equal(F_derivative(2.0)[0], -0.16)


def test_rational_function() -> None:
    """
    From original C++ code
    """
    P_coeffs: np.ndarray = np.array([[1], [1], [0]], dtype=np.float64)
    Q_coeffs: np.ndarray = np.array([1, 0, 1], dtype=np.float64)
    F = RationalFunction(2, 1, P_coeffs, Q_coeffs)
    # F_derivative = RationalFunction.from_zero_function(4, 1)
    F_derivative: RationalFunction = F.compute_derivative()

    assert float_equal(F_derivative(-1.0)[0], 0.5)
    assert float_equal(F_derivative(0.0)[0], 1.0)
    assert float_equal(F_derivative(1.0)[0], -0.5)
    assert float_equal(F_derivative(2.0)[0], -0.28)


def test_planar_rational_function() -> None:
    """
    From original C++ code
    """
    P_coeffs: Matrix3x2f = np.array([[1, 1],
                                     [0, 1],
                                     [0, 0]], dtype=np.float64)
    Q_coeffs: Vector3f = np.array([1, 0, 1], dtype=np.float64)
    F = RationalFunction(2, 2, P_coeffs, Q_coeffs)
    print(F)

    # F_derivative = RationalFunction.from_zero_function(4, 2)
    F_derivative: RationalFunction = F.compute_derivative()

    # FIXME: the whole non-scalar, non-vector F_derivative results might cause a lot of problems
    # -2t / (1 + t ^ 2) ^ 2
    # (1 - 2t - t ^ 2) / (1 + t ^ 2) ^ 2
    assert float_equal(F_derivative(-1.0)[0], 0.5)
    assert float_equal(F_derivative(0.0)[0], 0.0)
    assert float_equal(F_derivative(1.0)[0], -0.5)
    assert float_equal(F_derivative(2.0)[0], -0.16)

    assert float_equal(F_derivative(-1.0)[1], 0.5)
    assert float_equal(F_derivative(0.0)[1], 1.0)
    assert float_equal(F_derivative(1.0)[1], -0.5)
    assert float_equal(F_derivative(2.0)[1], -0.28)
