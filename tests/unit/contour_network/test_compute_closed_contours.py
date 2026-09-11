"""
test_compute_closed_contours.py 

Tests compute_closed_contours inside init_contour_network.
"""

import pathlib

import numpy as np
import pytest

from pyalgcon.contour_network.compute_closed_contours import (
    _point_distance_squared, compute_closed_contours)
from pyalgcon.contour_network.compute_closed_contours_c import \
    point_distance_squared as point_distance_squared_c
from pyalgcon.core.common import (compare_eigen_numpy_matrix,
                                  compare_list_list_varying_lengths,
                                  float_equal)
from pyalgcon.core.rational_function import RationalFunction
from pyalgcon.utils.rational_function_testing_utils import \
    deserialize_rational_functions_from_file


def test_compute_closed_contours(testing_fileinfo) -> None:
    """
    Testing part of init_contour_network. 
    """
    # Set up parameters
    folder_path: pathlib.Path
    folder_path, _ = testing_fileinfo
    filepath: pathlib.Path = (folder_path / "contour_network" /
                              "compute_closed_contours" / "compute_closed_contours")
    contour_segments: list[RationalFunction] = deserialize_rational_functions_from_file(
        filepath / "contour_segments.json")
    contours: list[list[int]]
    contour_labels: list[int]
    contours, contour_labels = compute_closed_contours(contour_segments)

    compare_list_list_varying_lengths(filepath / "contours.csv", contours)
    compare_eigen_numpy_matrix(filepath / "contour_labels.csv", np.array(contour_labels))


@pytest.mark.regression
def test_point_squared_distance():
    """
    Ensures that Cython version runs identically.
    """
    num_cases = 10000
    v = np.random.uniform(-100, 100, size=(num_cases, 3))
    w = np.random.uniform(-100, 100, size=(num_cases, 3))

    for i in range(num_cases):
        n_control = _point_distance_squared(v[i], w[i])
        n_test = point_distance_squared_c(v[i], w[i])

        assert float_equal(n_control, n_test, 1e-10), \
            f"Failed! {n_control} not equal {n_test}"
