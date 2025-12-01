"""
test_compute_closed_contours.py 

Tests compute_closed_contours inside init_contour_network.
"""

import pathlib

import numpy as np

from pyalgcon.contour_network.compute_closed_contours import \
    compute_closed_contours
from pyalgcon.core.common import (compare_eigen_numpy_matrix,
                                  compare_list_list_varying_lengths)
from pyalgcon.core.rational_function import RationalFunction
from pyalgcon.utils.rational_function_testing_utils import \
    deserialize_rational_functions


def test_compute_closed_contours(testing_fileinfo) -> None:
    """
    Testing part of init_contour_network. 
    """
    # Set up parameters
    folder_path: pathlib.Path
    folder_path, _ = testing_fileinfo
    filepath: pathlib.Path = (folder_path / "contour_network" /
                              "compute_closed_contours" / "compute_closed_contours")
    contour_segments: list[RationalFunction] = deserialize_rational_functions(
        filepath / "contour_segments.json")
    contours: list[list[int]]
    contour_labels: list[int]
    contours, contour_labels = compute_closed_contours(contour_segments)

    compare_list_list_varying_lengths(filepath / "contours.csv", contours)
    compare_eigen_numpy_matrix(filepath / "contour_labels.csv", np.array(contour_labels))
