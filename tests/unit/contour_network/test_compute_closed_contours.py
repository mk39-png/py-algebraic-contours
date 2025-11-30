"""
test_compute_closed_contours.py 

Tests compute_closed_contours inside init_contour_network.
"""

import pathlib

import numpy as np

from pyalgcon.contour_network.compute_closed_contours import \
    compute_closed_contours
from pyalgcon.core.rational_function import RationalFunction
from pyalgcon.debug.debug import (compare_eigen_numpy_matrix_absolute,
                                  compare_list_list_varying_lengths_absolute,
                                  deserialize_rational_functions_absolute)


def test_compute_closed_contours(obj_fileinfo) -> None:
    """
    Testing part of init_contour_network. 
    """
    # Set up parameters
    folder_path: pathlib.Path
    folder_path, _ = obj_fileinfo
    filepath: pathlib.Path = (folder_path / "contour_network" /
                              "compute_closed_contours" / "compute_closed_contours")
    contour_segments: list[RationalFunction] = deserialize_rational_functions_absolute(
        filepath / "contour_segments.json")
    contours: list[list[int]]
    contour_labels: list[int]
    contours, contour_labels = compute_closed_contours(contour_segments)

    compare_list_list_varying_lengths_absolute(filepath / "contours.csv", contours)
    compare_eigen_numpy_matrix_absolute(filepath / "contour_labels.csv", np.array(contour_labels))
