"""
test_compute_rational_bezier_curve_intersections.py

Method for testing the blackbox functions featured in compute_rational_bezier_curve_intersections()
"""

import pathlib

import numpy as np
import pytest

from pyalgcon.contour_network.compute_rational_bezier_curve_intersection import (
    _clipfatline, find_intersections_bezier_clipping)
from pyalgcon.core.common import (FIND_INTERSECTIONS_BEZIER_CLIPPING_PRECISION,
                                  Matrix5x3f, SpatialVector1d, Vector2f,
                                  compare_eigen_numpy_matrix,
                                  deserialize_eigen_matrix_csv_to_numpy)


@pytest.mark.filterwarnings("ignore:loadtxt")
def test_find_intersections_bezier_clipping(testing_fileinfo) -> None:
    """
    Tests semi black box function find intersections bezier clipping.
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / "contour_network" / \
        "compute_rational_bezier_curve_intersection" / "find_intersections_bezier_clipping"

    # i represents number of files
    for i in range(378):
        A: list[SpatialVector1d] = [
            np.array(point)
            for point in deserialize_eigen_matrix_csv_to_numpy(filepath / "A" / f"{i}.csv").tolist()
        ]
        B: list[SpatialVector1d] = [
            np.array(point)
            for point in deserialize_eigen_matrix_csv_to_numpy(filepath / "B" / f"{i}.csv").tolist()
        ]

        # Execute method
        xs: list[tuple[float, float]] = find_intersections_bezier_clipping(
            A, B, FIND_INTERSECTIONS_BEZIER_CLIPPING_PRECISION)

        # Compare results
        compare_eigen_numpy_matrix(filepath / "xs" / f"{i}.csv", np.array(xs).squeeze())


@pytest.mark.filterwarnings("ignore:loadtxt")
def test_clipfatline(testing_fileinfo) -> None:
    """
    Tests black box function clipfatline
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / "contour_network" / \
        "compute_rational_bezier_curve_intersection" / "clipfatline"

    # Number of files
    # precision: float = 1e-7
    for i in range(3634):
        P: Matrix5x3f = deserialize_eigen_matrix_csv_to_numpy(filepath / "P" / f"{i}.csv")
        Q: Matrix5x3f = deserialize_eigen_matrix_csv_to_numpy(filepath / "Q" / f"{i}.csv")
        clip_range_test: Vector2f = np.zeros(shape=(2, ), dtype=np.float64)

        # Execute method
        _clipfatline(P, Q, clip_range_test, FIND_INTERSECTIONS_BEZIER_CLIPPING_PRECISION)

        # Compare results
        compare_eigen_numpy_matrix(filepath / "clip_range" / f"{i}.csv", clip_range_test)
