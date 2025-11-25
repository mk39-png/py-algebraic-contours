#
# test_compute_rational_bezier_curve_intersections.py
# Method for testing the blackbox functions featured in compute_rational_bezier_curve_intersections()
#

import numpy as np
import pytest

from pyalgcon.contour_network.compute_rational_bezier_curve_intersection import (
    _clipfatline, find_intersections_bezier_clipping)
from pyalgcon.core.common import (FIND_INTERSECTIONS_BEZIER_CLIPPING_PRECISION,
                                  Matrix5x3f, SpatialVector1d, Vector2f,
                                  compare_eigen_numpy_matrix,
                                  deserialize_eigen_matrix_csv_to_numpy,
                                  deserialize_list_list_varying_lengths_float)


@pytest.mark.filterwarnings("ignore:loadtxt")
def test_find_intersections_bezier_clipping(root_folder) -> None:
    """

    """
    filepath: str = (
        f"{root_folder}\\contour_network\\compute_rational_bezier_curve_intersection\\find_intersections_bezier_clipping\\")

    # Number of files
    for i in range(378):
        A: list[SpatialVector1d] = [
            np.array(point)
            for point in deserialize_eigen_matrix_csv_to_numpy(filepath + f"A\\{i}.csv").tolist()]
        B: list[SpatialVector1d] = [
            np.array(point)
            for point in deserialize_eigen_matrix_csv_to_numpy(filepath + f"B\\{i}.csv").tolist()]

        xs: list[tuple[float, float]] = find_intersections_bezier_clipping(
            A, B, FIND_INTERSECTIONS_BEZIER_CLIPPING_PRECISION)
        compare_eigen_numpy_matrix(filepath+f"xs\\{i}.csv", np.array(xs).squeeze())


@pytest.mark.filterwarnings("ignore:loadtxt")
def test_clipfatline(root_folder) -> None:
    """


    """
    filepath: str = f"{root_folder}\\contour_network\\compute_rational_bezier_curve_intersection\\clipfatline\\"

    #  Number of files
    # precision: float = 1e-7
    for i in range(3634):
        P: Matrix5x3f = deserialize_eigen_matrix_csv_to_numpy(filepath+f"P\\{i}.csv")
        Q: Matrix5x3f = deserialize_eigen_matrix_csv_to_numpy(filepath+f"Q\\{i}.csv")
        clip_range_test: Vector2f = np.zeros(shape=(2, ), dtype=np.float64)
        _clipfatline(P, Q, clip_range_test, FIND_INTERSECTIONS_BEZIER_CLIPPING_PRECISION)
        compare_eigen_numpy_matrix(filepath+f"clip_range\\{i}.csv", clip_range_test)
