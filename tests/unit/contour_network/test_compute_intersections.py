"""
Test compute interesections
"""
import pathlib

import numpy as np
import pytest

from pyalgcon.contour_network.compute_intersections import (
    IntersectionParameters,
    _compute_bezier_clipping_planar_curve_intersections,
    _compute_planar_curve_intersections_from_bounding_box,
    _prune_intersection_points, compute_intersections,
    compute_planar_curve_intersections)
from pyalgcon.contour_network.intersection_data import IntersectionData
from pyalgcon.contour_network.intersection_heuristics import IntersectionStats
from pyalgcon.core.common import (Matrix5x2f, Matrix5x3f, PlanarPoint1d,
                                  Vector5f, compare_eigen_numpy_matrix,
                                  compare_list_list_varying_lengths,
                                  compare_list_list_varying_lengths_float,
                                  deserialize_eigen_matrix_csv_to_numpy,
                                  float_equal)
from pyalgcon.core.rational_function import RationalFunction
from pyalgcon.utils.compute_intersections_testing_utils import (
    compare_intersection_stats, compare_list_list_intersection_data_from_file,
    deserialize_intersection_stats, deserialize_list_list_intersection_data)
from pyalgcon.utils.rational_function_testing_utils import \
    deserialize_rational_functions_from_file

# ********************
# Main Testing Methods
# ********************


@pytest.mark.filterwarnings("ignore:loadtxt")
def test_compute_intersections(testing_fileinfo: tuple[pathlib.Path, pathlib.Path]) -> None:
    """
    Test compute_intersections() as it appears in init_contour_network() where 
    planar_contour_segments is another name for image_segments
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / \
        "contour_network" / "compute_intersections" / "compute_intersections"
    planar_contour_segments: list[RationalFunction] = deserialize_rational_functions_from_file(
        filepath / "image_segments.json")
    intersect_params: IntersectionParameters = IntersectionParameters()
    contour_intersections: list[list[IntersectionData]] = deserialize_list_list_intersection_data(
        filepath / "contour_intersections_in.json")
    num_intersections_in: int = 0

    # Execute method
    # Alias to match the method as it appears in init_contour_network
    intersection_knots: list[list[float]]
    intersection_indices: list[list[int]]
    intersection_call: int
    num_intersections_out: int
    (intersection_knots,
     intersection_indices,
     num_intersections_out,
     intersection_call) = compute_intersections(planar_contour_segments,
                                                intersect_params,
                                                contour_intersections,
                                                num_intersections_in)

    # Compare results
    assert num_intersections_out == 176
    assert intersection_call == 378
    compare_list_list_intersection_data_from_file(
        filepath / "contour_intersections_out.json",
        contour_intersections)
    compare_list_list_varying_lengths(
        filepath / "intersection_indices.csv",
        intersection_indices)
    compare_list_list_varying_lengths_float(
        filepath / "intersections.csv",
        intersection_knots)


@pytest.mark.filterwarnings("ignore:loadtxt")
def test_prune_intersection_points(testing_fileinfo: tuple[pathlib.Path, pathlib.Path]) -> None:
    """
    Tests prune_intersection_points with spot_control mesh
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / \
        "contour_network" / "compute_intersections" / "prune_intersection_points"

    for i in range(378):
        # Initialize parameters
        first_planar_curve: RationalFunction = deserialize_rational_functions_from_file(
            filepath / "first_planar_curve" / f"{i}.json")[0]
        second_planar_curve: RationalFunction = deserialize_rational_functions_from_file(
            filepath / "second_planar_curve" / f"{i}.json")[0]
        intersection_points:  np.ndarray = deserialize_eigen_matrix_csv_to_numpy(
            filepath / "intersection_points" / f"{i}.csv")

        # Making a 1D array for such situations
        if (intersection_points.ndim == 1 and intersection_points.size != 0):
            intersection_points = np.array([intersection_points])

        first_curve_intersections_test: list[float] = deserialize_eigen_matrix_csv_to_numpy(
            filepath / "first_curve_intersections_in" / f"{i}.csv").tolist()
        second_curve_intersections_test: list[float] = deserialize_eigen_matrix_csv_to_numpy(
            filepath / "second_curve_intersections_in" / f"{i}.csv").tolist()

        # Execute method
        _prune_intersection_points(first_planar_curve,
                                   second_planar_curve,
                                   intersection_points,
                                   first_curve_intersections_test,
                                   second_curve_intersections_test)

        # Compare results
        compare_eigen_numpy_matrix(filepath / "first_curve_intersections_out" / f"{i}.csv",
                                   np.array(first_curve_intersections_test))
        compare_eigen_numpy_matrix(filepath / "second_curve_intersections_out" / f"{i}.csv",
                                   np.array(second_curve_intersections_test))


@pytest.mark.filterwarnings("ignore:loadtxt")
def test_compute_bezier_clipping_planar_curve_intersections(
        testing_fileinfo: tuple[pathlib.Path, pathlib.Path]) -> None:
    """
    Test compute bezier clipping planar curve intersections
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / "contour_network" / \
        "compute_intersections" / "compute_bezier_clipping_planar_curve_intersections"

    for i in range(378):
        # Deserialize parameters
        first_planar_curve: RationalFunction = deserialize_rational_functions_from_file(
            filepath / "first_planar_curve" / f"{i}.json")[0]
        second_planar_curve: RationalFunction = deserialize_rational_functions_from_file(
            filepath / "second_planar_curve" / f"{i}.json")[0]
        first_bezier_control_points: Matrix5x3f = deserialize_eigen_matrix_csv_to_numpy(
            filepath / "first_bezier_control_points" / f"{i}.csv")
        second_bezier_control_points: Matrix5x3f = deserialize_eigen_matrix_csv_to_numpy(
            filepath / "second_bezier_control_points" / f"{i}.csv")

        # Execute method
        intersection_points_test: list[tuple[float, float]] = (
            _compute_bezier_clipping_planar_curve_intersections(first_planar_curve,
                                                                second_planar_curve,
                                                                first_bezier_control_points,
                                                                second_bezier_control_points))

        # Compare results
        compare_eigen_numpy_matrix(filepath / "intersection_points" / f"{i}.csv",
                                   np.array(intersection_points_test).squeeze())

# def test_compute_spline_surface_boundary_intersections_spot_control() -> None:
#     """

#     """
#     filepath: pathlib.Path = "spot_control\\contour_network\\compute_intersections\\compute_intersections\\"
#     tester: list[list[IntersectionData]] = deserialize_list_list_intersection_data(
#         filepath+"contour_intersections.json")


@pytest.mark.filterwarnings("ignore:loadtxt")
def test_compute_planar_curve_intersections(
        testing_fileinfo: tuple[pathlib.Path, pathlib.Path]) -> None:
    """
    Tests the planar curve intersections computation from bounding box
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / \
        "contour_network" / "compute_intersections" / "compute_planar_curve_intersections"

    for i in range(2773):
        # Deserialize parameters
        first_planar_curve: RationalFunction = deserialize_rational_functions_from_file(
            filepath / "first_planar_curve" / f"{i}.json")[0]
        second_planar_curve: RationalFunction = deserialize_rational_functions_from_file(
            filepath / "second_planar_curve" / f"{i}.json")[0]
        intersect_params = IntersectionParameters()
        first_curve_intersections: list[float] = []
        second_curve_intersections: list[float] = []
        intersection_stats: IntersectionStats = deserialize_intersection_stats(
            filepath / "intersection_stats_in" / f"{i}.json")
        first_bounding_box: tuple[PlanarPoint1d, PlanarPoint1d] = (
            deserialize_eigen_matrix_csv_to_numpy(
                filepath / "first_bounding_box" / f"{i}.csv"))
        second_bounding_box: tuple[PlanarPoint1d, PlanarPoint1d] = (
            deserialize_eigen_matrix_csv_to_numpy(
                filepath / "second_bounding_box" / f"{i}.csv"))
        first_bezier_control_points: Matrix5x3f = deserialize_eigen_matrix_csv_to_numpy(
            filepath / "first_bezier_control_points" / f"{i}.csv")
        second_bezier_control_points: Matrix5x3f = deserialize_eigen_matrix_csv_to_numpy(
            filepath / "second_bezier_control_points" / f"{i}.csv")

        # Execute method
        _compute_planar_curve_intersections_from_bounding_box(first_planar_curve,
                                                              second_planar_curve,
                                                              intersect_params,
                                                              first_curve_intersections,
                                                              second_curve_intersections,
                                                              intersection_stats,
                                                              first_bounding_box,
                                                              second_bounding_box,
                                                              first_bezier_control_points,
                                                              second_bezier_control_points)

        # Testing results
        compare_eigen_numpy_matrix(
            filepath / "first_curve_intersections" / f"{i}.csv",
            np.array(first_curve_intersections))
        compare_eigen_numpy_matrix(
            filepath / "second_curve_intersections" / f"{i}.csv",
            np.array(second_curve_intersections))
        compare_intersection_stats(
            filepath / "intersection_stats_out" / f"{i}.json",
            intersection_stats)


def test_compute_intersections_simple_linear_functions() -> None:
    """
    Test case from original Algebraic Contours code.
    Simple Linear Functions.
    """
    first_P_coeffs: Matrix5x2f = np.array([[1, 0],
                                           [2, 1],
                                           [0, 0],
                                           [0, 0],
                                           [0, 0],])
    first_Q_coeffs: Vector5f = np.array([1, 0, 0, 0, 0])
    second_P_coeffs: Matrix5x2f = np.array([[4, 0],
                                            [-1, 1],
                                            [0, 0],
                                            [0, 0],
                                            [0, 0]])
    second_Q_coeffs: Vector5f = np.array([1, 0, 0, 0, 0])
    first_curve_intersections: list[float] = []
    second_curve_intersections: list[float] = []
    intersection_stats: IntersectionStats = IntersectionStats()
    intersection_params: IntersectionParameters = IntersectionParameters()

    first_image_segment = RationalFunction(4, 2, first_P_coeffs, first_Q_coeffs)
    second_image_segment = RationalFunction(4, 2, second_P_coeffs, second_Q_coeffs)

    # Execute method
    compute_planar_curve_intersections(first_image_segment, second_image_segment,
                                       intersection_params,
                                       first_curve_intersections, second_curve_intersections,
                                       intersection_stats)

    # Compare results
    assert len(first_curve_intersections) == 1
    assert len(second_curve_intersections) == 1
    float_equal(first_curve_intersections[0], 0.0)
    float_equal(second_curve_intersections[0], 0.0)
