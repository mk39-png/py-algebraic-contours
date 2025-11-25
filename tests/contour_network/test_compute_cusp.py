"""
test_compute_cusp.py

Testing compute_cusp
"""

import numpy as np

from pyalgcon.contour_network.compute_cusps import (
    compute_boundary_cusps_testing, compute_cusp_by_one_patch_testing,
    compute_cusp_start_end_points_testing, compute_spline_surface_cusps)
from pyalgcon.contour_network.compute_intersections import \
    IntersectionParameters
from pyalgcon.contour_network.contour_network import InvisibilityParameters
from pyalgcon.core.common import (Matrix3x3f, Vector1D,
                                  compare_eigen_numpy_matrix,
                                  compare_list_list_varying_lengths_float,
                                  deserialize_eigen_matrix_csv_to_numpy,
                                  deserialize_list_list_varying_lengths)
from pyalgcon.core.conic import Conic
from pyalgcon.core.rational_function import RationalFunction
from pyalgcon.quadratic_spline_surface.quadratic_spline_surface import \
    QuadraticSplineSurface
from pyalgcon.quadratic_spline_surface.twelve_split_spline import \
    TwelveSplitSplineSurface
from pyalgcon.utils.conic_testing_utils import deserialize_conics
from pyalgcon.utils.rational_function_testing_utils import \
    deserialize_rational_functions


def test_compute_spline_surface_cusps(_initialize_contour_info_spot_mesh):
    """
    Tests all 3 methods inside the compute_spline_surface_cusps() method.
    """
    # Setting up pre-parameters
    spline_surface: QuadraticSplineSurface
    intersect_params: IntersectionParameters
    invisibility_params: InvisibilityParameters
    patch_boundary_edges: list[tuple[int, int]]
    frame: Matrix3x3f
    (spline_surface,
        intersect_params,
        invisibility_params,
        patch_boundary_edges,
        frame) = _initialize_contour_info_spot_mesh

    filepath: str = "spot_control\\contour_network\\compute_cusps\\compute_spline_surface_cusps\\"
    contour_domain_curve_segments: list[Conic] = deserialize_conics(
        filepath+"contour_domain_curve_segments.json")
    contour_segments: list[RationalFunction] = deserialize_rational_functions(
        filepath+"contour_segments.json")
    patch_indices: list[int] = np.array(deserialize_eigen_matrix_csv_to_numpy(
        filepath+"patch_indices.csv"), dtype=np.int64).tolist()  # aka contour_patch_indices
    closed_contours: list[list[int]] = deserialize_list_list_varying_lengths(
        filepath+"closed_contours.csv")  # aka contours

    # Execute actual method to test
    interior_cusps: list[list[float]]
    boundary_cusps: list[list[float]]
    has_cusp_at_base: list[bool]
    has_cusp_at_tip: list[bool]
    (interior_cusps,
     boundary_cusps,
     has_cusp_at_base,
     has_cusp_at_tip) = compute_spline_surface_cusps(spline_surface,
                                                     contour_domain_curve_segments,
                                                     contour_segments,
                                                     patch_indices,
                                                     closed_contours)

    # Now actually comparing.
    compare_list_list_varying_lengths_float(filepath+"interior_cusps.csv", interior_cusps)
    compare_list_list_varying_lengths_float(filepath+"boundary_cusps.csv", boundary_cusps)
    compare_eigen_numpy_matrix(filepath+"has_cusp_at_base.csv", np.array(has_cusp_at_base))
    compare_eigen_numpy_matrix(filepath+"has_cusp_at_tip.csv", np.array(has_cusp_at_tip))


def test_compute_cusp_by_one_patch_spot_control(_initialize_contour_info_spot_mesh) -> None:
    """
    Testing according to method usage in compute_spline_surface_cusps()
    """
    filepath: str = "spot_control\\contour_network\\compute_cusps\\compute_cusp_by_one_patch\\"
    contour_domain_curve_segments: list[Conic] = deserialize_conics(
        filepath + "contour_domain_curve_segments.json")
    patch_indices: Vector1D = np.array(
        deserialize_eigen_matrix_csv_to_numpy(
            filepath + "patch_indices.csv"),
        dtype=np.int64)
    spline_surface: TwelveSplitSplineSurface
    spline_surface, _, _, _, _ = _initialize_contour_info_spot_mesh

    interior_cusps: list[list[float]] = []
    for i, _ in enumerate(contour_domain_curve_segments):
        cusp: list[float] = compute_cusp_by_one_patch_testing(
            spline_surface.get_patch(patch_indices[i]),
            contour_domain_curve_segments[i])
        interior_cusps.append(cusp)

    compare_list_list_varying_lengths_float(filepath+"interior_cusps.csv", interior_cusps)


def test_compute_cusp_start_end_points_spot_control(_initialize_contour_info_spot_mesh) -> None:
    """

    """
    # Setting up pre-parameters
    spline_surface: QuadraticSplineSurface
    intersect_params: IntersectionParameters
    invisibility_params: InvisibilityParameters
    patch_boundary_edges: list[tuple[int, int]]
    frame: Matrix3x3f
    (spline_surface,
     intersect_params,
     invisibility_params,
     patch_boundary_edges,
     frame) = _initialize_contour_info_spot_mesh

    filepath: str = "spot_control\\contour_network\\compute_cusps\\compute_cusp_start_end_points\\"
    contour_domain_curve_segments: list[Conic] = deserialize_conics(
        filepath+"contour_domain_curve_segments.json")
    patch_indices: list[int] = np.array(deserialize_eigen_matrix_csv_to_numpy(
        filepath+"patch_indices.csv"), dtype=np.int64).tolist()  # aka contour_patch_indices

    # Compute cusp function endpoints
    function_start_points: list[float]
    function_end_points: list[float]
    function_start_points_param: list[float]
    function_end_points_param: list[float]
    (function_start_points, function_end_points, function_start_points_param,
     function_end_points_param) = compute_cusp_start_end_points_testing(spline_surface,
                                                                        contour_domain_curve_segments, patch_indices)

    compare_eigen_numpy_matrix(filepath+"function_start_points.csv",
                               np.array(function_start_points))
    compare_eigen_numpy_matrix(filepath+"function_end_points.csv",
                               np.array(function_end_points))
    compare_eigen_numpy_matrix(filepath+"function_start_points_param.csv",
                               np.array(function_start_points_param))
    compare_eigen_numpy_matrix(filepath+"function_end_points_param.csv",
                               np.array(function_end_points_param))


def test_compute_boundary_cusps_spot_control() -> None:
    """

    """
    filepath: str = "spot_control\\contour_network\\compute_cusps\\compute_boundary_cusps\\"

    function_start_points: list[float] = np.array(deserialize_eigen_matrix_csv_to_numpy(
        filepath+"function_start_points.csv")).tolist()
    function_end_points: list[float] = np.array(deserialize_eigen_matrix_csv_to_numpy(
        filepath+"function_end_points.csv")).tolist()
    function_start_points_param: list[float] = np.array(deserialize_eigen_matrix_csv_to_numpy(
        filepath+"function_start_points_param.csv")).tolist()
    function_end_points_param: list[float] = np.array(deserialize_eigen_matrix_csv_to_numpy(
        filepath+"function_end_points_param.csv")).tolist()
    closed_contours: list[list[int]] = deserialize_list_list_varying_lengths(
        filepath+"closed_contours.csv")  # aka contours

    boundary_cusps: list[list[float]]
    has_cusp_at_base: list[bool]
    has_cusp_at_tip: list[bool]
    (boundary_cusps,
     has_cusp_at_base,
     has_cusp_at_tip) = compute_boundary_cusps_testing(function_start_points,
                                                       function_end_points,
                                                       function_start_points_param,
                                                       function_end_points_param,
                                                       closed_contours)

    compare_eigen_numpy_matrix(filepath+"has_cusp_at_base.csv", np.array(has_cusp_at_base))
    compare_eigen_numpy_matrix(filepath+"has_cusp_at_tip.csv", np.array(has_cusp_at_tip))
    compare_list_list_varying_lengths_float(filepath+"boundary_cusps.csv", boundary_cusps)
