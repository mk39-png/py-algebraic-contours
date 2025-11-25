
import os

import numpy as np
import pytest

from pyalgcon.core.common import (
    MatrixNx3f, MatrixNx3i, SpatialVector, SpatialVector1d,
    compare_eigen_numpy_matrix, convert_nested_vector_to_matrix,
    convert_polylines_to_edges, todo)
from pyalgcon.quadratic_spline_surface.quadratic_spline_surface import (
    QuadraticSplineSurface, SurfaceDiscretizationParameters)


def test_spot_control_triangulate_patch_patch_index_0() -> None:
    """ 
    This is dependent on QuadraticSplineSurface.read_spline() working properly.
    """
    # Need to initialize QuadraticSplineSurface with our test file
    filename: str = "spot_control_mesh-cleaned_conf_simplified_with_uv_CONTROL.txt"
    filepath: str = os.path.abspath(f"src\\tests\\spot_control\\{filename}")
    spline_surface: QuadraticSplineSurface = QuadraticSplineSurface.from_file(filepath)

    patch_index: int = 0
    num_subdivisions: int = 2
    V_patch_0: np.ndarray[tuple[int], np.dtype[np.float64]]
    F_patch_0: np.ndarray[tuple[int], np.dtype[np.int64]]
    N_patch_0: np.ndarray[tuple[int], np.dtype[np.float64]]
    V_patch_0, F_patch_0, N_patch_0 = spline_surface.triangulate_patch(
        patch_index, num_subdivisions)

    compare_eigen_numpy_matrix(
        "spot_control\\quadratic_spline_surface\\discretize\\F_triangulate_patch_0.csv", F_patch_0)
    compare_eigen_numpy_matrix(
        "spot_control\\quadratic_spline_surface\\discretize\\V_triangulate_patch_0.csv", V_patch_0)
    compare_eigen_numpy_matrix(
        "spot_control\\quadratic_spline_surface\\discretize\\N_triangulate_patch_0.csv", N_patch_0)


def test_spot_control_discretize() -> None:
    """
    This is dependent on QuadraticSplineSurface.read_spline() working properly.
    TODO: this is redundant with test in test_quadratic_spline.py
    """
    # Need to initialize QuadraticSplineSurface with our test file
    filename: str = "spot_control_mesh-cleaned_conf_simplified_with_uv_CONTROL.txt"
    filepath: str = os.path.abspath(f"src\\tests\\spot_control\\{filename}")
    spline_surface: QuadraticSplineSurface = QuadraticSplineSurface.from_file(filepath)

    num_subdivisions: int = 2
    surface_disc_params = SurfaceDiscretizationParameters(num_subdivisions=num_subdivisions)
    V: MatrixNx3f
    F: MatrixNx3i
    N: MatrixNx3f

    V, F, N = spline_surface.discretize(surface_disc_params)
    compare_eigen_numpy_matrix(
        "spot_control\\quadratic_spline_surface\\add_surface_to_viewer\\F_discretized_2_subdiv.csv", F)
    compare_eigen_numpy_matrix(
        "spot_control\\quadratic_spline_surface\\add_surface_to_viewer\\V_discretized_2_subdiv.csv", V)
    compare_eigen_numpy_matrix(
        "spot_control\\quadratic_spline_surface\\add_surface_to_viewer\\N_discretized_2_subdiv.csv", N)


def test_spot_control_discretize_patch_boundaries() -> None:
    """
    This is dependent on QuadraticSplineSurface.read_spline() working properly.
    TODO: this is redundant with test in test_quadratic_spline.py
    """
    # Need to initialize QuadraticSplineSurface with our test file
    filename: str = "spot_control_mesh-cleaned_conf_simplified_with_uv_CONTROL.txt"
    filepath: str = os.path.abspath(f"src\\tests\\spot_control\\{filename}")
    spline_surface: QuadraticSplineSurface = QuadraticSplineSurface.from_file(filepath)

    boundary_points: list[SpatialVector1d]
    boundary_polylines: list[list[int]]
    boundary_points, boundary_polylines = spline_surface.discretize_patch_boundaries()

    compare_eigen_numpy_matrix(
        "spot_control\\quadratic_spline_surface\\add_surface_to_viewer\\boundary_points.csv",
        np.array(boundary_points))
    compare_eigen_numpy_matrix(
        "spot_control\\quadratic_spline_surface\\add_surface_to_viewer\\boundary_polylines.csv",
        np.array(boundary_polylines))


def test_spot_control_boundary_points_matrix() -> None:
    """
    This is dependent on QuadraticSplineSurface.read_spline() working properly.
    This is dependent on QuadraticSplineSurface.discretize_patch_boundaries() working properly.
    TODO: this is redundant with test in test_quadratic_spline.py
    """
    # Need to initialize QuadraticSplineSurface with our test file
    filename: str = "spot_control_mesh-cleaned_conf_simplified_with_uv_CONTROL.txt"
    filepath: str = os.path.abspath(f"src\\tests\\spot_control\\{filename}")
    spline_surface: QuadraticSplineSurface = QuadraticSplineSurface.from_file(filepath)

    boundary_points: list[SpatialVector1d]
    _boundary_polylines: list[list[int]]
    boundary_points, _boundary_polylines = spline_surface.discretize_patch_boundaries()

    boundary_points_matrix: np.ndarray = convert_nested_vector_to_matrix(boundary_points)
    compare_eigen_numpy_matrix(
        "spot_control\\quadratic_spline_surface\\add_surface_to_viewer\\boundary_points_mat.csv",
        np.array(boundary_points_matrix))


def test_spot_control_boundary_edges() -> None:
    """
    This is dependent on QuadraticSplineSurface.read_spline() working properly.
    This is dependent on QuadraticSplineSurface.discretize_patch_boundaries() working properly.
    TODO: this is redundant with test in test_quadratic_spline.py
    """
    # Need to initialize QuadraticSplineSurface with our test file
    filename: str = "spot_control_mesh-cleaned_conf_simplified_with_uv_CONTROL.txt"
    filepath: str = os.path.abspath(f"src\\tests\\spot_control\\{filename}")
    spline_surface: QuadraticSplineSurface = QuadraticSplineSurface.from_file(filepath)

    _boundary_points: list[SpatialVector]
    boundary_polylines: list[list[int]]
    _boundary_points, boundary_polylines = spline_surface.discretize_patch_boundaries()

    boundary_edges: list[tuple[int, int]] = convert_polylines_to_edges(boundary_polylines)
    compare_eigen_numpy_matrix(
        "spot_control\\quadratic_spline_surface\\add_surface_to_viewer\\boundary_edges.csv",
        np.array(boundary_edges))
