"""
Test quadratic spline surface
"""

import pathlib

import numpy as np

from pyalgcon.core.common import (MatrixNx3f, MatrixNx3i, PlanarPoint1d,
                                  SpatialVector1d, compare_eigen_numpy_matrix,
                                  convert_nested_vector_to_matrix,
                                  convert_polylines_to_edges,
                                  deserialize_eigen_matrix_csv_to_numpy)
from pyalgcon.quadratic_spline_surface.quadratic_spline_surface import (
    QuadraticSplineSurface, SurfaceDiscretizationParameters)
from pyalgcon.quadratic_spline_surface.twelve_split_spline import \
    TwelveSplitSplineSurface


def test_compute_hash_indices(testing_fileinfo,
                              twelve_split_spline_transformed) -> None:
    """
    Testing part of compute_spline_surface_ray_intersections for contour calculation as well

    NOTE: this test currently fails since the hash indices do not need to match 1-to-1
    between the Python and C++ code.
    """
    # Retrieving parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / "quadratic_spline_surface" / \
        "quadratic_spline_surface" / "compute_hash_indices"
    spline_surface: TwelveSplitSplineSurface = twelve_split_spline_transformed  # subclass inherits

    # Loop through files
    for i in range(198):
        # Deserialize parameter
        ray_plane_point: PlanarPoint1d = deserialize_eigen_matrix_csv_to_numpy(
            filepath / "ray_plane_point" / f"{i}.csv")

        # Execute method
        hash_indices: tuple[int, int] = spline_surface.compute_hash_indices(ray_plane_point)

        # Compare results
        compare_eigen_numpy_matrix(filepath / "hash_indices" / f"{i}.csv", np.array(hash_indices))


def test_triangulate_patch_patch_index_0(testing_fileinfo,
                                         quadratic_spline_surface_control_from_file) -> None:
    """ 
    This is dependent on QuadraticSplineSurface.read_spline() working properly.
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / "quadratic_spline_surface" / "discretize"

    # Need to initialize QuadraticSplineSurface with our test file
    spline_surface: QuadraticSplineSurface = quadratic_spline_surface_control_from_file

    # Execute method
    patch_index: int = 0
    num_subdivisions: int = 2
    V_patch_0: np.ndarray[tuple[int], np.dtype[np.float64]]
    F_patch_0: np.ndarray[tuple[int], np.dtype[np.int64]]
    N_patch_0: np.ndarray[tuple[int], np.dtype[np.float64]]
    V_patch_0, F_patch_0, N_patch_0 = spline_surface.triangulate_patch(
        patch_index, num_subdivisions)

    # Compare results
    compare_eigen_numpy_matrix(filepath / "F_triangulate_patch_0.csv", F_patch_0)
    compare_eigen_numpy_matrix(filepath / "V_triangulate_patch_0.csv", V_patch_0)
    compare_eigen_numpy_matrix(filepath / "N_triangulate_patch_0.csv", N_patch_0)


def test_discretize(testing_fileinfo,
                    quadratic_spline_surface_control_from_file) -> None:
    """
    This is dependent on QuadraticSplineSurface.read_spline() working properly.
    TODO: this is redundant with test in test_quadratic_spline.py
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    control_filepath: pathlib.Path = base_data_folderpath / \
        "quadratic_spline_surface" / "add_surface_to_viewer"

    # Need to initialize QuadraticSplineSurface with our test file
    spline_surface: QuadraticSplineSurface = quadratic_spline_surface_control_from_file

    # Execute method
    num_subdivisions: int = 2
    surface_disc_params = SurfaceDiscretizationParameters(num_subdivisions=num_subdivisions)
    V: MatrixNx3f
    F: MatrixNx3i
    N: MatrixNx3f
    V, F, N = spline_surface.discretize(surface_disc_params)

    # Compare results
    compare_eigen_numpy_matrix(control_filepath / "F_discretized_2_subdiv.csv", F)
    compare_eigen_numpy_matrix(control_filepath / "V_discretized_2_subdiv.csv", V)
    compare_eigen_numpy_matrix(control_filepath / "N_discretized_2_subdiv.csv", N)


def test_discretize_patch_boundaries(testing_fileinfo,
                                     quadratic_spline_surface_control_from_file) -> None:
    """
    This is dependent on QuadraticSplineSurface.read_spline() working properly.
    TODO: this is redundant with test in test_quadratic_spline.py
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / \
        "quadratic_spline_surface" / "add_surface_to_viewer"

    # Need to initialize QuadraticSplineSurface with our test file
    spline_surface: QuadraticSplineSurface = quadratic_spline_surface_control_from_file

    # Execute functions
    boundary_points: list[SpatialVector1d]
    boundary_polylines: list[list[int]]
    boundary_points, boundary_polylines = spline_surface.discretize_patch_boundaries()

    # Compare results
    compare_eigen_numpy_matrix(
        filepath / "boundary_points.csv",
        np.array(boundary_points))
    compare_eigen_numpy_matrix(
        filepath / "boundary_polylines.csv",
        np.array(boundary_polylines))


def test_boundary_points_matrix(testing_fileinfo,
                                quadratic_spline_surface_control_from_file) -> None:
    """
    This is dependent on QuadraticSplineSurface.read_spline() working properly.
    This is dependent on QuadraticSplineSurface.discretize_patch_boundaries() working properly.
    TODO: this is redundant with test in test_quadratic_spline.py
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / \
        "quadratic_spline_surface" / "add_surface_to_viewer"

    # Need to initialize QuadraticSplineSurface with our test file
    spline_surface: QuadraticSplineSurface = quadratic_spline_surface_control_from_file

    # Execute functions
    boundary_points: list[SpatialVector1d]
    boundary_points, _ = spline_surface.discretize_patch_boundaries()
    boundary_points_matrix: np.ndarray = convert_nested_vector_to_matrix(boundary_points)

    # Compare results
    compare_eigen_numpy_matrix(
        filepath / "boundary_points_mat.csv",
        np.array(boundary_points_matrix))


def test_boundary_edges(testing_fileinfo,
                        quadratic_spline_surface_control_from_file) -> None:
    """
    This is dependent on QuadraticSplineSurface.read_spline() working properly.
    This is dependent on QuadraticSplineSurface.discretize_patch_boundaries() working properly.
    TODO: this is redundant with test in test_quadratic_spline.py
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / "quadratic_spline_surface" / \
        "add_surface_to_viewer"

    # Need to initialize QuadraticSplineSurface with our test file
    spline_surface: QuadraticSplineSurface = quadratic_spline_surface_control_from_file

    # Execute method
    boundary_polylines: list[list[int]]
    _, boundary_polylines = spline_surface.discretize_patch_boundaries()
    boundary_edges: list[tuple[int, int]] = convert_polylines_to_edges(boundary_polylines)

    # Compare results
    compare_eigen_numpy_matrix(
        filepath / "boundary_edges.csv",
        np.array(boundary_edges))
