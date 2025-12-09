"""
Unit tests that test compute_twelve_split_energy_quadratic()
Primarily testing the variables inside of compute_twelve_split_energy_quadratic()
since the whole process is highly dependent on the value of the outputs being 
identical to the ground truth that is the original C++ code.

NOTE: a lot of these tests were originally inline debugging statements 
moved out here for testing.
So, they may be highly coupled to the structure, layout, and design of the 
compute_twelve_split_energy_quadratic() code

Major component of Quadratic Surface calculation that is susceptible to go wrong.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from pyalgcon.core.affine_manifold import AffineManifold
from pyalgcon.core.common import (Matrix2x3r, MatrixNx3f, MatrixXi,
                                  SpatialVector, SpatialVector1d,
                                  compare_eigen_numpy_matrix)
from pyalgcon.quadratic_spline_surface.optimize_spline_surface import (
    OptimizationParameters, build_twelve_split_energy_quadratic_params)
from pyalgcon.quadratic_spline_surface.powell_sabin_local_to_global_indexing import \
    build_face_variable_vector
from pyalgcon.quadratic_spline_surface.twelve_split_spline import \
    TwelveSplitSplineSurface


#
# Utility class
#
@dataclass
class TwelveSplitQuadraticParams:
    # """
    # To hold the long list of parameters
    # NOTE: the ordering of these member variables is different from the ordering of parameters
    # that compute_twelve_split_energy_quadratic() expects
    # """
    vertex_positions: list[SpatialVector1d]
    vertex_gradients: list[Matrix2x3r]
    edge_gradients: list[list[SpatialVector1d]]  # list[SpatialVector1d] of length 3
    global_vertex_indices: list[int]
    global_edge_indices: list[list[int]]  # list[int] of length 3
    initial_vertex_positions: list[SpatialVector]
    initial_face_normals: np.ndarray  # matrix
    affine_manifold: AffineManifold
    optimization_params: OptimizationParameters
    num_variable_vertices: int
    num_variable_edges: int


#
# Local fixture
#
@pytest.fixture(scope="session", params=[
    "fit", "full"
])
def energy_quadratic_filepath(request, testing_fileinfo) -> Path:
    """
    Returns a suffix to append to the filepath to address a particular
    folder we would like to use.
    Shifted case is for data after shift_local_energy_quadratic_vertices()
    fit case for testing build_twelve_split_spline_energy_system() for the
    fit_matrix with optimization_params_fit

    full case is for computing energy_hessian_inverse with optimization_params_full
    from build_twelve_split_spline_energy_system()
    """
    # Retrieve parameters
    filepath_suffix: str = request.param
    base_data_folderpath: Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: Path = base_data_folderpath / "quadratic_spline_surface" / \
        "optimize_spline_surface" / "compute_twelve_split_energy_quadratic" / filepath_suffix

    # Return values
    return filepath


@pytest.fixture(scope="session")
def compute_twelve_split_energy_quadratic_params(energy_quadratic_filepath,
                                                 parsed_control_mesh,
                                                 initialize_affine_manifold
                                                 ) -> TwelveSplitQuadraticParams:
    """
    Fit and Full cases.

    :return vertex_positions:         /
    :return vertex_gradients:         /
    :return edge_gradients:           /
    :return global_vertex_indices:    /
    :return global_edge_indices:      /
    :return initial_vertex_positions: /
    :return initial_face_normals:     /
    :return manifold:                 /
    :return optimization_params:      /
    :return num_variable_vertices:    /
    :return num_variable_edges:       /
    """
    # Retrieve parameters
    initial_V: np.ndarray
    initial_V, _, _, _ = parsed_control_mesh
    affine_manifold: AffineManifold = initialize_affine_manifold
    filepath: Path = energy_quadratic_filepath
    optimization_params: OptimizationParameters
    if filepath.name == "fit":
        # Fit case where mapping factor = 0.0
        optimization_params = OptimizationParameters(
            parametrized_quadratic_surface_mapping_factor=0.0)
    else:
        # Full case
        optimization_params = OptimizationParameters()

    initial_face_normals: MatrixNx3f = TwelveSplitSplineSurface.generate_face_normals(
        initial_V, affine_manifold)

    # HACK: long list of typing to get typing all working
    # TODO: return a dictionary where the keys are the strings of the variable names
    vertex_positions: list[SpatialVector1d]
    vertex_gradients: list[Matrix2x3r]
    edge_gradients: list[list[SpatialVector1d]]  # list[SpatialVector1d] of length 3
    global_vertex_indices: list[int]
    global_edge_indices: list[list[int]]  # list[int] of length 3
    initial_vertex_positions: list[SpatialVector]
    num_variable_vertices: int
    num_variable_edges: int
    (vertex_positions,
     vertex_gradients,
     edge_gradients,
     global_vertex_indices,
     global_edge_indices,
     initial_vertex_positions,
     num_variable_vertices,
     num_variable_edges) = build_twelve_split_energy_quadratic_params(
        initial_V,
        affine_manifold)

    return TwelveSplitQuadraticParams(
        vertex_positions,
        vertex_gradients,
        edge_gradients,
        global_vertex_indices,
        global_edge_indices,
        initial_vertex_positions,
        initial_face_normals,
        affine_manifold,
        optimization_params,
        num_variable_vertices,
        num_variable_edges)


def test_initial_vertex_positions_t(energy_quadratic_filepath,
                                    compute_twelve_split_energy_quadratic_params) -> None:
    """
    Test build_face_variable_vector for fit and full cases
    """
    # Retrieve parameters
    filepath: Path = energy_quadratic_filepath

    twelve_split_quadratic_params: TwelveSplitQuadraticParams = (
        compute_twelve_split_energy_quadratic_params)
    initial_vertex_positions: list[SpatialVector] = (
        twelve_split_quadratic_params.initial_vertex_positions)
    affine_manifold: AffineManifold = twelve_split_quadratic_params.affine_manifold

    for face_index in range(affine_manifold.num_faces):
        # Get face vertices
        F: MatrixXi = affine_manifold.faces
        __i: int = F[face_index, 0]
        __j: int = F[face_index, 1]
        __k: int = F[face_index, 2]

        # Bundle relevant global variables into per face local vectors (all list of length 3)
        initial_vertex_positions_T: list[SpatialVector1d] = build_face_variable_vector(
            initial_vertex_positions, __i, __j, __k)

        assert len(initial_vertex_positions_T) == 3
        compare_eigen_numpy_matrix(
            filepath / "initial_vertex_positions_T" / f"face_index_{face_index}.csv",
            np.array(initial_vertex_positions_T).squeeze())


def test_vertex_positions_t(energy_quadratic_filepath,
                            compute_twelve_split_energy_quadratic_params) -> None:
    """
    Test vertex_positions_T for fit and full cases
    """
    # Retrieve parameters
    filepath: Path = energy_quadratic_filepath
    twelve_split_quadratic_params: TwelveSplitQuadraticParams = (
        compute_twelve_split_energy_quadratic_params)
    affine_manifold: AffineManifold = twelve_split_quadratic_params.affine_manifold
    vertex_positions: list[SpatialVector1d] = twelve_split_quadratic_params.vertex_positions

    for face_index in range(affine_manifold.num_faces):
        # Get face vertices
        F: MatrixXi = affine_manifold.faces
        __i: int = F[face_index, 0]
        __j: int = F[face_index, 1]
        __k: int = F[face_index, 2]

        # Bundle relevant global variables into per face local vectors (all list of length 3)
        vertex_positions_T: list[SpatialVector] = build_face_variable_vector(vertex_positions,
                                                                             __i,
                                                                             __j,
                                                                             __k)

        assert len(vertex_positions_T) == 3
        compare_eigen_numpy_matrix(
            filepath / "vertex_positions_T" / f"face_index_{face_index}.csv",
            np.array(vertex_positions_T).squeeze())


def test_vertex_gradients_t(energy_quadratic_filepath,
                            compute_twelve_split_energy_quadratic_params) -> None:
    """
    Test vertex_gradients_T for fit and full cases
    """
    # Retrieve parameters
    filepath: Path = energy_quadratic_filepath
    twelve_split_quadratic_params: TwelveSplitQuadraticParams = (
        compute_twelve_split_energy_quadratic_params)
    affine_manifold: AffineManifold = twelve_split_quadratic_params.affine_manifold
    vertex_gradients: list[SpatialVector1d] = twelve_split_quadratic_params.vertex_gradients

    for face_index in range(affine_manifold.num_faces):
        # Get face vertices
        F: MatrixXi = affine_manifold.faces
        __i: int = F[face_index, 0]
        __j: int = F[face_index, 1]
        __k: int = F[face_index, 2]

        # Bundle relevant global variables into per face local vectors (all list of length 3)
        vertex_gradients_T: list[Matrix2x3r] = build_face_variable_vector(vertex_gradients,
                                                                          __i,
                                                                          __j,
                                                                          __k)

        assert len(vertex_gradients_T) == 3
        compare_eigen_numpy_matrix(
            filepath / "vertex_gradients_T" / f"face_index_{face_index}.csv",
            np.array(vertex_gradients_T),
            make_3d=True)


def test_edge_gradients_t(energy_quadratic_filepath,
                          compute_twelve_split_energy_quadratic_params) -> None:
    """
    Test edge gradients for fit and full cases
    """
    # Retrieve parameters
    filepath: Path = energy_quadratic_filepath
    twelve_split_quadratic_params: TwelveSplitQuadraticParams = (
        compute_twelve_split_energy_quadratic_params)
    affine_manifold: AffineManifold = twelve_split_quadratic_params.affine_manifold
    edge_gradients: list[list[SpatialVector1d]] = twelve_split_quadratic_params.edge_gradients

    for face_index in range(affine_manifold.num_faces):
        # Get face vertices
        F: MatrixXi = affine_manifold.faces
        __i: int = F[face_index, 0]
        __j: int = F[face_index, 1]
        __k: int = F[face_index, 2]

        # Bundle relevant global variables into per face local vectors (all list of length 3)
        edge_gradients_T: list[SpatialVector] = edge_gradients[face_index]

        assert len(edge_gradients_T) == 3
        compare_eigen_numpy_matrix(
            filepath / "edge_gradients_T" / f"face_index_{face_index}.csv",
            np.array(edge_gradients_T).squeeze())

#
# HIGHLY COUPLED TEST CASES
#
