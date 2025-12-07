"""
Unit tests that test compute_twelve_split_energy_quadratic()
Major component of Quadratic Surface calculation that is susceptible to go wrong.
"""

import numpy as np
import pytest
from quadratic_spline_surface.twelve_split_spline import \
    TwelveSplitSplineSurface

from pyalgcon.core.affine_manifold import AffineManifold
from pyalgcon.core.common import (Matrix2x3r, MatrixNx3f, SpatialVector,
                                  SpatialVector1d)
from pyalgcon.quadratic_spline_surface.optimize_spline_surface import (
    OptimizationParameters, build_twelve_split_energy_quadratic_params)

# @pytest.fixture(scope="session", params=[
#     "fit", "full"
# ])
# def energy_quadratic_suffix(request) -> pathlib.Path:
#     """
#     Returns a suffix to append to the filepath to address a particular
#     folder we would like to use.
#     Shifted case is for data after shift_local_energy_quadratic_vertices()
#     fit case for testing build_twelve_split_spline_energy_system() for the
#     fit_matrix with optimization_params_fit

#     full case is for computing energy_hessian_inverse with optimization_params_full
#     from build_twelve_split_spline_energy_system()
#     """
#     # Setup
#     filepath_suffix: str = request.param

#     # Return values
#     return pathlib.Path(filepath_suffix)


@pytest.fixture(params=["fit", "full"])
def compute_twelve_split_energy_quadratic_params(request,
                                                 parsed_control_mesh,
                                                 initialize_affine_manifold
                                                 ) -> tuple[list[SpatialVector1d],
                                                            list[Matrix2x3r],
                                                            # list[SpatialVector1d] of length 3
                                                            list[list[SpatialVector1d]],
                                                            list[int],
                                                            # list[int] of length 3
                                                            list[list[int]],
                                                            list[SpatialVector],
                                                            np.ndarray,  # matrix
                                                            AffineManifold,
                                                            OptimizationParameters,
                                                            int,
                                                            int]:
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
    optimization_params: OptimizationParameters
    if (request.param == "fit"):
        # Fit case where mapping factor = 0.0
        optimization_params = OptimizationParameters(
            parametrized_quadratic_surface_mapping_factor=0.0)
    else:
        # Full case
        optimization_params = OptimizationParameters()

    initial_face_normals: MatrixNx3f = TwelveSplitSplineSurface.generate_face_normals(
        initial_V, affine_manifold)

    return build_twelve_split_energy_quadratic_params(initial_V,
                                                      initial_face_normals,
                                                      affine_manifold,
                                                      optimization_params)


# @pytest.fixutre
# def pre():
#     """
#     Gathers the parameters needed for the test
#     """
#     # Only get fit or full from the path. That's it.
#     # but also, redirect to build_twelve_split_spline_energy_system() data folder
#     # since that is being used to pass in our


# def test_build_face_variable_vector(compute_twelve_split_energy_quadratic_params) -> None:
#     """
#     Test build_face_variable_vector for fit and full cases
#     """
#     # build_face_variable_vector()
#     vertex_positions: list[SpatialVector1d]
#     vertex_gradients: list[Matrix2x3r]
#     edge_gradients: list[list[SpatialVector1d]]  # list[SpatialVector1d] of length 3
#     global_vertex_indices: list[int]
#     global_edge_indices: list[list[int]]  # list[int] of length 3
#     initial_vertex_positions: list[SpatialVector]
#     initial_face_normals: np.ndarray  # matrix
#     affine_manifold: AffineManifold
#     optimization_params: OptimizationParameters
#     num_variable_vertices: int
#     num_variable_edges: int
#     (vertex_positions,
#      vertex_gradients,
#      edge_gradients,
#      global_vertex_indices,
#      global_edge_indices,
#      initial_vertex_positions,
#      initial_face_normals,
#      affine_manifold,
#      optimization_params,
#      num_variable_vertices,
#      num_variable_edges) = compute_twelve_split_energy_quadratic_params

#     for face_index in range(affine_manifold.num_faces):
#         # Get face vertices
#         F: MatrixXi = affine_manifold.faces
#         __i: int = F[face_index, 0]
#         __j: int = F[face_index, 1]
#         __k: int = F[face_index, 2]

#         # Bundle relevant global variables into per face local vectors (all list of length 3)
#         initial_vertex_positions_T: list[SpatialVector1d] = build_face_variable_vector(
#             initial_vertex_positions, __i, __j, __k)
