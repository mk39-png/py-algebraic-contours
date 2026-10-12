"""
Test powell sabin local to global indexing
"""

import pathlib

import numpy as np
import numpy.testing as npt

from pyalgcon.core.affine_manifold import AffineManifold
from pyalgcon.core.common import (ROWS, Index, Matrix2x3r, SpatialVector,
                                  SpatialVector1d, TwelveSplitGradient,
                                  TwelveSplitHessian, Vector1D, Vector36f,
                                  Vector36i, compare_eigen_numpy_matrix,
                                  index_vector_complement)
from pyalgcon.core.halfedge import Halfedge
from pyalgcon.quadratic_spline_surface.optimize_spline_surface import (
    generate_zero_edge_gradients, generate_zero_vertex_gradients)
from pyalgcon.quadratic_spline_surface.powell_sabin_local_to_global_indexing import (
    build_variable_edge_indices_map, build_variable_vertex_indices_map,
    generate_twelve_split_variable_value_vector, update_energy_quadratic)


def test_generate_twelve_split_variable_value_vector(testing_fileinfo,
                                                     parsed_control_mesh,
                                                     initialize_affine_manifold) -> None:
    """
    Tests generate_twelve_split_variable_value_vector from spot_control
    to make initial_variable_values inside optimize_twelve_split_spline_surface()
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / "quadratic_spline_surface" / \
        "optimize_spline_surface" / "optimize_twelve_split_spline_surface"

    # Retrieve mesh data
    initial_V: np.ndarray
    initial_V, _, _, _ = parsed_control_mesh
    affine_manifold: AffineManifold = initialize_affine_manifold
    num_vertices: int = initial_V.shape[ROWS]
    num_faces: int = affine_manifold.num_faces

    # Build halfedge
    he_to_corner: list[tuple[int, int]] = affine_manifold.he_to_corner
    halfedge: Halfedge = affine_manifold.halfedge
    num_edges: int = halfedge.num_edges

    # Initialize variables to optimize
    vertex_positions: list[SpatialVector1d] = []
    initial_vertex_positions: list[SpatialVector1d] = []
    for i in range(num_vertices):
        assert initial_V[i, :].shape == (3, )
        vertex_positions.append(initial_V[i, :])
        initial_vertex_positions.append(initial_V[i, :])
    assert len(vertex_positions) == num_vertices
    assert len(initial_vertex_positions) == num_vertices
    vertex_gradients: list[Matrix2x3r] = generate_zero_vertex_gradients(num_vertices)
    edge_gradients: list[list[SpatialVector]] = generate_zero_edge_gradients(num_faces)
    assert len(edge_gradients[0]) == 3

    # Assume all vertices and edges are variable
    # TODO: index_vector_complement does not need list parameter.
    fixed_vertices: list[int] = []
    fixed_edges: list[int] = []
    variable_vertices: list[int] = index_vector_complement(fixed_vertices, num_vertices)
    variable_edges: list[int] = index_vector_complement(fixed_edges, num_edges)

    # Execute method
    initial_variable_values: Vector1D = generate_twelve_split_variable_value_vector(
        vertex_positions,
        vertex_gradients,
        edge_gradients,
        variable_vertices,
        variable_edges,
        halfedge,
        he_to_corner)

    # Compare results
    compare_eigen_numpy_matrix(filepath / "initial_variable_values.csv", initial_variable_values)


def test_build_variable_vertex_indices_map(testing_fileinfo,
                                           parsed_control_mesh) -> None:
    """
    Test build_variable_vertex_indices_map()
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / "quadratic_spline_surface" / \
        "optimize_spline_surface" / "build_twelve_split_spline_energy_system"
    initial_V: np.ndarray
    initial_V, _, _, _ = parsed_control_mesh
    num_vertices: int = initial_V.shape[ROWS]

    # Assume all vertices and edges are variable
    fixed_vertices: list[int] = []
    variable_vertices: list[int] = index_vector_complement(fixed_vertices, num_vertices)

    # Execute method
    global_vertex_indices: list[int] = build_variable_vertex_indices_map(
        num_vertices, variable_vertices)

    # Compare results
    compare_eigen_numpy_matrix(filepath / "fit" / "global_vertex_indices.csv",
                               np.array(global_vertex_indices))
    compare_eigen_numpy_matrix(filepath / "full" / "global_vertex_indices.csv",
                               np.array(global_vertex_indices))


def test_build_variable_edge_indices_map(testing_fileinfo,
                                         initialize_affine_manifold) -> None:
    """
    Test build_variable_edge_indices_map()
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / "quadratic_spline_surface" / \
        "optimize_spline_surface" / "build_twelve_split_spline_energy_system"
    affine_manifold: AffineManifold = initialize_affine_manifold
    num_faces: int = affine_manifold.num_faces

    # Build halfedge
    he_to_corner: list[tuple[Index, Index]] = affine_manifold.he_to_corner
    halfedge: Halfedge = affine_manifold.halfedge
    num_edges: int = halfedge.num_edges

    # Assume all vertices and edges are variable
    fixed_edges: list[int] = []
    variable_edges: list[int] = index_vector_complement(fixed_edges, num_edges)

    # Build edge variable indices
    global_edge_indices: list[list[int]] = build_variable_edge_indices_map(
        num_faces, variable_edges, halfedge, he_to_corner)

    # Compare results
    compare_eigen_numpy_matrix(filepath / "fit" / "global_edge_indices.csv",
                               np.array(global_edge_indices))
    compare_eigen_numpy_matrix(filepath / "full" / "global_edge_indices.csv",
                               np.array(global_edge_indices))


def test_update_energy_quadratic():

    def update_energy_quadratic_control(local_energy: float,
                                        local_derivatives: TwelveSplitGradient,  # shape (36,)
                                        local_hessian: TwelveSplitHessian,  # shape (36, 36)
                                        local_to_global_map: Vector36i,
                                        energy: float,
                                        derivatives_ref: Vector1D,
                                        # hessian_entries_ref: HessianEntries
                                        hessian_entries_ref: list[tuple[int, int, float]]
                                        ) -> float:
        """
        Update global energy, derivatives, and hessian with local per face values

        :param local_energy:        [in] local energy value
        :param local_derivatives:   [in] local energy gradient
        :param local_hessian:       [in] local energy Hessian
        :param local_to_global_map: [in] map from local to global DOF indices

        :param  derivatives_ref:    [out] global energy gradient
        :param global_indices_i:    [out] global energy Hessian index i
        :param global_indices_j:    [out] global energy Hessian index j
        :param  hessian_values:     [out] global energy Hessian

        :return energy:           global energy value
        """
        assert derivatives_ref.ndim == 1  # shape (36, )
        assert local_derivatives.ndim == 1
        assert local_hessian.ndim == 2
        assert local_to_global_map.shape == (36, )

        # Update energy
        energy += local_energy

        # TODO: get valid local and globally indices to vectorize the below operation
        # Update derivatives
        num_local_indices: int = len(local_to_global_map)
        for local_index in range(num_local_indices):
            global_index: int = local_to_global_map[local_index]
            if global_index < 0:
                continue  # Skip fixed variables with no global index
            derivatives_ref[global_index] += local_derivatives[local_index]

        # Update hessian entries
        for local_index_i in range(num_local_indices):
            # Get global row index, skipping fixed variables with no global index
            global_index_i: int = local_to_global_map[local_index_i]
            if global_index_i < 0:
                continue

            for local_index_j in range(num_local_indices):
                # Get global column index, skipping fixed variables with no global index
                global_index_j: int = local_to_global_map[local_index_j]
                if global_index_j < 0:
                    continue

                # Get Hessian entry value
                hessian_value: float = local_hessian[local_index_i, local_index_j]

                # Assemble global Hessian entry
                hessian_entries_ref.append((global_index_i, global_index_j, hessian_value))

        return energy

    trials = 1000
    magnitude = 100

    derivatives_control: Vector1D = np.random.rand(36)
    hessian_entries_control: list[tuple[int, int, float]] = []
    derivatives_test = derivatives_control.copy()
    hessian_entries_test: list[tuple[int, int, float]] = []
    local_to_global_map = np.arange(-1, 35)
    # local_to_global_map = np.random.randint(-1, 36, size=36)
    local_derivatives: TwelveSplitGradient = magnitude * \
        np.random.rand(36)
    local_hessian: TwelveSplitHessian = magnitude * \
        np.random.rand(36, 36)
    energy: float = 0.0

    for i in range(trials):
        np.random.shuffle(local_to_global_map)

        update_energy_quadratic_control(0,
                                        local_derivatives,
                                        local_hessian,
                                        local_to_global_map,
                                        0,
                                        derivatives_control,
                                        hessian_entries_control)

        update_energy_quadratic(local_derivatives,
                                local_hessian,
                                local_to_global_map,
                                derivatives_test,
                                hessian_entries_test)

    # Checking original output versus Claude-optimized version
    npt.assert_allclose(derivatives_test, derivatives_control)
    npt.assert_allclose(np.array(hessian_entries_test),
                        np.array(hessian_entries_control))
