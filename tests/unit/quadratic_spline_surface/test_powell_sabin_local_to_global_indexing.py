"""
Test powell sabin local to global indexing
"""

import pathlib

import numpy as np

from pyalgcon.core.affine_manifold import AffineManifold
from pyalgcon.core.common import (ROWS, Index, Matrix2x3r, SpatialVector,
                                  SpatialVector1d, Vector1D,
                                  compare_eigen_numpy_matrix,
                                  index_vector_complement)
from pyalgcon.core.halfedge import Halfedge
from pyalgcon.quadratic_spline_surface.optimize_spline_surface import (
    generate_zero_edge_gradients, generate_zero_vertex_gradients)
from pyalgcon.quadratic_spline_surface.powell_sabin_local_to_global_indexing import (
    build_variable_edge_indices_map, build_variable_vertex_indices_map,
    generate_twelve_split_variable_value_vector)


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
