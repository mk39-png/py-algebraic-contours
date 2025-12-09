"""
Test optimize spline surface
"""

import copy
import pathlib
from collections import defaultdict

import numpy as np
import numpy.testing as npt
from cholespy import CholeskySolverD
from scipy.sparse import csr_matrix

from pyalgcon.core.affine_manifold import AffineManifold
from pyalgcon.core.common import (ROWS, Matrix2x3f, Matrix2x3r, MatrixNx3f,
                                  MatrixXf, SpatialVector, SpatialVector1d,
                                  Vector1D, compare_eigen_numpy_matrix,
                                  deserialize_eigen_matrix_csv_to_numpy,
                                  float_equal, index_vector_complement)
from pyalgcon.core.halfedge import Halfedge
from pyalgcon.quadratic_spline_surface.optimize_spline_surface import (
    OptimizationParameters, TriangleCornerData, TriangleMidpointData,
    build_twelve_split_spline_energy_system,
    generate_optimized_twelve_split_position_data,
    generate_zero_edge_gradients, generate_zero_vertex_gradients,
    optimize_twelve_split_spline_surface, shift_array)
from pyalgcon.quadratic_spline_surface.twelve_split_spline import \
    TwelveSplitSplineSurface

# def test_compute_twelve_split_energy_quadratic() -> None:
#     # TODO: make this a proper test case.
#     # For now, just lazily copied over my hasty testing.

#     manifold: AffineManifold = initialize_affine_manifold_from_spot_control()
#     num_independent_variables: int = 9 * num_variable_vertices + 3 * num_variable_edges
#     energy: float = 0.0
#     derivatives: Vector2D = np.zeros(shape=(num_independent_variables, 1), dtype=np.float64)
#     hessian_entries: list[tuple[int, int, float]] = []

#     # TODO: setup the tester to read from a file.
#     # Each element corresponds to face_index.
#     filepath: str = "spot_control\\optimize_spline_surface\\compute_twelve_split_energy_quadratic"

#     if (float_equal(optimization_params.parametrized_quadratic_surface_mapping_factor, 0.0)):
#         filepath += "\\fit_"
#         filepath_shifted += "\\fit_shifted"
#     else:
#         filepath += "\\full"
#         filepath_shifted += "\\full_shifted"

#     for face_index in range(manifold.num_faces):
#         # Get face vertices
#         F: MatrixXi = manifold.get_faces
#         __i: int = F[face_index, 0]
#         __j: int = F[face_index, 1]
#         __k: int = F[face_index, 2]

#         # Bundle relevant global variables into per face local vectors (all list of length 3)

#         # TODO: below is checked.
#         # TODO: this logic and if it could be checked


#         # Get the global uv values for the face vertices
#         # XXX: DO NOT MODIFY BELOW BY REFERNCE.
#         # CHANGES SHOULD BE LOCAL TO EACH ITERATION OF THE FOR LOOP
#         face_vertex_uv_positions: list[PlanarPoint] = copy.deepcopy(
#             manifold.get_face_global_uv(face_index))  # length 3
#         assert len(face_vertex_uv_positions) == 3

#         # Get corner uv positions for the given face corners.
#         # NOTE: These may differ from the edge difference vectors computed from the global
#         # uv by a rotation per vertex due to the local layouts performed at each vertex.
#         # Since vertex gradients are defined in terms of these local vertex charts, we must
#         # use these directions when computing edge direction gradients from the vertex uv
#         # gradients.
#         corner_to_corner_uv_positions: list[Matrix2x2r] = copy.deepcopy(
#             manifold.get_face_corner_charts(face_index))  # length 3
#         assert len(corner_to_corner_uv_positions) == 3

#         # Get edge orientations
#         # NOTE: The edge frame is oriented so that one basis vector points along the edge
#         # counterclockwise and the other points perpendicular into the interior of the
#         # triangle. If the given face is the bottom face in the edge chart, the sign of
#         # the midpoint gradient needs to be reversed.
#         reverse_edge_orientations: list[bool] = []  # length 3
#         for i in range(3):
#             chart: EdgeManifoldChart = manifold.get_edge_chart(face_index, i)
#             reverse_edge_orientations.append(chart.top_face_index != face_index)
#         assert len(reverse_edge_orientations) == 3

#         # Mark cone vertices
#         is_cone: list[bool] = []  # length 3
#         for i in range(3):
#             vi: int = F[face_index, i]
#             is_cone.append(manifold.get_vertex_chart(vi).is_cone)

#         # Mark cone adjacent vertices
#         is_cone_adjacent: list[bool] = []  # length 3
#         for i in range(3):
#             vi = F[face_index, i]
#             is_cone_adjacent.append(manifold.get_vertex_chart(vi).is_cone_adjacent)

#         # Get global indices of the local vertex and edge DOFs
#         face_global_vertex_indices: list[int] = build_face_variable_vector(
#             global_vertex_indices, __i, __j, __k)  # length 3
#         face_global_edge_indices: list[int] = global_edge_indices[face_index]  # length 3

#         # ---------------------------------
#         # ULTIMATE TESTER BELOW
#         # ---------------------------------
#         # TODO: below works for fit and non-fit cases. Move to separate testing case

#         # TESTING OF THESE AS WELL
#         compare_eigen_numpy_matrix(
#             f"{filepath}face_vertex_uv_positions\\_{face_index}.csv",
#             np.array(face_vertex_uv_positions).squeeze())
#         compare_eigen_numpy_matrix(
#             f"{filepath}corner_to_corner_uv_positions\\_{face_index}.csv",
#             np.array(corner_to_corner_uv_positions).squeeze(),
#             make_3d=True)
#         compare_eigen_numpy_matrix(
#             f"{filepath}reverse_edge_orientations\\_{face_index}.csv",
#             np.array(reverse_edge_orientations).squeeze())
#         compare_eigen_numpy_matrix(
#             f"{filepath}is_cone\\_{face_index}.csv",
#             np.array(is_cone).squeeze())
#         compare_eigen_numpy_matrix(
#             f"{filepath}is_cone_adjacent\\_{face_index}.csv",
#             np.array(is_cone_adjacent).squeeze())
#         compare_eigen_numpy_matrix(
#             f"{filepath}face_global_vertex_indices\\_{face_index}.csv",
#             np.array(face_global_vertex_indices).squeeze())
#         compare_eigen_numpy_matrix(
#             f"{filepath}face_global_edge_indices\\_{face_index}.csv",
#             np.array(face_global_edge_indices).squeeze())

#         control = deserialize_eigen_matrix_csv_to_numpy(
#             "spot_control\\optimize_spline_surface\\compute_twelve_split_energy_quadratic\\fit_face_vertex_uv_positions\\_{face_index}.csv")
#         # These 2 should be equal... I don't know why they are not equal....
#         # They both use the same manifold... right?
#         test = deserialize_eigen_matrix_csv_to_numpy(
#             f"spot_control\\optimize_spline_surface\\compute_twelve_split_energy_quadratic\\face_vertex_uv_positions\\_{face_index}.csv")
#         compare_eigen_numpy_matrix(f"spot_control\\optimize_spline_surface\\compute_twelve_split_energy_quadratic\\fit_face_vertex_uv_positions\\_{face_index}.csv",
#                                    test)

#         # Check if an edge is collapsing and make sure any collapsing edges have
#         # local vertex indices 0 and 1
#         # WARNING: This is a somewhat fragile operation that must occur after all
#         # of these arrays are build and before the local to global map is built
#         # and is not necessary in the current framework used in the paper but is for
#         # some deprecated experimental methods

#         is_cone_adjacent_face: bool = False
#         for i in range(3):
#             if is_cone[(i + 2) % 3]:
#                 shift_local_energy_quadratic_vertices(vertex_positions_T,
#                                                       vertex_gradients_T,
#                                                       edge_gradients_T,
#                                                       initial_vertex_positions_T,
#                                                       face_vertex_uv_positions,
#                                                       corner_to_corner_uv_positions,
#                                                       reverse_edge_orientations,
#                                                       is_cone,
#                                                       is_cone_adjacent,
#                                                       face_global_vertex_indices,
#                                                       face_global_edge_indices,
#                                                       i)
#                 is_cone_adjacent_face = True
#                 break

#         # TODO: below works for fit and non-fit cases. Move to separate testing case
#         compare_eigen_numpy_matrix(
#             f"{filepath_shifted}initial_vertex_positions_T\\_{face_index}.csv",
#             np.array(initial_vertex_positions_T).squeeze())
#         compare_eigen_numpy_matrix(
#             f"{filepath_shifted}vertex_positions_T\\_{face_index}.csv",
#             np.array(vertex_positions_T).squeeze())
#         compare_eigen_numpy_matrix(
#             f"{filepath_shifted}vertex_gradients_T\\_{face_index}.csv",
#             np.array(vertex_gradients_T),
#             make_3d=True)
#         compare_eigen_numpy_matrix(
#             f"{filepath_shifted}edge_gradients_T\\_{face_index}.csv",
#             np.array(edge_gradients_T).squeeze())
#         compare_eigen_numpy_matrix(
#             f"{filepath_shifted}face_vertex_uv_positions\\_{face_index}.csv",
#             np.array(face_vertex_uv_positions).squeeze())
#         compare_eigen_numpy_matrix(
#             f"{filepath_shifted}corner_to_corner_uv_positions\\_{face_index}.csv",
#             np.array(corner_to_corner_uv_positions).squeeze(),
#             make_3d=True)
#         compare_eigen_numpy_matrix(
#             f"{filepath_shifted}reverse_edge_orientations\\_{face_index}.csv",
#             np.array(reverse_edge_orientations).squeeze())
#         compare_eigen_numpy_matrix(
#             f"{filepath_shifted}is_cone\\_{face_index}.csv",
#             np.array(is_cone).squeeze())
#         compare_eigen_numpy_matrix(
#             f"{filepath_shifted}is_cone_adjacent\\_{face_index}.csv",
#             np.array(is_cone_adjacent).squeeze())
#         compare_eigen_numpy_matrix(
#             f"{filepath_shifted}face_global_vertex_indices\\_{face_index}.csv",
#             np.array(face_global_vertex_indices).squeeze())
#         compare_eigen_numpy_matrix(
#             f"{filepath_shifted}face_global_edge_indices\\_{face_index}.csv",
#             np.array(face_global_edge_indices).squeeze())

#         # Get normal for the face
#         normal: SpatialVector = np.zeros(shape=(1, 3))
#         if is_cone_adjacent_face:
#             normal = initial_face_normals[[face_index], :]
#             assert normal.shape == (1, 3)
#             logger.info("Weighting by normal %s", normal.T)

#         # Get local to global map
#         local_to_global_map: list[int] = generate_twelve_split_local_to_global_map(
#             face_global_vertex_indices,
#             face_global_edge_indices,
#             num_variable_vertices)  # length = 36
#         # TODO: is local_to_global_map always length 36?
#         assert len(local_to_global_map) == 36

#         # Compute local hessian data
#         local_hessian_data: LocalHessianData = LocalHessianData.generate_local_hessian_data(
#             face_vertex_uv_positions,
#             corner_to_corner_uv_positions,
#             reverse_edge_orientations,
#             is_cone,
#             is_cone_adjacent,
#             normal,
#             optimization_params)

#         # Compute local degree of freedom data
#         local_dof_data: LocalDOFData = LocalDOFData.generate_local_dof_data(
#             initial_vertex_positions_T,
#             vertex_positions_T,
#             vertex_gradients_T,
#             edge_gradients_T)

#         # Compute the local energy quadratic system for the face
#         local_energy: float
#         local_derivatives: TwelveSplitGradient
#         local_hessian: TwelveSplitHessian
#         # FIXME: local_derivatives adding to the global_derivative are WRONG!
#         local_energy, local_derivatives, local_hessian = compute_local_twelve_split_energy_quadratic(
#             local_hessian_data,
#             local_dof_data
#         )

#         # TODO: below works for fit and non-fit cases. Move to separate testing case
#         compare_eigen_numpy_matrix(f"{filepath}normal\\_{face_index}.csv", normal.squeeze())
#         compare_eigen_numpy_matrix(f"{filepath}local_to_global_map\\_{face_index}.csv",
#                                    np.array(local_to_global_map).squeeze())
#         compare_eigen_numpy_matrix(
#             f"{filepath}local_hessian_data_H_f\\_{face_index}.csv", local_hessian_data.H_f)
#         compare_eigen_numpy_matrix(
#             f"{filepath}local_hessian_data_H_s\\_{face_index}.csv", local_hessian_data.H_s)
#         compare_eigen_numpy_matrix(
#             f"{filepath}local_hessian_data_H_p\\_{face_index}.csv", local_hessian_data.H_p)
#         # assert local_hessian_data.w_f == 1  # magic numbers from ASOC code result
#         # assert local_hessian_data.w_s == 0  # magic numbers from ASOC code result
#         # assert local_hessian_data.w_p == 0  # magic numbers from ASOC code result
#         compare_eigen_numpy_matrix(
#             f"{filepath}local_dof_data_r_alpha_0\\_{face_index}.csv", local_dof_data.r_alpha_0)
#         compare_eigen_numpy_matrix(
#             f"{filepath}local_dof_data_r_alpha\\_{face_index}.csv", local_dof_data.r_alpha)
#         compare_eigen_numpy_matrix(
#             f"{filepath}local_dof_data_r_alpha_flat\\_{face_index}.csv", local_dof_data.r_alpha_flat.squeeze())
#         # assert float_equal(local_energy, 0.0)  # magic numbers from ASOC code result
#         compare_eigen_numpy_matrix(
#             f"{filepath}local_derivatives\\_{face_index}.csv", local_derivatives.squeeze())
#         compare_eigen_numpy_matrix(f"{filepath}local_hessian\\_{face_index}.csv", local_hessian)

#         # Update the energy quadratic with the new face energy
#         # NOTE: update_energy_quadratic is only used here.
#         # NOTE: derivatives and hessian_entries and are in the method below.
#         energy = update_energy_quadratic(local_energy,
#                                          local_derivatives,
#                                          local_hessian,
#                                          local_to_global_map,
#                                          energy,
#                                          derivatives,
#                                          hessian_entries)

#     # TODO: below works for fit and non-fit cases. Move to separate testing case
#     if (float_equal(optimization_params.parametrized_quadratic_surface_mapping_factor, 0.0)):
#         float_equal(energy, 0.0)
#     else:
#         float_equal(energy, 1269.98, eps=0.01)
#     compare_eigen_numpy_matrix(
#         f"{filepath}derivatives_energy_quadratic.csv", derivatives.squeeze())
#     compare_eigen_numpy_matrix(
#         f"{filepath}hessian_entries_energy_quadratic.csv", np.array(hessian_entries))


def test_initial_vertex_positions(testing_fileinfo,
                                  parsed_control_mesh) -> None:
    """
    Tests if vertex_positions and initial_vertex_positions are built properly 
    within build_twelve_split_spline_energy_system.
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / "quadratic_spline_surface" / \
        "optimize_spline_surface" / "build_twelve_split_spline_energy_system"
    V: np.ndarray
    V, _, _, _ = parsed_control_mesh
    num_vertices: int = V.shape[0]  # number of rows as well

    # Initialize variables to optimize
    vertex_positions: list[SpatialVector] = []
    initial_vertex_positions: list[SpatialVector] = []
    for i in range(num_vertices):
        assert V[i, :].shape == (3, )
        vertex_positions.append(V[i, :])  # shape (1, 3) for SpatialVectors
        initial_vertex_positions.append(V[i, :])

    # Compare results
    # NOTE: below should be the same for fit and full cases
    compare_eigen_numpy_matrix(
        filepath / "fit" / "vertex_positions.csv",
        np.array(vertex_positions))
    compare_eigen_numpy_matrix(
        filepath / "fit" / "initial_vertex_positions.csv",
        np.array(initial_vertex_positions))
    compare_eigen_numpy_matrix(
        filepath / "full" / "vertex_positions.csv",
        np.array(vertex_positions))
    compare_eigen_numpy_matrix(
        filepath / "full" / "initial_vertex_positions.csv",
        np.array(initial_vertex_positions))


def test_optimize_twelve_split_spline_surface(testing_fileinfo,
                                              parsed_control_mesh,
                                              initialize_affine_manifold) -> None:
    """
    Testing the following:
    optimized_V,
    optimized_vertex_gradients,
    optimized_reduced_edge_gradients
    """

    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / "quadratic_spline_surface" / \
        "optimize_spline_surface" / "generate_optimized_twelve_split_position_data"
    V: np.ndarray
    V, _, _, _ = parsed_control_mesh
    num_vertices: int = V.shape[ROWS]
    affine_manifold: AffineManifold = initialize_affine_manifold

    # Build halfedge
    he_to_corner: list[tuple[int, int]] = affine_manifold.he_to_corner
    halfedge: Halfedge = affine_manifold.halfedge
    num_edges: int = halfedge.num_edges

    # Assume all vertices and edges are variable
    # TODO: index_vector_complement does not need list parameter.
    fixed_vertices: list[int] = []
    fixed_edges: list[int] = []
    variable_vertices: list[int] = index_vector_complement(fixed_vertices, num_vertices)
    variable_edges: list[int] = index_vector_complement(fixed_edges, num_edges)

    # Generate normals
    # filepath = "spot_control\\12_split_spline\\generate_face_normals\\N.csv"
    # N: MatrixXf = deserialize_eigen_matrix_csv_to_numpy(filepath)
    N: MatrixXf = TwelveSplitSplineSurface.generate_face_normals(V, affine_manifold)

    # Generate fit matrix by setting the parametrized quadratic surface mapping factor to zero
    fit_matrix: csr_matrix
    # Make a deep since we don't want the same parameters between Fit vs Non-fit 12-split-splines
    optimization_params_fit: OptimizationParameters = OptimizationParameters(
        parametrized_quadratic_surface_mapping_factor=0.0)
    _, _, fit_matrix, _ = build_twelve_split_spline_energy_system(
        V,
        N,
        affine_manifold,
        optimization_params_fit)

    # Build full energy hessian system
    energy_hessian_inverse: CholeskySolverD
    optimization_params_full: OptimizationParameters = OptimizationParameters(
        parametrized_quadratic_surface_mapping_factor=1.0)
    _, _, _, energy_hessian_inverse = build_twelve_split_spline_energy_system(
        V, N,
        affine_manifold,
        optimization_params_full)

    # Execute primary function for testing.
    optimized_V: MatrixNx3f
    optimized_vertex_gradients: list[Matrix2x3r]
    optimized_reduced_edge_gradients: list[list[SpatialVector1d]]
    optimized_V, optimized_vertex_gradients, optimized_reduced_edge_gradients = (
        optimize_twelve_split_spline_surface(
            V,
            affine_manifold,
            halfedge,
            he_to_corner,
            variable_vertices,
            variable_edges,
            fit_matrix,
            energy_hessian_inverse))

    # Below line of code originally for comparing internally within the function.
    # compare_eigen_numpy_matrix(filepath+"right_hand_side.csv", right_hand_side)
    compare_eigen_numpy_matrix(
        filepath / "optimized_V.csv",
        optimized_V)
    compare_eigen_numpy_matrix(
        filepath / "optimized_vertex_gradients.csv",
        np.array(optimized_vertex_gradients),
        make_3d=True)
    compare_eigen_numpy_matrix(
        filepath / "optimized_reduced_edge_gradients.csv",
        np.array(optimized_reduced_edge_gradients),
        make_3d=True)


# def test_convert_reduced_edge_gradients_to_full() -> None:
#     """
#     convert_reduced_edge_gradients_to_full() used inside
#     generate_optimized_twelve_split_position_data()
#     """
#     # Build the full edge gradients with first gradient determined by the corner position data
#     optimized_edge_gradients: dict[int, dict[int, Matrix2x3r]] = convert_reduced_edge_gradients_to_full(
#         optimized_reduced_edge_gradients,
#         corner_data,
#         affine_manifold)  # list[Matrix2x3r] of length 3
#     assert len(optimized_edge_gradients[0]) == 3

#     # TODO: move test to separate case. passes.
#     compare_eigen_numpy_matrix(
#         filepath+"optimized_edge_gradients.csv",
#         np.array(optimized_edge_gradients).reshape(-1, 3))


def test_generate_optimized_twelve_split_position_data(testing_fileinfo,
                                                       parsed_control_mesh) -> None:
    """
    Testing generate_optimized_twelve_split_position_data() from the TwelveSplitSplineSurface()
    constructor on the spot_control mesh.

    NOTE: this method has the following dependencies to work properly:
    * build_twelve_split_spline_energy_system() for fit case
    * build_twelve_split_spline_energy_system() for full case
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath = base_data_folderpath / "quadratic_spline_surface" / "12_split_spline"

    # Retrieve mesh
    V: np.ndarray
    uv: np.ndarray
    F: np.ndarray
    FT: np.ndarray
    V, uv, F, FT = parsed_control_mesh
    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)
    optimization_params: OptimizationParameters = OptimizationParameters()
    corner_data: dict[int, dict[int, TriangleCornerData]] = defaultdict(dict)
    midpoint_data: dict[int, dict[int, TriangleMidpointData]] = defaultdict(dict)

    # Generate normals
    # filepath = "spot_control\\12_split_spline\\generate_face_normals\\N.csv"
    N: MatrixXf = deserialize_eigen_matrix_csv_to_numpy(
        filepath / "generate_face_normals" / "N.csv")

    # Generate fit matrix by setting the parametrized quadratic surface mapping factor to zero
    fit_matrix: csr_matrix
    # Make a deep since we don't want the same parameters between Fit vs Non-fit 12-split-splines
    optimization_params_fit: OptimizationParameters = copy.deepcopy(optimization_params)
    optimization_params_fit.parametrized_quadratic_surface_mapping_factor = 0.0
    _, _, fit_matrix, _ = build_twelve_split_spline_energy_system(
        V, N,
        affine_manifold,
        optimization_params_fit)

    # Build full energy hessian system
    energy_hessian_inverse: CholeskySolverD
    _, _, _, energy_hessian_inverse = build_twelve_split_spline_energy_system(
        V, N,
        affine_manifold,
        optimization_params)

    # Build optimized corner and midpoint data.
    # As in, initializes self.__corner_data and self.__midpoint_data
    # XXX: Below changes __corner_data and __midpoint_data by reference.
    # NOTE: testing the method below.
    generate_optimized_twelve_split_position_data(V,
                                                  affine_manifold,
                                                  fit_matrix,
                                                  energy_hessian_inverse,
                                                  corner_data,
                                                  midpoint_data)

    # Comparing results
    corner_data_test: list[np.ndarray] = []
    for outer_key in sorted(corner_data):
        for inner_key in sorted(corner_data[outer_key]):
            triangle_corner_data: TriangleCornerData = corner_data[outer_key][inner_key]
            corner_data_test.append(triangle_corner_data.function_value)
            corner_data_test.append(triangle_corner_data.first_edge_derivative)
            corner_data_test.append(triangle_corner_data.second_edge_derivative)
    compare_eigen_numpy_matrix(filepath / "corner_data.csv",
                               np.array(corner_data_test))

    midpoint_data_test: list[np.ndarray] = []
    for outer_key in sorted(midpoint_data):
        for inner_key in sorted(midpoint_data[outer_key]):
            triangle_midpoint_data: TriangleMidpointData = midpoint_data[outer_key][inner_key]
            midpoint_data_test.append(triangle_midpoint_data.normal_derivative)
    compare_eigen_numpy_matrix(filepath / "midpoint_data.csv",
                               np.array(midpoint_data_test))


def test_zero_vertex_gradients(testing_fileinfo, parsed_control_mesh) -> None:
    """
    Tests generate_zero_vertex_gradients() from spot_mesh.
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / "quadratic_spline_surface" / \
        "optimize_spline_surface" / "build_twelve_split_spline_energy_system"
    V: np.ndarray
    V, _, _, _ = parsed_control_mesh
    num_vertices: int = V.shape[0]  # rows

    # Execute method
    vertex_gradients: list[Matrix2x3f] = generate_zero_vertex_gradients(num_vertices)

    # Compare results
    # NOTE: both fit and full cases shouuld be the same.
    compare_eigen_numpy_matrix(
        filepath / "fit" / "vertex_gradients.csv",
        np.array(vertex_gradients),
        make_3d=True)
    compare_eigen_numpy_matrix(
        filepath / "full" / "vertex_gradients.csv",
        np.array(vertex_gradients),
        make_3d=True)


def test_zero_edge_gradients(testing_fileinfo,
                             initialize_affine_manifold) -> None:
    """
    Tests generate_zero_edge_gradients() from spot_mesh.
    Used in TwelveSplitSplineSurface generation.
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / "quadratic_spline_surface" / \
        "optimize_spline_surface" / "build_twelve_split_spline_energy_system"
    affine_manifold: AffineManifold = initialize_affine_manifold
    num_faces: int = affine_manifold.num_faces

    # Execute method
    edge_gradients: list[list[SpatialVector1d]] = generate_zero_edge_gradients(num_faces)

    # Compare results
    # NOTE: both fit and full cases should be the same.
    compare_eigen_numpy_matrix(
        filepath / "fit" / "edge_gradients.csv",
        np.array(edge_gradients),
        make_3d=True)
    compare_eigen_numpy_matrix(
        filepath / "full" / "edge_gradients.csv",
        np.array(edge_gradients),
        make_3d=True)


def test_shift_array() -> None:
    """
    Tests shift_array() method. with simple and complex objects.
    """
    # NOTE: I checked this behavior with the ASOC code.
    # In list[np.ndarray] = [0, 1, 2]
    # change such that it shifts to [2, 0, 1]
    # or like [10, 20, 30] becomes [30, 10, 20]
    # Basically, moving elements to the left by "shift" amount.

    # Simple array
    int_list: list[int] = [0, 1, 2]

    shift_array(int_list, 2)
    assert int_list[0] == 2
    assert int_list[1] == 0
    assert int_list[2] == 1

    # list of NumPy array
    numpy_list: list[np.ndarray[tuple[int, int], np.dtype[np.int64]]] = [
        np.full(shape=(2, 2), fill_value=0, dtype=np.int64),
        np.full(shape=(2, 2), fill_value=1, dtype=np.int64),
        np.full(shape=(2, 2), fill_value=2, dtype=np.int64)]

    shift_array(numpy_list, 1)
    npt.assert_array_equal(numpy_list[0], np.full(shape=(2, 2), fill_value=1))
    npt.assert_array_equal(numpy_list[1], np.full(shape=(2, 2), fill_value=2))
    npt.assert_array_equal(numpy_list[2], np.full(shape=(2, 2), fill_value=0))


def test_build_twelve_split_spline_energy_system_full(testing_fileinfo,
                                                      parsed_control_mesh) -> None:
    """
    This tests build_twelve_split_spline_energy(). 
    Used in TwelveSplitSplineSurface generation.

    NOTE: build_twelve_split_spline_energy_system() has the following dependencies:
    * AffineManifold.he_to_corner.
    * generate_face_normals()
    * index_vector_complement
    * generate_zero_vertex_gradients
    * generate_zero_edge_gradients
    * build_variable_vertex_indices_map
    * build_variable_edge_indices_map
    * compute_twelve_split_energy_quadratic
    NOTE: most of the above methods will result in the same values for fit 
    and full cases of the TwelveSplitSplineSurface constructor.
    But, compute_twelve_split_energy_quadratic will result in different values.

    Tests entirity of the build_twelve_split_spline_energy_system function.
    Retrieving proper he_to_corner from affine_manifold for spot_control mesh.
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / "quadratic_spline_surface" / "12_split_spline"

    # Get input mesh
    V, uv, F, FT = parsed_control_mesh
    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)
    optimization_params_fit = OptimizationParameters(
        parametrized_quadratic_surface_mapping_factor=0.0)
    optimization_params_full = OptimizationParameters(
        parametrized_quadratic_surface_mapping_factor=1.0)
    N: MatrixXf = TwelveSplitSplineSurface.generate_face_normals(V, affine_manifold)

    # ** Fit Energy Case **
    fit_energy: float
    fit_derivatives: Vector1D
    fit_matrix: csr_matrix
    fit_matrix_inverse: CholeskySolverD
    fit_energy, fit_derivatives, fit_matrix, fit_matrix_inverse = (
        build_twelve_split_spline_energy_system(
            V, N, affine_manifold, optimization_params_fit))

    # NOTE: magic number from ASOC code output for fit_energy
    assert float_equal(fit_energy, 0.0)
    compare_eigen_numpy_matrix(
        filepath / "fit_derivatives.csv", fit_derivatives)
    compare_eigen_numpy_matrix(
        filepath / "fit_matrix_dense.csv", fit_matrix.todense())

    # ** Full Energy Case **
    energy: float
    derivatives: Vector1D
    energy_hessian: csr_matrix
    energy_hessian_inverse: CholeskySolverD
    energy, derivatives, energy_hessian, energy_hessian_inverse = (
        build_twelve_split_spline_energy_system(V,
                                                N,
                                                affine_manifold,
                                                optimization_params_full))

    # NOTE: can't really test energy_hessian_inverse directly since it's not translatable
    # from the original C++ code
    # NOTE: magic number from ASOC code's output for energy
    assert float_equal(energy, 1269.9805595159069)
    compare_eigen_numpy_matrix(
        filepath / "derivatives.csv", derivatives)
    compare_eigen_numpy_matrix(
        filepath / "energy_hessian_dense.csv", energy_hessian.todense())
