"""
Test 12 split spline
"""

import copy
import logging
import pathlib
from collections import defaultdict

import numpy as np
import numpy.testing as npt
from cholespy import CholeskySolverD
from scipy.sparse import csr_matrix

from pyalgcon.core.affine_manifold import (AffineManifold,
                                           ParametricAffineManifold)
from pyalgcon.core.bivariate_quadratic_function import \
    evaluate_quadratic_mapping
from pyalgcon.core.common import (DISCRETIZATION_LEVEL, SKY_BLUE, Matrix3x2f,
                                  Matrix6x3r, MatrixNx3f, MatrixXf, MatrixXi,
                                  PlanarPoint1d, SpatialVector,
                                  SpatialVector1d, compare_eigen_numpy_matrix,
                                  vector_equal)
from pyalgcon.core.convex_polygon import ConvexPolygon
from pyalgcon.quadratic_spline_surface.optimize_spline_surface import (
    OptimizationParameters, build_twelve_split_spline_energy_system,
    generate_optimized_twelve_split_position_data)
from pyalgcon.quadratic_spline_surface.quadratic_spline_surface import \
    QuadraticSplineSurface
from pyalgcon.quadratic_spline_surface.quadratic_spline_surface_patch import \
    QuadraticSplineSurfacePatch
from pyalgcon.quadratic_spline_surface.twelve_split_spline import (
    TriangleCornerData, TriangleMidpointData, TwelveSplitSplineSurface,
    generate_twelve_split_spline_patch_patch_boundaries,
    generate_twelve_split_spline_patch_patch_to_corner_map,
    generate_twelve_split_spline_patch_surface_mapping)
from pyalgcon.utils.generate_position_data import (
    QuadraticGradientFunction, QuadraticPositionFunction,
    generate_parametric_affine_manifold_corner_data,
    generate_parametric_affine_manifold_midpoint_data)

logger: logging.Logger = logging.getLogger(__name__)

# ****************
# Test Methods
# ****************


def test_position_data(testing_fileinfo, parsed_control_mesh) -> None:
    """
    Tests corner_data and midpoint_data of TwelveSplitSpline.

    Dependencies:
    * build_twelve_split_spline_energy_system
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / "quadratic_spline_surface" / "12_split_spline"
    V_raw: np.ndarray
    uv: np.ndarray
    F: np.ndarray
    FT: np.ndarray
    V_raw, uv, F, FT = parsed_control_mesh
    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)
    optimization_params: OptimizationParameters = OptimizationParameters()
    spline_surface: TwelveSplitSplineSurface = TwelveSplitSplineSurface(V_raw,
                                                                        affine_manifold,
                                                                        optimization_params)

    # NOTE: corner_data.csv is set up such that it represents the below:
    # (either that or... just combine it all smoothly...)
    # CORNER 1
    # 0, 1, 2 function_value
    # 0, 1, 2 first_edge_derivative
    # 0, 1, 2 second_edge_derivative
    # CORNER 2
    # 0, 1, 2 function_value
    # 0, 1, 2 first_edge_derivative
    # 0, 1, 2 second_edge_derivative
    # CORNER 3
    # 0, 1, 2 function_value
    # 0, 1, 2 first_edge_derivative
    # 0, 1, 2 second_edge_derivative
    # ...
    # Nevermind, just bash it all together into 1 big chunk..

    corner_data_ref: dict[int, dict[int, TriangleCornerData]] = spline_surface.corner_data
    midpoint_data_ref: dict[int, dict[int, TriangleMidpointData]] = spline_surface.midpoint_data

    corner_data_test: list[np.ndarray] = []
    for outer_key in sorted(corner_data_ref):
        for inner_key in sorted(corner_data_ref[outer_key]):
            triangle_corner_data: TriangleCornerData = corner_data_ref[outer_key][inner_key]
            corner_data_test.append(triangle_corner_data.function_value.flatten())
            corner_data_test.append(triangle_corner_data.first_edge_derivative.flatten())
            corner_data_test.append(triangle_corner_data.second_edge_derivative.flatten())
    compare_eigen_numpy_matrix(filepath / "corner_data.csv",
                               np.array(corner_data_test))

    midpoint_data_test: list[np.ndarray] = []
    for outer_key in sorted(midpoint_data_ref):
        for inner_key in sorted(midpoint_data_ref[outer_key]):
            triangle_midpoint_data: TriangleMidpointData = midpoint_data_ref[outer_key][inner_key]
            midpoint_data_test.append(triangle_midpoint_data.normal_derivative.flatten())
    compare_eigen_numpy_matrix(filepath / "midpoint_data.csv",
                               np.array(midpoint_data_test))


def test_face_patch_indices(testing_fileinfo,
                            parsed_control_mesh) -> None:
    """
    Tests init_twelve_split_patches using spot_control mesh.
    NOTE: relies on generate_twelve_split_spline_patch_patch_boundaries,
    ConvexPolygon.init_from_boundary_segments_coeffs(patch_boundaries[i]),
    generate_twelve_split_spline_patch_patch_to_corner_map, and
    generate_twelve_split_spline_patch_surface_mapping
    on working.

    Relies on AffineManifold compute_cone_corners() on working.
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    V_raw: np.ndarray
    uv: np.ndarray
    F: np.ndarray
    FT: np.ndarray
    V_raw, uv, F, FT = parsed_control_mesh

    # Get input mesh
    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)
    optimization_params: OptimizationParameters = OptimizationParameters()

    optimization_params: OptimizationParameters = OptimizationParameters()
    corner_data: dict[int, dict[int, TriangleCornerData]] = defaultdict(dict)
    midpoint_data: dict[int, dict[int, TriangleMidpointData]] = defaultdict(dict)
    face_to_patch_indices: list[list[int]]
    patch_to_face_indices: list[int]
    patches: list[QuadraticSplineSurfacePatch]

    # Generate normals
    N: MatrixNx3f = TwelveSplitSplineSurface.generate_face_normals(V_raw, affine_manifold)

    # Generate fit matrix by setting the parametrized quadratic surface mapping factor to zero
    fit_matrix: csr_matrix
    # Make a deep since we don't want the same parameters between Fit vs Non-fit 12-split-splines
    optimization_params_fit: OptimizationParameters = copy.deepcopy(optimization_params)
    optimization_params_fit.parametrized_quadratic_surface_mapping_factor = 0.0
    _, _, fit_matrix, _ = build_twelve_split_spline_energy_system(
        V_raw,
        N,
        affine_manifold,
        optimization_params_fit)

    # Build full energy hessian system
    energy_hessian_inverse: CholeskySolverD
    _, _, _, energy_hessian_inverse = (
        build_twelve_split_spline_energy_system(V_raw,
                                                N,
                                                affine_manifold,
                                                optimization_params))

    # Build optimized corner and midpoint data.
    # As in, initializes self.__corner_data and self.__midpoint_data
    # XXX: Below changes __corner_data and __midpoint_data by reference.
    generate_optimized_twelve_split_position_data(V_raw,
                                                  affine_manifold,
                                                  fit_matrix,
                                                  energy_hessian_inverse,
                                                  corner_data,
                                                  midpoint_data)

    # Get cone corners
    is_cone_corner: list[list[bool]] = affine_manifold.compute_cone_corners()
    split_spline = TwelveSplitSplineSurface
    # Initialize position data and patches.
    face_to_patch_indices, patch_to_face_indices, patches = split_spline.build_twelve_split_patches(
        corner_data,
        midpoint_data,
        is_cone_corner)

    folder_path: pathlib.Path = base_data_folderpath / \
        "quadratic_spline_surface" / "12_split_spline" / "init_twelve_split_patches"
    compare_eigen_numpy_matrix(folder_path / "face_to_patch_indices.csv",
                               np.array(face_to_patch_indices))
    compare_eigen_numpy_matrix(folder_path / "patch_to_face_indices.csv",
                               np.array(patch_to_face_indices))


def test_patch_to_corner_map(testing_fileinfo) -> None:
    """
    Tests generate_twelve_split_spline_patch_patch_to_corner_map().
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / "quadratic_spline_surface" / \
        "12_split_spline" / "init_twelve_split_patches"

    # Execute method
    patch_to_corner_map: list[tuple[int, int]]  # list of length 12
    patch_to_corner_map = generate_twelve_split_spline_patch_patch_to_corner_map()

    # Compare results
    assert len(patch_to_corner_map) == 12
    compare_eigen_numpy_matrix(filepath / "patch_to_corner_map.csv",
                               np.array(patch_to_corner_map))


def test_patch_boundaries(testing_fileinfo) -> None:
    """
    Tests generate_twelve_split_spline_patch_patch_boundaries().
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / "quadratic_spline_surface" / \
        "12_split_spline" / "init_twelve_split_patches"

    # Execute method
    patch_boundaries: list[list[np.ndarray]] = (
        generate_twelve_split_spline_patch_patch_boundaries())

    # Compare results
    assert len(patch_boundaries) == 12
    assert len(patch_boundaries[0]) == 3
    assert patch_boundaries[0][0].shape == (3, )  # lazy shape checking
    compare_eigen_numpy_matrix(filepath / "patch_boundaries.csv",
                               np.array(patch_boundaries).squeeze(),
                               make_3d=True)


def test_generate_face_normals(testing_fileinfo,
                               parsed_control_mesh,
                               initialize_affine_manifold) -> None:
    """
    This tests generate_face_normals
    """

    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / "quadratic_spline_surface" / \
        "12_split_spline" / "generate_face_normals"
    V_raw: np.ndarray
    F: np.ndarray
    V_raw, _, F, _ = parsed_control_mesh
    affine_manifold: AffineManifold = initialize_affine_manifold

    # Execute method
    N_test: MatrixXf = TwelveSplitSplineSurface.generate_face_normals(V_raw, affine_manifold)

    # Compare results
    compare_eigen_numpy_matrix(filepath / "V.csv", V_raw)
    compare_eigen_numpy_matrix(filepath / "F.csv", F)
    # XXX: N was fixed since there was a typo with angle_from_positions() in common.py
    compare_eigen_numpy_matrix(filepath / "N.csv", N_test)


def test_view_mesh(twelve_split_spline_raw) -> None:
    """
    This is used to test and view the spot model.
    """
    color: tuple[float, float, float] = SKY_BLUE
    num_subdivisions: int = DISCRETIZATION_LEVEL

    # Generate quadratic spline
    # NOTE: must input a mesh that is already UV unwrapped.
    spline_surface: TwelveSplitSplineSurface = twelve_split_spline_raw

    # View the mesh
    spline_surface.view(color, num_subdivisions)


def test_patches(testing_fileinfo,
                 parsed_control_mesh,
                 quadratic_spline_surface_control_from_file) -> None:
    """
    Testing to see if parent class QuadraticSplineSurface is utilized properly by TwelveSplitSpline.
    As in, the patches in TwelveSplitSpline subclass are the same as QuadraticSplineSurface
    parent class.

    As in, we can deserialize and reserialize
    FIXME: make this modular and usable for any .obj
    """
    logger.setLevel(logging.DEBUG)
    optimization_params: OptimizationParameters = OptimizationParameters()

    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo

    # Get input mesh
    V: MatrixXf
    uv: MatrixXf
    F: MatrixXi
    FT: MatrixXi
    V, uv, F, FT = parsed_control_mesh

    # Generate quadratic spline
    logger.info("Computing spline surface")
    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)
    spline_surface: TwelveSplitSplineSurface = TwelveSplitSplineSurface(
        V, affine_manifold,  optimization_params)

    # TODO: Making sure that the 12 split spline patches are the same as quadratic patch...
    # First open files to convert into list[QuadraticSplineSurfacePatch]
    filename_control: str = "spot_control_mesh-cleaned_conf_simplified_with_uv_CONTROL.txt"
    filepath_control: pathlib.Path = base_data_folderpath / filename_control

    # NOTE: need the placeholder to utilize its deserialize() method
    spline_surface_placeholder: QuadraticSplineSurface = quadratic_spline_surface_control_from_file
    with open(filepath_control, "r", encoding="utf-8") as file_control:
        patches_control: list[QuadraticSplineSurfacePatch] = spline_surface_placeholder.deserialize(
            file_control)
        file_control.close()

    # Now, grabbing the patches made from twelve_split_spline (i.e. our patches to test)
    patches_test: list[QuadraticSplineSurfacePatch] = spline_surface._patches
    assert len(patches_control) == len(patches_test)
    num_patches: int = len(patches_control)

    # Now, checking the values that have been saved
    for i in range(num_patches):
        # cx, cy, cz
        surface_mapping_coeffs_control: Matrix6x3r = patches_control[i].surface_mapping
        domain_control: ConvexPolygon = patches_control[i].domain
        vertices_control: Matrix3x2f = domain_control.vertices  # p1, p2, p3

        surface_mapping_coeffs_test: Matrix6x3r = patches_test[i].surface_mapping  # cx, cy, cz
        domain_test: ConvexPolygon = patches_test[i].domain
        vertices_test: Matrix3x2f = domain_test.vertices  # p1, p2, p3

        # lower precision because serialization loses precision
        npt.assert_allclose(vertices_control, vertices_test)
        npt.assert_allclose(surface_mapping_coeffs_control, surface_mapping_coeffs_test)

    # View the mesh
    # color: tuple[float, float, float] = SKY_BLUE
    # num_subdivisions: int = DISCRETIZATION_LEVEL
    # spline_surface.view(color, num_subdivisions)


# *******************
# Original Test Cases
# *******************

def twelve_split_quadratic_reproduction(uv_coeff: float,
                                        uu_coeff: float,
                                        vv_coeff: float
                                        ) -> bool:
    """
    Test that a quadratic surface can be reproduced from 
    analytic corner and midpoint data.
    This is more of a test that goes through the process
      and makes sure that everything operates normally.
    """

    V: np.ndarray = np.array([
        [1.0,  0.0],
        [0.0,  1.0],
        [0.0,  0.0]
    ], dtype=float)  # shape (3, 2)
    assert V.shape == (3, 2)

    # NOTE: dtype of F must be np.int64 or else it will fail the test case
    F: np.ndarray = np.array([
        [0, 1, 2]
    ], dtype=np.int64)  # shape (1, 3)
    assert F.shape == (1, 3)
    parametric_affine_manifold = ParametricAffineManifold(F, V)
    position_func = QuadraticPositionFunction(uv_coeff, uu_coeff, vv_coeff)
    gradient_func = QuadraticGradientFunction(uv_coeff, uu_coeff, vv_coeff)

    # Generate function data
    corner_data: dict[int, dict[int, TriangleCornerData]] = (
        generate_parametric_affine_manifold_corner_data(
            position_func, gradient_func, parametric_affine_manifold))
    midpoint_data: dict[int, dict[int, TriangleMidpointData]] = (
        generate_parametric_affine_manifold_midpoint_data(
            gradient_func, parametric_affine_manifold))
    surface_mappings: list[Matrix6x3r] = (
        generate_twelve_split_spline_patch_surface_mapping(
            corner_data[0],
            midpoint_data[0]))  # length 12 list

    assert len(surface_mappings) == 12

    domain_point: PlanarPoint1d = np.array([0.2, 0.3])
    assert domain_point.shape == (2, )
    q: SpatialVector = evaluate_quadratic_mapping(surface_mappings[0], domain_point)
    assert q.shape == (3, )

    if len(surface_mappings) != 12:
        return False

    if not vector_equal(q, position_func(0.2, 0.3)):
        return False

    return True


def test_twelve_split_spline_constant_surface() -> None:
    """
    Build constant function triangle data
    """
    p: SpatialVector1d = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    zero: SpatialVector1d = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    assert p.shape == (3,)
    assert zero.shape == (3,)

    corner_data: dict[int, TriangleCornerData] = {
        0: TriangleCornerData(p, zero, zero),
        1: TriangleCornerData(p, zero, zero),
        2: TriangleCornerData(p, zero, zero)
    }

    midpoint_data: dict[int, TriangleMidpointData] = {
        0: TriangleMidpointData(zero),
        1: TriangleMidpointData(zero),
        2: TriangleMidpointData(zero)
    }

    surface_mappings: list[Matrix6x3r]  # length 12 array with matrices shape (6, 3)
    surface_mappings = generate_twelve_split_spline_patch_surface_mapping(
        corner_data,
        midpoint_data)

    domain_point: PlanarPoint1d = np.array([0.25, 0.25], dtype=np.float64)
    q: SpatialVector = evaluate_quadratic_mapping(surface_mappings[0], domain_point)

    assert len(surface_mappings) == 12
    assert vector_equal(q, p)


def test_twelve_split_spline_linear_surface() -> None:
    """
    Build linear "quadratic" functionals
    """
    assert twelve_split_quadratic_reproduction(0.0, 0.0, 0.0)


def test_twelve_split_spline_quadratic_surface() -> None:
    """
    Test linear "quadratic" functionals
    """
    assert twelve_split_quadratic_reproduction(1.0, 0.0, 0.0)
    assert twelve_split_quadratic_reproduction(0.0, 1.0, 0.0)
    assert twelve_split_quadratic_reproduction(0.0, 0.0, 1.0)
    assert twelve_split_quadratic_reproduction(1.0, 2.0, -1.0)
