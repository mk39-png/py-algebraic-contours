"""
Holding utility methods and "global" constants that pertain to all tests.
"""


import logging
import os

import igl
import numpy as np
import numpy.typing as npty
import pytest

from pyalgcon.contour_network.compute_intersections import \
    IntersectionParameters
from pyalgcon.contour_network.contour_network import (ContourNetwork,
                                                      InvisibilityParameters)
from pyalgcon.core.affine_manifold import AffineManifold
from pyalgcon.core.apply_transformation import (
    apply_camera_frame_transformation_to_vertices,
    apply_transformation_to_vertices)
from pyalgcon.core.common import MatrixNx3f
from pyalgcon.quadratic_spline_surface.optimize_spline_surface import \
    OptimizationParameters
from pyalgcon.quadratic_spline_surface.twelve_split_spline import (
    TwelveSplitSplineSurface, compute_twelve_split_spline_patch_boundary_edges)

# from utils.projected_curve_networks_utils import SVGOutputMode


logger: logging.Logger = logging.getLogger(__name__)


FILE_BASE = "algebraic_contours"


def initialize_spot_control_mesh() -> tuple[npty.ArrayLike, npty.ArrayLike, npty.ArrayLike, npty.ArrayLike]:
    """ 
    Used for testing spot_control mesh in generating 
    the TwelveSplitSplineSurface.
    Returns only the parts of the mesh that are needed

    :return: tuple V, uv, F, FT
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    # Get input mesh
    filename: str = "spot_control_mesh-cleaned_conf_simplified_with_uv.obj"
    filepath: str = os.path.abspath(f"spot_control\\{filename}")

    V_temp: npty.ArrayLike
    uv_temp: npty.ArrayLike
    N_temp: npty.ArrayLike
    F_temp: npty.ArrayLike  # int
    FT_temp: npty.ArrayLike  # int
    FN_temp: npty.ArrayLike  # int
    V_temp, uv_temp, N_temp, F_temp, FT_temp, FN_temp = igl.readOBJ(filepath)

    # Wrapping inside np.array for typing
    V: np.ndarray = np.array(V_temp)
    uv: np.ndarray = np.array(uv_temp)
    F: np.ndarray = np.array(F_temp)
    FT: np.ndarray = np.array(FT_temp)

    return V, uv, F, FT


# **************
# Helper Methods Compute Contours
# **************
def _initialize_contour_info_spot_mesh() -> tuple[TwelveSplitSplineSurface,
                                                  IntersectionParameters,
                                                  InvisibilityParameters,
                                                  list[tuple[int, int]],
                                                  np.ndarray]:
    """
    Helper function to initialize information for Contour Network testing.

    :return: spline_surface
    :return: intersect_params
    :return: invisibility_params
    :return: patch_boundary_edges
    :return: frame
    """
    # svg_output_mode: SVGOutputMode = SVGOutputMode.UNIFORM_VISIBLE_CURVES
    optimization_params = OptimizationParameters()
    intersect_params = IntersectionParameters()
    invisibility_params = InvisibilityParameters()
    # weight: float = optimization_params.position_difference_factor
    # trim: float = intersect_params.trim_amount
    # pad: float = invisibility_params.pad_amount
    # invisibility_method: InvisibilityMethod = invisibility_params.invisibility_method
    # show_nodes: bool = False
    logger.setLevel(logging.INFO)

    # Set up the camera
    frame_3x3f: np.ndarray = np.array([[1, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 1]])
    logger.info("Projecting onto frame:\n%s", frame_3x3f)
    V:  np.ndarray
    uv: np.ndarray
    F:  np.ndarray
    FT: np.ndarray
    V, uv, F, FT = initialize_spot_control_mesh()
    V_transformed: MatrixNx3f = apply_camera_frame_transformation_to_vertices(V, frame_3x3f)

    # Generate quadratic spline
    logger.info("Comnputing spline surface")
    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)
    spline_surface: TwelveSplitSplineSurface = TwelveSplitSplineSurface(V_transformed,
                                                                        affine_manifold,
                                                                        optimization_params)

    # Get the boundary edges
    patch_boundary_edges: list[tuple[int, int]]
    patch_boundary_edges: list[tuple[int, int]] = (
        compute_twelve_split_spline_patch_boundary_edges(F, spline_surface.face_to_patch_indices))

    return spline_surface, intersect_params, invisibility_params, patch_boundary_edges, frame_3x3f


#
# Affine Manifold
#
@pytest.fixture
def initialize_affine_manifold_from_spot_control() -> AffineManifold:
    """
    Helper function to initialize AffineManifold from spot_control mesh.
    """
    # Get input mesh
    V, uv, F, FT = initialize_spot_control_mesh()
    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)
    affine_manifold_filepath: str = "spot_control\\affine_manifold\\"

    return affine_manifold


#
# TESTING PARAMETERS
#
@pytest.fixture
def obj_filepaths() -> list[str]:
    print(os.path.abspath(f'{FILE_BASE}'))
    return [
        # (os.path.abspath(
        #     f'{FILE_BASE}\\tests\\spot_control\\spot_control_mesh-cleaned_conf_simplified_with_uv.obj')),
        (os.path.abspath("default_cube\\temp_out.obj"))
    ]
# @pytest.mark.parametrize(
#     "filepath",
#     [
#         # (os.path.abspath(
#         #     f'{FILE_BASE}\\tests\\spot_control\\spot_control_mesh-cleaned_conf_simplified_with_uv.obj')),
#         (os.path.abspath(
#             f'{FILE_BASE}\\tests\\default_cube\\temp_out.obj')),
#     ]
# )

# XXX: change to proper mark


@pytest.fixture
def projection_matrices() -> list[np.ndarray]:
    return [np.array([
        [2.777777671813965, 0.0, 0.0, 0.0],
        [0.0, 4.938271522521973, 0.0, 0.0],
        [0.0, 0.0, -1.0020020008087158, -0.20020020008087158],
        [0.0, 0.0, -1.0, 0.0]], dtype=np.float64)]
# @pytest.mark.parametrize(
#     "projection_matrices",
#     [
#         # TODO: this may not correspond to the Blender's actual projection matrix.
#         # That I need to test.
#         np.array([
#             [2.777777671813965, 0.0, 0.0, 0.0],
#             [0.0, 4.938271522521973, 0.0, 0.0],
#             [0.0, 0.0, -1.0020020008087158, -0.20020020008087158],
#             [0.0, 0.0, -1.0, 0.0]], dtype=np.float64)
#     ]
# )
#
# TESTING FIXTURES
#


@pytest.fixture
def root_folder() -> str:
    # FIXME: this should be limited in scope to the fixtures below rather than it being
    # accessible by all other pytest cases
    return "spot_control"


@pytest.fixture(name="load_mesh_testing")
def load_mesh_testing(obj_filepaths) -> tuple[npty.ArrayLike,
                                              npty.ArrayLike,
                                              npty.ArrayLike,
                                              npty.ArrayLike,
                                              npty.ArrayLike,
                                              npty.ArrayLike]:
    """
    Returns deserialized .obj file.
    """
    return igl.readOBJ(obj_filepaths[0])


@pytest.fixture
def initialize_affine_manifold(load_mesh_testing) -> AffineManifold:
    """
    Fixture to calculate the AffineManifold from the load_mesh_testing fixture.
    """
    V_raw, uv, N, F, FT, FN = load_mesh_testing
    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)
    return affine_manifold


@pytest.fixture(name="projection_on_vertices")
def projection_on_vertices(load_mesh_testing, projection_matrices) -> np.ndarray:
    """
    Returns vertices under projection matrix.
    """
    V_raw, uv, N, F, FT, FN = load_mesh_testing
    projection_matrix: np.ndarray = projection_matrices[0]

    V_transformed: MatrixNx3f = apply_transformation_to_vertices(V_raw, projection_matrix)
    return V_transformed


@pytest.fixture
def initialize_twelve_split_spline_transformed(initialize_affine_manifold,
                                               projection_on_vertices) -> TwelveSplitSplineSurface:
    """
    This is used to test the member variables of TwevleSplitSplineSurface().
    Helper function that initialized TwelveSplitSplineSurface from the spot_control mesh.
    Also returns the affine_manifold used to build the TwelveSplitSplineSurface object.
    Also returns vertices used to initialize TwelveSplitSplineSurface
    (i.e. vertices of the spot_control mesh)

    NOTE: initializes 12-split spline for use in contour network.
    """
    affine_manifold: AffineManifold = initialize_affine_manifold
    V_transformed: np.ndarray = projection_on_vertices
    optimization_params: OptimizationParameters = OptimizationParameters()

    # Generate quadratic spline
    spline_surface_transformed: TwelveSplitSplineSurface = TwelveSplitSplineSurface(
        V_transformed,
        affine_manifold,
        optimization_params)
    return spline_surface_transformed


@pytest.fixture
def initialize_contour_network(
        load_mesh_testing,
        initialize_twelve_split_spline_transformed) -> ContourNetwork:
    """
    Used for testing of contour network generat
    """
    # Retrieve parameters
    V_raw, uv, N, F, FT, FN = load_mesh_testing
    spline_surface: TwelveSplitSplineSurface = initialize_twelve_split_spline_transformed
    intersect_params = IntersectionParameters()
    invisibility_params = InvisibilityParameters()

    # Get the boundary edges
    patch_boundary_edges: list[tuple[int, int]] = (
        compute_twelve_split_spline_patch_boundary_edges(F, spline_surface.face_to_patch_indices))

    # Build the contours
    contour_network = ContourNetwork(
        spline_surface,
        intersect_params,
        invisibility_params,
        patch_boundary_edges
    )

    return contour_network
