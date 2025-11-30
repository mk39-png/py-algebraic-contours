"""
Holding utility methods and "global" constants that pertain to all tests.
So, this entails .obj reading an filepath resolving, along with NumPy test comparisons.
A more concrete use case for this file is to initialize objects that are
used throughout various tests (e.g. Spot Control .obj file)
"""

import logging
import os
import pathlib

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
from pyalgcon.core.common import Matrix3x3f, MatrixNx3f
from pyalgcon.quadratic_spline_surface.optimize_spline_surface import \
    OptimizationParameters
from pyalgcon.quadratic_spline_surface.twelve_split_spline import (
    TwelveSplitSplineSurface, compute_twelve_split_spline_patch_boundary_edges)

logger: logging.Logger = logging.getLogger(__name__)


FILE_BASE = "algebraic_contours"

#
# UTILITY METHODS
#


@pytest.fixture(scope="session", params=[
    ("spot_control", "spot_control_mesh-cleaned_conf_simplified_with_uv.obj")
])
def obj_fileinfo(request) -> tuple[pathlib.Path, pathlib.Path]:
    """ Flexible method to resolve filepath of test data.

    :returns: tuple of folder path and filepath to a given obj file
    """
    folder_name: str
    file_name: str
    folder_name, file_name = request.param
    base_directory: pathlib.Path = pathlib.Path(__file__).parent / "data"

    return (base_directory / folder_name), (base_directory / folder_name / file_name)


@pytest.fixture(scope="session")
def parsed_control_mesh(obj_fileinfo) -> tuple[pathlib.Path,
                                               tuple[np.ndarray,
                                                     np.ndarray,
                                                     np.ndarray,
                                                     np.ndarray]]:
    """ 
    Used for testing control mesh in generating 
    the TwelveSplitSplineSurface.
    Returns only the parts of the mesh that are needed.
    Returns the root folder of the mesh and its associated parsed
    data.

    :return: folder path and tuple V, uv, F, FT
    :rtype: pathlib.Path, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    folder_path: pathlib.Path
    file_path: pathlib.Path
    folder_path, file_path = obj_fileinfo

    # Get input mesh
    V_temp: npty.ArrayLike
    uv_temp: npty.ArrayLike
    N_temp: npty.ArrayLike
    F_temp: npty.ArrayLike  # int
    FT_temp: npty.ArrayLike  # int
    FN_temp: npty.ArrayLike  # int
    V_temp, uv_temp, N_temp, F_temp, FT_temp, FN_temp = igl.readOBJ(file_path)

    # Wrapping inside np.array for typing
    V: np.ndarray = np.array(V_temp)
    uv: np.ndarray = np.array(uv_temp)
    F: np.ndarray = np.array(F_temp)
    FT: np.ndarray = np.array(FT_temp)

    return folder_path, (V, uv, F, FT)


@pytest.fixture(scope="session")
def initialize_affine_manifold(parsed_control_mesh) -> AffineManifold:
    """
    Fixture to calculate the AffineManifold from the load_mesh_testing fixture.
    """
    folder_path, (V, uv, F, FT) = parsed_control_mesh
    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)
    return affine_manifold


# @pytest.fixture
# def projection_matrices() -> list[np.ndarray]:
#     """Preset matrices from Blender to test with

#     :return: perspective matrix
#     :rtype: list[np.ndarray]
#     """
#     return [np.array([
#         [2.777777671813965, 0.0, 0.0, 0.0],
#         [0.0, 4.938271522521973, 0.0, 0.0],
#         [0.0, 0.0, -1.0020020008087158, -0.20020020008087158],
#         [0.0, 0.0, -1.0, 0.0]], dtype=np.float64)]

# TODO: modify the below to support various projection matrices.
# @pytest.fixture(scope="session")
# def projection_on_vertices(parsed_control_mesh, projection_matrices) -> np.ndarray:
#     """
#     Returns vertices under projection matrix.
#     """
#     folder_path, (V_raw, uv, F, FT) = parsed_control_mesh

#     # TODO: have ability to test with various projection matrices
#     projection_matrix: np.ndarray = projection_matrices[0]
#     V_transformed: MatrixNx3f = apply_transformation_to_vertices(V_raw, projection_matrix)
#     return V_transformed


@pytest.fixture(scope="session")
def projection_frame_on_vertices(parsed_control_mesh) -> np.ndarray:
    """
    Returns vertices projected onto camera default camera.
    """
    folder_path, (V_raw, uv, F, FT) = parsed_control_mesh

    # TODO: have ability to test with various projection matrices
    frame: Matrix3x3f = np.array([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]])
    V_transformed: MatrixNx3f = apply_camera_frame_transformation_to_vertices(V_raw, frame)
    return V_transformed


@pytest.fixture(scope="session")
def twelve_split_spline_transformed(initialize_affine_manifold,
                                    projection_frame_on_vertices) -> TwelveSplitSplineSurface:
    """
    This is used to test the member variables of TwevleSplitSplineSurface().
    Helper function that initialized TwelveSplitSplineSurface from the spot_control mesh.
    Also returns the affine_manifold used to build the TwelveSplitSplineSurface object.
    Also returns vertices used to initialize TwelveSplitSplineSurface
    (i.e. vertices of the spot_control mesh)

    NOTE: initializes 12-split spline for use in contour network.
    """
    affine_manifold: AffineManifold = initialize_affine_manifold
    V_transformed: np.ndarray = projection_frame_on_vertices
    optimization_params: OptimizationParameters = OptimizationParameters()

    # Generate quadratic spline
    spline_surface_transformed: TwelveSplitSplineSurface = TwelveSplitSplineSurface(
        V_transformed,
        affine_manifold,
        optimization_params)
    return spline_surface_transformed


@pytest.fixture(scope="session")
def initialize_contour_network(
        parsed_control_mesh,
        twelve_split_spline_transformed) -> tuple[pathlib.Path, ContourNetwork]:
    """
    Used for testing of contour network generat
    """
    # Retrieve parameters
    folder_path, (V_raw, uv, F, FT) = parsed_control_mesh
    spline_surface: TwelveSplitSplineSurface = twelve_split_spline_transformed
    intersect_params = IntersectionParameters()
    invisibility_params = InvisibilityParameters()

    # Get the boundary edges
    patch_boundary_edges: list[tuple[int, int]] = (
        compute_twelve_split_spline_patch_boundary_edges(F, spline_surface.face_to_patch_indices)
    )

    # Build the contours
    logger.info("Computing contours")
    contour_network = ContourNetwork(
        spline_surface,
        intersect_params,
        invisibility_params,
        patch_boundary_edges
    )

    output_filepath: pathlib.Path = folder_path / "contour_network" / "output.svg"
    return output_filepath, contour_network
