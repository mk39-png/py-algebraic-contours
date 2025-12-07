"""
Holding utility methods and "global" constants that pertain to all tests.
So, this entails .obj reading an filepath resolving, along with NumPy test comparisons.
A more concrete use case for this file is to initialize objects that are
used throughout various tests (e.g. Spot Control .obj file)
"""

import logging
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
from pyalgcon.core.apply_transformation import \
    apply_camera_frame_transformation_to_vertices
from pyalgcon.core.common import Matrix3x3f, MatrixNx3f
from pyalgcon.quadratic_spline_surface.optimize_spline_surface import \
    OptimizationParameters
from pyalgcon.quadratic_spline_surface.quadratic_spline_surface import \
    QuadraticSplineSurface
from pyalgcon.quadratic_spline_surface.twelve_split_spline import (
    TwelveSplitSplineSurface, compute_twelve_split_spline_patch_boundary_edges)

logger: logging.Logger = logging.getLogger(__name__)


@pytest.fixture(scope="session", params=[
    ("spot_control", "spot_control_mesh-cleaned_conf_simplified_with_uv.obj")
])
def testing_fileinfo(request) -> tuple[pathlib.Path, pathlib.Path]:
    """ Flexible method to resolve filepath of test data.

    :returns: tuple of folderpath and filepath to a given obj file
    """
    # Setup
    foldername: str
    obj_filename: str
    foldername, obj_filename = request.param
    base_folderpath: pathlib.Path = pathlib.Path(__file__).parent / "data"

    # Return values
    base_data_folderpath: pathlib.Path = base_folderpath / foldername
    obj_filepath: pathlib.Path = base_folderpath / foldername / obj_filename
    return base_data_folderpath, obj_filepath


@pytest.fixture(scope="session")
def parsed_control_mesh(testing_fileinfo: tuple[pathlib.Path, pathlib.Path]
                        ) -> tuple[np.ndarray,
                                   np.ndarray,
                                   np.ndarray,
                                   np.ndarray]:
    """ 
    Used for testing control mesh in generating 
    the TwelveSplitSplineSurface.
    Returns only the parts of the mesh that are needed.
    Returns the root folder of the mesh and its associated parsed
    data.

    :return: folder path and tuple of vertices, uv coordinates, face indices, 
             and face indices into texture coordinates
    :rtype: pathlib.Path, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    obj_filepath: pathlib.Path
    _, obj_filepath = testing_fileinfo

    # Get input mesh
    V_temp: npty.ArrayLike
    uv_temp: npty.ArrayLike
    N_temp: npty.ArrayLike
    F_temp: npty.ArrayLike  # int
    FT_temp: npty.ArrayLike  # int
    FN_temp: npty.ArrayLike  # int
    V_temp, uv_temp, N_temp, F_temp, FT_temp, FN_temp = igl.readOBJ(obj_filepath)

    # Wrapping inside np.array for typing
    V: np.ndarray = np.array(V_temp)
    uv: np.ndarray = np.array(uv_temp)
    F: np.ndarray = np.array(F_temp)
    FT: np.ndarray = np.array(FT_temp)

    return V, uv, F, FT


@pytest.fixture(scope="session")
def initialize_affine_manifold(parsed_control_mesh) -> AffineManifold:
    """
    Fixture to calculate the AffineManifold from the load_mesh_testing fixture.
    """
    V: np.ndarray
    uv: np.ndarray
    F: np.ndarray
    FT: np.ndarray
    V, uv, F, FT = parsed_control_mesh
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
    V_raw: np.ndarray
    uv: np.ndarray
    F: np.ndarray
    FT: np.ndarray
    V_raw, uv, F, FT = parsed_control_mesh

    # TODO: have ability to test with various projection matrices
    frame: Matrix3x3f = np.array([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]])
    V_transformed: MatrixNx3f = apply_camera_frame_transformation_to_vertices(V_raw, frame)
    return V_transformed


@pytest.fixture(scope="session")
def quadratic_spline_surface_control_from_file(testing_fileinfo) -> QuadraticSplineSurface:
    """
    Utilizes the .from_file() class method to create a QuadraticSplineSurface from file.
    This follows the format of QuadraticSplineSurface serialization that the original C++ code
    uses
    """
    obj_filepath: pathlib.Path
    _, obj_filepath = testing_fileinfo

    # Need to initialize QuadraticSplineSurface with our test file
    # simply gets the original .obj filename and appends to it "_CONTROL.txt"
    surface_filepath: pathlib.Path = obj_filepath.with_name(obj_filepath.stem + "_CONTROL.txt")
    spline_surface: QuadraticSplineSurface = QuadraticSplineSurface.from_file(surface_filepath)

    return spline_surface


@pytest.fixture(scope="session")
def twelve_split_spline_raw(parsed_control_mesh,
                            initialize_affine_manifold) -> TwelveSplitSplineSurface:
    """
    Creates a TwelveSplitSpline surface WITHOUT transformed vertices
    NOTE: initializes 12-split spline for use in quadratic surface tests.
    """
    # Retrieve parameters
    V_raw: np.ndarray
    uv: np.ndarray
    F: np.ndarray
    FT: np.ndarray
    V_raw, uv, F, FT = parsed_control_mesh
    affine_manifold: AffineManifold = initialize_affine_manifold
    optimization_params: OptimizationParameters = OptimizationParameters()

    # Generate quadratic spline
    spline_surface: TwelveSplitSplineSurface = TwelveSplitSplineSurface(
        V_raw,
        affine_manifold,
        optimization_params)

    return spline_surface


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
def initialize_patch_boundary_edges(parsed_control_mesh: tuple[np.ndarray,
                                                               np.ndarray,
                                                               np.ndarray,
                                                               np.ndarray],
                                    twelve_split_spline_transformed: TwelveSplitSplineSurface
                                    ) -> list[tuple[int, int]]:
    """ Calculates patch boundary edges

    :param parsed_control_mesh: mesh information
    :type parsed_control_mesh: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    :param twelve_split_spline_transformed: twelve split splined from projected vertices
    :type twelve_split_spline_transformed: TwelveSplitSplineSurface
    :return: patch boundary edges
    :rtype: list[tuple[int, int]]
    """

    # Retrieve parameters
    V: np.ndarray
    uv: np.ndarray
    F: np.ndarray
    FT: np.ndarray
    V, uv, F, FT = parsed_control_mesh
    spline_surface: TwelveSplitSplineSurface = twelve_split_spline_transformed

    # Get the boundary edges
    patch_boundary_edges: list[tuple[int, int]] = (
        compute_twelve_split_spline_patch_boundary_edges(F, spline_surface.face_to_patch_indices)
    )

    return patch_boundary_edges


@pytest.fixture(scope="session")
def initialize_contour_network(
        testing_fileinfo,
        twelve_split_spline_transformed,
        initialize_patch_boundary_edges) -> tuple[pathlib.Path,
                                                  ContourNetwork]:
    """
    Used for testing of contour network generation
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    spline_surface: TwelveSplitSplineSurface = twelve_split_spline_transformed
    patch_boundary_edges: list[tuple[int, int]] = initialize_patch_boundary_edges
    intersect_params = IntersectionParameters()
    invisibility_params = InvisibilityParameters()

    # Build the contours
    logger.info("Computing contours")
    contour_network = ContourNetwork(
        spline_surface,
        intersect_params,
        invisibility_params,
        patch_boundary_edges
    )

    output_contour_filepath: pathlib.Path = base_data_folderpath / "contour_network" / "output.svg"
    return output_contour_filepath, contour_network
