"""
Test apply_transformations
"""

import logging
import pathlib

import numpy as np
import numpy.testing as npt
import pytest

from pyalgcon.core.apply_transformation import (
    apply_camera_frame_transformation_to_vertices,
    apply_camera_matrix_transformation_to_vertices,
    apply_transformation_to_vertices,
    apply_transformation_to_vertices_in_place)
from pyalgcon.core.common import (Matrix3x3f, Matrix4x4f, MatrixNx3f,
                                  compare_eigen_numpy_matrix,
                                  deserialize_eigen_matrix_csv_to_numpy)

logger: logging.Logger = logging.getLogger(__name__)


@pytest.mark.parametrize("seed", range(100))
def test_apply_transformation_to_vertices(seed, parsed_control_mesh) -> None:
    """
    Test to ensure that the output apply_transformation_to_vertices_in_place is 
    the same as apply_transformation_to_vertices()
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    projective_transformation: Matrix4x4f = rng.normal(size=(4, 4))
    V, uv, F, FT = parsed_control_mesh

    V_control: MatrixNx3f = apply_transformation_to_vertices(V, projective_transformation)
    V_test: MatrixNx3f = np.copy(V)
    apply_transformation_to_vertices_in_place(V_test, projective_transformation)

    npt.assert_allclose(V_control, V_test)


# def test_apply_camera_matrix_transformation_to_vertices(
#         testing_fileinfo: tuple[pathlib.Path, pathlib.Path],
#         parsed_control_mesh: tuple[np.ndarray,
#                                    np.ndarray,
#                                    np.ndarray,
#                                    np.ndarray]) -> None:
#     """
#     Tests applying camera matrix on vertices.

#     NOTE: this test seems similar to the projection_matrix_on_vertices() fixture,
#     but this test serves to validate the output of apply_camera_matrix_transformation_on_vertices()
#     with the output from the original C++ code as part of the process to debug
#     the mismatching .svg contours output.
#     """
#     # Retrieve parameters
#     base_data_folderpath: pathlib.Path
#     base_data_folderpath, _ = testing_fileinfo
#     filepath: pathlib.Path = base_data_folderpath / "core" / "apply_transformations" / \
#         "apply_camera_matrix_transformation_to_vertices"
#     camera_matrix: Matrix4x4f = deserialize_eigen_matrix_csv_to_numpy(
#         filepath / "camera_matrix.csv")
#     assert camera_matrix.shape == (4, 4)
#     V, uv, F, FT = parsed_control_mesh

#     # Execute method
#     logger.info("Projecting onto camera matrix:\n%s", camera_matrix)
#     V_transformed: MatrixNx3f = apply_camera_matrix_transformation_to_vertices(V, camera_matrix)

#     # Compare results
#     compare_eigen_numpy_matrix(filepath / "V_transformed.csv", V_transformed)


def test_apply_camera_frame_transformation_to_vertices(
        testing_fileinfo: tuple[pathlib.Path, pathlib.Path],
        parsed_control_mesh: tuple[np.ndarray,
                                   np.ndarray,
                                   np.ndarray,
                                   np.ndarray]) -> None:
    """
    Tests applying camera frame on vertices.
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / "core" / "apply_transformations" / \
        "apply_camera_frame_transformation_to_vertices" / "V_transformed.csv"
    V, uv, F, FT = parsed_control_mesh
    frame: Matrix3x3f = np.array([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]],
                                 dtype=np.float64)

    # Execute method
    logger.info("Projecting onto frame:\n%s", frame)
    V_copy: MatrixNx3f = apply_camera_frame_transformation_to_vertices(V, frame)

    # Compare results
    compare_eigen_numpy_matrix(filepath, V_copy)
