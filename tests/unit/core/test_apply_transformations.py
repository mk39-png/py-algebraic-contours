"""
Test apply_transformations
"""

import logging
import pathlib

import numpy as np

from pyalgcon.core.apply_transformation import \
    apply_camera_frame_transformation_to_vertices
from pyalgcon.core.common import (Matrix3x3f, MatrixNx3f,
                                  compare_eigen_numpy_matrix)

logger: logging.Logger = logging.getLogger(__name__)


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
