"""
Test apply_transformations
"""

import logging
import pathlib

import numpy as np

from pyalgcon.core.apply_transformation import \
    apply_camera_frame_transformation_to_vertices
from pyalgcon.core.common import Matrix3x3f, MatrixNx3f
from pyalgcon.debug.debug import compare_eigen_numpy_matrix_absolute

logger: logging.Logger = logging.getLogger(__name__)


def test_apply_camera_frame_transformation_to_vertices(
        parsed_control_mesh: tuple[pathlib.Path,
                                   tuple[np.ndarray,
                                         np.ndarray,
                                         np.ndarray,
                                         np.ndarray]]) -> None:
    """
    Tests applying camera frame on vertices.
    """
    folder_path: pathlib.Path
    folder_path, (V, uv, F, FT) = parsed_control_mesh

    frame: Matrix3x3f = np.array([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]],
                                 dtype=np.float64)

    logger.info("Projecting onto frame:\n%s", frame)
    V_copy: MatrixNx3f = apply_camera_frame_transformation_to_vertices(V, frame)

    compare_eigen_numpy_matrix_absolute(
        folder_path / "core" / "apply_transformations" /
        "apply_camera_frame_transformation_to_vertices" / "V_transformed.csv",
        V_copy)
