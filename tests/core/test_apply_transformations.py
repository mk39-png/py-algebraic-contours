

import logging

import numpy as np

from pyalgcon.core.apply_transformation import \
    apply_camera_frame_transformation_to_vertices
from pyalgcon.core.common import (
    Matrix3x3f, compare_eigen_numpy_matrix, initialize_spot_control_mesh)

logger: logging.Logger = logging.getLogger(__name__)


def test_apply_camera_frame_transformation_to_vertices():
    """
    """
    frame: Matrix3x3f = np.array([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]])
    logger.info("Projecting onto frame:\n%s", frame)
    V, uv, F, FT = initialize_spot_control_mesh()
    V_copy = apply_camera_frame_transformation_to_vertices(V, frame)
    compare_eigen_numpy_matrix(
        "spot_control\\core\\apply_transformations\\apply_camera_frame_transformation_to_vertices\\V_transformed.csv",
        V_copy)
