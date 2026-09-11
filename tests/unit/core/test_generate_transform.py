"""
Test generate transformation
"""

import numpy as np
import numpy.testing as npt
import pytest

from pyalgcon.core.common import Matrix4x4f
from pyalgcon.core.generate_transformation import \
    origin_to_infinity_projective_matrix


@pytest.mark.regression
def test_origin_to_infinity_projective_matrix() -> None:
    """
    Testing with values from original C++ code
    Esnure that infinity projection does not deviate from its core implementation.
    """
    camera_to_plane_distance: float = 1.0
    projection_matrix_test: Matrix4x4f = origin_to_infinity_projective_matrix(
        camera_to_plane_distance)
    projection_matrix_control: Matrix4x4f = np.array([[1, 0, 0, 0],
                                                      [0, 1, 0, 0],
                                                      [0, 0, 0, -1],
                                                      [0, 0, 1, 0]],
                                                     dtype=np.float64)
    npt.assert_allclose(projection_matrix_control, projection_matrix_test)
