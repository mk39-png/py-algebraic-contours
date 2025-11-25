import numpy as np
import pytest

from pyalgcon.core.bivariate_quadratic_function import \
    compute_quadratic_cross_product
from pyalgcon.core.common import Matrix3x3f


def test_compute_quadratic_cross_product() -> None:
    V_coeffs: Matrix3x3f = np.array([[0, 1, 2], [3, 4, 5], [7, 8, 9]], dtype=np.float64)
    W_coeffs: Matrix3x3f = np.array([[0, 1, 2], [3, 4, 5], [7, 8, 9]], dtype=np.float64)
    assert V_coeffs.shape == (3, 3)
    assert W_coeffs.shape == (3, 3)

    N_coeffs: np.ndarray = compute_quadratic_cross_product(V_coeffs, W_coeffs)
    assert N_coeffs.shape == (6, 3)
