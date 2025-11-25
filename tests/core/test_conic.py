
import numpy as np

from pyalgcon.core.common import (Matrix3x2f, Vector3f,
                                  Vector6f, float_equal)
from pyalgcon.core.conic import Conic, ConicType
from pyalgcon.core.rational_function import RationalFunction


def test_zero_case() -> None:
    """
    Testing 2D vector with F_coeffs to see if it's all good.
    """
    P_coeffs: Matrix3x2f = np.array([[0, 2],
                                     [0, 3],
                                     [0, 1]], dtype=np.float64)
    Q_coeffs: Vector3f = np.array([1, 0, 0], dtype=np.float64)
    F_coeffs: Vector6f = np.array([[0],
                                   [0],
                                   [0],
                                   [0],
                                   [0],
                                   [0]], dtype=np.float64)
    assert P_coeffs.shape == (3, 2)
    assert Q_coeffs.shape == (3, )
    assert F_coeffs.shape == (6, 1)

    conic = Conic(ConicType.UNKNOWN, P_coeffs, Q_coeffs)
    pullback: RationalFunction = conic.pullback_quadratic_function(1, F_coeffs)

    assert (float_equal(pullback(-1.0)[0], 0.0))
    assert (float_equal(pullback(0.0)[0], 0.0))
    assert (float_equal(pullback(1.0)[0], 0.0))


def test_unit_pullback_case() -> None:
    P_coeffs: Matrix3x2f = np.array([[0.0, 2.0],
                                     [0.0, 3.0],
                                     [0.0, 1.0]], dtype=np.float64)
    Q_coeffs: Vector3f = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    # NOTE: first element in F_coeffs different from zero case
    #   (for those looking for anything different between this case and zero case)
    F_coeffs: Vector6f = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    assert P_coeffs.shape == (3, 2)
    assert Q_coeffs.shape == (3, )
    assert F_coeffs.shape == (6, )

    conic = Conic(ConicType.UNKNOWN, P_coeffs, Q_coeffs)
    pullback: RationalFunction = conic.pullback_quadratic_function(1, F_coeffs)

    assert (float_equal(pullback(-1.0)[0], 1.0))
    assert (float_equal(pullback(0.0)[0], 1.0))
    assert (float_equal(pullback(1.0)[0], 1.0))


def test_u_projection_case() -> None:
    P_coeffs: Matrix3x2f = np.array([[1.0, 1.0],
                                     [2.0, -2.0],
                                     [1.0, 1.0]], dtype=np.float64)
    Q_coeffs: Vector3f = np.array([1.0, 0.0, 1.0], dtype=np.float64)
    F_coeffs: Vector6f = np.array([0, 1, 0, 0, 0, 0], dtype=np.float64)
    assert P_coeffs.shape == (3, 2)
    assert Q_coeffs.shape == (3, )
    assert F_coeffs.shape == (6, )

    conic = Conic(ConicType.UNKNOWN, P_coeffs, Q_coeffs)
    pullback: RationalFunction = conic.pullback_quadratic_function(1, F_coeffs)

    assert float_equal(pullback(-1.0)[0], 0.0)
    assert float_equal(pullback(0.0)[0], 1.0)
    assert float_equal(pullback(1.0)[0], 2.0)
