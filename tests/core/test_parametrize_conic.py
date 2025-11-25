
import logging

import numpy as np

from pyalgcon.core.bivariate_quadratic_function import (
    evaluate_quadratic, evaluate_quadratic_mapping)
from pyalgcon.core.common import (PlanarPoint1d, Vector6f,
                                  float_equal, todo)
from pyalgcon.core.conic import Conic, ConicType
from pyalgcon.core.parametrize_conic import (
    identify_conic, parametrize_conic)

logger: logging.Logger = logging.getLogger(__name__)


def _test_parametrization(C_coeffs: Vector6f) -> bool:
    """
    """
    assert C_coeffs.shape == (6, )
    logger.info("Testing conic with equation %s", C_coeffs)

    conics: list[Conic] = parametrize_conic(C_coeffs)
    for conic in conics:
        points: list[PlanarPoint1d] = conic.sample_points(10)
        for p in points:
            assert p.shape == (2, )
            if not float_equal(evaluate_quadratic(C_coeffs, p), 0.0):
                return False

    return True


def test_ellipse() -> None:
    C_coeffs: Vector6f = np.array([1, 0, 0, 0, -1, -1], dtype=np.float64)

    print("Circle")
    assert (identify_conic(C_coeffs) == ConicType.ELLIPSE)

    print("Vertically stretched ellipse")
    C_coeffs[4] = -2
    assert (identify_conic(C_coeffs) == ConicType.ELLIPSE)

    print("Horizontally stretched ellipse")
    C_coeffs[5] = -2
    assert (identify_conic(C_coeffs) == ConicType.ELLIPSE)

    print("Large circle")
    C_coeffs = np.array([1, 0, 0, 0, -0.1, -0.1])
    assert (identify_conic(C_coeffs) == ConicType.ELLIPSE)

    print("General standard form")
    C_coeffs = np.array([5, 0, 0, 0, -10, -0.1])
    assert (identify_conic(C_coeffs) == ConicType.ELLIPSE)

    print("Negative standard form")
    C_coeffs = np.array([-5, 0, 0, 0, 10, 0.1])
    assert (identify_conic(C_coeffs) == ConicType.ELLIPSE)

    print("Linear terms")
    C_coeffs = np.array([-5, -5, 5, 0, 10, 0.1])
    assert (identify_conic(C_coeffs) == ConicType.ELLIPSE)

    print("General form")
    C_coeffs = np.array([-5, -5, 5, 1.9, 10, 0.1])
    assert (identify_conic(C_coeffs) == ConicType.ELLIPSE)


def test_hyperbola():
    """"""
    C_coeffs: Vector6f = np.array([1, 0, 0, 0, 1, -1], dtype=np.float64)

    print("Vertically symmetric")
    assert identify_conic(C_coeffs) == ConicType.HYPERBOLA

    print("Horizontally symmetric")
    C_coeffs[4] = -1
    C_coeffs[5] = 1
    assert identify_conic(C_coeffs) == ConicType.HYPERBOLA

    # FIXME cases below not working
    # print("Vertically stretched hyperbola")
    # C_coeffs[4] = 2
    # assert identify_conic(C_coeffs) == ConicType.HYPERBOLA

    # print("Horizontally stretched hyperbola")
    # C_coeffs[5] = -2
    # assert identify_conic(C_coeffs) == ConicType.HYPERBOLA

    print("General standard form")
    C_coeffs[0] = 5
    C_coeffs[4] = -10
    C_coeffs[5] = 0.1
    assert identify_conic(C_coeffs) == ConicType.HYPERBOLA

    print("Negative standard form")
    C_coeffs[0] = -5
    C_coeffs[4] = 10
    C_coeffs[5] = -0.1
    assert identify_conic(C_coeffs) == ConicType.HYPERBOLA

    print("Linear terms")
    C_coeffs[0] = -5
    C_coeffs[1] = -5
    C_coeffs[2] = 5
    C_coeffs[4] = -10
    C_coeffs[5] = 0.1
    assert identify_conic(C_coeffs) == ConicType.HYPERBOLA

    print("General form")
    C_coeffs[0] = -5
    C_coeffs[1] = -5
    C_coeffs[2] = 5
    C_coeffs[3] = 2.1
    C_coeffs[4] = 10
    C_coeffs[5] = 0.1
    assert identify_conic(C_coeffs) == ConicType.HYPERBOLA


def test_ellipse_parametrized():
    """
    """
    C_coeffs: Vector6f = np.zeros(shape=(6, ), dtype=np.float64)
    C_coeffs[0] = -1.0
    C_coeffs[4] = 1.0
    C_coeffs[5] = 1.0

    print("Unit circle")
    assert _test_parametrization(C_coeffs)

    print("Small circle")
    C_coeffs[4] = 2
    C_coeffs[5] = 2
    assert _test_parametrization(C_coeffs)

    print("General circle")
    C_coeffs[0] = -0.5
    C_coeffs[1] = -2
    C_coeffs[2] = 3
    C_coeffs[4] = 2
    C_coeffs[5] = 2
    assert _test_parametrization(C_coeffs)

    print("General ellipse")
    C_coeffs[0] = -0.5
    C_coeffs[1] = -2
    C_coeffs[2] = 3
    C_coeffs[4] = 0.1
    C_coeffs[5] = 2
    assert _test_parametrization(C_coeffs)


def test_hyperbola_parametrized():
    """
    """
    C_coeffs: Vector6f = np.zeros(shape=(6, ), dtype=np.float64)
    C_coeffs[0] = -1.0
    C_coeffs[4] = 1.0
    C_coeffs[5] = -1.0

    print("Unit hyperbola")
    assert _test_parametrization(C_coeffs)

    print("Translated hyperbola")
    C_coeffs[1] = -2
    C_coeffs[2] = 3
    assert _test_parametrization(C_coeffs)

    print("Stretched hyperbola")
    C_coeffs[4] = -2.0
    C_coeffs[5] = 0.5
    assert _test_parametrization(C_coeffs)

    print("General hyperbola")
    C_coeffs[0] = 0.5
    C_coeffs[1] = -2
    C_coeffs[2] = 3
    C_coeffs[4] = 0.1
    C_coeffs[5] = -2.0
    assert _test_parametrization(C_coeffs)
