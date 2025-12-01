"""
Testing parametrize conic
"""
import logging

import numpy as np

from pyalgcon.core.bivariate_quadratic_function import evaluate_quadratic
from pyalgcon.core.common import PlanarPoint1d, Vector6f, float_equal
from pyalgcon.core.conic import Conic, ConicType
from pyalgcon.core.parametrize_conic import identify_conic, parametrize_conic

logger: logging.Logger = logging.getLogger(__name__)


def _test_parametrization(C_coeffs: Vector6f) -> bool:
    """
    Utility method for testing
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
    """
    From original C++ code.
    """
    C_coeffs: Vector6f = np.array([1, 0, 0, 0, -1, -1], dtype=np.float64)

    logger.info("Circle")
    assert (identify_conic(C_coeffs) == ConicType.ELLIPSE)

    logger.info("Vertically stretched ellipse")
    C_coeffs[4] = -2
    assert (identify_conic(C_coeffs) == ConicType.ELLIPSE)

    logger.info("Horizontally stretched ellipse")
    C_coeffs[5] = -2
    assert (identify_conic(C_coeffs) == ConicType.ELLIPSE)

    logger.info("Large circle")
    C_coeffs = np.array([1, 0, 0, 0, -0.1, -0.1])
    assert (identify_conic(C_coeffs) == ConicType.ELLIPSE)

    logger.info("General standard form")
    C_coeffs = np.array([5, 0, 0, 0, -10, -0.1])
    assert (identify_conic(C_coeffs) == ConicType.ELLIPSE)

    logger.info("Negative standard form")
    C_coeffs = np.array([-5, 0, 0, 0, 10, 0.1])
    assert (identify_conic(C_coeffs) == ConicType.ELLIPSE)

    logger.info("Linear terms")
    C_coeffs = np.array([-5, -5, 5, 0, 10, 0.1])
    assert (identify_conic(C_coeffs) == ConicType.ELLIPSE)

    logger.info("General form")
    C_coeffs = np.array([-5, -5, 5, 1.9, 10, 0.1])
    assert (identify_conic(C_coeffs) == ConicType.ELLIPSE)


def test_hyperbola() -> None:
    """
    From original C++ code.
    """
    C_coeffs: Vector6f = np.array([1, 0, 0, 0, 1, -1], dtype=np.float64)

    logger.info("Vertically symmetric")
    assert identify_conic(C_coeffs) == ConicType.HYPERBOLA

    logger.info("Horizontally symmetric")
    C_coeffs[4] = -1
    C_coeffs[5] = 1
    assert identify_conic(C_coeffs) == ConicType.HYPERBOLA

    # FIXME case below not working
    logger.info("Vertically stretched hyperbola")
    C_coeffs[4] = 2
    assert identify_conic(C_coeffs) == ConicType.HYPERBOLA

    # FIXME case below not working
    logger.info("Horizontally stretched hyperbola")
    C_coeffs[5] = -2
    assert identify_conic(C_coeffs) == ConicType.HYPERBOLA

    logger.info("General standard form")
    C_coeffs[0] = 5
    C_coeffs[4] = -10
    C_coeffs[5] = 0.1
    assert identify_conic(C_coeffs) == ConicType.HYPERBOLA

    logger.info("Negative standard form")
    C_coeffs[0] = -5
    C_coeffs[4] = 10
    C_coeffs[5] = -0.1
    assert identify_conic(C_coeffs) == ConicType.HYPERBOLA

    logger.info("Linear terms")
    C_coeffs[0] = -5
    C_coeffs[1] = -5
    C_coeffs[2] = 5
    C_coeffs[4] = -10
    C_coeffs[5] = 0.1
    assert identify_conic(C_coeffs) == ConicType.HYPERBOLA

    logger.info("General form")
    C_coeffs[0] = -5
    C_coeffs[1] = -5
    C_coeffs[2] = 5
    C_coeffs[3] = 2.1
    C_coeffs[4] = 10
    C_coeffs[5] = 0.1
    assert identify_conic(C_coeffs) == ConicType.HYPERBOLA


def test_ellipse_parametrized() -> None:
    """
    From original C++ code.
    """
    C_coeffs: Vector6f = np.zeros(shape=(6, ), dtype=np.float64)
    C_coeffs[0] = -1.0
    C_coeffs[4] = 1.0
    C_coeffs[5] = 1.0

    logger.info("Unit circle")
    assert _test_parametrization(C_coeffs)

    logger.info("Small circle")
    C_coeffs[4] = 2
    C_coeffs[5] = 2
    assert _test_parametrization(C_coeffs)

    logger.info("General circle")
    C_coeffs[0] = -0.5
    C_coeffs[1] = -2
    C_coeffs[2] = 3
    C_coeffs[4] = 2
    C_coeffs[5] = 2
    assert _test_parametrization(C_coeffs)

    logger.info("General ellipse")
    C_coeffs[0] = -0.5
    C_coeffs[1] = -2
    C_coeffs[2] = 3
    C_coeffs[4] = 0.1
    C_coeffs[5] = 2
    assert _test_parametrization(C_coeffs)


def test_hyperbola_parametrized() -> None:
    """
    From original C++ code
    """
    C_coeffs: Vector6f = np.zeros(shape=(6, ), dtype=np.float64)
    C_coeffs[0] = -1.0
    C_coeffs[4] = 1.0
    C_coeffs[5] = -1.0

    logger.info("Unit hyperbola")
    assert _test_parametrization(C_coeffs)

    logger.info("Translated hyperbola")
    C_coeffs[1] = -2
    C_coeffs[2] = 3
    assert _test_parametrization(C_coeffs)

    logger.info("Stretched hyperbola")
    C_coeffs[4] = -2.0
    C_coeffs[5] = 0.5
    assert _test_parametrization(C_coeffs)

    logger.info("General hyperbola")
    C_coeffs[0] = 0.5
    C_coeffs[1] = -2
    C_coeffs[2] = 3
    C_coeffs[4] = 0.1
    C_coeffs[5] = -2.0
    assert _test_parametrization(C_coeffs)
