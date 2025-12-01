"""
This file basically test bivariate_quadratic_function.py
"""
import logging

import numpy as np

from pyalgcon.core.bivariate_quadratic_function import \
    generate_quadratic_monomials
from pyalgcon.core.common import Vector6f

logger: logging.Logger = logging.getLogger(__name__)


def test_patch_monomials() -> None:
    """
    From original C++ code
    """
    logger.info("Patch monomials are computed at (0,0)")
    w: Vector6f = generate_quadratic_monomials(np.array([0, 0]))
    assert w.shape == (6, )
    assert w[0] == 1.0
    assert w[1] == 0.0
    assert w[2] == 0.0
    assert w[3] == 0.0
    assert w[4] == 0.0
    assert w[5] == 0.0

    logger.info("Patch monomials are computed at (1,0)")
    w = generate_quadratic_monomials(np.array([1, 0]))
    assert w[0] == 1.0
    assert w[1] == 1.0
    assert w[2] == 0.0
    assert w[3] == 0.0
    assert w[4] == 1.0
    assert w[5] == 0.0

    logger.info("Patch monomials are computed at (0,1)")
    w = generate_quadratic_monomials(np.array([0, 1]))
    assert w.size == 6
    assert w[0] == 1.0
    assert w[1] == 0.0
    assert w[2] == 1.0
    assert w[3] == 0.0
    assert w[4] == 0.0
    assert w[5] == 1.0

    logger.info("Patch monomials are computed at (1,1)")
    w = generate_quadratic_monomials(np.array([1, 1]))
    assert w.size == 6
    assert w[0] == 1.0
    assert w[1] == 1.0
    assert w[2] == 1.0
    assert w[3] == 1.0
    assert w[4] == 1.0
    assert w[5] == 1.0

    logger.info("Patch monomials are computed at (0.5,1)")
    w = generate_quadratic_monomials(np.array([0.5, 1]))
    assert w.size == 6
    assert w[0] == 1.0
    assert w[1] == 0.5
    assert w[2] == 1.0
    assert w[3] == 0.5
    assert w[4] == 0.25
    assert w[5] == 1.0

    logger.info("Patch monomials are computed at (0.5,0.5)")
    w = generate_quadratic_monomials(np.array([0.5, 0.5]))
    assert w[0] == 1.0
    assert w[1] == 0.5
    assert w[2] == 0.5
    assert w[3] == 0.25
    assert w[4] == 0.25
    assert w[5] == 0.25
