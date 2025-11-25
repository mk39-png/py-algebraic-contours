"""
project_curves.py

Given a rational spatial curve and view direction, project the curve to a planar curve
"""

import logging

import numpy as np

from pyalgcon.contour_network.validity import is_valid_frame
from pyalgcon.core.common import (Matrix3x3f, Matrix5x2f,
                                  Matrix5x3f, Vector5f)
from pyalgcon.core.rational_function import RationalFunction

logger: logging.Logger = logging.getLogger(__name__)


def _project_curve(spatial_curve: RationalFunction, frame: Matrix3x3f) -> RationalFunction:
    """
    Given a rational spatial curve and view direction, project the curve to a
    planar curve

    :param spatial_curve: [in] curve in R^3 to project (i.e. degree 4, dimension 3)
    :param frame: [in] 3x3 matrix defining the projection
    :return planar_curve: projected planar curve in R^2 (i.e. degree 4, dimension 2)
    """
    assert (spatial_curve.degree, spatial_curve.dimension) == (4, 3)
    assert is_valid_frame(frame)

    # Get coordinates for spatial curve
    logger.debug("Getting spatial curve coefficients")
    spatial_P_coeffs: Matrix5x3f = spatial_curve.numerators
    assert spatial_P_coeffs.shape == (5, 3)
    Q_coeffs: Vector5f = spatial_curve.denominator
    assert Q_coeffs.shape == (5, )

    # Build planar curve coefficients
    logger.debug("Getting planar curve coefficients")
    # FIXME: probably problem with shaping... as in we may not need to transpose.
    # shape (5, 1) = (5, 3) @ (1, 3).T --> (5, 3) @ (3, 1) --> (5, 1)
    # TODO: remove unnecessary transpose
    planar_P_coeffs: Matrix5x2f = np.column_stack([spatial_P_coeffs @ frame[0, :].T,
                                                   spatial_P_coeffs @ frame[1, :].T])
    assert planar_P_coeffs.shape == (5, 2)

    # Build planar curve
    planar_curve = RationalFunction(4, 2, planar_P_coeffs, Q_coeffs, spatial_curve.domain)

    return planar_curve


def project_curves(spatial_curves: list[RationalFunction],
                   frame: Matrix3x3f) -> list[RationalFunction]:
    """
    Given a list of rational spatial curves and view direction, project the
    curves to a list of planar curves.

    :param spatial_curves: [in] curves in R^3 to project (i.e. degree 4, dimension 3)
    :param frame: [in] 3x3 matrix defining the projection

    :return planar_curves: projected planar curves in R^2 (i.e. degree 4, dimension 2)
    """
    assert is_valid_frame(frame)

    num_curves: int = len(spatial_curves)
    planar_curves: list[RationalFunction] = []

    for i in range(num_curves):
        planar_curve: RationalFunction = _project_curve(spatial_curves[i], frame)
        assert (planar_curve.degree, planar_curve.dimension) == (4, 2)
        planar_curves.append(planar_curve)

    return planar_curves
