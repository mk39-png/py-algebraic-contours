"""
Representation of a line segment, which is a rational function
with degree 1 numerator and degree 0 denominator.
TODO: how is the degree 0 denominator possible?
"""

import logging

import numpy as np

from pyalgcon.core.bivariate_quadratic_function import \
    formatted_bivariate_linear_mapping
from pyalgcon.core.common import (Matrix3x2r, Vector2D,
                                  unreachable)
from pyalgcon.core.conic import Conic
from pyalgcon.core.interval import Interval
from pyalgcon.core.polynomial_function import \
    formatted_polynomial
from pyalgcon.core.rational_function import RationalFunction

logger: logging.Logger = logging.getLogger(__name__)


class LineSegment(Conic):
    # ************
    # Constructors
    # ************
    def __init__(self, numerator_coeffs: np.ndarray = None, input_domain: Interval = None) -> None:
        # TODO: call Conic super() constructor...

        # There's not much logic going on for the constructor in the C++ code, so just going to put them all here for simplicity.
        if (numerator_coeffs is None) and (input_domain is None):
            unreachable(
                "Attempted LineSegment constructions with no parameters.")

        if numerator_coeffs is not None:
            self.__init_conic_coefficients(numerator_coeffs)

        if input_domain is not None:
            self.m_domain: Interval = input_domain

        # NOTE: these numbers are inherited from Conic, which has these numbers as the default.
        # TODO: do I call the default constructor for Conic?
        self.m_degree = 2
        self.m_dimension = 2

    def pullback_linear_function(self, dimension: int,
                                 F_coeffs: np.ndarray) -> RationalFunction:
        """
        Pulles back line segment by linear function.
        Used in discretize.py.

        :param dimension: [in] dimension for use in parameters
        :param F_coeffs: [in]: shape (3, dimension)
        :return pullback_function: degree = 1, dimension = dimension

        """
        logger.info("Pulling back line segment by linear function %s",
                    formatted_bivariate_linear_mapping(dimension, F_coeffs))

        # Separate the individual polynomial coefficients from the rational
        # function
        # TODO: typedef Matrix2x2r
        P_coeffs = self.numerators
        u_coeffs = P_coeffs[:, 0]
        assert u_coeffs.shape == (3, 1)
        v_coeffs = P_coeffs[:, 1]
        assert v_coeffs.shape == (3, 1)
        Q_coeffs = np.array([1.0, 0.0])

        logger.info("u function before pullback: %s",
                    formatted_polynomial(3, 1, u_coeffs))
        logger.info("v function before pullback: %s",
                    formatted_polynomial(3, 1, v_coeffs))

        # Combine quadratic monomial functions into a matrix
        monomial_coeffs = np.zeros(shape=(2, 3))
        monomial_coeffs[0, 0] = 1.0
        monomial_coeffs[0, 1] = u_coeffs[0]
        monomial_coeffs[1, 1] = u_coeffs[1]
        monomial_coeffs[0, 2] = v_coeffs[0]
        monomial_coeffs[1, 2] = v_coeffs[1]
        logger.info("Monomial coefficient matrix:\n%s", monomial_coeffs)

        # Compute the pulled back rational function numerator
        logger.info("Linear coefficient matrix:\n%s", F_coeffs)
        pullback_coeffs = monomial_coeffs * F_coeffs
        assert pullback_coeffs.shape == (2, self.m_dimension)
        logger.info("Pullback function: %s",
                    formatted_polynomial(2, self.m_dimension, pullback_coeffs))

        pullback_function = RationalFunction(
            1, self.m_dimension, pullback_coeffs, Q_coeffs, self.m_domain)

        return pullback_function

    def __init_conic_coefficients(self, numerator_coeffs: np.ndarray):
        # Build conic numerator with trivial quadratic term
        # TODO: typedef matrix3x2r
        conic_numerator_coeffs: Matrix3x2r = np.ndarray(shape=(3, 2))

        # FIXME: blocking operators need correct translation
        # Set top two rows to numerator_coeff
        conic_numerator_coeffs[:2, :2] = numerator_coeffs
        # Set bottom row to 0s.
        conic_numerator_coeffs[2, :2] = 0

        # Build constant 1 denominator
        # XXX: conic_denominator_coeffs MUST be [3, 1] since it is used in discretize_patch_boundaries() loop, which
        # is then used in RationalFunction.sample_point(), which then called __evaluate(), which *then* relies on conic_denominator_coeffs
        # to have shape (m_degree + 1, 1) to work. Since LineSegment inherits from Conic, it has m_degree = 2 and m_dimension = 2
        conic_denominator_coeffs: Vector2D = np.array([[1.0], [0.0], [0.0]], dtype=np.float64)

        # Set conic coefficients
        self.m_numerator_coeffs = conic_numerator_coeffs
        self.m_denominator_coeffs = conic_denominator_coeffs

        # self.set_numerators(conic_numerator_coeffs)
        # self.set_denominator(conic_denominator_coeffs)
