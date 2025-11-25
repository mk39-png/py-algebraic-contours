"""

"""
import copy
import logging
from enum import Enum

import numpy as np

from pyalgcon.core.bivariate_quadratic_function import \
    formatted_bivariate_quadratic_mapping
from pyalgcon.core.common import (COLS, Matrix2x2f,
                                  Matrix3x2f, Matrix6xNi,
                                  PlanarPoint1d, Vector1D,
                                  Vector2D, Vector3f)
from pyalgcon.core.interval import Interval
from pyalgcon.core.polynomial_function import (
    compute_polynomial_mapping_product, formatted_polynomial)
from pyalgcon.core.rational_function import RationalFunction

logger: logging.Logger = logging.getLogger(__name__)


class ConicType(Enum):
    ELLIPSE = 1
    HYPERBOLA = 2
    PARABOLA = 3
    PARALLEL_LINES = 4
    INTERSECTING_LINES = 5
    LINE = 6
    POINT = 7
    EMPTY = 8
    PLANE = 9
    ERROR = 10
    UNKNOWN = 11


class Conic(RationalFunction):
    """
    Explicit representation of a conic segment
    """

    # ************
    # Constructors
    # ************
    def __init__(self,
                 m_type=ConicType.UNKNOWN,
                 numerator_coeffs: Matrix3x2f | None = None,
                 denominator_coeffs: Vector3f | None = None,
                 domain: Interval | None = None) -> None:
        """
        Constructor for Conic class.
        NOTE: Conic has degree == 2 and dimension == 2.

        :param numerator_coeffs: shape (3, 2)
        :param denominator_coeffs: shape (3, ) 
        """
        self.__type: ConicType = m_type
        super().__init__(2, 2, numerator_coeffs, denominator_coeffs, domain)
        assert self.__is_valid()

    # ******
    # PUBLIC
    # ******
    @property
    def type(self) -> ConicType:
        """
        Get the type (e.g. hyperbola, line, etc.) of the conic.

        :return: type identifier
        """
        return self.__type

    def split_at_knot_conic(self, knot: float) -> tuple["Conic", "Conic"]:
        """
        Split the Conic into two Conic at some knotin the domain.

        Used by contour network.

        :param knot: [in] point in the domain to split the conic at
        :return lower_segment: [out] conic with lower domain
        :return upper_segment: [out] conic with upper domain
        """
        #  Build lower segment
        t0: float = self.domain.lower_bound
        assert t0 <= knot

        # TODO: when  building interval, need to set is_upper_bound and is_lower_bound
        lower_domain = Interval(t0, knot)
        lower_segment = Conic(numerator_coeffs=self.numerator_coeffs,
                              denominator_coeffs=self.denominator_coeffs,
                              domain=lower_domain)

        # Build upper segment
        t1: float = self.domain.upper_bound
        assert knot <= t1
        upper_domain = Interval(knot, t1)
        upper_segment = Conic(numerator_coeffs=self.numerator_coeffs,
                              denominator_coeffs=self.denominator_coeffs,
                              domain=upper_domain)

        return lower_segment, upper_segment

    def transform(self, rotation: Matrix2x2f, translation: PlanarPoint1d) -> None:
        """
        NOTE: assumes row vector points... somewhat.
        """
        assert rotation.shape == (2, 2)
        assert translation.shape == (2, )

        # HACK: forcing translation to 1D shape
        # FIXME: potentially bad C++ translation with 2D array now 1D
        assert translation.ndim == 1
        # assert translation.shape == (1, 2)

        # (3, 2) @ (2, 2) + (3, 1) @ (1, 2)
        P_rot_coeffs: Matrix3x2f = (self.numerators @ rotation +
                                    (self.denominator.reshape(3, 1) @ translation.reshape(1, 2)))
        assert P_rot_coeffs.shape == (3, 2)
        self.numerators = P_rot_coeffs

    def pullback_quadratic_function(self, dimension: int,
                                    F_coeffs_ref: Vector1D | Matrix6xNi) -> RationalFunction:
        """
        Compute the pulled back rational function numerator.
        """
        F_coeffs: Matrix6xNi = F_coeffs_ref.reshape((6, dimension))
        assert F_coeffs.shape == (6, dimension)
        assert F_coeffs.dtype == np.float64

        logger.debug("Pulling back conic by quadratic function %s",
                     formatted_bivariate_quadratic_mapping(dimension, F_coeffs))

        # Separate the individual polynomial coefficients from the rational function
        P_coeffs: Matrix3x2f = self.numerators
        u_coeffs: Vector3f = P_coeffs[:, 0]
        v_coeffs: Vector3f = P_coeffs[:, 1]
        Q_coeffs: Vector3f = self.denominator
        # Asserting to make sure we are getting the shape we want.
        assert u_coeffs.shape == (3, )
        assert v_coeffs.shape == (3, )
        assert Q_coeffs.shape == (3, )

        logger.debug("u function before pullback: (%s)/(%s)",
                     u_coeffs, Q_coeffs)

        logger.debug("v function before pullback: (%s)/(%s)",
                     v_coeffs, Q_coeffs)
        # formatted_polynomial(self.get_degree, self.get_dimension, u_coeffs),
        # formatted_polynomial(self.get_degree, self.get_dimension, Q_coeffs))

        # Compute (homogenized) polynomial coefficients for the quadratic terms
        QQ_coeffs: Vector1D = np.ndarray(shape=(5, ))
        Qu_coeffs: Vector1D = np.ndarray(shape=(5, ))
        Qv_coeffs: Vector1D = np.ndarray(shape=(5, ))
        uv_coeffs: Vector1D = np.ndarray(shape=(5, ))
        uu_coeffs: Vector1D = np.ndarray(shape=(5, ))
        vv_coeffs: Vector1D = np.ndarray(shape=(5, ))

        # TODO: change the  function below to reemove the dimension parameter... maybe...
        QQ_coeffs = compute_polynomial_mapping_product(
            2, 2, Q_coeffs, Q_coeffs)
        Qu_coeffs = compute_polynomial_mapping_product(
            2, 2, Q_coeffs, u_coeffs)
        Qv_coeffs = compute_polynomial_mapping_product(
            2, 2, Q_coeffs, v_coeffs)
        uv_coeffs = compute_polynomial_mapping_product(
            2, 2, u_coeffs, v_coeffs)
        uu_coeffs = compute_polynomial_mapping_product(
            2, 2, u_coeffs, u_coeffs)
        vv_coeffs = compute_polynomial_mapping_product(
            2, 2, v_coeffs, v_coeffs)

        # Combine quadratic monomial functions into a matrix
        # NOTE: need to flatten the NP matrices into vectors from (5,1) to (5,)
        # shape for broadcasting to work
        # NOTE: Also transposing with .T to be shape (5,6) rather than (6,5)
        monomial_coeffs = np.array([QQ_coeffs,
                                    Qu_coeffs,
                                    Qv_coeffs,
                                    uv_coeffs,
                                    uu_coeffs,
                                    vv_coeffs]).T
        assert monomial_coeffs.shape == (5, 6)
        logger.debug("Monomial coefficients matrix:\n%s", monomial_coeffs)

        # Compute the pulled back rational function numerator
        logger.debug("Quadratic coefficient matrix:\n%s", F_coeffs)
        pullback_coeffs = monomial_coeffs @ F_coeffs
        assert pullback_coeffs.shape == (5, dimension)

        logger.debug("Pullback numerator: %s",
                     pullback_coeffs)
        # formatted_polynomial(self.get_degree, self.get_dimension, pullback_coeffs))
        logger.debug("Pullback denominator: %s",
                     QQ_coeffs)
        # formatted_polynomial(self.get_degree, self.get_dimension, QQ_coeffs))

        # XXX: domain() is in the C++. Meanwhile, Python code here uses self.m_domain...
        # may cause problems.
        # XXX: specifically, just be wary of the getter that's used in the C++ code.
        # NOTE: below RationalFunction is made from interval
        pullback_function = RationalFunction(
            4, dimension, pullback_coeffs, QQ_coeffs, self.domain)

        # Redundant checks just to be extra safe
        assert pullback_function.degree == 4
        assert pullback_function.dimension == dimension
        return pullback_function

    def __is_valid(self) -> bool:
        if self.m_numerator_coeffs.shape[1] == 0:
            return False
        if self.m_denominator_coeffs.size == 0:
            return False

        return True

    def formatted_conic(self) -> str:
        """
        Return the string representation of the Conic object.
        """
        conic_string: str = "1/("
        conic_string += formatted_polynomial(2, 1, self.m_denominator_coeffs)
        conic_string += ") [\n  "

        # Going through rows of m_numerator_coeffs
        for i in range(self.m_numerator_coeffs.shape[COLS]):
            # conic_string += f"{self.m_numerator_coeffs[:, i]}"
            conic_string += formatted_polynomial(2, 1, self.m_numerator_coeffs[:, i])
            conic_string += ",\n  "

        conic_string += "], t in "
        conic_string += self.m_domain.formatted_interval()

        return conic_string

    def __repr__(self) -> str:
        return self.formatted_conic()
