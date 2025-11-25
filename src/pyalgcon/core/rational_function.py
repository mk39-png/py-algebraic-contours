"""@package docstring
    Quotient of scalar or vector valued polynomial functions over an interval.
"""

import logging
from dataclasses import dataclass

import numpy as np
import polyscope

from pyalgcon.core.common import (
    COLS, MatrixXf, Vector1D, Vector2D, convert_nested_vector_to_matrix,
    convert_polylines_to_edges, interval_lerp, todo, unimplemented,
    unreachable)
from pyalgcon.core.interval import Interval
from pyalgcon.core.polynomial_function import (
    compute_polynomial_mapping_derivative, compute_polynomial_mapping_product,
    compute_polynomial_mapping_scalar_product, evaluate_polynomial,
    formatted_polynomial)

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class CurveDiscretizationParameters:
    """
    Parameters for generating curve discretization in the case of contour network.
    """
    num_samples: int = 5
    num_tangents_per_segment: int = 5


class RationalFunction:
    """
    Representation of a vector valued rational function f: R -> R^n.
    """
    # ************
    # Constructors
    # ************
    # FIXME: is there ever a scenario where the default RationalFunction constructor is NOT used?
    # AS in,

    def __init__(self,
                 degree: int,
                 dimension: int,
                 numerator_coeffs: np.ndarray | None = None,
                 denominator_coeffs: np.ndarray | None = None,
                 domain: Interval | None = None) -> None:
        """ 
        General constructor over given interval.
        ### Possible combinations include 
        Default constructor for 0 function R^n: 
        numerator_coeffs == None, denominator_coeffs = None, domain == None \n
        Constructor for vector polynomial: denominator_coeffs == None, domain == None \n
        General constructor over entire real line: domain == None \n
        General constructor over given interval: all arguments are NOT None \n

        :param degree (int): [in]
        :param dimension (int): [in]
        :param numerator_coeffs (np.ndarray): [in] coefficients of the numerator polynomial.
            numerator_coeffs shape (degree + 1, dimension)
        :param denominator_coeffs (np.ndarray): [in] coefficients of the denominator polynomial.
            denominator_coeffs shape (degree + 1, )
        :param domain (Interval): [in] domain interval for the mapping
        """
        self.m_degree: int = degree
        self.m_dimension: int = dimension
        self.m_numerator_coeffs: Vector2D
        self.m_denominator_coeffs: Vector1D
        self.m_domain: Interval
        # TODO: assert the shape for numerator_coeffs and denominator_coeffs

        if (degree is None) or (dimension is None):
            raise ValueError("degree and dimension cannot be None.")

        if numerator_coeffs is None:
            self.m_numerator_coeffs = np.zeros(
                shape=(degree+1, dimension), dtype='float64')
        else:
            self.m_numerator_coeffs = numerator_coeffs

        if denominator_coeffs is None:
            self.m_denominator_coeffs = np.zeros(
                shape=(degree+1, ), dtype='float64')
            self.m_denominator_coeffs[0] = 1.0
        else:
            self.m_denominator_coeffs = denominator_coeffs

        if domain is None:
            self.m_domain = Interval()
        else:
            self.m_domain = domain

        assert self.m_numerator_coeffs.shape == (degree + 1, dimension)
        assert self.m_denominator_coeffs.shape == (degree + 1, )
        assert self.__is_valid()

    # *******************
    # Getters and setters
    # *******************
    @property
    def degree(self) -> int:
        """
        Compute the degree of the polynomial mapping as the max of the degrees
        of the numerator and denominator degrees.
        :return: degree of the rational mapping
        """
        return self.m_degree

    @property
    def dimension(self) -> int:
        """
        Compute the dimension of the rational mapping.
        :return: dimension of the rational mapping
        """
        return self.m_dimension

    @property
    def domain(self) -> Interval:
        """Retrives domain of rational function"""
        return self.m_domain

    @property
    def numerator_coeffs(self) -> np.ndarray:
        """Retrives numerator coefficients"""
        return self.m_numerator_coeffs

    @property
    def denominator_coeffs(self) -> np.ndarray:
        """Retrieves denominator coefficients"""
        return self.m_denominator_coeffs

    # ***************
    # Utility Methods
    # ***************

    def compute_derivative(self) -> "RationalFunction":
        """
        Compute the derivative of the rational function, which is also a rational function,
        using the quotient rule.

        :return derivative (RationalFunction<2*degree, dimension>): [out] derivative rational 
            function.
        """
        # Compute the derivatives of the numerator and denominator polynomials
        logger.info("Taking derivative of rational function")
        logger.info("Numerator:\n%s", self.m_numerator_coeffs)
        logger.info("Denominator:\n%s", self.m_denominator_coeffs)
        numerator_deriv_coeffs: MatrixXf = compute_polynomial_mapping_derivative(
            self.m_degree, self.m_dimension, self.m_numerator_coeffs)
        assert numerator_deriv_coeffs.shape == (self.m_degree, self.m_dimension)

        denominator_deriv_coeffs: Vector1D = compute_polynomial_mapping_derivative(
            self.m_degree, 1, self.m_denominator_coeffs).flatten()
        assert denominator_deriv_coeffs.shape == (self.m_degree, )
        logger.info("Numerator derivative:\n%s", numerator_deriv_coeffs)
        logger.info("Denominator derivative:\n%s", denominator_deriv_coeffs)

        # FIXME (ASOC): 0 degree case?

        # Compute the derivative numerator and denominator from the quotient rule
        term_0: MatrixXf = compute_polynomial_mapping_scalar_product(self.m_degree,
                                                                     self.m_degree - 1,
                                                                     self.m_dimension,
                                                                     self.m_denominator_coeffs,
                                                                     numerator_deriv_coeffs)
        assert term_0.shape == (2 * self.m_degree, self.m_dimension)

        term_1: MatrixXf = compute_polynomial_mapping_scalar_product(self.m_degree - 1,
                                                                     self.m_degree,
                                                                     self.m_dimension,
                                                                     denominator_deriv_coeffs,
                                                                     self.m_numerator_coeffs)
        assert term_1.shape == (2 * self.m_degree, self.m_dimension)

        logger.info("First term: \n%s", term_0)
        logger.info("Second term: \n%s", term_1)

        num_coeffs: MatrixXf = np.zeros(shape=(2 * self.m_degree + 1, self.m_dimension),
                                        dtype=np.float64)
        # FIXME: something might go wrong with the slicing.
        num_coeffs[0:2 * self.m_degree, 0:self.m_dimension] = term_0 - term_1

        denom_coeffs: Vector1D = compute_polynomial_mapping_product(self.m_degree,
                                                                    self.m_degree,
                                                                    self.m_denominator_coeffs,
                                                                    self.m_denominator_coeffs)
        assert denom_coeffs.shape == (2 * self.m_degree + 1, )

        # Build the derivative
        derivative = RationalFunction(2 * self.m_degree,
                                      self.m_dimension,
                                      num_coeffs,
                                      denom_coeffs,
                                      self.m_domain)

        return derivative

    def apply_one_form(self, one_form: Vector1D) -> "RationalFunction":
        """
        Compose the rational mapping f: R -> R^n with a one form to obtain
        a rational scalar.

        :param one_form: [in] One form w: R^n -> R to apply to the rational mapping
        :return scalar_function: composed scalar rational function
        """
        assert one_form.shape == (self.dimension, )

        # Compute the scalar polynomial numerator coefficients
        numerator_coeffs = self.m_numerator_coeffs @ one_form
        assert numerator_coeffs.shape == (self.degree + 1, )

        # Create a scalar rational function with the same domain and denominator
        # but the new numerator
        scalar_function = RationalFunction(self.m_degree, 1,
                                           numerator_coeffs,
                                           self.m_denominator_coeffs,
                                           self.m_domain)

        return scalar_function

    def split_at_knot(self, knot: float) -> tuple["RationalFunction", "RationalFunction"]:
        """
        Split the rational function into two rational function at some knot in the domain.

        WARNING: Do not try to use this method for the Conic subclass since it does not include
        the "type" member variable

        Used by contour network.

        :param knot: [in] point in the domain to split the function at
        :return lower_segment: [out] rational function with lower domain
        :return upper_segment: [out] rational function with upper domain
        """
        #  Build lower segment
        t0: float = self.m_domain.lower_bound
        assert t0 <= knot

        # TODO: when  building interval, need to set is_upper_bound and is_lower_bound
        lower_domain = Interval(t0, knot)
        lower_segment = RationalFunction(self.m_degree,
                                         self.m_dimension,
                                         self.m_numerator_coeffs,
                                         self.m_denominator_coeffs,
                                         lower_domain)

        # Build upper segment
        t1: float = self.m_domain.upper_bound
        assert knot <= t1
        upper_domain = Interval(knot, t1)
        upper_segment = RationalFunction(self.m_degree,
                                         self.m_dimension,
                                         self.m_numerator_coeffs,
                                         self.m_denominator_coeffs,
                                         upper_domain)

        return lower_segment, upper_segment

    def sample_points(self, num_points: int) -> list[Vector1D]:
        """
        Sample points in the rational function. 

        NOTE:  sample_points() seen with PlanarPoint and SpatialVector.

        :param num_points: [in] number of points to sample
        :return points: [out] vector of sampled points.
        """
        # Get sample of the domain
        t_samples: list[float] = self.m_domain.sample_points(num_points)
        points: list[Vector1D] = []

        for i in range(num_points):
            evaluated_rational_function: Vector1D = self.evaluate(t_samples[i])
            assert evaluated_rational_function.ndim == 1
            points.append(evaluated_rational_function)

        assert len(points) == num_points
        return points

    def start_point(self) -> Vector1D:
        """
        Get the point at the start of the rational mapping curve.
        Used in projected_curve_network.py

        :return: curve start point in R^n.
        :rtype: np.ndarray of dimension (self.dimension, )
        """
        # Return the default constructor point if the domain is not bounded below
        if not self.domain.is_bounded_below():
            return np.zeros(shape=(self.dimension, ), dtype=np.float64)
            # return np.zeros(shape=(1, self.dimension), dtype=np.float64)

        t0: float = self.domain.lower_bound
        return self.evaluate(t0)

    def mid_point(self) -> Vector2D:
        """
        Get the point of the rational mapping curve sampled at the midpoint
        of the domain interval (or some interior point in an unbounded interval).

        :return: curve mid point in R^n (self.dimension, )
        """
        if not self.domain.is_bounded_below():
            return np.zeros(shape=(self.dimension, ), dtype=np.float64)
        if not self.domain.is_bounded_above():
            return np.zeros(shape=(self.dimension, ), dtype=np.float64)

        t0: float = self.domain.lower_bound
        t1: float = self.domain.upper_bound
        return self.evaluate((t0 + t1) / 2.0)

    def end_point(self) -> Vector1D:
        """
        Get the point at the end of the rational mapping curve.
        Used in projected_curve_network.py

        :return: curve end point in R^n
        :rtype: np.ndarray of dimension (self.dimension, )
        """
        # Return the default constructor point if the domain is not bounded below
        if not self.domain.is_bounded_above():
            return np.zeros(shape=(self.dimension, ), dtype=np.float64)

        t1: float = self.domain.upper_bound
        return self.evaluate(t1)

    def evaluate_normalized_coordinate(self, t: float) -> Vector1D:
        """
        Evaluate the function at an normalized parameter in [0, 1]

        :param t: [in] normalized coordinate
        :return point: rational function evaluated at normalized coordinate t
        """
        # Check if domain is bounded
        # FIXME: potential issue returning (0, 0) shape array...
        if not self.domain.is_bounded_below():
            return np.ndarray(shape=(0, 0))
        if not self.domain.is_bounded_above():
            return np.ndarray(shape=(0, 0))

        # Linearly interpolate the coordinate
        t0: float = self.domain.lower_bound
        t1: float = self.domain.upper_bound
        s: float = interval_lerp(0.0, 1.0, t0, t1, t)

        # Evaluate at the given domain coordinate
        point: Vector2D = self.evaluate(s)

        # HACK: flattening to 1D when evaluate should natively return 1D
        return point.flatten()

    def is_in_domain(self, t: float) -> bool:
        """
        Determine if a point is in the domain of the rational mapping.

        :return: true iff t is in the domain
        """
        return self.m_domain.contains(t)

    def is_in_domain_interior(self, t: float) -> bool:
        """
        Determine if a point is in the interior of the domain of the
        rational mapping.

        :return: true iff t is in the domain interior
        """
        return self.m_domain.is_in_interior(t)

    def discretize(self, curve_disc_params: CurveDiscretizationParameters
                   ) -> tuple[list[Vector2D], list[int]]:
        """
        Discretize the given rational curve as a polyline curve network

        :param[in] curve_disc_params: parameters for the curve discretization
        :return points: points of the curve network
        :return polyline: polyline indices of the curve network
        """
        # Write curves
        num_samples: int = curve_disc_params.num_samples
        points: list[Vector2D] = self.sample_points(num_samples)

        # Build polyline for the given curve
        polyline: list[int] = []
        for l, _ in enumerate(points):
            polyline.append(l)

        return points, polyline

    # TODO: this is where I need to interact with the Blender API since *that* is now my viewer.
    def add_curve_to_viewer(self, curve_name="rational_function_curve") -> None:
        """
        Add the rational function curve to the polyscope viewer.

        Note that this method only works for rational space curves.

        :param curve_name: [in] name to assign the curve in the viewer
        """
        if (self.m_dimension != 3):
            logger.error("Cannot view nonspatial curve")

        # Generate curve discretization
        curve_disc_params = CurveDiscretizationParameters()
        points: list[Vector2D]
        polyline: list[int]
        points, polyline = self.discretize(curve_disc_params)

        # Add curve mesh
        # FIXME: there was a warning in the function docstring below. check out
        points_mat: MatrixXf = convert_nested_vector_to_matrix(points)
        polylines: list[list[int]] = [polyline]
        edges: list[tuple[int, int]] = convert_polylines_to_edges(polylines)
        polyscope.init()
        rational_function_curve: polyscope.CurveNetwork = polyscope.register_curve_network(
            curve_name, points_mat, edges)
        rational_function_curve.set_radius(0.0025)

    def finite_difference_derivative(self) -> None:
        """
        @brief Compute the derivative at domain point t with finite differences
        with finite difference step size h.
        This method should only be used for validation; the derivative method is
        exact.
        @param[in] t: point to evaluate the derivative at
        @param[in] h: finite difference step size
        @return finite difference derivative
        """
        unimplemented()

    # *******************
    # Getters and setters
    # *******************
    @property
    def numerators(self) -> Vector2D:
        """Retrieves numerator of RationalFunction"""

        assert self.m_numerator_coeffs.shape == (
            self.m_degree + 1, self.m_dimension)
        return self.m_numerator_coeffs

    @numerators.setter
    def numerators(self, numerator: np.ndarray) -> None:
        """Sets numerator of RationalFunction"""

        assert numerator.shape == (self.m_degree + 1, self.m_dimension)
        self.m_numerator_coeffs = numerator

    @property
    def denominator(self) -> Vector2D:
        """
        Retrieves denominator of RationalFunction
        NOTE: denominator is 1-dimensional
        """

        assert self.m_denominator_coeffs.shape == (self.m_degree + 1, )
        return self.m_denominator_coeffs

    @denominator.setter
    def denominator(self, denominator: np.ndarray) -> None:
        """Sets denominator of RationalFunction"""
        assert denominator.shape == (self.m_degree + 1, 1)
        self.m_denominator_coeffs = denominator

    # TODO: then have the domain accessible.
    # TODO: do equivalent to "friend class Conic;"

    def __is_valid(self) -> bool:
        # Making sure that numerator is shape (n,) array
        # NOTE: m_numerator_coeffs can have multiple dimensions...
        # It's just the denominator that must be 1 dimensional....
        # if (self.m_numerator_coeffs.ndim != 1 or self.m_numerator_coeffs.ndim != 1):
        # return False

        # This ensures that we're still dealing with matrices.
        # Because numerator can be shape (n, m) and not (n, )
        if self.m_numerator_coeffs.shape[COLS] == 0:
            return False

        # Making sure that denominator is NOT empty.
        if self.m_denominator_coeffs.size == 0:
            return False

        return True

    # ******************************
    # Helper functions for operators
    # ******************************
    # NOTE: this is equivalent to operator() in C++ code
    def __call__(self, t: float) -> np.ndarray:
        """
        Evaluate the rational mapping at domain point t.

        Args:
            t (float): [in] domain point to evaluate at.

        Returns:
            evaluated point.
        """
        return self.evaluate(t)

    def evaluate(self, t: float) -> Vector1D:
        """
        Evaluate the function at a domain coordinate

        NOTE: Pt = np.ndarray(shape=(1, self.m_dimension)) \n
        NOTE: Qt = np.ndarray(shape=(1, 1))

        :param t: [in] coordinate
        :return point: rational function evaluated at coordinate t. Shape = (, self.dimension)
        """
        # NOTE: using evaluate_polynomial_mapping() rather than evaluate_polynomial() for cases where m_dimension > 1
        # NOTE: keep the modification by reference since that helps showcase what shape Pt and Qt should be.

        # FIXME: Wait a minute... why is numerator all 0s with test_unit_pullback_case?
        # FIXME: inheriting degree from
        Pt: np.ndarray = evaluate_polynomial(degree=self.m_degree,
                                             dimension=self.m_dimension,
                                             polynomial_coeffs_ref=self.m_numerator_coeffs,
                                             t=t)

        Qt: np.ndarray = evaluate_polynomial(degree=self.m_degree,
                                             dimension=1,
                                             polynomial_coeffs_ref=self.m_denominator_coeffs,
                                             t=t)

        assert Pt.shape == (self.m_dimension, )
        assert Qt.shape == (1, )

        point: np.ndarray = Pt / Qt[0]
        assert point.shape == (self.dimension, )
        return point

    # TODO: finish a lot of these things for PolynomialFunction and Interval classes

    def __repr__(self) -> str:
        rational_function_string: str = "1/("
        rational_function_string += (
            formatted_polynomial(self.m_degree, 1, self.m_denominator_coeffs, 17))
        rational_function_string += ") [\n  "

        # for each column in m_numerator_coeffs
        for i in range(self.m_numerator_coeffs.shape[COLS]):
            rational_function_string += (
                formatted_polynomial(self.m_degree, 1, self.m_numerator_coeffs[:, i], 17))
            rational_function_string += ",\n  "

        rational_function_string += "], t in " + self.m_domain.formatted_interval()

        return rational_function_string
