"""@package docstring
    Quotient of scalar or vector valued polynomial functions over an interval.
"""

import logging
from dataclasses import dataclass

import numpy as np
import polyscope

from pyalgcon.core.common import (COLS, MatrixXf, Vector1D, Vector2D,
                                  convert_nested_vector_to_matrix,
                                  convert_polylines_to_edges, interval_lerp,
                                  unimplemented)
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
        self.__degree: int = degree
        self.__dimension: int = dimension
        self.__numerator_coeffs: Vector2D
        self.__denominator_coeffs: Vector1D
        self.__domain: Interval
        # TODO: assert the shape for numerator_coeffs and denominator_coeffs

        if (degree is None) or (dimension is None):
            raise ValueError("degree and dimension cannot be None.")

        if numerator_coeffs is None:
            self.__numerator_coeffs = np.zeros(
                shape=(degree+1, dimension), dtype='float64')
        else:
            self.__numerator_coeffs = numerator_coeffs

        if denominator_coeffs is None:
            self.__denominator_coeffs = np.zeros(
                shape=(degree+1, ), dtype='float64')
            self.__denominator_coeffs[0] = 1.0
        else:
            self.__denominator_coeffs = denominator_coeffs

        if domain is None:
            self.__domain = Interval()
        else:
            self.__domain = domain

        assert self.__numerator_coeffs.shape == (degree + 1, dimension)
        assert self.__denominator_coeffs.shape == (degree + 1, )
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
        return self.__degree

    @property
    def dimension(self) -> int:
        """
        Compute the dimension of the rational mapping.
        :return: dimension of the rational mapping
        """
        return self.__dimension

    @property
    def domain(self) -> Interval:
        """Retrives domain of rational function"""
        return self.__domain

    @property
    def numerator_coeffs(self) -> np.ndarray:
        """Retrives numerator coefficients"""
        return self.__numerator_coeffs

    @property
    def denominator_coeffs(self) -> np.ndarray:
        """Retrieves denominator coefficients"""
        return self.__denominator_coeffs

    @degree.setter
    def degree(self, value: int) -> None:
        """Sets degree"""
        assert value > 0
        self.__degree = value

    @dimension.setter
    def dimension(self, value: int) -> None:
        """Sets dimension"""
        assert value > 0
        self.__dimension = value

    @domain.setter
    def domain(self, domain: Interval) -> None:
        """Sets domain"""
        self.__domain = domain

    @numerator_coeffs.setter
    def numerator_coeffs(self, coeffs: np.ndarray) -> None:
        """Sets numerator coefficients (by deepcopy value)"""
        assert coeffs.shape == (self.__degree + 1, self.__dimension)
        self.__numerator_coeffs = np.copy(coeffs)

    @denominator_coeffs.setter
    def denominator_coeffs(self, coeffs: np.ndarray) -> None:
        """Sets numerator coefficients (by deepcopy value)"""
        assert coeffs.shape == (self.__degree + 1, )
        self.__denominator_coeffs = np.copy(coeffs)

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
        logger.debug("Taking derivative of rational function")
        logger.debug("Numerator:\n%s", self.__numerator_coeffs)
        logger.debug("Denominator:\n%s", self.__denominator_coeffs)
        numerator_deriv_coeffs: MatrixXf = compute_polynomial_mapping_derivative(
            self.__degree, self.__dimension, self.__numerator_coeffs)
        assert numerator_deriv_coeffs.shape == (self.__degree, self.__dimension)

        denominator_deriv_coeffs: Vector1D = compute_polynomial_mapping_derivative(
            self.__degree, 1, self.__denominator_coeffs).flatten()
        assert denominator_deriv_coeffs.shape == (self.__degree, )
        logger.debug("Numerator derivative:\n%s", numerator_deriv_coeffs)
        logger.debug("Denominator derivative:\n%s", denominator_deriv_coeffs)

        # FIXME (ASOC): 0 degree case?

        # Compute the derivative numerator and denominator from the quotient rule
        term_0: MatrixXf = compute_polynomial_mapping_scalar_product(self.__degree,
                                                                     self.__degree - 1,
                                                                     self.__dimension,
                                                                     self.__denominator_coeffs,
                                                                     numerator_deriv_coeffs)
        assert term_0.shape == (2 * self.__degree, self.__dimension)

        term_1: MatrixXf = compute_polynomial_mapping_scalar_product(self.__degree - 1,
                                                                     self.__degree,
                                                                     self.__dimension,
                                                                     denominator_deriv_coeffs,
                                                                     self.__numerator_coeffs)
        assert term_1.shape == (2 * self.__degree, self.__dimension)

        logger.debug("First term: \n%s", term_0)
        logger.debug("Second term: \n%s", term_1)

        num_coeffs: MatrixXf = np.zeros(shape=(2 * self.__degree + 1, self.__dimension),
                                        dtype=np.float64)
        # FIXME: something might go wrong with the slicing.
        num_coeffs[0:2 * self.__degree, 0:self.__dimension] = term_0 - term_1

        denom_coeffs: Vector1D = compute_polynomial_mapping_product(self.__degree,
                                                                    self.__degree,
                                                                    self.__denominator_coeffs,
                                                                    self.__denominator_coeffs)
        assert denom_coeffs.shape == (2 * self.__degree + 1, )

        # Build the derivative
        derivative = RationalFunction(2 * self.__degree,
                                      self.__dimension,
                                      num_coeffs,
                                      denom_coeffs,
                                      self.__domain)

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
        numerator_coeffs = self.__numerator_coeffs @ one_form
        assert numerator_coeffs.shape == (self.degree + 1, )

        # Create a scalar rational function with the same domain and denominator
        # but the new numerator
        scalar_function = RationalFunction(self.__degree, 1,
                                           numerator_coeffs,
                                           self.__denominator_coeffs,
                                           self.__domain)

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
        t0: float = self.__domain.lower_bound
        assert t0 <= knot

        # TODO: when  building interval, need to set is_upper_bound and is_lower_bound
        lower_domain = Interval(t0, knot)
        lower_segment = RationalFunction(self.__degree,
                                         self.__dimension,
                                         self.__numerator_coeffs,
                                         self.__denominator_coeffs,
                                         lower_domain)

        # Build upper segment
        t1: float = self.__domain.upper_bound
        assert knot <= t1
        upper_domain = Interval(knot, t1)
        upper_segment = RationalFunction(self.__degree,
                                         self.__dimension,
                                         self.__numerator_coeffs,
                                         self.__denominator_coeffs,
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
        t_samples: list[float] = self.__domain.sample_points(num_points)
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
        # NOTE: need to throw value error since ASOC code has this always bounded when calling
        #       evaluate_normalized_coordinate
        if not self.domain.is_bounded_below():
            raise ValueError(f"ATTEMPTED TO PASS IN UNBOUNDED DOMAIN {self.domain}")
        if not self.domain.is_bounded_above():
            raise ValueError(f"ATTEMPTED TO PASS IN UNBOUNDED DOMAIN {self.domain}")

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
        return self.__domain.contains(t)

    def is_in_domain_interior(self, t: float) -> bool:
        """
        Determine if a point is in the interior of the domain of the
        rational mapping.

        :return: true iff t is in the domain interior
        """
        return self.__domain.is_in_interior(t)

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
        if (self.__dimension != 3):
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

        assert self.__numerator_coeffs.shape == (
            self.__degree + 1, self.__dimension)
        return self.__numerator_coeffs

    @numerators.setter
    def numerators(self, numerator: np.ndarray) -> None:
        """Sets numerator of RationalFunction"""

        assert numerator.shape == (self.__degree + 1, self.__dimension)
        self.__numerator_coeffs = numerator

    @property
    def denominator(self) -> Vector2D:
        """
        Retrieves denominator of RationalFunction
        NOTE: denominator is 1-dimensional
        """
        assert self.__denominator_coeffs.shape == (self.__degree + 1, )
        return self.__denominator_coeffs

    @denominator.setter
    def denominator(self, denominator: np.ndarray) -> None:
        """Sets denominator of RationalFunction"""
        assert denominator.shape == (self.__degree + 1, 1)
        self.__denominator_coeffs = denominator

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
        if self.__numerator_coeffs.shape[COLS] == 0:
            return False

        # Making sure that denominator is NOT empty.
        if self.__denominator_coeffs.size == 0:
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
        Pt: np.ndarray = evaluate_polynomial(degree=self.__degree,
                                             dimension=self.__dimension,
                                             polynomial_coeffs_ref=self.__numerator_coeffs,
                                             t=t)

        Qt: np.ndarray = evaluate_polynomial(degree=self.__degree,
                                             dimension=1,
                                             polynomial_coeffs_ref=self.__denominator_coeffs,
                                             t=t)

        assert Pt.shape == (self.__dimension, )
        assert Qt.shape == (1, )

        point: np.ndarray = Pt / Qt[0]
        assert point.shape == (self.dimension, )
        return point

    # TODO: finish a lot of these things for PolynomialFunction and Interval classes

    def __repr__(self) -> str:
        rational_function_string: str = "1/("
        rational_function_string += (
            formatted_polynomial(self.__degree, 1, self.__denominator_coeffs, 17))
        rational_function_string += ") [\n  "

        # for each column in m_numerator_coeffs
        for i in range(self.__numerator_coeffs.shape[COLS]):
            rational_function_string += (
                formatted_polynomial(self.__degree, 1, self.__numerator_coeffs[:, i], 17))
            rational_function_string += ",\n  "

        rational_function_string += "], t in " + self.__domain.formatted_interval()

        return rational_function_string
