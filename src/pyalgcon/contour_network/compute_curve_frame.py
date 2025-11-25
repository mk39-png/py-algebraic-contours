"""
compute_curve_frame.py

Methods to compute a curve aligned frame for quadratic and spline surfaces.
"""

import logging

from pyalgcon.core.common import (Matrix5x3f, Matrix6x3f,
                                  Matrix9x3f, Matrix13x3f,
                                  Vector5f, Vector9f,
                                  Vector13f)
from pyalgcon.core.conic import Conic
from pyalgcon.core.polynomial_function import (
    compute_polynomial_mapping_cross_product,
    compute_polynomial_mapping_product)
from pyalgcon.core.rational_function import RationalFunction
from pyalgcon.quadratic_spline_surface.quadratic_spline_surface_patch import \
    QuadraticSplineSurfacePatch

logger: logging.Logger = logging.getLogger(__name__)

# ****************
# Helper Methods
# ****************


def _compute_quadratic_surface_curve_tangent(surface_mapping_coeffs: Matrix6x3f,
                                             domain_curve: Conic) -> RationalFunction:
    """
    Compute tangent of a curve on a quadratic surface. degree = 8, dimension = 3
    """
    # Lift the domain curve to a surface curve
    logger.info("Computing quadratic surface curve tangent")
    logger.info("Domain curve: %s", domain_curve)
    surface_curve: RationalFunction  # degree = 4, dimension = 3
    surface_curve = domain_curve.pullback_quadratic_function(3, surface_mapping_coeffs)
    assert surface_curve.degree == 4
    assert surface_curve.dimension == 3
    logger.info("Surface curve: %s", surface_curve)

    # Compute the tangent of the surface curve directly
    surface_curve_tangent: RationalFunction = surface_curve.compute_derivative()
    assert (surface_curve_tangent.degree, surface_curve_tangent.dimension) == (8, 3)

    return surface_curve_tangent


def _compute_quadratic_surface_curve_normal(normal_mapping_coeffs: Matrix6x3f,
                                            domain_curve: Conic) -> RationalFunction:
    """
    Compute surface normal along a curve on a quadratic surface. degree = 4, dimension = 3
    """
    surface_curve_normal: RationalFunction
    surface_curve_normal = domain_curve.pullback_quadratic_function(3, normal_mapping_coeffs)
    assert (surface_curve_normal.degree, surface_curve_normal.dimension) == (4, 3)
    return surface_curve_normal


def compute_surface_curve_tangent_normal(surface_curve_tangent: RationalFunction,
                                         surface_curve_normal: RationalFunction
                                         ) -> RationalFunction:
    """
    General method to compute the surface curve tangent normal from the tangent and normal
    :param surface_curve_tangent: degree = 8, dimension = 3
    :param surface_curve_normal: degree = 4, dimension = 3

    :return surface_curve_tangent_normal: degree = 12, dimension = 3
    """
    # Compute the cross product of the (vector valued) numerators
    tangent_numerators: Matrix9x3f = surface_curve_tangent.numerators
    normal_numerators: Matrix5x3f = surface_curve_normal.numerators
    tangent_normal_numerators: Matrix13x3f = compute_polynomial_mapping_cross_product(
        8, 4, tangent_numerators, normal_numerators)

    # Checking shapes
    assert tangent_numerators.shape == (9, 3)
    assert normal_numerators.shape == (5, 3)
    assert tangent_normal_numerators.shape == (13, 3)

    # Compute the scalar product of the denominators
    tangent_denominator: Vector9f = surface_curve_tangent.denominator
    normal_denominator: Vector5f = surface_curve_normal.denominator
    tangent_normal_denominator: Vector13f = compute_polynomial_mapping_product(
        8, 4, tangent_denominator, normal_denominator)

    # Checking shapes
    assert tangent_denominator.shape == (9, )
    assert normal_denominator.shape == (5, )
    assert tangent_normal_denominator.shape == (13, )

    # Build rational function from the numerators, denominators, and the common domain
    surface_curve_tangent_normal: RationalFunction = RationalFunction(
        12, 3, tangent_normal_numerators, tangent_normal_denominator, surface_curve_tangent.domain)
    assert (surface_curve_tangent_normal.degree, surface_curve_tangent_normal.dimension) == (12, 3)

    return surface_curve_tangent_normal

# ****************
# Primary Methods
# ****************


def compute_quadratic_surface_curve_frame(surface_mapping_coeffs: Matrix6x3f,
                                          normal_mapping_coeffs: Matrix6x3f,
                                          domain_curve_segment: Conic) -> tuple[RationalFunction,
                                                                                RationalFunction,
                                                                                RationalFunction]:
    """
    Compute the frame aligned to a curve on a quadratic surface.

    :param surface_mapping_coeffs: [in] coefficients for the quadratic surface
    :param normal_mapping_coeffs:  [in] coefficients for the quadratic surface normal
    :param domain_curve_segment:   [in] curve segment in the parametric domain of the surface

    :return surface_curve_tangent: function for the tangent on the curve.
        surface_curve_tangent with degree 8, dimension 3
    :return surface_curve_normal: function for the surface normal on the curve.
        surface_curve_normal with degree 4, dimension 3
    :return surface_curve_tangent_normal: function for the tangent normal on the curve.
        surface_curve_tangent_normal with degree 12, dimension 3
    """
    surface_curve_tangent: RationalFunction = _compute_quadratic_surface_curve_tangent(
        surface_mapping_coeffs, domain_curve_segment)
    surface_curve_normal: RationalFunction = _compute_quadratic_surface_curve_normal(
        normal_mapping_coeffs, domain_curve_segment)
    surface_curve_tangent_normal: RationalFunction = compute_surface_curve_tangent_normal(
        surface_curve_tangent, surface_curve_normal)

    return surface_curve_tangent, surface_curve_normal, surface_curve_tangent_normal


def compute_spline_surface_patch_curve_frame(spline_surface_patch: QuadraticSplineSurfacePatch,
                                             domain_curve_segment: Conic
                                             ) -> tuple[RationalFunction,
                                                        RationalFunction,
                                                        RationalFunction]:
    """
    Compute the frame aligned to a given curve on a quadratic spline surface patch.
    :param spline_surface_patch: [in] quadratic spline surface patch
    :param domain_curve_segment: [in] curve segment in the parametric domain of the surface

    :return surface_curve_tangent: function for the tangent on the curve.
        surface_curve_tangent with degree 8, dimension 3
    :return surface_curve_normal: function for the surface normal on the curve.
        surface_curve_normal with degree 4, dimension 3
    :return surface_curve_tangent_normal: function for the tangent normal on the curve.
        surface_curve_tangent_normal with degree 12, dimension 3
    """
    logger.debug("Computing spline surface patch curve frame")
    #  Get surface and normal mappings for the given patch
    surface_mapping_coeffs: Matrix6x3f = spline_surface_patch.surface_mapping
    normal_mapping_coeffs: Matrix6x3f = spline_surface_patch.normal_mapping

    # Compute frame for quadratic surface patch
    surface_curve_tangent: RationalFunction  # degree 8, dimension = 3
    surface_curve_normal: RationalFunction  # degree 4, dimension = 3
    surface_curve_tangent_normal: RationalFunction  # degree 12, dimension = 3
    (surface_curve_tangent,
     surface_curve_normal,
     surface_curve_tangent_normal) = compute_quadratic_surface_curve_frame(surface_mapping_coeffs,
                                                                           normal_mapping_coeffs,
                                                                           domain_curve_segment)

    logger.debug("Surface curve tangent: %s", surface_curve_tangent)
    logger.debug("Surface curve normal: %s", surface_curve_normal)
    logger.debug("Surface curve tangent normal: %s",  surface_curve_tangent_normal)

    return surface_curve_tangent, surface_curve_normal, surface_curve_tangent_normal
