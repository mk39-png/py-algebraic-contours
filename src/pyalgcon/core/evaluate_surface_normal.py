"""
Methods to evaluate normals to a quadratic surface with Zwart-Powell basis
coefficients.
"""
import numpy as np

from pyalgcon.core.bivariate_quadratic_function import (
    compute_quadratic_cross_product, u_derivative_matrix, v_derivative_matrix)
from pyalgcon.core.common import (Matrix3x3r, Matrix3x6r,
                                  Matrix6x3r)


def generate_quadratic_surface_normal_coeffs(surface_mapping_coeffs: Matrix6x3r) -> Matrix6x3r:
    """
    Compute the quadratic coefficients of the normal vector to a quadratic surface.

    :param surface_mapping_coeffs: [in] coefficients for the quadratic surface
    :return: Coefficients for the quadratic polynomial defining the normal vector
    on the surface
    """
    # Get directional derivatives
    D_u: Matrix3x6r = u_derivative_matrix()
    D_v: Matrix3x6r = v_derivative_matrix()
    u_derivative_coeffs: Matrix3x3r = D_u @ surface_mapping_coeffs
    v_derivative_coeffs: Matrix3x3r = D_v @ surface_mapping_coeffs
    assert D_u.shape == (3, 6)
    assert D_v.shape == (3, 6)
    assert u_derivative_coeffs.shape == (3, 3)
    assert v_derivative_coeffs.shape == (3, 3)

    # Compute normal from the cross product
    normal_mapping_coeffs: Matrix6x3r = compute_quadratic_cross_product(u_derivative_coeffs,
                                                                        v_derivative_coeffs)

    return normal_mapping_coeffs
