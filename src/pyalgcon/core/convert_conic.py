"""
File for conic conversion methods
"""

import logging
import math
from typing import Any

import numpy as np
import numpy.testing as npt

from pyalgcon.core.common import (Matrix2x2f, PlanarPoint1d, Vector2f,
                                  Vector6f, float_equal, todo)

logger: logging.Logger = logging.getLogger(__name__)


def compute_symmetric_matrix_eigen_decomposition(A: Matrix2x2f) -> tuple[Vector2f,
                                                                         Matrix2x2f]:
    """
    Given a symmetric 2x2 matrix A, compute the eigenvalues and a rotation matrix
    U of eigenvectors so that A = U * diag(eigenvalues) * U^T

    :param A: [in] symmetric matrix to decompose
    :return eigenvalues: length 2 vector of eigenvalues with the largest first
    :return rotation: rotation matrix such the the columns are the eigenvalues
    """
    assert float_equal(A[0, 1], A[1, 0])

    # Compute eigenvalues sigma_1, sigma_2
    # FIXME Remove pow everywhere trace^2 - 4 det
    discriminant: float = pow(A[0, 0] - A[1, 1], 2) + 4 * A[0, 1] * A[1, 0]
    sigma_1: float = 0.5 * (A[0, 0] + A[1, 1] + math.sqrt(discriminant))
    sigma_2: float = 0.5 * (A[0, 0] + A[1, 1] - math.sqrt(discriminant))
    assert sigma_1 >= sigma_2
    eigenvalues: Vector2f = np.array([sigma_1, sigma_2], dtype=np.float64)

    # Compute rotation matrix U such that A = U diag(sigma_1, sigma_2) U^T
    # First row of rotation matrix is the first eigenvector
    # FIXME Check if discriminant zero instead
    # Otherwise, if a00 < a11, use current formula, if not use other (with
    # A(1,1))
    eigenvector_1: PlanarPoint1d
    if not float_equal(A[0, 1], 0.0):
        eigenvector_1 = np.array([A[0, 1], sigma_1 - A[0, 0]])
        assert not float_equal(np.linalg.norm(eigenvector_1), 0.0)
        eigenvector_1 /= np.linalg.norm(eigenvector_1)
    # This can be removed
    elif A[0, 0] < A[1, 1]:
        eigenvector_1 = np.array([0, 1], dtype=np.float64)
    else:
        eigenvector_1 = np.array([1, 0], dtype=np.float64)

    # Second column of rotation matrix is the first eigenvector rotated 90
    # degrees
    eigenvector_2: PlanarPoint1d
    eigenvector_2 = np.array([-eigenvector_1[1], eigenvector_1[0]], dtype=np.float64)

    # Assemble rotation matrix
    rotation: Matrix2x2f = np.array(
        [eigenvector_1,
         eigenvector_2], dtype=np.float64)
    assert rotation.shape == (2, 2)

    # FIXME: potentially incorrect C++ translation
    npt.assert_allclose(
        A,
        rotation.T @ np.diag([eigenvalues[0], eigenvalues[1]]) @ rotation, atol=1e-5)
    assert float_equal(np.linalg.det(rotation), 1.0)

    return eigenvalues, rotation


def convert_conic_to_matrix_form(conic_coeffs: Vector6f) -> tuple[Matrix2x2f,
                                                                  Vector2f,
                                                                  float]:
    """
    Given a conic C represented by coefficients conic_coeffs corresponding to
    1, u, v, uv, u^2, v^2, express the quadratic equation in the form
    0.5 r^T A r + b^T r + c
    :param conic_coeffs: [in] coefficients for the conic C in terms of r
    :return A: quadratic terms symmetric matrix
    :return b: linear terms vector
    :return c: constant term
    """
    assert conic_coeffs.shape == (6, )

    #  Compute A
    A: Matrix2x2f = np.array([[2.0 * conic_coeffs[4], conic_coeffs[3]],
                              [conic_coeffs[3], 2.0 * conic_coeffs[5]]],
                             dtype=np.float64)

    #  Compute b
    b: Vector2f = np.array([conic_coeffs[1], conic_coeffs[2]], dtype=np.float64)

    #  Compute c
    c: float = conic_coeffs[0]

    return A, b, c


def convert_conic_to_standard_form(conic_coeffs: Vector6f) -> tuple[Vector6f,
                                                                    Matrix2x2f,
                                                                    PlanarPoint1d]:
    """
    Given coefficients conic_coeffs for a conic C, rotate and translate the
    conic so that it is centered at the origin and axis aligned. A point r in
    the original conic is mapped to U(r - r_0) in the standard form conic.

    :param conic_coeffs: [in] original coefficients for the conic C
    :return conic_standard_form: coefficients for the standard form conic C
    :return rotation: rotation U to convert to standard form
    :return translation: translation r_0 to convert to standard form.
    """
    conic_standard_form: Vector6f = np.zeros(shape=(6, ), dtype=np.float64)
    rotation: Matrix2x2f
    translation: PlanarPoint1d = np.zeros(shape=(2, ), dtype=np.float64)

    # Put contour coefficients in quadratic form (1/2)r^T A r + b^T r + c
    A: Matrix2x2f
    b: Vector2f
    c: float
    A, b, c = convert_conic_to_matrix_form(conic_coeffs)
    # FIXME Make sure XT A X is identity

    # Get rotation and singular values
    singular_values: Vector2f
    singular_values, rotation = compute_symmetric_matrix_eigen_decomposition(A)
    det: float = singular_values[0] * singular_values[1]

    if float_equal(singular_values[0], 0.0) or float_equal(singular_values[1], 0.0):
        # FIXME Add warning so we know this case is occurring
        logger.warning("Singular conic with det %s  and singular values %s , %s ",
                       det,
                       singular_values[0],
                       singular_values[1])

        # Ensure the y coordinate is singular
        if (float_equal(singular_values[0], 0.0)):
            # FIXME: potentially incompatible C++ translation
            singular_values[0], singular_values[1] = singular_values[1], singular_values[0]
            rotation[0, 0], rotation[1, 0] = rotation[1, 0], rotation[0, 0]
            rotation[0, 1], rotation[1, 1] = rotation[1, 1], rotation[0, 1]

            # Ensure the rotation is still orientation preserving
            rotation[1, :] *= -1
            assert float_equal(np.linalg.det(rotation), 1.0)

        # The conic is a line or plane if A = 0
        if float_equal(singular_values[0], 0.0):
            #  Normalize the equation
            # HACK: float wrapping to avoid Pylance error
            # FIXME: does this introduce any precision error?
            normalization_factor: float = np.linalg.norm(b)
            b /= normalization_factor
            c /= normalization_factor

            # Translate b to c
            translation = -c * b

            # Rotate e1 to the linear term b
            rotation[0, :] = b
            rotation[1, 0] = -b[1]
            rotation[1, 1] = b[0]
            assert (float_equal(np.linalg.det(rotation), 1.0))

            # The conic equation is just x = 0
            conic_standard_form[1] = 1.0
        else:
            conic_standard_form[4] = 0.5 * singular_values[0]
            Ub: Vector2f = rotation @ b
            assert Ub.shape == (2, )

            # Quadratic in a single variable
            if float_equal(Ub[1], 0.0):
                conic_standard_form[0] = c
                conic_standard_form[1] = Ub[0]
            # Translate a parabola so that its vertex is at the origin
            else:
                logger.debug("Parabola is %s  u^2 + %s  u + %s  v + %s ",
                             0.5 * singular_values[0],
                             Ub[0],
                             Ub[1],
                             c)
                translation[0] = -Ub[0] / singular_values[0]
                translation[1] = -(c - 0.5 * singular_values[0] * Ub[0] * Ub[0]) / (Ub[1])
                logger.debug("Using translation %s for parabola", translation)

                translation = translation @ rotation
                logger.debug("Rotating to translation %s", translation)
                conic_standard_form[2] = Ub[1]

            # Ensure nonzero quadratic term is positive
            if conic_standard_form[4] < 0.0:
                conic_standard_form *= -1.0
    # Nonsingular conics (ellipse, hyperbola, etc.)
    else:
        # Get translation r_0 = -A^{-1} b
        translation = -np.linalg.inv(A) @ b
        assert translation.shape == (2, )

        # Compute standard form coefficients for the conic
        # FIXME: potentially incompatible C++ translation below
        # shape below: (2, ) @ (2, 2) @ (1, 2) --> (2, ) ....
        conic_standard_form[0] = c - 0.5 * (translation @ A) @ translation.T
        conic_standard_form[4] = 0.5 * singular_values[0]
        conic_standard_form[5] = 0.5 * singular_values[1]
        # todo("double check the matmul above")

    logger.debug("Standard form: %s", conic_standard_form)
    logger.debug("Rotation:\n%s", rotation)
    logger.debug("Translation:\n%s", translation)

    return conic_standard_form, rotation, translation
