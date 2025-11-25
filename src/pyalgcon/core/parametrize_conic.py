"""
Parameterize Conic.
Quite important for computing contours.
"""
import copy
import logging
import math

import numpy as np
import numpy.testing as npt

from pyalgcon.core.bivariate_quadratic_function import (
    evaluate_line, generate_quadratic_coordinate_affine_transformation_matrix,
    is_conic_standard_form, u_derivative_matrix, v_derivative_matrix)
from pyalgcon.core.common import (Matrix2x2f, Matrix3x2f,
                                  Matrix6x6r, PlanarPoint1d,
                                  Vector3f, Vector6f,
                                  compute_discriminant,
                                  float_equal)
from pyalgcon.core.conic import Conic, ConicType
from pyalgcon.core.convert_conic import \
    convert_conic_to_standard_form
from pyalgcon.core.interval import Interval
from pyalgcon.core.rational_function import RationalFunction

logger: logging.Logger = logging.getLogger(__name__)


# **************
# Helper Methods
# **************


def identify_standard_form_conic(conic_standard_form: Vector6f) -> ConicType:
    """
    Identify the type of conic C represented by C_standard form, which is assumed
    to be in standard form.
    """
    # Extract standard form coefficients
    c: float = conic_standard_form[0]
    b_1: float = conic_standard_form[1]
    b_2: float = conic_standard_form[2]
    sigma_1: float = conic_standard_form[4]
    sigma_2: float = conic_standard_form[5]
    det: float = sigma_1 * sigma_2

    # No quadratic terms: Conic is an empty set, line, or plane equation
    if float_equal(sigma_1, 0.0) and float_equal(sigma_2, 0.0):
        if (not float_equal(b_1, 0.0)) or (not float_equal(b_2, 0.0)):
            return ConicType.LINE
        if not float_equal(c, 0.0):
            return ConicType.EMPTY
        else:
            return ConicType.PLANE

    # Singular conic: sigma_1 z_1^2 + b_1 z_1 + b_2 z_2 + c = 0
    if float_equal(sigma_1, 0.0) or float_equal(sigma_2, 0.0):
        if float_equal(b_2, 0.0):
            return ConicType.PARALLEL_LINES
        else:
            return ConicType.PARABOLA

    # Nonsingular conic with no constant term: sigma_1 z_1^2 + sigma_2 z_2^2 = 0
    if float_equal(c, 0.0):
        if det > 0:
            return ConicType.POINT
        else:
            return ConicType.INTERSECTING_LINES

    # Nonsingular conic with constant term: sigma_1 z_1^2 + sigma_2 z_2^2 + c = 0
    if ((sigma_1 > 0) and (sigma_2 > 0) and (c > 0)):
        return ConicType.EMPTY
    elif ((sigma_1 < 0) and (sigma_2 < 0) and (c < 0)):
        return ConicType.EMPTY
    elif ((sigma_1 > 0) and (sigma_2 > 0) and (c < 0)):
        return ConicType.ELLIPSE
    elif ((sigma_1 < 0) and (sigma_2 < 0) and (c > 0)):
        return ConicType.ELLIPSE
    else:
        return ConicType.HYPERBOLA


def parametrize_ellipse(conic_standard_form: Vector6f, conics_ref: list[Conic]) -> None:
    """
    Parametrize ellipse conics for parametrize_standard_form_conic()
    """
    assert identify_conic(conic_standard_form) == ConicType.ELLIPSE

    # Get axes lengths k1, k2
    c: float = conic_standard_form[0]
    sigma_1: float = conic_standard_form[4]
    sigma_2: float = conic_standard_form[5]
    k1: float = math.sqrt(abs(c / sigma_1))
    k2: float = math.sqrt(abs(c / sigma_2))

    # Determine sign of c used for orientation
    sign: float = np.sign(c)

    # Parametrize ellipse as (1/(1 + t^2)) * [2 k1 t, k2 (1 - t^2)]
    P_coeffs: Matrix3x2f = np.zeros(shape=(3, 2), dtype=np.float64)
    Q_coeffs: Vector3f = np.array([1.0, 0.0, 1.0], dtype=np.float64)
    elliptic_basis_1: Vector3f = np.array([1, 0, -1], dtype=np.float64)
    elliptic_basis_2: Vector3f = np.array([0, 2, 0], dtype=np.float64)

    if sign < 0:
        P_coeffs[:, 0] = k1 * elliptic_basis_1
        P_coeffs[:, 1] = k2 * elliptic_basis_2
    else:
        P_coeffs[:, 0] = -k1 * elliptic_basis_1
        P_coeffs[:, 1] = k2 * elliptic_basis_2

    # Parametrize upper half over finite interval [-1,1]
    semi_ellipse_upper = Conic(ConicType.ELLIPSE, np.copy(P_coeffs), np.copy(Q_coeffs))
    semi_ellipse_upper.domain.set_lower_bound(-1.0, False)
    semi_ellipse_upper.domain.set_upper_bound(1.0, False)
    conics_ref.append(semi_ellipse_upper)

    # Parametrize lower half (also with closed endpoints)
    semi_ellipse_lower = Conic(ConicType.ELLIPSE, -np.copy(P_coeffs), np.copy(Q_coeffs))
    semi_ellipse_lower.domain.set_lower_bound(-1.0, False)
    semi_ellipse_lower.domain.set_upper_bound(1.0,  False)
    conics_ref.append(semi_ellipse_lower)


def parametrize_hyperbola(conic_standard_form: Vector6f, conics_ref: list[Conic]) -> None:
    """
    Parametrize hyperbola conics for parametrize_standard_form_conic()
    """
    # TODO: for some reason, the line below was commented out
    assert identify_conic(conic_standard_form) == ConicType.HYPERBOLA

    c: float = conic_standard_form[0]
    sigma_1: float = conic_standard_form[4]
    sigma_2: float = conic_standard_form[5]
    k1: float = math.sqrt(abs(c / sigma_1))
    k2: float = math.sqrt(abs(c / sigma_2))

    # FIXME (ASOC): Clean up
    # Parametrize ellipse as (1/(1 - t^2)) * [k1 (1 + t^2), 2 k2 t]
    P_coeffs: Matrix3x2f = np.zeros(shape=(3, 2), dtype=np.float64)
    Q_coeffs: Vector3f = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    hyperbolic_basis_1: Vector3f = np.array([1, 0, 1], dtype=np.float64)
    hyperbolic_basis_2: Vector3f = np.array([0, 2, 0], dtype=np.float64)
    sign: float = np.sign(c)
    if (sigma_1 < 0) and (sign < 0):
        P_coeffs[:, 0] = -k1 * hyperbolic_basis_2
        P_coeffs[:, 1] = k2 * hyperbolic_basis_1
    elif (sigma_1 > 0) and (sign < 0):
        P_coeffs[:, 0] = k1 * hyperbolic_basis_1
        P_coeffs[:, 1] = k2 * hyperbolic_basis_2
    elif (sigma_1 < 0) and (sign > 0):
        P_coeffs[:, 0] = k1 * hyperbolic_basis_1
        P_coeffs[:, 1] = -k2 * hyperbolic_basis_2
    elif (sigma_1 > 0) and (sign > 0):
        P_coeffs[:, 0] = k1 * hyperbolic_basis_2
        P_coeffs[:, 1] = k2 * hyperbolic_basis_1

    # Parametrize one branch over finite interval (-1,1)
    hyperbola_branch = Conic(ConicType.HYPERBOLA, np.copy(P_coeffs), np.copy(Q_coeffs))
    hyperbola_branch.domain.set_lower_bound(-1.0, True)
    hyperbola_branch.domain.set_upper_bound(1.0, True)
    conics_ref.append(hyperbola_branch)

    # Parametrize other branch
    hyperbola_branch_2 = Conic(ConicType.HYPERBOLA, -np.copy(P_coeffs), np.copy(Q_coeffs))
    hyperbola_branch_2.domain.set_lower_bound(-1.0, True)
    hyperbola_branch_2.domain.set_upper_bound(1.0, True)
    conics_ref.append(hyperbola_branch_2)


def parametrize_parabola(conic_standard_form: Vector6f) -> list[Conic]:
    """
    Parametrize parabola conics for parametrize_standard_form_conic()
    """
    assert identify_conic(conic_standard_form) == ConicType.INTERSECTING_LINES

    # Get parabolic coefficient
    sigma: float = conic_standard_form[4]

    # Parametrize ellipse as (1/(1 - t^2)) * [k1 (1 + t^2), 2 k2 t]
    P_coeffs: Matrix3x2f = np.array([
        [0.0, 0.0],
        [-1.0, 0.0],
        [0.0, -sigma]
    ], dtype=np.float64)
    Q_coeffs: Vector3f = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    conics: list[Conic] = [Conic(ConicType.PARABOLA, P_coeffs, Q_coeffs)]
    return conics


def parametrize_intersecting_lines(conic_standard_form: Vector6f) -> list[Conic]:
    """
    Add the intersecting lines as four rays with consistent orientation.
    """
    assert conic_standard_form.shape == (6, )
    result = identify_conic(conic_standard_form)
    assert result == ConicType.INTERSECTING_LINES
    conics: list[Conic] = []

    # Get absolute value of the slope of the lines
    sigma_1: float = conic_standard_form[4]
    sigma_2: float = conic_standard_form[5]

    # Get intervals for the positive and negative rays
    nonpositives = Interval()
    nonnegatives = Interval()
    nonpositives.set_upper_bound(0.0, False)
    nonnegatives.set_lower_bound(0.0, False)

    P_coeffs: Matrix3x2f = np.array([
        [0.0, 0.0],
        [math.sqrt(abs(sigma_2)), math.sqrt(abs(sigma_1))],
        [0.0, 0.0]
    ], dtype=np.float64)
    Q_coeffs: Vector3f = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    # FIXME: ensure that the order of the arguments to Conic is correct
    conics.append(
        Conic(ConicType.INTERSECTING_LINES,
              np.copy(P_coeffs), np.copy(Q_coeffs), copy.deepcopy(nonnegatives)))
    conics.append(
        Conic(ConicType.INTERSECTING_LINES,
              -1.0 * np.copy(P_coeffs), np.copy(Q_coeffs), copy.deepcopy(nonnegatives)))

    P_coeffs[1, 1] *= -1.0
    conics.append(
        Conic(ConicType.INTERSECTING_LINES,
              np.copy(P_coeffs), np.copy(Q_coeffs), copy.deepcopy(nonpositives)))
    conics.append(
        Conic(ConicType.INTERSECTING_LINES,
              -1.0 * np.copy(P_coeffs), np.copy(Q_coeffs), copy.deepcopy(nonpositives)))

    return conics


def parametrize_parallel_lines(conic_standard_form: Vector6f) -> list[Conic]:
    """
    Parametrize parallel lines conics for parametrize_standard_form_conic()
    """
    assert identify_conic(conic_standard_form) == ConicType.PARALLEL_LINES
    conics: list[Conic] = []

    # Get absolute value of the slope of the lines
    c: float = conic_standard_form[0]
    b1: float = conic_standard_form[1]
    sigma: float = conic_standard_form[4]
    discriminant: float = compute_discriminant(sigma, b1, c)
    x0: float = (0.5 / sigma) * (-b1 - math.sqrt(discriminant))
    x1: float = (0.5 / sigma) * (-b1 + math.sqrt(discriminant))

    # Parametrize ellipse as (1/(1 - t^2)) * [k1 (1 + t^2), 2 k2 t]
    P_coeffs: Matrix3x2f = np.array([
        [x0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0]
    ], dtype=np.float64)

    Q_coeffs: Vector3f = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    conics.append(Conic(ConicType.PARALLEL_LINES, np.copy(P_coeffs), np.copy(Q_coeffs)))
    P_coeffs[0, 0] = x1
    P_coeffs[1, 1] = -1.0
    conics.append(Conic(ConicType.PARALLEL_LINES, np.copy(P_coeffs), np.copy(Q_coeffs)))

    return conics


def parametrize_line() -> list[Conic]:
    """
    Parametrize line conics for parametrize_standard_form_conic()
    """
    conics: list[Conic] = []

    # All lines are x = 0 in standard form
    P_coeffs: Matrix3x2f = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0]], dtype=np.float64)
    Q_coeffs: Vector3f = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    conics.append(Conic(ConicType.LINE, P_coeffs, Q_coeffs))

    return conics


def parametrize_standard_form_conic(conic_standard_form: Vector6f) -> list[Conic]:
    """
    Parametrizes standard conic form
    """
    assert is_conic_standard_form(conic_standard_form)
    conics: list[Conic] = []

    # Parametrize conic based on type
    conic_type: ConicType = identify_standard_form_conic(conic_standard_form)
    assert conic_type == identify_standard_form_conic(-1 * conic_standard_form)

    if conic_type == ConicType.ELLIPSE:
        logger.info("Parametrizing ellipse")
        parametrize_ellipse(conic_standard_form, conics)
    elif conic_type == ConicType.HYPERBOLA:
        logger.info("Parametrizing hyperbola")
        parametrize_hyperbola(conic_standard_form, conics)
    elif conic_type == ConicType.PARABOLA:
        logger.info("Parametrizing parabola")
        conics = parametrize_parabola(conic_standard_form)
    elif conic_type == ConicType.PARALLEL_LINES:
        logger.info("Parametrizing parallel lines")
        conics = parametrize_parallel_lines(conic_standard_form)
    elif conic_type == ConicType.INTERSECTING_LINES:
        logger.info("Parametrizing intersecting lines")
        conics = parametrize_intersecting_lines(conic_standard_form)
    elif conic_type == ConicType.LINE:
        logger.info("Parametrizing line")
        conics = parametrize_line()
    elif conic_type == ConicType.POINT:
        logger.info("Skipping degenerate point")
    elif conic_type == ConicType.EMPTY:
        logger.info("Skipping degenerate empty set")
    elif conic_type == ConicType.PLANE:
        logger.error("Entire plane is the solution")
        raise ValueError("Entire plane is the solution")
    elif conic_type == ConicType.ERROR:
        logger.error("Error in conic")
        raise ValueError("Error in conic")
    else:
        logger.error("Unknown conic")
        raise ValueError("Unknown conic")

    return conics


def check_standard_form(conic_coeffs: Vector6f,
                        conic_standard_form: Vector6f,
                        rotation: Matrix2x2f,
                        translation: PlanarPoint1d) -> None:
    """Checks if parameters are in standard form"""
    logger.info("Checking standard form %s for implicit function %s",
                conic_standard_form, conic_coeffs)

    change_of_basis_matrix: Matrix6x6r
    change_of_basis_matrix = generate_quadratic_coordinate_affine_transformation_matrix(rotation,
                                                                                        translation)
    test_conic_coeffs: Vector6f = change_of_basis_matrix @ conic_coeffs
    assert test_conic_coeffs.ndim == 1
    logger.info("Implicit function after rotation is %s",
                test_conic_coeffs)

    npt.assert_allclose(test_conic_coeffs, conic_standard_form, atol=1e-4)


def check_parametrized_conic(conic: Conic, conic_coeffs: Vector6f) -> bool:
    """
    Checking conic parametrization
    """
    logger.info("Checking parametrization for conic %s with implicit function %s",
                conic,
                conic_coeffs)

    pullback: RationalFunction  # <4, 1>
    pullback = conic.pullback_quadratic_function(1, conic_coeffs)
    logger.info("Pullback by implicit function is %s", pullback)

    return float_equal(np.linalg.norm(pullback.numerators), 0.0, 1e-4)


def check_orientation(conic: Conic, conic_coeffs: Vector6f) -> bool:
    """
    FIXME (ASOC): rename
    """
    t: float = 0.1
    if not conic.is_in_domain(t):
        t = -0.1

    logger.debug("Checking consistency for conic %s with implicit function %s at %s",
                 conic,
                 conic_coeffs,
                 t)
    tangent: RationalFunction  # <4, 2>
    tangent = conic.compute_derivative()
    assert (tangent.degree, tangent.dimension) == (4, 2)
    u_derivative: Vector3f = u_derivative_matrix() @ conic_coeffs
    assert u_derivative.shape == (3, )
    v_derivative: Vector3f = v_derivative_matrix() @ conic_coeffs

    # HACK: flattening potentially 2D array
    point: PlanarPoint1d = conic(t).flatten()
    assert point.shape == (2, )
    perp_u: float = evaluate_line(u_derivative, point)
    perp_v: float = evaluate_line(v_derivative, point)
    logger.debug("Contour gradient: [%s, %s]", perp_u, perp_v)
    # HACK: flattening potentially 2D array
    point_tangent: PlanarPoint1d = tangent(t).flatten()
    assert point_tangent.shape == (2, )
    tu: float = point_tangent[0]
    tv: float = point_tangent[1]
    logger.debug("Contour parametric tangent: [%s, %s]", tu, tv)

    return ((-tv * perp_u + tu * perp_v) < 0)


# ***************
# Primary Methods
# ***************

def parametrize_conic(conic_coeffs: Vector6f) -> list[Conic]:
    """
    Given an implicit quadratic function for a conic, parameterize the
    conic with explicit rational mappings.

    :param conic_coeffs: [in] implicit conic equation
    :return conics: segments of the parametrized conic
    """
    logger.info("Parametrizing conic with equation: %s", conic_coeffs)
    conics: list[Conic] = []

    # Get standard form with rotation and translation
    rotation: Matrix2x2f
    translation: PlanarPoint1d
    conic_standard_form: Vector6f
    conic_standard_form, rotation, translation = convert_conic_to_standard_form(conic_coeffs)
    check_standard_form(conic_coeffs, conic_standard_form, rotation, translation)

    # Get parameterization of conic
    standard_form_conics: list[Conic] = parametrize_standard_form_conic(conic_standard_form)
    # FIXME: check to see that modification of conic in the loop is reflected back inside
    # standard_form_conics
    for conic in standard_form_conics:
        assert check_parametrized_conic(conic, conic_standard_form)
        assert check_orientation(conic, conic_standard_form)

        conic.transform(rotation, translation)
        logger.debug("Parametrized conic:\n%s", conic)

        assert check_parametrized_conic(conic, conic_coeffs)
        assert check_orientation(conic, conic_coeffs)

        conics.append(conic)

    return conics


def identify_conic(conic_coeffs: Vector6f) -> ConicType:
    """
    Identify the type of conic C represented by conic_coeffs.

    :param conic_coeffs: [in] quadratic coefficients for the conic
    :return: type identifier of the conic
    """
    conic_standard_form: Vector6f
    rotation: Matrix2x2f
    translation: PlanarPoint1d
    conic_standard_form, rotation, translation = convert_conic_to_standard_form(conic_coeffs)
    return identify_standard_form_conic(conic_standard_form)


def parametrize_cone_patch_conic(conic_coeffs: Vector6f) -> list[Conic]:
    """
    Given an implicit quadratic function for a conic from a singular cone
    patch, parametrize the conic with explicit rational mappings.
    In this case, the conic can only be a point or intersecting lines.

    :param conic_coeffs: [in] implicit conic equation
    :return conics: segments of the parametrized conic
    """
    conics: list[Conic] = []
    logger.debug("Parametrizing conic from cone patch with equation: %s",
                 conic_coeffs)

    # Get standard form with rotation and translation
    conic_standard_form: Vector6f
    rotation: Matrix2x2f
    translation: PlanarPoint1d
    conic_standard_form, rotation, translation = convert_conic_to_standard_form(conic_coeffs)

    # Get singular values
    sigma_1: float = conic_standard_form[4]
    sigma_2: float = conic_standard_form[5]
    det: float = sigma_1 * sigma_2

    # If the determinant is positive, the contour is a point and can be skipped
    if (det > 0.0) or float_equal(det, 0.0):
        logger.debug("Skipping degenerate point contour in cone patch with determinant %s", det)
        return conics

    # Get parametrization of conic
    logger.debug("Parametrizing intersecting lines in cone patch")
    standard_form_conics: list[Conic] = parametrize_intersecting_lines(conic_standard_form)
    # FIXME: check to see that modification of conic in the loop is reflected back inside
    # standard_form_conics
    for conic in standard_form_conics:
        assert check_parametrized_conic(conic, conic_standard_form)
        assert check_orientation(conic, conic_standard_form)
        conic.transform(rotation, translation)
        if not check_parametrized_conic(conic, conic_coeffs):
            logger.error("Did not parametrize implicit cone patch conic %s with %s",
                         conic_coeffs,
                         conic)
        assert check_orientation(conic, conic_coeffs)
        logger.debug("Parametrized conic:\n%s", conic)
        conics.append(conic)

    return conics
