"""
Methods to operate on bivariate quadratics represented by coefficient vectors.
"""
import numpy as np

from pyalgcon.core.common import (Matrix2x2f, Matrix3x3r,
                                  Matrix3x6r, Matrix6x3f,
                                  Matrix6x6r, MatrixXf,
                                  PlanarPoint, PlanarPoint1d,
                                  SpatialVector1d, Vector1D,
                                  Vector2D, Vector3f,
                                  Vector6f, cross_product,
                                  float_equal, unimplemented)
from pyalgcon.core.polynomial_function import formatted_term


def const_coeffs(dimension: int, quadratic_coeffs: np.ndarray, col: int) -> float:
    """
    Retrieves quadratic_coeffs const element.
    """
    assert quadratic_coeffs.shape == (6, dimension)
    assert col > dimension
    return quadratic_coeffs[0, col]


def u_coeffs(dimension: int, quadratic_coeffs: np.ndarray, col: int) -> float:
    """
    Retrieves quadratic_coeffs u element.
    """
    assert quadratic_coeffs.shape == (6, dimension)
    assert col >= 0
    assert col < dimension
    return quadratic_coeffs[1, col]


def v_coeffs(dimension: int, quadratic_coeffs: np.ndarray, col: int) -> float:
    """
    Retrieves quadratic_coeffs v element.
    """
    assert quadratic_coeffs.shape == (6, dimension)
    assert col >= 0
    assert col < dimension
    return quadratic_coeffs[2, col]


def uv_coeffs(dimension: int, quadratic_coeffs: np.ndarray, col: int) -> float:
    """
    Retrieves quadratic_coeffs uv element.
    """
    assert quadratic_coeffs.shape == (6, dimension)
    assert col >= 0
    assert col < dimension
    return quadratic_coeffs[3, col]


def uu_coeffs(dimension: int, quadratic_coeffs: np.ndarray, col: int) -> float:
    """
    Retrieves quadratic_coeffs uu element.
    """
    assert quadratic_coeffs.shape == (6, dimension)
    assert col >= 0
    assert col < dimension
    return quadratic_coeffs[4, col]


def vv_coeffs(dimension: int, quadratic_coeffs: np.ndarray, col: int) -> float:
    """
    Retrieves quadratic_coeffs vv element.
    """
    assert quadratic_coeffs.shape == (6, dimension)
    assert col >= 0
    assert col < dimension
    return quadratic_coeffs[5, col]


def generate_quadratic_monomials(domain_point: PlanarPoint1d) -> Vector6f:
    """
    Generate monomial variable terms for a quadratic in order [1, u, v, uv, uu, vv]

    :param domain_point: uv coordinates to generate the monomials for
    :returns: quadratic monomials of shape (6, )
    """
    assert domain_point.shape == (2, )
    u: float = domain_point[0]
    v: float = domain_point[1]

    w: Vector1D = np.array([1, u, v, u * v, u * u, v * v], dtype=np.float64)
    assert w.shape == (6, )
    return w


def generate_linear_monomials(domain_point: PlanarPoint1d) -> SpatialVector1d:
    """
    Generate monomial variable terms for a line in order [1, u, v]

    :param domain_point: uv coordinates to generate the monomials for
    :return: linear monomials of shape (3, )
    """
    assert domain_point.shape == (2, )
    u: float = domain_point[0]
    v: float = domain_point[1]

    w: Vector2D = np.array([1, u, v], dtype=np.float64)
    assert w.shape == (3, )
    return w


def evaluate_quadratic_mapping(quadratic_coeffs: Matrix6x3f,
                               domain_point: PlanarPoint) -> SpatialVector1d:
    """
    Evaluate a quadratic bivariate equation with scalar coefficients.
    Dimension can be greater than 1, as seen in quadratic_spline_surface_patch.py
    NOTE: This supports any dimension of quadratic coefficient, hence why it's used.

    :param dimension: [in] dimension of quadratic_coeffs
    :type dimension: int
    :param quadratic_coeffs: [in] quadratic coefficients in order [1, u, v, uv, uu, vv]. 
    quadratic_coeffs has shape (6, dimension)
    :param domain_point: uv coordinates to evaluate the quadratic at
    :return: quadratic_evaluation: quadratic function evaluation
    """

    assert quadratic_coeffs.shape == (6, 3)
    assert domain_point.shape == (2, )
    w: Vector6f = generate_quadratic_monomials(domain_point)
    assert w.shape == (6, )

    # shapes: (6, ) @ (6, 3)
    quadratic_evaluation: Vector3f = w @ quadratic_coeffs
    assert quadratic_evaluation.shape == (3, )

    return quadratic_evaluation


def evaluate_quadratic(quadratic_coeffs: Vector6f, domain_point: PlanarPoint1d) -> float:
    """
    Function not used.
    Aside from testing in test_parametrize_conic.py
    """
    w: Vector6f = generate_quadratic_monomials(domain_point)
    return w @ quadratic_coeffs


def evaluate_line(line_coeffs: Vector3f, domain_point: PlanarPoint1d) -> float:
    """
    Evaluate a linear bivariate equation with scalar coefficients.

    :param line_coeffs:  [in] line coefficients in order [1, u, v]
    :param domain_point: [in] uv coordinates to evaluate the line at
    :return: linear function evaluation
    """
    assert line_coeffs.shape == (3, )
    assert domain_point.shape == (2, )

    w: SpatialVector1d = generate_linear_monomials(domain_point)
    assert w.shape == (3, )

    return (w @ line_coeffs)


def compute_linear_product(L1_coeffs: Vector3f, L2_coeffs: Vector3f) -> Vector6f:
    """
    Compute the quadratic coefficients for the scalar product of two linear
    scalar functions V(u,v) and W(u,v) with coefficients in order [1, u, v]

    :param[in] V_coeffs: coefficients for the first linear vector function
    :param[in] W_coeffs: coefficients for the second linear vector function
    :return coefficients for the quadratic product function of shape (6, )
    """
    assert L1_coeffs.shape == (3, )
    assert L2_coeffs.shape == (3, )

    product_coeffs: Vector2D = np.array([
        (L1_coeffs[0] * L2_coeffs[0]),
        (L1_coeffs[1] * L2_coeffs[0] + L1_coeffs[0] * L2_coeffs[1]),
        (L1_coeffs[2] * L2_coeffs[0] + L1_coeffs[0] * L2_coeffs[2]),
        (L1_coeffs[1] * L2_coeffs[2] + L1_coeffs[2] * L2_coeffs[1]),
        (L1_coeffs[1] * L2_coeffs[1]),
        (L1_coeffs[2] * L2_coeffs[2])])
    assert product_coeffs.shape == (6, )

    return product_coeffs


def compute_quadratic_cross_product(V_coeffs: Matrix3x3r, W_coeffs: Matrix3x3r) -> Matrix6x3f:
    """
    Compute the quadratic coefficients for the cross product of two linear row
    vector functions V(u,v) and W(u,v) with coefficients in order [1, u, v]

    :param V_coeffs: [in] coefficients for the first linear vector function
    :param W_coeffs: [in] coefficients for the second linear vector function
    :return: coefficients for the quadratic cross function
    """
    assert V_coeffs.shape == (3, 3)
    assert W_coeffs.shape == (3, 3)

    N_coeffs: Matrix6x3f = np.array([
        # 1 coefficient
        cross_product(V_coeffs[0, :], W_coeffs[0, :]),  # row 0

        # u coefficient
        cross_product(V_coeffs[0, :], W_coeffs[1, :]) +  # row 1
        cross_product(V_coeffs[1, :], W_coeffs[0, :]),

        # v coefficient
        cross_product(V_coeffs[0, :], W_coeffs[2, :]) +  # row 2
        cross_product(V_coeffs[2, :], W_coeffs[0, :]),

        # uv coefficient
        cross_product(V_coeffs[1, :], W_coeffs[2, :]) +  # row 3
        cross_product(V_coeffs[2, :], W_coeffs[1, :]),

        # u^2 coefficient
        cross_product(V_coeffs[1, :], W_coeffs[1, :]),  # row 4

        # v^2 coefficient
        cross_product(V_coeffs[2, :], W_coeffs[2, :])  # row 5
    ], dtype=np.float64)

    assert N_coeffs.shape == (6, 3)
    return N_coeffs


@staticmethod
def u_derivative_matrix() -> Matrix3x6r:
    """
    Build matrix from quadratic coefficients to linear coefficients representing
    the derivative in the u direction

    @return u derivative matrix
    """
    D_u: Matrix3x6r = np.zeros(shape=(3, 6), dtype=np.float64)

    # Set nonzero elements explicitly
    D_u[0, 1] = 1
    D_u[1, 4] = 2
    D_u[2, 3] = 1

    assert D_u.shape == (3, 6)
    return D_u


@staticmethod
def v_derivative_matrix() -> Matrix3x6r:
    """
    Build matrix from quadratic coefficients to linear coefficients representing
    the derivative in the v direction

    @return v derivative matrix
    """
    D_v: Matrix3x6r = np.zeros(shape=(3, 6), dtype=np.float64)

    # Set non-zero elements explicitly
    D_v[0, 2] = 1
    D_v[1, 3] = 1
    D_v[2, 5] = 2
    assert D_v.shape == (3, 6)
    return D_v


def generate_bezier_to_monomial_matrix() -> Matrix6x6r:
    """
    Generate the matrix to go from Bezier control points to quadratic
    coefficients over the standard u + v <= 1 triangle in the positive quadrant.

    @param[out] change_of_basis_matrix: matrix going from bezier points to
    monomial coefficients
    """
    change_of_basis_matrix: Matrix6x6r = np.array([
        [0, 0, 0, 0, 0, 1],
        [0, 0, 2, 0, 0, -2],
        [0, 2, 0, 0, 0, -2],
        [2, -2, -2, 0, 0, 2],
        [0, 0, -2, 1, 0, 1],
        [0, -2, 0, 0, 1, 1]
    ], dtype=np.float64)

    assert change_of_basis_matrix.shape == (6, 6)
    return change_of_basis_matrix


def generate_monomial_to_bezier_matrix() -> Matrix6x6r:
    """
    /// Generate the matrix to go from quadratic coefficients over the standard u +
    /// v <= 1 triangle in the positive quadrant to Bezier control points.
    ///
    /// @param[out] change_of_basis_matrix: matrix going from monomial coefficients
    /// to bezier points
    """
    change_of_basis_matrix: Matrix6x6r = np.array([
        [1, 0.5, 0.5, 0.5, 0, 0],
        [1, 0, 0.5, 0, 0, 0],
        [1, 0.5, 0, 0, 0, 0],
        [1, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0]
    ], dtype=np.float64)
    assert change_of_basis_matrix.shape == (6, 6)
    return change_of_basis_matrix


# FIXME: This is not fully general.
def is_conic_standard_form(C_coeffs: Vector1D) -> bool:
    """
    Return true iff the conic with quadratic coefficients C_coeffs is in
    standard form with no mixed terms

    :param C_coeffs: [in] quadratic coefficients for the conic
    :return: true iff C_coeffs is a conic in standard form
    """
    assert C_coeffs.ndim == 1

    #  Mixed term must be zero
    if not float_equal(C_coeffs[3], 0.0):
        return False

    return True


def formatted_bivariate_quadratic_mapping(dimension: int,
                                          quadratic_coeffs_ref: np.ndarray,
                                          precision: int = 16) -> str:
    """
    Generate a human readable format of a quadratic mapping

    @param[in] quadratic_coeffs: quadratic coefficients in order [1, u, v, uv, uu, vv]
    @return formatted quadratic mapping
    """
    # Making sure to make vector into 2D vector for sake of iterating through it
    if quadratic_coeffs_ref.ndim == 1:
        quadratic_coeffs = quadratic_coeffs_ref.reshape((quadratic_coeffs_ref.size, dimension))
    else:
        quadratic_coeffs = quadratic_coeffs_ref

    quadratic_string: str = ""
    for i in range(dimension):
        quadratic_string += f"{quadratic_coeffs[0, i]}:.{precision}f"
        quadratic_string += formatted_term(
            quadratic_coeffs[1, i], "u", precision)
        quadratic_string += formatted_term(
            quadratic_coeffs[2, i], "v", precision)
        quadratic_string += formatted_term(
            quadratic_coeffs[3, i], "uv", precision)
        quadratic_string += formatted_term(
            quadratic_coeffs[4, i], "u^2", precision)
        quadratic_string += formatted_term(
            quadratic_coeffs[5, i], "v^2", precision)
        quadratic_string += "\n"

    return quadratic_string


def formatted_bivariate_linear_mapping(dimension: int,
                                       line_coeffs: np.ndarray,
                                       precision: int = 16) -> str:
    """
    Generate a human readable format of a linear mapping

    :param line_coeffs: [in] linear coefficients in order [1, u, v]
    :return formatted linear mapping
    """
    line_string: str = ""
    for i in range(dimension):
        line_string += f"{line_coeffs[0, i]}:.{precision}f"
        line_string += formatted_term(line_coeffs[1, i], "u", precision)
        line_string += formatted_term(line_coeffs[2, i], "v", precision)
        line_string += "\n"
    return line_string


def generate_quadratic_coordinate_affine_transformation_matrix(linear_transformation: Matrix2x2f,
                                                               translation: PlanarPoint1d
                                                               ) -> Matrix6x6r:
    """
    Given an affine transformation [u, v]^T = A*[u', v']^T + b of R^2, generate
    the change of basis matrix C for the bivariate quadratic monomial
    coefficients vector Q with respect to u, v so that Q' = C * Q is the
    coefficient vector for the bivariate quadratic monomials with respect to u',
    v'.

    @param[in] linear_transformation: linear part of the affine transformation
    @param[in] translation: translation part of the affine transformation
    @param[out] change_of_basis_matrix: change of coefficient basis matrix
    """
    assert linear_transformation.shape == (2, 2)
    assert translation.shape == (2, )

    # Get matrix information
    b11: float = linear_transformation[0, 0]
    b12: float = linear_transformation[0, 1]
    b21: float = linear_transformation[1, 0]
    b22: float = linear_transformation[1, 1]
    b1: float = translation[0]
    b2: float = translation[1]

    # Set matrix
    # TODO: check order that change_of_basis_matrix is being filled.
    change_of_basis_matrix: Matrix6x6r = np.array([
        [1.0, b1, b2, b1 * b2, b1 * b1, b2 * b2],
        [0.0, b11, b12, b11 * b2 + b12 * b1, 2 * b1 * b11, 2 * b2 * b12],
        [0.0, b21, b22, b22 * b1 + b21 * b2, 2 * b1 * b21, 2 * b2 * b22],
        [0.0, 0.0, 0.0, b11 * b22 + b21 * b12, 2 * b11 * b21, 2 * b12 * b22],
        [0.0, 0.0, 0.0, b11 * b12, b11 * b11, b12 * b12],
        [0.0, 0.0, 0.0, b21 * b22, b21 * b21, b22 * b22]],
        dtype=np.float64)

    assert change_of_basis_matrix.shape == (6, 6)
    return change_of_basis_matrix


def generate_quadratic_coordinate_translation_matrix(du: float, dv: float) -> Matrix6x6r:
    """
    Generate the matrix to transform quadratic monomial coefficients for
    coordinates (u, v) to coefficients for translated coordinates (u', v') = (u
    + du, v + dv).

    @param[in] du: change in u coordinate
    @param[in] dv: change in v coordinate
    @param[out] change_of_basis_matrix: change of coefficient basis matrix
    """
    # Build (inverse) translation and linear identity matrix
    translation: PlanarPoint1d = np.array([-du, -dv], dtype=np.float64)
    assert translation.shape == (2, )
    identity: Matrix2x2f = np.identity(2, dtype=np.float64)
    # TODO: redundant check below
    assert identity.shape == (2, 2)

    # Generate the translation transformation as a special case of an affine transformation
    change_of_basis_matrix: Matrix6x6r = generate_quadratic_coordinate_affine_transformation_matrix(
        identity, translation)
    return change_of_basis_matrix


def generate_quadratic_coordinate_barycentric_transformation_matrix(
    barycentric_transformation: Matrix3x3r
) -> Matrix6x6r:
    """
    Given an barycentric transformation [u, v, w]^T = A*[u', v', w']^T of RP^2
    with coordinates normalized so that u + v + w = u' + v' + w' = 1, generate
    the change of basis matrix C for the bivariate quadratic monomial
    coefficients vector Q with respect to u, v so that Q' = C * Q is the
    coefficient vector for the bivariate quadratic monomials with respect to u',
    v'.

    :param barycentric_transformation: [in] transformation of the coordinates
    :return change_of_basis_matrix: change of coefficient basis matrix
    """
    assert barycentric_transformation.shape == (3, 3)

    # Get relevant barycentric matrix components. Since the w coordinate is
    # discarded, we only use the first two rows of the barycentric transformation
    a11: float = barycentric_transformation[0, 0]
    a12: float = barycentric_transformation[0, 1]
    a13: float = barycentric_transformation[0, 2]
    a21: float = barycentric_transformation[1, 0]
    a22: float = barycentric_transformation[1, 1]
    a23: float = barycentric_transformation[1, 2]

    # Build affine transformation from barycentric transformation with w = 1 - u - v
    # TODO: double check that the shape is made as expected. As per ASOC code, that is.
    linear_transformation: Matrix2x2f = np.array(
        [[a11 - a13, a21 - a23], [a12 - a13, a22 - a23]], dtype=np.float64)
    assert linear_transformation.shape == (2, 2)
    translation: PlanarPoint1d = np.array([a13, a23], dtype=np.float64)
    assert translation.shape == (2, )

    # Get change of basis matrix from the affine transformation
    change_of_basis_matrix: Matrix6x6r = generate_quadratic_coordinate_affine_transformation_matrix(
        linear_transformation, translation)
    assert change_of_basis_matrix.shape == (6, 6)
    return change_of_basis_matrix


def generate_quadratic_coordinate_domain_triangle_normalization_matrix(v0: PlanarPoint1d,
                                                                       v1: PlanarPoint1d,
                                                                       v2: PlanarPoint1d
                                                                       ) -> Matrix6x6r:
    """
    of basis matrix C for the bivariate quadratic monomial coefficients vector
    Given vertex positions in R^2 for a domain triangle, generate the change
    Q with respect to u, v so that Q' = C * Q is the coefficient vector for the
    surface mapping over the triangle in the positive quadrant with u + v <= 1
    that has the same image as the surface mapping over the input domain.

    :param v0: [in] first domain triangle vertex position
    :param v1: [in] second domain triangle vertex position
    :param v2: [in] third domain triangle vertex position
    :param change_of_basis_matrix: [out] change of coefficient basis matrix
    """
    assert v0.shape == (2, )
    assert v1.shape == (2, )
    assert v2.shape == (2, )

    # Generate affine transformation mapping the standard triangle to the new
    # triangle
    linear_transformation: Matrix2x2f = np.array(
        [(v1 - v0),
         (v2 - v0)],
        dtype=np.float64)
    translation: PlanarPoint = v0
    assert linear_transformation.shape == (2, 2)

    # Get change of basis matrix from the affine transformation
    change_of_basis_matrix: Matrix6x6r = generate_quadratic_coordinate_affine_transformation_matrix(
        linear_transformation, translation)
    assert change_of_basis_matrix.shape == (6, 6)

    return change_of_basis_matrix


def generate_reparameterization() -> None:
    """
    Function not used.
    """
    unimplemented()
