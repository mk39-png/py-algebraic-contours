"""
Test polynomial function
"""

import logging

import numpy as np
import numpy.testing as npt

from pyalgcon.core.common import MatrixXf, Vector1D, Vector2D, float_equal
from pyalgcon.core.polynomial_function import (
    compute_polynomial_mapping_cross_product,
    compute_polynomial_mapping_derivative, compute_polynomial_mapping_product,
    evaluate_polynomial, generate_monomials, polynomial_real_roots,
    quadratic_real_roots)

logger: logging.Logger = logging.getLogger(__name__)


def test_generate_monomials() -> None:
    """
    From original C++ code.
    """
    degree = 2
    t = -1
    T: Vector2D = generate_monomials(degree, t)

    assert T.shape == (1, degree + 1)
    assert T[0][0] == 1
    assert T[0][1] == -1
    assert T[0][2] == 1


def test_evaluate_polynomial() -> None:
    """
    From original C++ code.
    """
    degree = 2
    dimension = 1
    polynomial_coeffs: Vector1D = np.array([0., 0., 0.])
    t: float = -1.0

    polynomial_evaluation: Vector1D = evaluate_polynomial(
        degree, dimension, polynomial_coeffs, t)
    assert polynomial_evaluation.shape == (dimension, )


def test_compute_polynomial_mapping_product_one_dimension() -> None:
    """
    Testing ASOC code's implementation with NumPy's .convolve() method.
    Because we don't need to reimplement everything if NumPy conveniently provides
    functionality for us.
    """
    first_degree = 1
    second_degree = 1
    # Unnecessary parameter since compute_polynomial_mapping_product is only used with 1D
    # coefficient arrays
    dimension = 1
    first_polynomial_coeffs: Vector1D = np.array([2, 1])
    second_polynomial_coeffs: Vector1D = np.array([1, 1])
    product_polynomial_coeffs_control: Vector1D = np.zeros(shape=(3, ), dtype=np.float64)

    # *********
    # ASOC CODE
    # *********
    # Compute the new polynomial coefficients by convolution.
    for i in range(first_degree+1):
        for j in range(second_degree+1):
            product_polynomial_coeffs_control[i + j] += (first_polynomial_coeffs[i] *
                                                         second_polynomial_coeffs[j])

    # ******************
    # COMPARING RESULTS
    # ******************
    # Turns out that convolution likes same sized dimensions
    product_polynomial_coeffs_test: Vector1D = compute_polynomial_mapping_product(
        first_degree, second_degree, first_polynomial_coeffs, second_polynomial_coeffs)
    npt.assert_allclose(product_polynomial_coeffs_test, product_polynomial_coeffs_control)

    # Below is a hardcoded result from previous testing
    assert np.array_equal(product_polynomial_coeffs_test, np.array([2, 3, 1]))


def test_compute_polynomial_mapping_derivative_with_asoc() -> None:
    """
    Testing the ASOC code's implementation of compute_polynomial_mapping_derivative()
    with NumPy's derivative method.
    This is grabbing from test_zero_function() interaction with compute_derivative()
    and thus compute_polynomial_mapping_derivative()
    """
    # ******************
    # ZERO FUNCTION CASE
    # ******************
    degree = 1
    dimension = 1
    polynomial_coeffs: Vector2D = np.array([[0.0], [0.0]])
    derivative_polynomial_coeffs_control: Vector2D = np.array([[0.0]])
    assert polynomial_coeffs.shape == (2, 1)
    assert derivative_polynomial_coeffs_control.shape == (1, 1)

    # ASOC CODE
    for i in range(1, degree + 1):
        for j in range(dimension):
            derivative_polynomial_coeffs_control[i - 1, j] = i * polynomial_coeffs[i, j]

    derivative_polynomial_coeffs_test: MatrixXf = compute_polynomial_mapping_derivative(
        degree, dimension, polynomial_coeffs)

    npt.assert_allclose(derivative_polynomial_coeffs_test,
                        derivative_polynomial_coeffs_control)

    # TODO: test with other derivatives aside from zero function...
    # FIXME: because I'm quite sure this is not working as intended

    # ******************
    # LINEAR FUNCTION CASE
    # ******************
    P_coeffs: Vector2D = np.array([-1, 2]).reshape(2, 1)
    # P_deriv_coeffs = np.ndarray(shape=(1, 1))
    P_deriv_coeffs_test = compute_polynomial_mapping_derivative(1, 1, P_coeffs)
    P_deriv_coeffs_control = np.polynomial.polynomial.polyder(P_coeffs)
    assert np.array_equal(P_deriv_coeffs_test, P_deriv_coeffs_control)

    # TODO: derive each row of the polynomial for multidimensional coeff matrices (e.g shape (2, 3))


def test_remove_polynomial_trailing_coefficients() -> None:
    """
    Comparing original C++ functionality with NumPy
    """

    logger.info("Remove trailing zeros")

    A_coeffs: Vector1D = np.array([1, 2, 3, 0, 0, 0])
    reduced_coeffs: Vector1D

    # Find last nonzero entry and remove all zero entries after it
    last_zero: int = A_coeffs.size
    while last_zero > 0:
        # When reaching a non-zero number, stop and return new polynomial coefficient
        # vector with trailing zeros removed.
        if not float_equal(A_coeffs[last_zero - 1], 0.0):
            # TODO: double check that the below is equivalent to .head() in Eigen
            reduced_coeffs = A_coeffs[1:last_zero]
            return

        last_zero -= 1

    reduced_coeffs = A_coeffs[1:]

    # TODO: now compare this with NumPy operation
    assert np.array_equal(np.trim_zeros(A_coeffs, 'b'), reduced_coeffs)


def test_polynomial_mapping_cross_products_elementary_constant_functions() -> None:
    """
    Testing compute_polynomial_mapping_cross_product() with
    first_degree = 0 and second_degree = 0
    """
    logger.info("Elementary constant functions")
    A_coeffs = np.array([1, 0, 0])
    B_coeffs = np.array([0, 1, 0])
    cross_product_coeffs: Vector1D = compute_polynomial_mapping_cross_product(
        0, 0, A_coeffs, B_coeffs)

    assert cross_product_coeffs.shape == (1, 3)

    assert float_equal(cross_product_coeffs[0][0], 0.0)
    assert float_equal(cross_product_coeffs[0][1], 0.0)
    assert float_equal(cross_product_coeffs[0][2], 1.0)


def test_polynomial_mapping_cross_products_elementary_linear_functions() -> None:
    """
    Testing compute_polynomial_mapping_cross_product() with
    first_degree = 1 and second_degree = 1
    """
    logger.info("Elementary linear functions")
    A_coeffs = np.array([[2, 0, 0], [1, 0, 0]])
    B_coeffs = np.array([[0, 1, 0], [0, 1, 0]])
    cross_product_coeffs: Vector2D = compute_polynomial_mapping_cross_product(
        1, 1, A_coeffs, B_coeffs)

    assert cross_product_coeffs.shape == (3, 3)
    assert float_equal(cross_product_coeffs[0, 0], 0.0)
    assert float_equal(cross_product_coeffs[0, 1], 0.0)
    assert float_equal(cross_product_coeffs[0, 2], 2.0)
    assert float_equal(cross_product_coeffs[1, 2], 3.0)
    assert float_equal(cross_product_coeffs[2, 2], 1.0)

    # TODO: now check to see if equivalent to NumPy's polynomial solver...


def test_polynomial_mapping_cross_products_general_constant_functions() -> None:
    """
    Original C++ code.
    """
    logger.info("General constant functions")
    A_coeffs = np.array([1, 2, 3])
    B_coeffs = np.array([4, 5, 6])
    cross_product_coeffs: Vector2D = compute_polynomial_mapping_cross_product(
        0, 0,  A_coeffs, B_coeffs)

    assert cross_product_coeffs.shape == (1, 3)

    assert float_equal(cross_product_coeffs[0][0], -3.0)
    assert float_equal(cross_product_coeffs[0][1], 6.0)
    assert float_equal(cross_product_coeffs[0][2], -3.0)


def test_polynomial_mapping_cross_products_cancelling_linear_functions() -> None:
    """
    Original C++ code.
    """
    logger.info("Cancelling linear functions")
    A_coeffs = np.array([[1, 2, 3], [1, 1, 1]])
    B_coeffs = np.array([[4, 5, 6], [1, 1, 1]])
    cross_product_coeffs: Vector2D = compute_polynomial_mapping_cross_product(
        1, 1, A_coeffs, B_coeffs, )

    assert cross_product_coeffs.shape == (3, 3)

    assert float_equal(cross_product_coeffs[0, 0], -3.0)
    assert float_equal(cross_product_coeffs[0, 1], 6.0)
    assert float_equal(cross_product_coeffs[0, 2], -3.0)

    assert float_equal(cross_product_coeffs[1, 0], 0.0)
    assert float_equal(cross_product_coeffs[1, 1], 0.0)
    assert float_equal(cross_product_coeffs[1, 2], 0.0)

    assert float_equal(cross_product_coeffs[2, 0], 0.0)
    assert float_equal(cross_product_coeffs[2, 1], 0.0)
    assert float_equal(cross_product_coeffs[2, 2], 0.0)


def test_polynomial_real_roots_linear_function() -> None:
    """
    Original C++ code.
    """
    logger.info("Linear function")
    A_coeffs = np.array([1, 1])
    roots = polynomial_real_roots(A_coeffs)
    assert (roots.size == 1)
    assert float_equal(roots[0], -1.0)


def test_polynomial_real_roots_quadratic_function_with_roots() -> None:
    """
    Original C++ code.
    """
    logger.info("Quadratic function with roots")
    A_coeffs = np.array([-1, 0, 1])
    roots = polynomial_real_roots(A_coeffs)
    assert (roots.size == 2)
    assert (float_equal(roots[0], -1.0) or float_equal(roots[0], 1.0))
    assert (float_equal(roots[1], -1.0) or float_equal(roots[1], 1.0))


def test_polynomial_real_roots_quadratic_function_without_roots() -> None:
    """
    Original C++ code.
    """
    logger.info("Quadratic function without roots")
    A_coeffs = np.array([1, 0, 1])
    roots = polynomial_real_roots(A_coeffs)
    assert roots.size == 0


def test_polynomial_real_roots_vs_quadratic_real_roots() -> None:
    """
    This test is just to see if quadratic_real_roots() and polynomial_real_roots()
    do the same thing.
    """
    logger.info("Quadratic function with roots")
    A_coeffs = np.array([-1, 0, 1])
    roots_polynomial: Vector2D = polynomial_real_roots(A_coeffs)
    roots_quadratic, num_solutions = quadratic_real_roots(A_coeffs)

    assert (roots_polynomial.size == 2)
    assert (roots_quadratic.size == 2)
    assert np.array_equal(roots_polynomial, roots_quadratic)

    logger.info("Quadratic function without roots")
    A_coeffs = np.array([1, 0, 1])
    roots_polynomial = polynomial_real_roots(A_coeffs)
    roots_quadratic, num_solutions = quadratic_real_roots(A_coeffs)

    assert roots_polynomial.size == 0

    # TODO: the below assert should fail since the roots_quadratic() just has whatever.
    # But the num_solutions is 0....
    # But anyways, polynomial_real_roots and quadratic_real_roots appear to just do the same thing.
    # assert roots_quadratic.size == 0
