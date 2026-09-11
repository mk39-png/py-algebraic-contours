import numpy as np
cimport numpy as cnp
cimport cython
cnp.import_array()
DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t

cdef generate_monomials(int degree, double t):
    """
    Generate the row vector T of n + 1 monomials 1, t, ... , t^n.

    :param int degree: [in] maximum monomial degree.
    :param float t: [in] evaluation point for the monomials.

    :return T: row vector of monomials of shape (degree + 1, ).
    :rtype: np.ndarray
    """
    cdef cnp.ndarray T = np.empty(degree + 1, dtype=DTYPE) 

    T[0] = 1.0
    for i in range(1, degree + 1):
        T[i] = t * T[i - 1]
    return T


# TODO: typed memoryviews
# TODO: these are still slow below...
def evaluate_polynomial(int degree,
                        int dimension,
                        cnp.ndarray polynomial_coeffs_ref,
                        double t):
    """
    Evaluate the polynomial with given coefficients at t.
    NOTE: this has been modified from the ASOC code to support any dimension.

    :param int degree: [in] maximum monomial degree.
    :param int dimension: [in] polynomial dimension.
    :param polynomial_coeffs: [in] coefficients of the polynomial.
    :param t: [in] evaluation point for the polynomial.

    :return polynomial_evaluation: evaluation of the polynomial of shape (dimension, ) or float
    """

    # Perform calculation
    cdef cnp.ndarray T = generate_monomials(degree, t)
    assert T.size == degree + 1
    polynomial_evaluation = T @ polynomial_coeffs_ref
    return polynomial_evaluation


