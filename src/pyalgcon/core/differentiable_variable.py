"""
Original ASOC code utilizes the AutoDiff library by Wenzel Jakob. Based on code by Jon Kaldor
and Eitan Grinspun.

ORIGINAL PURPOSE: So, this file focuses on utilizing automatic differentiation via JAX. 
UPDATED PURPOSE: the methods in this class do not seem to be used elsewhere aside from testing.
"""
from pyalgcon.core.common import unimplemented


def generate_local_variable_matrix_index(row: int, col: int, dimension=3) -> int:
    """
    Used in powell_sabin_local_to_global.py.
    """
    return dimension * row + col


def generate_independent_variable() -> None:
    # This is only used by build_independent_variable_vector() and build_independent_variable_matrix()
    """
     Build an independent variable with a given value and variable index.
    ///
     @param[in] value: initial value of the variable
     @param[in] variable_index: global index of the variable in the full system
     @param[in] total_independent_variables: total number of variables in the
     system
     @return constructed differentiable variable
    """
    unimplemented(
        "Method is used by build_independent_variable_vector() and"
        " build_independent_variable_matrix(), both of which are unimplemented. "
        "Is also used in tests (which are commented out in original ASOC code)")


def build_independent_variable_vector():
    # This is only used in testing... no where else.
    """
    Build a vector of independent variables with a given initial value and
    contiguous variable indices from some starting index.
    @param[in] value_vector: initial values of the variables
    @param[out] variable_vector: constructed differentiable variables
    @param[in] start_variable_index: global index of the first variable in the
    full system
    @param[in] total_independent_variables: total number of variables in the
    system
    """
    unimplemented("Only used in tests (that have been commented out in original ASOC code)")


def build_independent_variable_matrix():
    """
     Build a matrix of independent variables with a given initial value and
    contiguous row-major variable indices from some starting index.
    ///
    @param[in] value_matrix: initial values of the variables
    @param[out] value_matrix: constructed differentiable variables
    @param[in] start_variable_index: global index of the first variable in the
    full system
    @param[in] total_independent_variables: total number of variables in the
    system
    """
    unimplemented("Only used in tests (that have been commented out in original ASOC code)")


def generate_constant_variable():
    """
    Build a differentiable constant with a given value.
    @param[in] value: value of the constant
    @return constructed differentiable variable
    """
    unimplemented(
        "Method is used by build_constant_variable_vector() and build_constant_variable_matrix(), "
        "both of which are unimplemented.")


def build_constant_variable_vector():
    """
    Build a vector of differentiable constants with given values.
    @param[in] value_vector: values of the constants
    @param[out] constant_variable_vector: vector of constant variables
    """
    unimplemented("Method is not used anywhere.")


def build_constant_variable_matrix():
    # This isn't used anywhere...
    """
    Build a matrix of differentiable constants with given values.
    @param[in] value_matrix: values of the constants
    @param[out] constant_variable_matrix: matrix of constant variables
    """
    unimplemented("Method is not used anywhere.")


def compute_variable_value():
    """
    Extract the value of a differentiable variable.
    @param[in] variable: differentiable variable
    @return value of the variable
    """
    unimplemented("Only used in tests and internally within differentiable_variable.py")


def compute_variable_gradient():
    """
    Extract the gradient of a differentiable variable with respect to the
    independent variables.
    @param[in] variable: differentiable variable
    @return gradient of the variable
    """
    unimplemented("Only used in tests (that have been commented out in original ASOC code)")


def compute_variable_hessian():
    """
    Extract the hessian of a differentiable variable with respect to the
    independent variables.
    @param[in] variable: differentiable variable
    @return hessian of the variable
    """
    unimplemented("Only used in tests (that have been commented out in original ASOC code)")


def extract_variable_vector_values():
    """
    Extract the values of a vector of differentiable variables.
    @param[in] variable_vector: vector of differentiable variables
    @param[out] values_vector: vector of the values of the variables
    """
    unimplemented("Method is not used anywhere.")


def extract_variable_matrix_values():
    """
    Extract the values of a matrix of differentiable variables.
    @param[in] variable_matrix: matrix of differentiable variables
    @param[out] values_matrix: matrix of the values of the variables
    """
    unimplemented("Method is not used anywhere.")


def vector_contains_nan():
    """
     Determine if a vector of differentiable variables contains NaN.
    @param[in] variable_vector: vector of differentiable variables
    @return true iff the vector contains NaN
    """
    unimplemented("Overrides common.py method of the same name. Not needed.")


def matrix_contains_nan():
    """
    Determine if a matrix of differentiable variables contains NaN.
    @param[in] variable_matrix: matrix of differentiable variables
    @return true iff the matrix contains NaN
    """
    unimplemented("Overrides common.py method of the same name. Not needed.")


def variable_square():
    """
    Compute the square of a variable
    @param[in] x: differentiable variables
    @return square of the variable
    """
    unimplemented("Helper to variable_square_norm()")


def variable_square_norm():
    """
    Compute the square norm of a variable vector.
    @param[in] variable_vector: vector of differentiable variables
    @return square norm of the variable
    """
    unimplemented("Method is not used anywhere.")
