# from typing import NewType
import csv
import json  # for testing
import logging
import math
import os
import pathlib  # used for testing
from io import StringIO

import igl
import numpy as np
import numpy.linalg as LA
import numpy.testing as npt
import numpy.typing as npty

logger: logging.Logger = logging.getLogger(__name__)


# *******
# GLOBALS
# *******

# Epsilon for default float
FLOAT_EQUAL_PRECISION: float = 1e-10
# Epsilon for chaining contours
ADJACENT_CONTOUR_PRECISION: float = 1e-6
# Epsilon for curve-curve bounding box padding
PLANAR_BOUNDING_BOX_PRECISION: float = 0
# Epsilon for Bezier clipping intersections
# Particular precision below was chosen since it gave satisfactory results (as in, number of intersections closer to C++ code)
# Though, confidence in this statement is varying...
# FIND_INTERSECTIONS_BEZIER_CLIPPING_PRECISION: float = 8.250000000e-7
FIND_INTERSECTIONS_BEZIER_CLIPPING_PRECISION: float = 1e-7

# FIND_INTERSECTIONS_BEZIER_CLIPPING_PRECISION: float = np.nextafter(1e-7, 0)  # 1e-7
# FIND_INTERSECTIONS_BEZIER_CLIPPING_PRECISION: float = 9.999999999995e-8
# FIND_INTERSECTIONS_BEZIER_CLIPPING_PRECISION: float = 9.999999999995e-8


# Spline surface discretization level
DISCRETIZATION_LEVEL: int = 2
# Size of spline surface hash table
HASH_TABLE_SIZE: int = 70

# *** Real number representations ***

# Including typing here for better code.
# https://stackoverflow.com/questions/71109838/numpy-typing-with-specific-shape-and-datatype
OneFormXr = np.ndarray  # TODO: what shape is this... I forget
PlanarPoint = np.ndarray[tuple[int, int], np.dtype[np.float64]]  # shape (1, 2)
# PlanarPoint1d = np.ndarray[tuple[int], np.dtype[np.float64]]  # shape (2, )
PlanarPoint1d = np.ndarray  # shape (2, )
SpatialVector = np.ndarray[tuple[int, int],
                           np.dtype[np.float64]]  # shape (1, 3)
SpatialVector1d = np.ndarray  # shape (3, )
Index = int
FaceIndex = int
VertexIndex = int
PatchIndex = int

# Typedefs for readability.
NodeIndex = int
SegmentIndex = int


# NOTE: Since NumPu does not have typing, it has been done so as below for readability reasons.
Color = tuple[float, float, float, float]
Vector3i = np.ndarray
Vector2f = np.ndarray
Vector3f = np.ndarray
Vector4f = np.ndarray
Vector5f = np.ndarray
Vector6f = np.ndarray
Vector9f = np.ndarray
Vector12f = np.ndarray
Vector13f = np.ndarray
Vector36f = np.ndarray
# shape (n, )... sometimes. Oftentimes it's shape (n, 1) or (1, n)...
VectorX = np.ndarray
# shape (1, n) (or sometimes shape (n, 1) as in the case of optimize_spline_surface)... might just be easier to flatten and use Vector1D... I don't see the point of having Vector2D if it will just be more confusing.
Vector2D = np.ndarray
Vector1D = np.ndarray  # shape (n, )
MatrixNx2f = np.ndarray
MatrixNx3 = np.ndarray[tuple[int, int], np.dtype[np.float64]]
Matrix6xNi = np.ndarray
Matrix3x6r = np.ndarray
# Matrix2x2r = np.ndarray[tuple[int, int], np.dtype[np.float64]]  # shape (2, 2)
# Matrix2x3r = np.ndarray[tuple[int, int], np.dtype[np.float64]]  # shape (2, 3)
# Matrix2x3f = np.ndarray[tuple[int, int], np.dtype[np.float64]]  # shape (2, 3)
# Matrix3x1r = np.ndarray[tuple[int, int], np.dtype[np.float64]]  # shape (3, 1)
# Matrix3x2r = np.ndarray[tuple[int, int], np.dtype[np.float64]]  # shape (3, 2)
# Matrix3x2f = np.ndarray[tuple[int, int], np.dtype[np.float64]]
# Matrix3x3r = np.ndarray[tuple[int, int], np.dtype[np.float64]]  # shape (3, 3)
# Matrix6x3r = np.ndarray[tuple[int, int], np.dtype[np.float64]]  # shape (6, 3)
# Matrix6x3f = np.ndarray[tuple[int, int], np.dtype[np.float64]]  # shape (6, 3)
# Matrix6x6r = np.ndarray[tuple[int, int], np.dtype[np.float64]]
# Matrix6x12f = np.ndarray[tuple[int, int], np.dtype[np.float64]]
# Matrix12x3f = np.ndarray[tuple[int, int], np.dtype[np.float64]]
# Matrix12x12r = np.ndarray[tuple[int, int], np.dtype[np.float64]]
# TwelveSplitGradient = np.ndarray[tuple[int, int], np.dtype[np.float64]]  # shape (36, 1)
# TwelveSplitHessian = np.ndarray[tuple[int, int], np.dtype[np.float64]]  # shape (36, 36)
Matrix2x2f = np.ndarray  # shape (2, 2)
Matrix2x3r = np.ndarray  # shape (2, 3)
Matrix2x3f = np.ndarray  # shape (2, 3)
Matrix3x1r = np.ndarray  # shape (3, 1)
Matrix3x2r = np.ndarray  # shape (3, 2)
Matrix3x2f = np.ndarray
Matrix3x3r = np.ndarray  # shape (3, 3)
Matrix3x3f = np.ndarray

Matrix4x2f = np.ndarray
Matrix4x4f = np.ndarray

Matrix5x2f = np.ndarray
Matrix5x3f = np.ndarray
Matrix5x5f = np.ndarray

Matrix6x3r = np.ndarray  # shape (6, 3)
Matrix6x3f = np.ndarray  # shape (6, 3)
Matrix6x6r = np.ndarray
Matrix6x12f = np.ndarray
Matrix12x3f = np.ndarray
Matrix12x12r = np.ndarray

Matrix9x3f = np.ndarray
Matrix13x3f = np.ndarray

Matrix36x36f = np.ndarray
TwelveSplitGradient = np.ndarray  # shape (36, 1)
TwelveSplitHessian = np.ndarray  # shape (36, 36)


MatrixXi = np.ndarray[tuple[int, int], np.dtype[np.int64]]
MatrixXf = np.ndarray[tuple[int, int], np.dtype[np.float64]]
MatrixXr = np.ndarray[tuple[int, int], np.dtype[np.float64]]  # shape (n, m)

# NOTE: N denotes any arbitrary number of rows.
# This is not suggesting that MatrixNx3f and MatrixNx3i has the same number of rows.
MatrixNx3f = np.ndarray  # np.ndarray[tuple[int, int], np.dtype[np.float64]]
MatrixNx3i = np.ndarray  # np.ndarray[tuple[int, int], np.dtype[np.int64]]

# Used for accessing numpy shape for clarity sake
ROWS = 0
COLS = 1
PLACEHOLDER_VALUE = -1
PLACEHOLDER_INDEX = -1
PLACEHOLDER_BOOL = False

# ***********************
# TESTING FLAGS
CHECK_VALIDITY: bool = False

#
# ***********************

# Algebraic constrained values
MAX_PATCH_RAY_INTERSECTIONS = 4

# **********************
# Debug/Helper Methods (keep here for now, especially when comparing later contours code)
# **********************


def initialize_spot_control_mesh() -> tuple[npty.ArrayLike, npty.ArrayLike, npty.ArrayLike, npty.ArrayLike]:
    """ 
    Used for testing spot_control mesh in generating 
    the TwelveSplitSplineSurface.
    Returns only the parts of the mesh that are needed

    :return: tuple V, uv, F, FT
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    # Get input mesh
    filename: str = "spot_control_mesh-cleaned_conf_simplified_with_uv.obj"
    filepath: str = os.path.abspath(f"tests\\data\\spot_control\\{filename}")

    V_temp: npty.ArrayLike
    uv_temp: npty.ArrayLike
    N_temp: npty.ArrayLike
    F_temp: npty.ArrayLike  # int
    FT_temp: npty.ArrayLike  # int
    FN_temp: npty.ArrayLike  # int
    V_temp, uv_temp, N_temp, F_temp, FT_temp, FN_temp = igl.readOBJ(filepath)

    # Wrapping inside np.array for typing
    V: np.ndarray = np.array(V_temp)
    uv: np.ndarray = np.array(uv_temp)
    F: np.ndarray = np.array(F_temp)
    FT: np.ndarray = np.array(FT_temp)

    return V, uv, F, FT


def load_json(filepath: pathlib.Path) -> list[dict] | list[list[dict]] | list[list[list]]:
    """
    Used by affine_manifold tests.
    Parses a list of dataclasses.
    """
    obj: list[list[dict]] | list[dict] | list[list[list]] | None = None

    with open(filepath, 'r', encoding='utf-8') as file:
        try:
            obj = json.load(file)
        except Exception as e:
            raise OSError(f"Error in JSON parsing at {filepath}") from e

    if obj is None:
        raise ValueError(f"Failure to read obj at file {filepath}")

    return obj


def deserialize_list_list_varying_lengths_float(filepath: pathlib.Path) -> list[list[float]]:
    """
    Used when the csv list contains list with varying list lengths.
    e.g. 
    [
    [1, 2, 3, 4],
    [1, 2],
    [5, 7, 8, 8, 19, 1],
    ]

    # TODO: change to be list of list of any datatype
    """
    rows_control: list[list[float]] = []

    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            parsed: list[float] = [float(x) for x in row if x.strip() != '']
            rows_control.append(parsed)

    return rows_control


def deserialize_list_list_varying_lengths(filepath: pathlib.Path) -> list[list[int]]:
    """
    Used when the csv list contains list with varying list lengths.
    e.g. 
    [
    [1, 2, 3, 4],
    [1, 2],
    [5, 7, 8, 8, 19, 1],
    ]

    # TODO: change to be list of list of any datatype
    """
    rows_control: list[list[int]] = []

    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            parsed: list[int] = [int(x) for x in row if x.strip() != '']
            rows_control.append(parsed)

    return rows_control


def compare_list_list_varying_lengths_float(filepath: pathlib.Path,
                                            rows_test: list[list[float]],
                                            precision=0.0) -> None:
    """
    Used when the csv list contains list with varying list lengths.
    e.g. 
    [
    [1, 2, 3, 4],
    [1, 2],
    [5, 7, 8, 8, 19, 1],
    ]

    NOTE: default precision is 0.0 to match the default atol value of 
    assert_allclose
    """
    rows_control: list[list[float]] = []

    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            parsed: list[float] = [float(x) for x in row if x.strip() != '']
            rows_control.append(parsed)
    # Make sure that both are the same length
    assert len(rows_control) == len(rows_test)
    num_rows: int = len(rows_control)

    for i in range(num_rows):
        npt.assert_allclose(np.array(rows_test[i]),
                            np.array(rows_control[i]),
                            atol=precision)


def compare_list_list_varying_lengths(filepath: pathlib.Path, rows_test: list[list[int]]) -> None:
    """
    Used when the csv list contains list with varying list lengths.
    e.g. 
    [
    [1, 2, 3, 4],
    [1, 2],
    [5, 7, 8, 8, 19, 1],
    ]

    Used for integer datatypes.
    """
    rows_control: list[list[int]] = []

    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            parsed: list[int] = [int(x) for x in row if x.strip() != '']
            rows_control.append(parsed)
    # Make sure that both are the same length
    assert len(rows_control) == len(rows_test)
    num_rows: int = len(rows_control)

    for i in range(num_rows):
        npt.assert_array_equal(np.array(rows_test[i]),
                               np.array(rows_control[i]))


def compare_eigen_numpy_matrix(filepath: pathlib.Path,
                               numpy_array: np.ndarray,
                               make_3d: bool = False) -> None:
    """ 
    Standardized function for comparing matrices for testing.
    Takes a filename and creates an absolute filepath to src/tests/
    """
    eigen_array: np.ndarray = deserialize_eigen_matrix_csv_to_numpy(
        filepath, make_3d)
    npt.assert_allclose(numpy_array, eigen_array)


def compare_intersection_points(filepath: pathlib.Path,
                                numpy_array: np.ndarray,
                                num_intersections: int) -> None:
    """
    Special function of compare_eigen_numpy() that takes in num_intersections and ignores the 
    rest of the irrelevant parts of the matrix to compare.
    Since the original C++ code only initializes the values of the points that are relevant.
    e.g
    ACTUAL: array([[0.108474, 0.195645],
              [0.      , 0.      ],
              [0.      , 0.      ],
              [0.      , 0.      ]])
    DESIRED: array([[1.084738e-001, 1.956454e-001],
          [6.953336e-310, 2.470328e-323],
          [6.953336e-310, 4.728331e-310],
          [0.000000e+000, 1.976263e-323]])
    But we only care about the first row since this only has 1 intersection.
    """
    eigen_array: np.ndarray = deserialize_eigen_matrix_csv_to_numpy(filepath)
    npt.assert_allclose(
        numpy_array[:num_intersections], eigen_array[:num_intersections], atol=1e-5)


def deserialize_eigen_matrix_csv_to_numpy(filepath: pathlib.Path, make_3d: bool = False) -> np.ndarray:
    """
    Turns Eigen matrix .csv into NumPy array
    Used for testing.

    :param make_3d: parses a 3D csv into a 3D nnumpy array. 
    usually equivalent structure is a list of matrices.
    """
    arr: np.ndarray

    if make_3d:
        with open(filepath, 'r', encoding='utf-8') as file:
            file_lines_raw: str = file.read()

            # In "3D" files, sections are separated by double newlines
            file_lines: list[str] = file_lines_raw.split("\n\n")

            # build this list 3d then convert to np
            arr_list: list[np.ndarray] = []
            for line_raw in file_lines:
                # NOTE: sometimes the last line can be '' empty string since we remove "\n\n"
                if len(line_raw) == 0:
                    continue
                # convert to stringio because that's what loadtxt works with
                s = StringIO(line_raw)
                # get matrix values to put in list of matrices
                line: np.ndarray = np.loadtxt(s, delimiter=',')
                arr_list.append(line)

            arr = np.array(arr_list)
            assert arr.ndim == 3
        file.close()
    else:
        arr = np.loadtxt(filepath, dtype=np.float64, delimiter=',')

    return arr


def deprecated(msg: str = "Method no longer in use") -> Exception:
    """
    Used to show methods that are no longer in use.

    :param str: Message to be displayed for the error. 
    """
    raise NotImplementedError(msg)


def unimplemented(msg: str = "Method yet to be implemented") -> Exception:
    """
    Used to show methods that are not yet implemented.

    :param str: Message to be displayed for the error. 
    """
    raise NotImplementedError(msg)


def todo(msg: str = "Method needs some work") -> Exception:
    """
    Used to show methods that need to be implemented.

    :param str: Message to be displayed for the error. 
    """
    raise NotImplementedError(msg)


def unreachable(msg: str = "Method should never reach this part") -> Exception:
    """
    Used to show methods that need to be implemented.

    :param str: Message to be displayed for the error. 
    """
    raise RuntimeError(msg)


# Colors for PolyScope display.
MINT_GREEN: tuple[float, float, float] = (0.170, 0.673, 0.292)
SKY_BLUE: tuple[float, float, float] = (0.297, 0.586, 0.758)
OFF_WHITE:  tuple[float, float, float] = (0.896, 0.932, 0.997)
GOLD_YELLOW:  tuple[float, float, float] = (0.670, 0.673, 0.292)

# -----------------------------------------


def float_equal_zero(x: float, eps=FLOAT_EQUAL_PRECISION) -> bool:
    """
    Check if some floating point value is numerically zero.

    :param x:   [in] value to compare with zero
    :param eps: [in] threshold for equality
    :return true iff x is below 1e-10
    """

    # NOTE: Use absolute tolerance! Relative tolerance is not suited for our purpose.
    return math.isclose(x, 0.0, abs_tol=eps)


def float_equal(x: float, y: float, eps=FLOAT_EQUAL_PRECISION) -> bool:
    """
    @brief Check if two floating point values are numerically equal

    @param[in] x: first value to compare
    @param[in] y: second value to compare
    @param[in] eps: threshold for equality
    @return true iff x - y is numerically zero
    """

    # NOTE: Use absolute tolerance! Relative tolerance is not suited for our purpose.
    # return np.isclose(x, y, atol=eps)
    return math.isclose(x, y, abs_tol=eps)


def vector_equal(v: np.ndarray, w: np.ndarray, eps: float = FLOAT_EQUAL_PRECISION) -> bool:
    """
    @brief Check if two row vectors of floating point values are numerically
    equal

    @param[in] v: first vector of values to compare
    @param[in] w: second vector of values to compare
    @param[in] eps: threshold for equality
    @return true iff v - w is numerically the zero vector
    """
    return np.allclose(v, w, atol=eps)


def column_vector_equal() -> None:
    """
    Method no longer in use.
    """
    deprecated()


def matrix_equal() -> None:
    """
    Method no longer in use.
    """
    deprecated()


def view_mesh() -> None:
    """
    Method no longer in use.
    """
    deprecated()


def view_parameterized_mesh() -> None:
    """
    Method no longer in use.
    """
    deprecated()


def screenshot_mesh() -> None:
    todo()

# ****************
# Basic arithmetic
# ****************


def sgn() -> None:
    todo()


def power() -> None:
    todo()


def compute_discriminant(a: float, b: float, c: float) -> float:
    """
    Compute the discriminant of the quadratic ax^2 + bx + c.

    :param a: [in] x^2 coefficient
    :param b: [in] x coefficient
    :param c: [in] constant coefficient
    :return: discriminate b^2 - 4ac
    """
    return (b * b) - (4.0 * a * c)


def dot_product(v: Vector1D, w: Vector1D) -> float:
    """
    Compute the dot product of two vectors of arbitrary scalars.

    :param v: [in] first vector to dot product
    :param w: [in] second vector to dot product
    :return: dot product of v and w
    """

    return v.dot(w)


def cross_product(v: Vector3f, w: Vector3f) -> Vector3f:
    """
    Compute the cross product of two vectors of arbitrary scalars.
    Dedicated method to check for NumPy shapes before cross product calculation.

    :param v: [in] first vector to cross product
    :param w: [in] second vector to cross product
    :return: cross product v x w in shape (3, )
    """
    # TODO: make this 1D only.
    assert v.shape == (3, )
    assert w.shape == (3, )
    assert v.size == 3
    assert w.size == 3

    # TODO: use NumPy's version of cross products
    n: Vector3f = np.array([
        (v[1] * w[2] - v[2] * w[1]),
        (-(v[0] * w[2] - v[2] * w[0])),
        (v[0] * w[1] - v[1] * w[0])],
        dtype=np.float64)
    assert n.shape == (3, )

    return n


def triple_product():
    todo()


def normalize():
    todo()


def elementary_basis_vector():
    todo()


def reflect_across_x_axis(vector: PlanarPoint1d) -> PlanarPoint1d:
    """
    @brief  Reflect a vector in the plane across the x-axis.

    @param[in] vector: vector to reflect
    @return reflected vector of shape (1, 2)
    """
    assert vector.shape == (2, )
    reflected_vector: PlanarPoint1d = np.array(
        [vector[0], -vector[1]], dtype=np.float64)
    return reflected_vector


# this is a void.
def rotate_vector():
    todo()

# this returns a SpatialVector class
# def rotate_vector():
#     todo()


def project_vector_to_plane():
    todo()


def vector_min():
    todo()


def vector_max():
    todo()


def column_vector_min():
    todo()


def column_vector_max():
    todo()


def vector_contains(vec: list, item) -> bool:
    return item in vec


def convert_index_vector_to_boolean_array(index_vector: list[int], num_indices: int) -> list[bool]:
    boolean_array: list[bool] = [False for _ in range(num_indices)]

    for i, _ in enumerate(index_vector):
        boolean_array[index_vector[i]] = True

    return boolean_array


def convert_boolean_array_to_index_vector(boolean_array: list[bool]) -> list[int]:
    """
    @brief From a boolean array, build a vector of the indices that are true.
    @param[in] boolean_array: array of boolean values
    @param[out] index_vector: indices where the array is true
    """
    num_indices: int = len(boolean_array)
    index_vector: list[int] = []
    for i in range(num_indices):
        if (boolean_array[i]):
            index_vector.append(i)

    return index_vector


def index_vector_complement(index_vector: list[int], num_indices: int) -> list[int]:
    """
    Returns the complement of the index_vector as a list of int.

    :param index_vector: vector to take the complement of
    :type index_vector: list[int]

    :param num_indices: determines the size of the complement_vector
    :type num_indices: int

    :return: complement_vector
    :rtype: list[int]
    """
    # Build index boolean array
    boolean_array: list[bool] = convert_index_vector_to_boolean_array(
        index_vector, num_indices)

    # Build complement
    complement_vector: list[int] = []
    for i in range(num_indices):
        if not boolean_array[i]:
            complement_vector.append(i)

    return complement_vector


def convert_signed_vector_to_unsigned() -> None:
    """
    Method no longer in use.
    """
    deprecated()


def convert_unsigned_vector_to_signed() -> None:
    """
    Method no longer in use.
    """
    deprecated()


def remove_vector_values(indices_to_remove: list[Index], vec: list) -> list:
    """
    Removes elements from vec with indices specified in indices_to_remove.

    :param indices_to_remove: indices to remove
    :type indices_to_remove: list[Index]

    :param vec: vector to remove from
    :type vec: list

    :return: vector with indices removed
    :rtype: list
    """
    # Removes indices from vev

    # Remove faces adjacent to cones
    indices_to_keep: list[Index] = index_vector_complement(
        indices_to_remove, len(vec))
    subvec: list = []

    # TODO: double check logic here with ASOC code
    for index_to_keep in indices_to_keep:
        subvec.append(vec[index_to_keep])

    assert len(subvec) == len(indices_to_keep)

    return subvec


def copy_to_planar_point():
    todo()


def copy_to_spatial_vector():
    todo()


# TODO: don't think we need this since Python prints out vectors just fine.... maybe
# Unless there's an extra fancy vector type in the C++ code like vector<RationalFunction>
#  or something like that.
def formatted_vector(vec: list, delim: str = "\n") -> str:
    # raise Exception(
    # "formatted_vector() is not implmemented. Print out object as-is instead.")
    vector_string: str = ""
    for i, _ in enumerate(vec):
        vector_string += (str(vec[i]) + delim)

    return vector_string


def write_vector():
    todo()


def write_float_vector():
    todo()


def append():
    todo()


def nested_vector_size(v: list[list]) -> int:
    """
    """
    count: int = 0

    for inner_v in v:
        count += len(inner_v)

    return count


def convert_nested_vector_to_matrix(vec: list[Vector2D]) -> np.ndarray:
    """
    WARNING: Do not use this method, implementation does not generalize to a list types.
    Especially with list[np.ndarray] where ndarray is some shape (n, 1) or (1, n)
    """
    # n: int = len(vec)
    # if (n <= 0):
    #     return np.ndarray(shape=(0, 0))

    # # TODO: problem may arise when vec is list of np.ndarray ndim >= 2
    # # TODO: below is supposed to be size3...
    # matrix = np.ndarray(shape=(len(vec), vec[0].size))
    # for i in range(n):
    #     # inner_vec_size = len(vec[i])
    #     inner_vec: np.ndarray = vec[i].flatten()
    #     for j in range(inner_vec.size):
    #         matrix[i, j] = vec[i].flatten()[j]

    matrix: np.ndarray = np.array(vec).squeeze()
    # assert matrix.shape == (len(vec))

    return matrix


def append_matrix():
    todo()


def flatten_matrix_by_row():
    todo()


def read_camera_matrix(filepath: pathlib.Path) -> Matrix4x4f:
    unimplemented("see deserialize_eigen_matrix_csv_to_numpy()")


def generate_linspace(t_0: float, t_1: float, num_points: int) -> Vector1D:
    """
    Originally under "Pythonic methods" in ASOC code.

    :param t_0: starting value
    :type t_0: float
    :param t_1: ending value
    :type t_1: float
    :param num_points: number of points to sample between interval.
    :type num_points: int
    :return: linspace vector
    :rtype: Vector1D
    """
    # TODO: compare NumPy linspace with ASOC linspace
    return np.linspace(t_0, t_1, num_points)


def arange(size: int) -> list:
    """Equivalent to Python's range."""
    # TODO: compare this with Python's range method
    arange_vec: list = []

    for i in range(size):
        arange_vec.append(i)

    return arange_vec

#  *******************
#  Basic mesh topology
#  *******************


def contains_vertex(face: Vector1D, vertex_index: int) -> bool:
    """
    Returns true iff the face contains the given vertex.

    :param face: 1D NumPy array of integers. Shape is (n, )
    :type face: np.ndarray

    :param vertex_index: the index to check for inside face
    :type vertex_index: int

    :return: boolean if vertex_index is in face
    """
    return vertex_index in face


def find_face_vertex_index(face: Vector1D, vertex_index: int) -> int:
    """
    :param face: np.ndarray of shape (n, ) of ndim = 1
    :type face: np.ndarray

    :param vertex_index:
    :type vertex_index: int

    :return:  face vertex index
    :rtype: int
    """
    # TODO: test this numpy-esque implementation with the ASOC version...
    # NOTE: we want to check that face is a vector rather than a matrix
    assert face.ndim == 1

    vertex_indices = np.argwhere(face == vertex_index)
    if vertex_indices.size > 0:
        return vertex_indices[0][0]

    return -1


def is_manifold(F: MatrixXi) -> bool:
    """
    Check if F describes a manifold mesh with a single component

    :param F: [in] mesh faces
    :return: true iff the mesh is manifold
    """

    # Check edge manifold condition
    # Checks the tuple of elements that are returned.
    # first element tells us if all edges are manifold or not.
    if not igl.is_edge_manifold(F)[0]:
        logger.error("Mesh is not edge manifold")
        return False

    # Check vertex manifold condition

    invalid_vertices: np.ndarray = np.asarray(
        igl.is_vertex_manifold(F), dtype=np.bool)
    if not invalid_vertices.any():
        logger.error("Mesh is not vertex manifold")
        return False

    # Check single component
    # TODO: check datatype on component_ids and if it's a numpy array
    component_ids: np.ndarray = np.asarray(
        igl.vertex_components(F), dtype=np.int64)

    if (component_ids.max() - component_ids.min()) > 0:
        logger.error("Mesh has multiple components")
        return False

    # Manifold otherwise
    return True


#  *******************
#  Basic mesh geometry
#  *******************
def area_from_length(l0: float, l1: float, l2: float) -> float:
    """
    @brief Compute the area of a triangle from the edge lengths.

    @param[in] l0: first edge length
    @param[in] l1: second edge length
    @param[in] l2: third edge length
    @return area of the triangle
    """
    # Return the area (or zero if there is a triangle inequality violation)
    s: float = 0.5 * (l0 + l1 + l2)  # semi-perimeter
    area: float = math.sqrt(max(s * (s - l0) * (s - l1) * (s - l2), 0.0))
    assert not math.isnan(area)
    return area


def area_from_positions() -> None:
    """
    Compute the area of a triangle from the vertex positions

    :param p0: [in] first vertex position
    :param p1: [in] second vertex position
    :param p2: [in] third vertex position
    :return: triangle area
    """
    # p0: PlanarPoint, p1: PlanarPoint, p2: PlanarPoint) -> float:
    deprecated(
        "Method only used in generate_twelve_split_domain_areas, which is no longer in use.")

    # assert p0.shape == (1, 2)
    # assert p1.shape == (1, 2)
    # assert p2.shape == (1, 2)
    # assert p0.shape[0] == 1  # making sure that p0 is shape (1, n)
    # assert p0.shape == p1.shape
    # assert p1.shape == p2.shape

    # TODO: double check that numpy norm is doing what we want
    # l0: float = LA.norm(p2 - p1)
    # l1: float = LA.norm(p0 - p2)
    # l2: float = LA.norm(p1 - p0)

    # assert isinstance(l0, float)

    # return area_from_length(l0, l1, l2)


def angle_from_length(edge_length_opposite_corner: float,
                      first_adjacent_edge_length: float,
                      second_adjacent_edge_length: float
                      ) -> float:
    """
    Compute the angle of a triangle corner with given edge lengths

    :param edge_length_opposite_corner: [in]  length of the edge opposite the corner
    :param first_adjacent_edge_length:  [in] length of one of the edges adjacent to the corner
    :param first_adjacent_edge_length:  [in] length of the other edge adjacent to the corner
    :return: angle of the corner
    """
    # Rename variables for readability
    l0: float = edge_length_opposite_corner
    l1: float = first_adjacent_edge_length
    l2: float = second_adjacent_edge_length

    # Compute the angle
    # FIXME Avoid potential division by 0
    ijk: float = (-l0 * l0 + l1 * l1 + l2 * l2)
    return math.acos(min(max(ijk / (2.0 * l1 * l2), -1.0), 1.0))


def angle_from_positions(angle_corner_position: Vector2f,
                         second_corner_position: Vector2f,
                         third_corner_position: Vector2f) -> float:
    """
    Compute the angle of a triangle corner with given positions

    :param angle_corner_position:  [in] position of the corner to compute the angle for
    :param second_corner_position: [in] position of one of the other two corners of the triangle
    :param third_corner_position:  [in] position of the final corner of the triangle
    :return angle of the corner
    """
    assert angle_corner_position.shape == (2, )
    assert second_corner_position.shape == (2, )
    assert third_corner_position.shape == (2, )

    # TODO: double check that the below are going to be floats...
    # FIXME: really silly error was here
    l0: float = LA.norm(third_corner_position - second_corner_position)
    l1: float = LA.norm(second_corner_position - angle_corner_position)
    l2: float = LA.norm(third_corner_position - angle_corner_position)

    return angle_from_length(l0, l1, l2)


def interval_lerp(t_min_0: float,
                  t_max_0: float,
                  t_min_1: float,
                  t_max_1: float,
                  t_0: float) -> float:
    """"
    Map [t_min_0, t_max_0] -> [t_min_1, t_max_1] with the unique linear
    isomorphism

    :param t_min_0: [in] minimum of the domain interval
    :param t_max_0: [in] maximum of the domain interval
    :param t_min_1: [in] minimum of the image interval
    :param t_max_1: [in] maximum of the domain interval
    :param t_0:     [in] point in the domain interval
    :return mapped point in the iamge
    """
    # Return the midpoint of the image if the input domain is trivial
    if float_equal(t_min_0, t_max_0):
        return 0.5 * (t_min_1 + t_max_1)

    # Perform the interpolation
    r_0: float = t_max_0 - t_min_0
    r_1: float = t_max_1 - t_min_1
    t_1: float = t_min_1 + (r_1 / r_0) * (t_0 - t_min_0)
    return t_1


def compute_point_cloud_bounding_box(points: MatrixNx3f) -> tuple[Vector1D, Vector1D]:
    """ 
    Compute the bounding box for a matrix of points in R^n.
    The points are assumed to be the rows of the points matrix.

    :param points: points to compute the bounding box for.
    :type points: np.ndarray

    :return (min_point, max_point): tuple of (point with minimum coordinates for the bounding box,
    point with maximum coordinates for the bounding box).
    :rtype: tuple[Vector1d, Vector1d]
    """
    num_points: int = points.shape[ROWS]
    dimension: int = points.shape[COLS]

    if num_points == 0:
        raise ValueError("num_points cannot be 0")
    if dimension == 0:
        raise ValueError("dimension cannot be 0")

    # Get minimum and maximum coordinates for the points
    min_point: Vector1D = points.min(axis=0)
    max_point: Vector1D = points.max(axis=0)
    assert min_point.ndim == 1
    assert max_point.ndim == 1
    return min_point, max_point


def remove_mesh_faces(V: MatrixNx3f,
                      F: MatrixNx3i,
                      faces_to_remove: list[FaceIndex]) -> tuple[MatrixNx3f, MatrixNx3i]:
    """
    Using igl to remove unreferenced vertices from V using faces_to_remove and updating F accordingly.

    :param V: vertices to remove unreferenced vertices from. np.ndarray of float
    :type V: np.ndarray
    :param F: faces with np.ndarray of int
    :type F: np.ndarray
    :param faces_to_remove: index of faces to remove
    :type faces_to_remove: list[int]

    :return: tuple of V and F submeshes (V, F)
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    faces_to_keep: list[FaceIndex] = index_vector_complement(
        faces_to_remove, F.shape[ROWS])  # rows

    # TODO: in the ASOC code, this F_unsimplified_submesh was initialized to shape
    # (faces_to_keep.size(), F.cols()) and then immediately resized.
    F_unsimplified_submesh: np.ndarray = np.ndarray(
        shape=(len(faces_to_keep), 3), dtype=int)

    for i, _ in enumerate(faces_to_keep):
        F_unsimplified_submesh[i, :] = F[faces_to_keep[i], :]

    # Remove unreferenced vertices and update face indices
    # FIXME: not sure if the below is doing what it needs to do
    F_submesh: np.ndarray
    V_submesh: np.ndarray
    # TODO: Pylint shows error with igl not having member, but I'm sure it is fine.
    F_submesh, V_submesh, _, _ = igl.remove_unreferenced(F, V)

    logger.info("Final mesh has %s faces and %s vertices",
                F_submesh.shape[ROWS], V_submesh.shape[ROWS])

    return V_submesh, F_submesh


def remove_mesh_vertices(V: MatrixNx3f,
                         F: MatrixNx3i,
                         vertices_to_remove: list[VertexIndex]
                         ) -> tuple[np.ndarray, np.ndarray, list[FaceIndex]]:
    """
    Removes mesh vertices from V based on the indices inside vertices_to_remove 
    and updates F accordingly.

    :param V: vertices matrix of floats
    :type V: np.ndarray
    :param F: faces matrix of integers
    :type F: np.ndarray
    :param vertices_to_remove: list of indices of vertices to remove
    :type vertices_to_remove: list[int]

    :return: tuple of vertex matrix with vertices removed, updated faces, and list 
    of face indices that were removed
    :rtype: tuple[np.ndarray, np.ndarray, list[FaceIndex]]
    """
    logger.info("Removing %s vertices from mesh with %s faces and %s vertices",
                len(vertices_to_remove), F.shape[ROWS], V.shape[ROWS])

    # Tag faces adjacent to the vertices to remove
    # TODO: implement some numpy version of of finding a vertex in a row of F
    faces_to_remove: list[FaceIndex] = []
    faces_to_remove.clear()
    for face_index in range(F.shape[0]):
        for i, _ in enumerate(vertices_to_remove):
            # NOTE: contains_vertex expects NumPy array of 1 dimension (i.e. shape (n , ))
            if contains_vertex(F[face_index, :], vertices_to_remove[i]):
                faces_to_remove.append(face_index)
                break
    logger.info("Remove %s faces", len(faces_to_remove))

    # Remove faces adjacent to cones
    V_submesh: np.ndarray
    F_submesh: np.ndarray
    V_submesh, F_submesh = remove_mesh_faces(V, F, faces_to_remove)

    return V_submesh, F_submesh, faces_to_remove


def join_path():
    todo()


def matrix_contains_nan(mat: np.ndarray) -> bool:
    assert mat.ndim > 1

    # TODO: add test case to check this function with ASOC code version
    return np.isnan(mat).any()


def vector_contains_nan(vec: Vector1D) -> bool:
    # assert vec.shape[0] == 1
    assert vec.ndim == 1
    # TODO: add test case to check this function with ASOC code version
    return np.isnan(vec).any()


def convert_polylines_to_edges(polylines: list[list[int]]) -> list[tuple[int, int]]:
    """
    Polylines are vector<vector<int>> in the original C++ code. 
    Meanwhile, this function returns list[array[int, 2]]. 
    So this takes the arbitrary length polylines and converts to list of list of 2 int elements.
    """

    # TODO: check to see if functionality of NumPy version is the same as the old one.
    edges: list[tuple[int, int]] = []
    for _, polyline in enumerate(polylines):
        # polyline equivalent to polylines[i]
        edge_length: int = len(polyline)
        for j in range(1, edge_length):
            edge: tuple[int, int] = (polyline[j - 1], polyline[j])
            edges.append(edge)

    return edges
