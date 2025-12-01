"""
debug.py

Holds methods for comparing NumPy matrices to Eigen matrices, 
Python types to C++ types (i.e. AffineManifold from PYASOC to ASOC), and
also data used for inline testing.

So, these are conversions of the common.py methods that parse from file.

Utilizes and centralizes absolute file paths.
"""

import csv
import logging
import pathlib  # used for testing

import numpy as np
import numpy.testing as npt

logger: logging.Logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG)

# **********************
# Debug/Helper Methods (keep here for now, especially when comparing later contours code)
# **********************


# def compare_intersection_points(filename: str,
#                                 numpy_array: np.ndarray,
#                                 num_intersections: int) -> None:
#     """
#     Special function of compare_eigen_numpy() that takes in num_intersections and ignores the
#     rest of the irrelevant parts of the matrix to compare.
#     Since the original C++ code only initializes the values of the points that are relevant.
#     e.g
#     ACTUAL: array([[0.108474, 0.195645],
#               [0.      , 0.      ],
#               [0.      , 0.      ],
#               [0.      , 0.      ]])
#     DESIRED: array([[1.084738e-001, 1.956454e-001],
#           [6.953336e-310, 2.470328e-323],
#           [6.953336e-310, 4.728331e-310],
#           [0.000000e+000, 1.976263e-323]])
#     But we only care about the first row since this only has 1 intersection.
#     """
#     eigen_array: np.ndarray = deserialize_eigen_matrix_csv_to_numpy(filename)
#     npt.assert_allclose(
#         numpy_array[:num_intersections], eigen_array[:num_intersections], atol=1e-5)


#
# LIST COMPARISON
#
def compare_list_list_varying_lengths_float_from_file(filepath: pathlib.Path, rows_test: list[list[float]], precision=0.0) -> None:
    """
    Used when the csv list contains list with varying list lengths.
    e.g. 
    [
    [1, 2, 3, 4],
    [1, 2],
    [5, 7, 8, 8, 19, 1],
    ]
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


def deserialize_list_list_varying_lengths_from_file(filepath: pathlib.Path) -> list[list[int]]:
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
