"""
debug.py

Holds methods for comparing NumPy matrices to Eigen matrices, 
Python types to C++ types (i.e. AffineManifold from PYASOC to ASOC), and
also data used for inline testing.

So, these are conversions of the common.py methods that parse from file.

Utilizes and centralizes absolute file paths.
"""

import csv
import json  # for testing
import logging
import os
import pathlib  # used for testing
from io import StringIO

import igl
import numpy as np
import numpy.testing as npt
import numpy.typing as npty

from pyalgcon.core.rational_function import RationalFunction
from pyalgcon.utils.rational_function_testing_utils import \
    deserialize_rational_function

logger: logging.Logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG)

# **********************
# Debug/Helper Methods (keep here for now, especially when comparing later contours code)
# **********************


def resolve_filepath():
    """
    Finds 


    """


def load_json_absolute(filepath: pathlib.Path) -> list[dict] | list[list[dict]] | list[list[list]]:
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


def compare_eigen_numpy_matrix_absolute(file_path: pathlib.Path,
                                        numpy_array: np.ndarray,
                                        make_3d: bool = False) -> None:
    """ 
    Standardized function for comparing matrices for testing.
    Takes a filename and creates an absolute filepath to src/tests/
    """
    eigen_array: np.ndarray = deserialize_eigen_matrix_csv_to_numpy_absolute(
        file_path, make_3d)
    npt.assert_allclose(numpy_array, eigen_array)


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


def deserialize_eigen_matrix_csv_to_numpy_absolute(file_path: pathlib.Path,
                                                   make_3d: bool = False) -> np.ndarray:
    """
    Turns Eigen matrix .csv into NumPy array
    Used for testing.

    :param make_3d: parses a 3D csv into a 3D nnumpy array. 
    usually equivalent structure is a list of matrices.
    """

    arr: np.ndarray

    if make_3d:
        with open(file_path, 'r', encoding='utf-8') as file:
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
        arr = np.loadtxt(file_path, dtype=np.float64, delimiter=',')

    return arr


def deserialize_rational_functions_absolute(filepath: pathlib.Path) -> list[RationalFunction]:
    """
    Takes in a JSON file and deserializes it to list of RationalFunction objects.
    """
    # Intermediate processing.
    # TODO: add proper typehinting for dict
    rational_functions_intermediate: list[dict[str, int | float | bool | list]] = load_json_absolute(
        filepath)
    rational_functions_final: list[RationalFunction] = []

    for rational_function_intermediate in rational_functions_intermediate:
        rational_function_final: RationalFunction = deserialize_rational_function(
            rational_function_intermediate)

        # # Extract the following:
        # degree: int = rational_function.get("degree")
        # dimension: int = rational_function.get("dimension")

        # # NOTE: must transpose to be (degree + 1, dimension) shape.
        # numerator_coeffs: list[list[float]] = np.array(
        #     rational_function.get("numerator_coeffs"),
        #     dtype=np.float64).T

        # denominator_coeffs: list[list[float]] = np.array(
        #     rational_function.get("denominator_coeffs"),
        #     dtype=np.float64).squeeze()

        # # Getting the interval.
        # t0: float = rational_function.get("domain").get("t0")
        # t1: float = rational_function.get("domain").get("t1")

        # # NOTE: the below probably is not needed, but it's good to confirm nonetheless
        # bounded_below: bool = rational_function.get("domain").get("bounded_below")
        # bounded_above: bool = rational_function.get("domain").get("bounded_above")
        # open_below: bool = rational_function.get("domain").get("open_below")
        # open_above: bool = rational_function.get("domain").get("open_above")

        # domain: Interval = Interval(t0, t1)
        # domain.set_lower_bound(t0)
        # domain.set_upper_bound(t1)

        # rational_function_final: RationalFunction = RationalFunction(
        #     degree,
        #     dimension,
        #     numerator_coeffs,
        #     denominator_coeffs,
        #     domain
        # )

        rational_functions_final.append(rational_function_final)

    return rational_functions_final


def compare_list_list_varying_lengths_absolute(filepath: pathlib.Path, rows_test: list[list[int]]) -> None:
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


#
# Segment Labels -- Deserialization Methods
#
def compare_segment_labels_absolute(filepath: pathlib.Path, segment_labels_test: list[dict[str, int]]) -> None:
    """

    """
    segment_labels_control: list[dict[str, int]] = deserialize_segment_labels_absolute(filepath)
    assert len(segment_labels_control) == len(segment_labels_test)

    # Compare element by element
    for _, (control, test) in enumerate(zip(segment_labels_control, segment_labels_test)):
        assert control == test


def deserialize_segment_labels_absolute(filepath: pathlib.Path) -> list[dict[str, int]]:
    """
    Simple deserialization of segment_labels.json.
    Since segment_labels used to be a vector<map<string, int>>
    """
    # TODO: fix type hint
    segment_labels_intermediate: list[dict[str, int]] = load_json_absolute(filepath)
    return segment_labels_intermediate
