"""
Test abstract curve network
"""

import pathlib

import numpy as np
import pytest

from pyalgcon.core.abstract_curve_network import AbstractCurveNetwork
from pyalgcon.core.common import (NodeIndex, SegmentIndex,
                                  compare_eigen_numpy_matrix,
                                  deserialize_eigen_matrix_csv_to_numpy)


#
# Fixtures
#
@pytest.fixture(scope="module")
def initialize_to_out_arrays(testing_fileinfo: tuple[pathlib.Path, pathlib.Path]
                             ) -> tuple[list[NodeIndex],
                                        list[SegmentIndex]]:
    """ Initializes to_array and out_array and returns them for testing

    :param testing_fileinfo: used for folderpath of the current testing mesh
    :type testing_fileinfo: tuple[pathlib.Path, pathlib.Path]
    :return: to_array, out_array
    :rtype: tuple[list[NodeIndex], list[SegmentIndex]]
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / "core" / \
        "abstract_curve_network" / "init_abstract_curve_network"

    # Execute method
    to_array: list[NodeIndex] = np.array(deserialize_eigen_matrix_csv_to_numpy(
        filepath / "to_array.csv"),
        dtype=np.int64).tolist()
    out_array: list[SegmentIndex] = np.array(deserialize_eigen_matrix_csv_to_numpy(
        filepath / "out_array.csv"),
        dtype=np.int64).tolist()

    # Return results
    return to_array, out_array


#
# Tests
#

def test_build_next_array(testing_fileinfo,
                          initialize_to_out_arrays) -> None:
    """ 
    Unit test for build next array
    Indirectly tests the abtract_curve_network constructor
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / "core" / \
        "abstract_curve_network" / "build_next_array"
    to_array: list[NodeIndex]
    out_array: list[SegmentIndex]
    to_array, out_array = initialize_to_out_arrays

    # Execute method
    next_array: list[SegmentIndex] = AbstractCurveNetwork.build_next_array(to_array, out_array)

    # Compare results
    compare_eigen_numpy_matrix(filepath / "next_array.csv",
                               np.array(next_array, dtype=np.int64))


def test_build_prev_array(testing_fileinfo,
                          initialize_to_out_arrays) -> None:
    """ 
    Unit test for build next array
    Indirectly tests the abtract_curve_network constructor
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / "core" / \
        "abstract_curve_network" / "build_prev_array"
    to_array: list[NodeIndex]
    out_array: list[SegmentIndex]
    to_array, out_array = initialize_to_out_arrays

    # Execute method
    prev_array: list[SegmentIndex] = AbstractCurveNetwork.build_prev_array(to_array, out_array)

    # Compare results
    compare_eigen_numpy_matrix(filepath / "prev_array.csv",
                               np.array(prev_array, dtype=np.int64))


def test_build_from_array(testing_fileinfo,
                          initialize_to_out_arrays) -> None:
    """ 
    Unit test for build next array
    Indirectly tests the abtract_curve_network constructor
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / "core" /  \
        "abstract_curve_network" / "build_from_array"
    to_array: list[NodeIndex]
    out_array: list[SegmentIndex]
    to_array, out_array = initialize_to_out_arrays

    # Execute method
    from_array: list[NodeIndex] = AbstractCurveNetwork.build_from_array(to_array, out_array)

    # Compare results
    compare_eigen_numpy_matrix(filepath / "from_array.csv",
                               np.array(from_array, dtype=np.int64))


def test_build_in_array(testing_fileinfo,
                        initialize_to_out_arrays) -> None:
    """ 
    Unit test for build next array
    Indirectly tests the abtract_curve_network constructor
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / "core" / \
        "abstract_curve_network" / "build_in_array"
    to_array: list[NodeIndex]
    out_array: list[SegmentIndex]
    to_array, out_array = initialize_to_out_arrays

    # Execute method
    in_array: list[SegmentIndex] = AbstractCurveNetwork.build_in_array(to_array, out_array)

    # Compare results
    compare_eigen_numpy_matrix(filepath / "in_array.csv",
                               np.array(in_array, dtype=np.int64))
