"""
test_project_curves.py

Used to test project_curves.
"""


import pathlib

import numpy as np

from pyalgcon.contour_network.project_curves import project_curves
from pyalgcon.core.common import Matrix3x3f
from pyalgcon.core.rational_function import RationalFunction
from pyalgcon.utils.rational_function_testing_utils import (
    compare_rational_functions_from_file,
    deserialize_rational_functions_from_file)


def test_project_curves(testing_fileinfo) -> None:
    """
    Testing project_curves with spot mesh and identity matrix as the frame.
    """
    # Retrieving parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = (
        base_data_folderpath / "contour_network" / "project_curves")

    # Reading parameter from file
    spatial_curves: list[RationalFunction] = deserialize_rational_functions_from_file(
        filepath / "spatial_curves.json")
    frame: Matrix3x3f = np.array([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]])

    # Execute method
    planar_curves_test: list[RationalFunction] = project_curves(spatial_curves, frame)

    # Compare results
    compare_rational_functions_from_file(
        filepath / "planar_curves.json", planar_curves_test)
