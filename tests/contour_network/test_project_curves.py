"""
test_project_curves.py

Used to test project_curves with spot mesh.
"""


import numpy as np

from pyalgcon.contour_network.project_curves import \
    project_curves
from pyalgcon.core.common import Matrix3x3f
from pyalgcon.core.rational_function import RationalFunction

from pyalgcon.utils.rational_function_testing_utils import (
    compare_rational_functions, deserialize_rational_functions)


def test_project_curves_spot_mesh() -> None:
    """
    Testing project_curves with spot mesh and identity matrix as the frame.
    """
    # Setting up parameters
    filepath: str = "spot_control\\contour_network\\project_curves\\"
    spatial_curves: list[RationalFunction] = deserialize_rational_functions(
        filepath + "spatial_curves.json")
    frame: Matrix3x3f = np.array([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]])

    planar_curves_control: list[RationalFunction] = deserialize_rational_functions(
        filepath + "planar_curves.json")
    planar_curves_test: list[RationalFunction] = project_curves(spatial_curves, frame)
    compare_rational_functions(planar_curves_control, planar_curves_test)
