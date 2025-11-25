import argparse
import logging
import os
import sys

import numpy as np
import numpy.testing as npt
import pytest

from pyalgcon.contour_network.contour_network import *
from pyalgcon.core.apply_transformation import *
from pyalgcon.core.common import *
from pyalgcon.core.compute_boundaries import *
from pyalgcon.core.convex_polygon import ConvexPolygon
from pyalgcon.core.generate_transformation import *
from pyalgcon.quadratic_spline_surface.optimize_spline_surface import *
from pyalgcon.quadratic_spline_surface.quadratic_spline_surface import *
from pyalgcon.quadratic_spline_surface.quadratic_spline_surface import \
    QuadraticSplineSurface
from pyalgcon.quadratic_spline_surface.quadratic_spline_surface_patch import \
    QuadraticSplineSurfacePatch
from pyalgcon.quadratic_spline_surface.twelve_split_spline import *


def test_read_write_spline_surface_serialization() -> None:
    """
    # test by deserializing control file, then serialize numpy code and compare to see if get the same file back
    Though, we would have to read through the files and compare the inputs...
    Deserialize, serialize, then check if true values..

    Utilizes write_spline and read_spline
    """
    filename_control: str = "spot_control_mesh-cleaned_conf_simplified_with_uv_CONTROL.txt"
    filepath_control: str = os.path.abspath(f"src\\tests\\spot_control\\{filename_control}")
    filename_test: str = "spot_control_mesh-cleaned_conf_simplified_with_uv_NUMPY.txt"
    filepath_test: str = os.path.abspath(f"src\\tests\\spot_control\\{filename_test}")

    # NOTE: need a placeholder to call write_spline() and deserialize()...
    # TODO: better to separate deserialize() and write_spline() as separate from, QuadraticSplineSurface class and take QuadraticSplineSurface as the parameter or whatnot.
    spline_surface_placeholder: QuadraticSplineSurface = QuadraticSplineSurface.from_file(
        filepath=filepath_control)
    # Write the saved spline data to an external file. So, converting Eigen TXT -> Numpy Implementation -> NumPy TXT
    spline_surface_placeholder.write_spline(filepath_test)

    # FIXME: make deserialize() independent of QudaraticSplineSurface...
    # TODO: now compare the files to see that they ghave the same contents
    control_patches: list[QuadraticSplineSurfacePatch]
    test_patches: list[QuadraticSplineSurfacePatch]

    # First open files to convert into list[QuadraticSplineSurfacePatch]
    with open(filepath_control, "r", encoding="utf-8") as file_control:
        control_patches: list[QuadraticSplineSurfacePatch] = spline_surface_placeholder.deserialize(
            file_control)
        file_control.close()

    with open(filepath_test, "r", encoding="utf-8") as file_test:
        test_patches: list[QuadraticSplineSurfacePatch] = spline_surface_placeholder.deserialize(
            file_test)
        file_test.close()

    assert len(control_patches) == len(test_patches)
    num_patches: int = len(control_patches)

    # Now, checking the values that have been saved
    for i in range(num_patches):
        # cx, cy, cz
        surface_mapping_coeffs_control: Matrix6x3r = control_patches[i].surface_mapping
        domain_control: ConvexPolygon = control_patches[i].domain
        vertices_control: Matrix3x2r = domain_control.vertices  # p1, p2, p3

        surface_mapping_coeffs_test: Matrix6x3r = test_patches[i].surface_mapping  # cx, cy, cz
        domain_test: ConvexPolygon = test_patches[i].domain
        vertices_test: Matrix3x2r = domain_test.vertices  # p1, p2, p3

        npt.assert_allclose(surface_mapping_coeffs_control,
                            surface_mapping_coeffs_test, atol=FLOAT_EQUAL_PRECISION)
        npt.assert_allclose(vertices_control, vertices_test, atol=FLOAT_EQUAL_PRECISION)


def test_read_view_spline_surface_deserialization() -> None:
    """
    Testing to see that the file can be deserialized and the surface displayed properly.
    """

    filename: str = "spot_control_mesh-cleaned_conf_simplified_with_uv_CONTROL.txt"
    filepath: str = os.path.abspath(f"src\\tests\\spot_control\\{filename}")

    spline_surface: QuadraticSplineSurface = QuadraticSplineSurface.from_file(filepath=filepath)
    spline_surface.view()
