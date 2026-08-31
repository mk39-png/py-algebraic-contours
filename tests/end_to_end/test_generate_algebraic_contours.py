"""
Testing Blender-facing generate_algebraic_contours.py
Essentially is an integration test
"""

import pathlib

import numpy as np
import pytest

from pyalgcon.contour_network.contour_network import (InvisibilityMethod,
                                                      InvisibilityParameters)
from pyalgcon.pipelines.generate_algebraic_contours import \
    generate_algebraic_contours

invisibility_params = InvisibilityParameters()

# NOTE: separate camera matrices to avoid sharing the same camera matrix used by the other test
# cases since the other test cases have ther inputs and outputs limited to that of
# spot_quadrangulated_tri_clean_camera.csv


@pytest.mark.parametrize(argnames="camera_matrix",
                         argvalues=[np.array(
                             [[1.0000, 0.0000,  0.0000, 0.0000],
                              [0.0000, 1.0000,  0.0000, 0.0000],
                              [0.0000, 0.0000,  1.0000, 5.0000],
                              [0.0000, 0.0000,  0.0000, 1.0000]])])
@pytest.mark.parametrize("method", list(InvisibilityMethod))
@pytest.mark.parametrize("show_nodes", [True, False])
def test_generate_algebraic_contours(camera_matrix: np.ndarray,
                                     testing_fileinfo: tuple[pathlib.Path, pathlib.Path],
                                     method: InvisibilityMethod,
                                     show_nodes: bool) -> None:
    """
    Generates algebraic contours.svg and .png based on a camera file.
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, obj_filepath = testing_fileinfo
    folderpath: pathlib.Path = base_data_folderpath / "exec" / "generate_algebraic_contours"
    output_filepath: pathlib.Path = folderpath / f"{method.name}_show_nodes-{show_nodes}.svg"
    assert type(method) is InvisibilityMethod
    invisibility_params.invisibility_method = method

    # Remove pre-existing file
    output_filepath.unlink(missing_ok=True)

    generate_algebraic_contours(camera_matrix, obj_filepath, output_filepath,
                                invisibility_params=invisibility_params,
                                show_nodes=show_nodes)

    assert output_filepath.is_file()
