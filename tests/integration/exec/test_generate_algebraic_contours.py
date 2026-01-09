"""
Testing Blender-facing generate_algebraic_contours.py
Essentially is an integration test
"""

import pathlib

import numpy as np
import pytest

from pyalgcon.exec.generate_algebraic_contours import \
    generate_algebraic_contours


@pytest.mark.parametrize("camera_matrix", np.array(((1.0000,  0.0000,  0.0000, 0.0000),
                                                    (0.0000, -0.7071,  -0.7071, 0.0000),
                                                    (0.0000,  -0.7071,  0.7071, 10.000),
                                                    (0.0000,  0.0000,  0.0000, 1.0000),)))
def test_generate_algebraic_contours(camera_matrix: np.ndarray,
                                     testing_fileinfo: tuple[pathlib.Path, pathlib.Path]) -> None:
    """
    Generates algebraic contours.svg and .png based on a camera file.
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    folderpath: pathlib.Path = base_data_folderpath / "exec" / "generate_algebraic_contours"

    generate_algebraic_contours(camera_matrix, base_data_folderpath)
