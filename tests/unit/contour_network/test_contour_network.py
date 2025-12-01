"""
Test cases for contour network
"""

import logging
import pathlib

import numpy as np

from pyalgcon.contour_network.contour_network import (ContourNetwork,
                                                      _build_contour_labels)
from pyalgcon.core.common import deserialize_eigen_matrix_csv_to_numpy
from pyalgcon.utils.projected_curve_networks_utils import (
    SVGOutputMode, compare_segment_labels)

logger: logging.Logger = logging.getLogger(__name__)


# TODO: the deserialization of rational functions and then printing of rational
#  functions should be the same as the whole rational functions.txt file

def test_build_contour_labels(testing_fileinfo: tuple[pathlib.Path, pathlib.Path]) -> None:
    """
    Testing build contour labels for contour network.
    """
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = (base_data_folderpath / "contour_network" /
                              "contour_network" / "build_contour_labels")

    contour_patch_indices: list[int] = deserialize_eigen_matrix_csv_to_numpy(
        filepath / "contour_patch_indices.csv").tolist()
    contour_is_boundary: list[bool] = np.array(deserialize_eigen_matrix_csv_to_numpy(
        filepath / "contour_is_boundary.csv"), dtype=bool).tolist()

    contour_segment_labels_test: list[dict[str, int]] = _build_contour_labels(
        contour_patch_indices,
        contour_is_boundary)

    compare_segment_labels(filepath / "contour_segment_labels.json",
                           contour_segment_labels_test)


def test_write(initialize_contour_network: tuple[pathlib.Path,
                                                 ContourNetwork,
                                                 list[tuple[int, int]]]) -> None:
    """
    Testing write for contour network.
    """
    output_filepath: pathlib.Path
    contour_network: ContourNetwork
    output_filepath, contour_network = initialize_contour_network
    svg_output_mode: SVGOutputMode = SVGOutputMode.UNIFORM_VISIBLE_CURVES
    show_nodes: bool = False

    # Save the contours to file
    logger.info("Saving contours to %s", output_filepath.resolve())
    contour_network.write(output_filepath, svg_output_mode, show_nodes)
