"""
Test cases for contour network
"""

import logging
import pathlib

import numpy as np
import pytest

from pyalgcon.contour_network.contour_network import (ContourNetwork,
                                                      _build_contour_labels)
from pyalgcon.core.common import (Matrix2x3f, Matrix4x4f, SpatialVector1d,
                                  compare_eigen_numpy_matrix,
                                  deserialize_eigen_matrix_csv_to_numpy)
from pyalgcon.exec.generate_algebraic_contours import \
    generate_algebraic_contours
from pyalgcon.utils.projected_curve_networks_utils import (
    SVGOutputMode, compare_segment_labels)

logger: logging.Logger = logging.getLogger(__name__)


def test_matrix_from_file_write(testing_fileinfo) -> None:
    """
    Testing writing contours with a camera matrix read in from file.
    """
    # Initialize parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = (base_data_folderpath / "contour_network" /
                              "FROM_MATRIX_FILE.svg")

    camera_matrix: Matrix4x4f = np.array(
        ((3.346195029192903236e-01, 3.371676824530447491e-03,
          -1.931632359086701556e-01, 3.634420946012050652e-02),
         (0.000000000000000000e+00, 3.863264718173403112e-01,
            6.743353649060895849e-03, -4.317130968153862908e-02),
         (1.931926600865509769e-01, -5.839915566789230343e-03,
          3.345685387482297268e-01, 1.937049982654145852e+00),
         (0.000000000000000000e+00, 0.000000000000000000e+00,
          0.000000000000000000e+00, 1.000000000000000000e+00)))

    generate_algebraic_contours(camera_matrix, filepath)


@pytest.mark.filterwarnings("ignore:loadtxt")
def test_compute_quantitative_invisibility_from_ray_intersections(testing_fileinfo) -> None:
    """

    """
    # Initialize parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = (base_data_folderpath / "contour_network" / "contour_network" /
                              "compute_quantitative_invisibility_from_ray_intersections")

    # Number from how many files there are
    for i in range(198):
        ray_mapping_coeffs: Matrix2x3f = deserialize_eigen_matrix_csv_to_numpy(
            filepath / "ray_mapping_coeffs" / f"{i}.csv")
        point: SpatialVector1d = deserialize_eigen_matrix_csv_to_numpy(
            filepath / "point" / f"{i}.csv")
        ray_intersections: list[float] = deserialize_eigen_matrix_csv_to_numpy(
            filepath / "ray_intersections" / f"{i}.csv").tolist()

        # Execute method
        qi_poll_element_test: int = (
            ContourNetwork._compute_quantitative_invisibility_from_ray_intersections(
                ray_mapping_coeffs, point, ray_intersections))

        # Compare results
        compare_eigen_numpy_matrix(filepath / "qi_poll_element" / f"{i}.csv",
                                   np.array(qi_poll_element_test))


def test_quantitative_invisibility(testing_fileinfo: tuple[pathlib.Path, pathlib.Path],
                                   initialize_contour_network: tuple[pathlib.Path, ContourNetwork]
                                   ) -> None:
    """
    Tests to see if the quantitative invisibility of PYAC is the same as ASOC. 
    If not, then the visibility of the contours will not be identical in PYAC to that of ASOC.
    """

    # Initialize parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = (base_data_folderpath / "contour_network" /
                              "compute_quantitative_invisibility")
    contour_network: ContourNetwork
    _, contour_network = initialize_contour_network

    # Execute method
    quantitative_invisibility: list[int] = contour_network.enumerate_quantitative_invisibility()

    # Compare results
    compare_eigen_numpy_matrix(filepath / "quantitative_invisibility.csv",
                               np.array(quantitative_invisibility))


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


@pytest.mark.parametrize("svg_output_mode", SVGOutputMode)
def test_write(svg_output_mode: SVGOutputMode,
               initialize_contour_network: tuple[pathlib.Path, ContourNetwork]) -> None:
    """
    Testing write for contour network.
    """
    # Retrieve parameters
    output_contour_folderpath: pathlib.Path
    contour_network: ContourNetwork
    output_contour_folderpath, contour_network = initialize_contour_network
    output_filepath: pathlib.Path = output_contour_folderpath / f"{svg_output_mode.name}.svg"
    show_nodes: bool = False

    # Save the contours to file
    logger.info("Saving contours to %s", output_filepath.resolve())
    contour_network.write(output_filepath,
                          svg_output_mode,
                          show_nodes)

    # TODO: Check if file has been written. But, be sure to check if this is a new output.svg.
    # i.e. remove the old output.svg safely


def test_rasterize(initialize_contour_network: tuple[pathlib.Path, ContourNetwork]) -> None:
    """
    Testing writing rasterized contour network.
    """
    # Retrieve parameters
    output_contour_folderpath: pathlib.Path
    contour_network: ContourNetwork
    output_contour_folderpath, contour_network = initialize_contour_network
    output_filepath: pathlib.Path = output_contour_folderpath / "contours.png"
    show_nodes: bool = False

    # Save the contours to file
    logger.info("Saving contours to %s", output_filepath.resolve())
    contour_network.write_rasterized_contours(output_filepath)

    # TODO: Check if file has been written. But, be sure to check if this is a new output.svg.
    # i.e. remove the old output.svg safely


def test_view_contours(initialize_contour_network) -> None:
    """ 
    Tests to see if we can view the contour network in Polyscope
    """
    # Retrieve parameters
    output_contour_folderpath: pathlib.Path
    contour_network: ContourNetwork
    output_contour_folderpath, contour_network = initialize_contour_network
    contour_network.view_contours()
