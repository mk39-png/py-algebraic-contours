"""
Test cases for contour network
"""

import logging
import pathlib

import numpy as np
import pytest

from pyalgcon.contour_network.contour_network import (ContourNetwork,
                                                      _build_contour_labels)
from pyalgcon.core.common import (Matrix2x3f, SpatialVector1d,
                                  compare_eigen_numpy_matrix,
                                  deserialize_eigen_matrix_csv_to_numpy)
from pyalgcon.utils.projected_curve_networks_utils import (
    SVGOutputMode, compare_segment_labels)

logger: logging.Logger = logging.getLogger(__name__)


@pytest.mark.filterwarnings("ignore:loadtxt")
def test_compute_quantitative_invisibility_from_ray_intersections(
        testing_fileinfo) -> None:
    """
    Testing quantitative invisibility function.
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

    # NOTE: it is expected that quantitative_invisibility will have differing inputs between
    # ASOC and PYAC, especially for the "propagation" method.
    # As a result, the below has been commented out.
    # Compare results
    # compare_eigen_numpy_matrix(filepath / "quantitative_invisibility.csv",
    #                            np.array(quantitative_invisibility))


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
@pytest.mark.parametrize("show_nodes", [True, False])
def test_write(svg_output_mode: SVGOutputMode,
               show_nodes: bool,
               initialize_contour_network: tuple[pathlib.Path, ContourNetwork],
               ) -> None:
    """
    Testing write for contour network.
    """
    # Retrieve parameters
    output_contour_folderpath: pathlib.Path
    contour_network: ContourNetwork
    output_contour_folderpath, contour_network = initialize_contour_network
    output_filepath: pathlib.Path = output_contour_folderpath / \
        "projected_curve_network" / "write" / f"{svg_output_mode.name}_show_nodes-{show_nodes}.svg"

    # Remove pre-existing file
    output_filepath.unlink(missing_ok=True)

    # Save the contours to file
    logger.info("Saving contours to %s", output_filepath.resolve())
    contour_network.write(output_filepath,
                          svg_output_mode,
                          show_nodes)

    assert output_filepath.is_file()

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
    output_filepath: pathlib.Path = output_contour_folderpath / \
        "contour_network" / "write_rasterized_contours" / "rasterized_contours.png"
    show_nodes: bool = False

    # Remove pre-existing result
    output_filepath.unlink(missing_ok=True)

    # Save the contours to file
    logger.info("Saving contours to %s", output_filepath.resolve())
    contour_network.write_rasterized_contours(output_filepath)

    assert output_filepath.exists()


def test_view_contours(initialize_contour_network, no_gui) -> None:
    """ 
    Tests to see if we can view the contour network in Polyscope
    """
    # NOTE: remove no_gui if wanting to see the polyscope viewer and its contours
    # Retrieve parameters
    output_contour_folderpath: pathlib.Path
    contour_network: ContourNetwork
    output_contour_folderpath, contour_network = initialize_contour_network
    contour_network.view_contours()
