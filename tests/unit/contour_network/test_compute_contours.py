"""
Test compute contours    
"""
import logging
import pathlib

import numpy as np

from pyalgcon.contour_network.compute_contours import (
    compute_spline_surface_boundaries,
    compute_spline_surface_boundary_intersections,
    compute_spline_surface_contours,
    compute_spline_surface_contours_and_boundaries, pad_contours)
from pyalgcon.contour_network.contour_network import InvisibilityParameters
from pyalgcon.contour_network.intersection_data import IntersectionData
from pyalgcon.core.common import (Matrix3x3f, PatchIndex,
                                  compare_eigen_numpy_matrix,
                                  deserialize_eigen_matrix_csv_to_numpy)
from pyalgcon.core.conic import Conic
from pyalgcon.core.rational_function import RationalFunction
from pyalgcon.quadratic_spline_surface.twelve_split_spline import \
    TwelveSplitSplineSurface
from pyalgcon.utils.compute_intersections_testing_utils import \
    compare_list_list_intersection_data_from_file
from pyalgcon.utils.conic_testing_utils import (compare_conics_from_file,
                                                deserialize_conics_from_file)
from pyalgcon.utils.rational_function_testing_utils import (
    compare_rational_functions_from_file,
    deserialize_rational_functions_from_file)

logger: logging.Logger = logging.getLogger(__name__)


# ********************
# Main Testing Methods
# ********************

def test_pad_contours(testing_fileinfo: tuple[pathlib.Path, pathlib.Path]) -> None:
    """
    Test major part of init_contour_network.
    """
    # Retrieve parameters
    folder_path: pathlib.Path
    folder_path, _ = testing_fileinfo

    filepath: pathlib.Path = folder_path / "contour_network" / "compute_contours" / "pad_contours"
    contour_domain_curve_segments: list[Conic] = deserialize_conics_from_file(
        filepath / "contour_domain_curve_segments.json")
    contour_segments: list[RationalFunction] = deserialize_rational_functions_from_file(
        filepath / "contour_segments.json")
    planar_contour_segments: list[RationalFunction] = deserialize_rational_functions_from_file(
        filepath / "planar_contour_segments.json")
    invisibility_params = InvisibilityParameters()

    # Execute method to test
    pad_contours(contour_domain_curve_segments,
                 contour_segments,
                 planar_contour_segments,
                 invisibility_params.pad_amount)

    # TODO: make a function that compares from file.
    # contour_domain_curve_segments_padded: list[Conic] = deserialize_conics_absolute(
    #     filepath / "contour_domain_curve_segments_PADDED.json")
    # contour_segments_padded: list[RationalFunction] = deserialize_rational_functions_absolute(
    #     filepath / "contour_segments_PADDED.json")
    # planar_contour_segments_padded: list[RationalFunction] = deserialize_rational_functions_absolute(
    #     filepath / "planar_contour_segments_PADDED.json")

    # Compare results
    compare_conics_from_file(
        filepath / "contour_domain_curve_segments_PADDED.json", contour_domain_curve_segments)
    compare_rational_functions_from_file(
        filepath / "contour_segments_PADDED.json", contour_segments)
    compare_rational_functions_from_file(
        filepath / "planar_contour_segments_PADDED.json", planar_contour_segments)


# ********************************************************************************
# compute_spline_surface_contours_and_boundaries Testing Methods
# ********************************************************************************


def test_compute_spline_surface_contours(testing_fileinfo,
                                         twelve_split_spline_transformed) -> None:
    """
    Test first major part of compute_spline_surface_contours_and_boundaries.

    As in, testing compute_spline_surface_contours as a part of 
    compute_spline_surface_contours_and_boundaries.
    """
    # Parameter setup
    base_folderpath: pathlib.Path
    base_folderpath, _ = testing_fileinfo
    spline_surface: TwelveSplitSplineSurface = twelve_split_spline_transformed
    # FIXME: take in frame as parameter from fixture parameter
    frame: Matrix3x3f = np.identity(3)

    # Compute contours
    contour_domain_curve_segments: list[Conic]
    contour_segments: list[RationalFunction]  # <degree 4, dimension 3>
    contour_patch_indices: list[PatchIndex]
    line_intersection_indices: list[tuple[int, int]]
    (contour_domain_curve_segments,
     contour_segments,
     contour_patch_indices,
     line_intersection_indices) = compute_spline_surface_contours(spline_surface,
                                                                  frame)

    # Comparing results
    data_folderpath: pathlib.Path = base_folderpath / "contour_network" / \
        "compute_contours" / "compute_spline_surface_contours"
    compare_conics_from_file(
        data_folderpath / "contour_domain_curve_segments.json", contour_domain_curve_segments)
    compare_rational_functions_from_file(
        data_folderpath / "contour_segments.json", contour_segments)
    compare_eigen_numpy_matrix(
        data_folderpath / "line_intersection_indices.csv", np.array(line_intersection_indices))
    compare_eigen_numpy_matrix(
        data_folderpath / "contour_patch_indices.csv", np.array(contour_patch_indices))


def test_compute_spline_surface_boundaries(testing_fileinfo,
                                           twelve_split_spline_transformed,
                                           initialize_patch_boundary_edges) -> None:
    """
    Test second major part of compute_spline_surface_contours_and_boundaries.

    NOTE: the parameters and return values for compute_spline_surface_boundaries() for this case
    should be empty for this spot mesh control case.

    As in, testing compute_spline_surface_boundaries as a part of 
    compute_spline_surface_contours_and_boundaries.
    """
    # Retrieving parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    spline_surface: TwelveSplitSplineSurface = twelve_split_spline_transformed
    patch_boundary_edges: list[tuple[int, int]] = initialize_patch_boundary_edges
    filepath: pathlib.Path = base_data_folderpath / "contour_network" / \
        "compute_contours" / "compute_spline_surface_boundaries"

    # Compute boundaries
    boundary_domain_curve_segments: list[Conic]
    boundary_segments: list[RationalFunction]  # <degree 4, dimension 3>
    boundary_patch_indices: list[PatchIndex]
    (boundary_domain_curve_segments,
     boundary_segments,
     boundary_patch_indices) = compute_spline_surface_boundaries(spline_surface,
                                                                 patch_boundary_edges)

    # NOTE: Both should be empty
    # Compare results
    compare_conics_from_file(filepath / "boundary_domain_curve_segments.json",
                             boundary_domain_curve_segments)
    compare_rational_functions_from_file(filepath / "boundary_segments.json", boundary_segments)
    compare_eigen_numpy_matrix(filepath / "boundary_patch_indices.csv",
                               np.array(boundary_patch_indices))


def test_compute_spline_surface_boundary_intersections(
        testing_fileinfo,
        twelve_split_spline_transformed,
        initialize_patch_boundary_edges) -> None:
    """ 
    Testing compute_spline_surface_boundary_intersection() as part of 
    compute_spline_surface_contours_and_boundaries()
    """
    # Retrieving parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    spline_surface: TwelveSplitSplineSurface = twelve_split_spline_transformed
    patch_boundary_edges: list[tuple[int, int]] = initialize_patch_boundary_edges
    filepath: pathlib.Path = base_data_folderpath / "contour_network" / \
        "compute_contours" / "compute_spline_surface_boundary_intersections"

    # Reading parameters from file
    contour_domain_curve_segments: list[Conic] = deserialize_conics_from_file(
        filepath / "contour_domain_curve_segments.json")
    contour_patch_indices: list[PatchIndex] = np.array(
        deserialize_eigen_matrix_csv_to_numpy(filepath / "contour_patch_indices.csv"),
        dtype=np.int64).tolist()
    line_intersection_indices: list[tuple[int, int]] = np.array(
        deserialize_eigen_matrix_csv_to_numpy(filepath / "line_intersection_indices.csv"),
        dtype=np.int64).tolist()
    boundary_domain_curve_segments: list[Conic] = deserialize_conics_from_file(
        filepath / "boundary_domain_curve_segments.json")

    # Execute method
    contour_intersections: list[list[IntersectionData]]
    num_intersections: int
    contour_intersections, num_intersections = compute_spline_surface_boundary_intersections(
        spline_surface,
        contour_domain_curve_segments,
        contour_patch_indices,
        line_intersection_indices,
        patch_boundary_edges,
        boundary_domain_curve_segments)

    # Compare results
    compare_list_list_intersection_data_from_file(
        filepath / "contour_intersections.json", contour_intersections)
    assert num_intersections == 0


def test_compute_spline_surface_contours_and_boundaries(
        testing_fileinfo: tuple[pathlib.Path, pathlib.Path],
        twelve_split_spline_transformed: TwelveSplitSplineSurface) -> None:
    """
    Part of testing for init_contour_network
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    spline_surface: TwelveSplitSplineSurface = twelve_split_spline_transformed
    filepath: pathlib.Path = base_data_folderpath / "contour_network" / \
        "compute_contours" / "compute_spline_surface_contours_and_boundaries"

    # Execute method
    # TODO: deal with case when not using identity(3) frame?
    contour_domain_curve_segments: list[Conic]
    contour_segments: list[RationalFunction]
    contour_patch_indices: list[PatchIndex]
    contour_is_boundary: list[bool]
    contour_intersections: list[list[IntersectionData]]
    num_intersections: int
    (contour_domain_curve_segments,
     contour_segments,
     contour_patch_indices,
     contour_is_boundary,
     contour_intersections,
     num_intersections) = compute_spline_surface_contours_and_boundaries(
        spline_surface,
        np.identity(3),
        deserialize_eigen_matrix_csv_to_numpy(filepath / "patch_boundary_edges.csv").tolist())

    # Comparing results
    compare_conics_from_file(filepath / "contour_domain_curve_segments.json",
                             contour_domain_curve_segments)
    compare_rational_functions_from_file(filepath / "contour_segments.json",
                                         contour_segments)
    compare_eigen_numpy_matrix(filepath / "contour_patch_indices.csv",
                               np.array(contour_patch_indices))
    compare_eigen_numpy_matrix(filepath / "contour_is_boundary.csv",
                               np.array(contour_is_boundary))
    compare_list_list_intersection_data_from_file(filepath / "contour_intersections.json",
                                                  contour_intersections)
    assert num_intersections == 0
