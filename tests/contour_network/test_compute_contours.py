
import logging
import os

import numpy as np

from pyalgcon.contour_network.compute_contours import (
    compute_spline_surface_boundaries,
    compute_spline_surface_boundary_intersections,
    compute_spline_surface_contours,
    compute_spline_surface_contours_and_boundaries, pad_contours)
from pyalgcon.contour_network.compute_intersections import \
    IntersectionParameters
from pyalgcon.contour_network.contour_network import (
    ContourNetwork, InvisibilityMethod, InvisibilityParameters)
from pyalgcon.contour_network.intersection_data import \
    IntersectionData
from pyalgcon.core.affine_manifold import AffineManifold
from pyalgcon.core.apply_transformation import \
    apply_camera_frame_transformation_to_vertices
from pyalgcon.core.common import (
    Matrix3x3f, MatrixNx3f, PatchIndex, compare_eigen_numpy_matrix,
    deserialize_eigen_matrix_csv_to_numpy, initialize_spot_control_mesh,
    unimplemented)
from pyalgcon.core.conic import Conic
from pyalgcon.core.rational_function import RationalFunction
from pyalgcon.quadratic_spline_surface.optimize_spline_surface import \
    OptimizationParameters
from pyalgcon.quadratic_spline_surface.quadratic_spline_surface import \
    QuadraticSplineSurface
from pyalgcon.quadratic_spline_surface.twelve_split_spline import (
    TwelveSplitSplineSurface, compute_twelve_split_spline_patch_boundary_edges)

from pyalgcon.utils.compute_intersections_testing_utils import (
    compare_list_list_intersection_data,
    compare_list_list_intersection_data_from_file,
    deserialize_list_list_intersection_data)
from pyalgcon.utils.conic_testing_utils import (compare_conics,
                                                compare_conics_from_file,
                                                deserialize_conics)
from pyalgcon.utils.projected_curve_networks_utils import SVGOutputMode
from pyalgcon.utils.rational_function_testing_utils import (
    compare_rational_functions, compare_rational_functions_from_file,
    deserialize_rational_functions)

logger: logging.Logger = logging.getLogger(__name__)


# **************
# Helper Methods
# **************
def _initialize_contour_info_spot_mesh() -> tuple[TwelveSplitSplineSurface,
                                                  IntersectionParameters,
                                                  InvisibilityParameters,
                                                  list[tuple[int, int]],
                                                  Matrix3x3f]:
    """
    Helper function to initialize information for Contour Network testing.

    :return: spline_surface
    :return: intersect_params
    :return: invisibility_params
    :return: patch_boundary_edges
    :return: frame
    """
    svg_output_mode = SVGOutputMode.UNIFORM_VISIBLE_CURVES
    optimization_params = OptimizationParameters()
    intersect_params = IntersectionParameters()
    invisibility_params = InvisibilityParameters()
    weight: float = optimization_params.position_difference_factor
    trim: float = intersect_params.trim_amount
    pad: float = invisibility_params.pad_amount
    invisibility_method: InvisibilityMethod = invisibility_params.invisibility_method
    show_nodes: bool = False
    logger.setLevel(logging.INFO)

    # Set up the camera
    frame: Matrix3x3f = np.array([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]])
    logger.info("Projecting onto frame:\n%s", frame)
    V:  np.ndarray
    uv: np.ndarray
    F:  np.ndarray
    FT: np.ndarray
    V, uv, F, FT = initialize_spot_control_mesh()
    V_transformed: MatrixNx3f = apply_camera_frame_transformation_to_vertices(V, frame)

    # Generate quadratic spline
    logger.info("Comnputing spline surface")
    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)
    spline_surface: TwelveSplitSplineSurface = TwelveSplitSplineSurface(V_transformed,
                                                                        affine_manifold,
                                                                        optimization_params)

    # Get the boundary edges
    patch_boundary_edges: list[tuple[int, int]]
    patch_boundary_edges: list[tuple[int, int]] = (
        compute_twelve_split_spline_patch_boundary_edges(F, spline_surface.face_to_patch_indices))

    return spline_surface, intersect_params, invisibility_params, patch_boundary_edges, frame

# ********************
# Main Testing Methods
# ********************


def test_pad_contours_spot_mesh() -> None:
    """
    Test major part of init_contour_network.
    """
    filepath: str = "spot_control\\contour_network\\compute_contours\\pad_contours\\"
    contour_domain_curve_segments: list[Conic] = deserialize_conics(
        filepath+"contour_domain_curve_segments.json")
    contour_segments: list[RationalFunction] = deserialize_rational_functions(
        filepath+"contour_segments.json")
    planar_contour_segments: list[RationalFunction] = deserialize_rational_functions(
        filepath+"planar_contour_segments.json")
    invisibility_params = InvisibilityParameters()

    pad_contours(contour_domain_curve_segments,
                 contour_segments,
                 planar_contour_segments,
                 invisibility_params.pad_amount)

    contour_domain_curve_segments_padded: list[Conic] = deserialize_conics(
        filepath+"contour_domain_curve_segments_PADDED.json")
    contour_segments_padded: list[RationalFunction] = deserialize_rational_functions(
        filepath+"contour_segments_PADDED.json")
    planar_contour_segments_padded: list[RationalFunction] = deserialize_rational_functions(
        filepath+"planar_contour_segments_PADDED.json")

    compare_conics(contour_domain_curve_segments_padded, contour_domain_curve_segments)
    compare_rational_functions(contour_segments_padded, contour_segments)
    compare_rational_functions(planar_contour_segments_padded, planar_contour_segments)


# ********************************************************************************
# compute_spline_surface_contours_and_boundaries Testing Methods
# ********************************************************************************


def test_compute_spline_surface_contours_spot_control() -> None:
    """
    Test first major part of compute_spline_surface_contours_and_boundaries.

    As in, testing compute_spline_surface_contours as a part of 
    compute_spline_surface_contours_and_boundaries.
    """

    spline_surface: QuadraticSplineSurface
    intersect_params: IntersectionParameters
    invisibility_params: InvisibilityParameters
    patch_boundary_edges: list[tuple[int, int]]
    frame: Matrix3x3f
    (spline_surface,
        intersect_params,
        invisibility_params,
        patch_boundary_edges,
        frame) = _initialize_contour_info_spot_mesh()

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

    filepath = "spot_control\\contour_network\\compute_contours\\compute_spline_surface_contours\\"
    contour_domain_curve_segments_control: list[Conic] = deserialize_conics(
        filepath+"contour_domain_curve_segments.json")
    compare_conics(contour_domain_curve_segments_control, contour_domain_curve_segments)
    contour_segments_control: list[RationalFunction] = deserialize_rational_functions(
        filepath+"contour_segments.json")
    compare_rational_functions(contour_segments_control, contour_segments)
    compare_eigen_numpy_matrix(filepath+"line_intersection_indices.csv",
                               np.array(line_intersection_indices))
    compare_eigen_numpy_matrix(filepath+"contour_patch_indices.csv",
                               np.array(contour_patch_indices))


def test_compute_spline_surface_boundaries_spot_control() -> None:
    """
    Test second major part of compute_spline_surface_contours_and_boundaries.

    NOTE: the parameters and return values for compute_spline_surface_boundaries() for this case
    should be empty for this spot mesh control case.

    As in, testing compute_spline_surface_boundaries as a part of 
    compute_spline_surface_contours_and_boundaries.
    """

    spline_surface: QuadraticSplineSurface
    intersect_params: IntersectionParameters
    invisibility_params: InvisibilityParameters
    patch_boundary_edges: list[tuple[int, int]]
    frame: Matrix3x3f
    (spline_surface,
        intersect_params,
        invisibility_params,
        patch_boundary_edges,
        frame) = _initialize_contour_info_spot_mesh()

    # Compute boundaries
    boundary_domain_curve_segments: list[Conic]
    boundary_segments: list[RationalFunction]  # <degree 4, dimension 3>
    boundary_patch_indices: list[PatchIndex]
    (boundary_domain_curve_segments,
     boundary_segments,
     boundary_patch_indices) = compute_spline_surface_boundaries(spline_surface,
                                                                 patch_boundary_edges)

    filepath: str = "spot_control\\contour_network\\compute_contours\\compute_spline_surface_boundaries\\"
    # Both should be empty
    boundary_domain_curve_segments_control: list[Conic] = deserialize_conics(
        filepath+"boundary_domain_curve_segments.json")
    boundary_segments_control: list[RationalFunction] = deserialize_rational_functions(
        filepath+"boundary_segments.json")
    compare_conics(boundary_domain_curve_segments_control, boundary_domain_curve_segments)
    compare_rational_functions(boundary_segments_control, boundary_segments)
    compare_eigen_numpy_matrix(filepath+"boundary_patch_indices.csv",
                               np.array(boundary_patch_indices))


def test_compute_spline_surface_boundary_intersections_spot_control() -> None:
    """ 
    Testing compute_spline_surface_boundary_intersection() as part of 
    compute_spline_surface_contours_and_boundaries()
    """
    spline_surface: QuadraticSplineSurface
    intersect_params: IntersectionParameters
    invisibility_params: InvisibilityParameters
    patch_boundary_edges: list[tuple[int, int]]
    frame: Matrix3x3f
    (spline_surface,
        intersect_params,
        invisibility_params,
        patch_boundary_edges,
        frame) = _initialize_contour_info_spot_mesh()

    filepath: str = "spot_control\\contour_network\\compute_contours\\compute_spline_surface_boundary_intersections\\"
    contour_domain_curve_segments: list[Conic] = deserialize_conics(
        filepath+"contour_domain_curve_segments.json")
    contour_patch_indices: list[PatchIndex] = np.array(deserialize_eigen_matrix_csv_to_numpy(
        filepath+"contour_patch_indices.csv"),
        dtype=np.int64).tolist()
    line_intersection_indices: list[tuple[int, int]] = np.array(deserialize_eigen_matrix_csv_to_numpy(
        filepath+"line_intersection_indices.csv"), dtype=np.int64).tolist()
    boundary_domain_curve_segments: list[Conic] = deserialize_conics(

        filepath+"boundary_domain_curve_segments.json")
    contour_intersections: list[list[IntersectionData]]
    num_intersections: int
    contour_intersections, num_intersections = compute_spline_surface_boundary_intersections(
        spline_surface,
        contour_domain_curve_segments,
        contour_patch_indices,
        line_intersection_indices,
        patch_boundary_edges,
        boundary_domain_curve_segments)

    contour_intersections_control: list[list[IntersectionData]] = (
        deserialize_list_list_intersection_data(filepath+"contour_intersections.json"))
    compare_list_list_intersection_data(contour_intersections, contour_intersections_control)
    assert num_intersections == 0


def test_compute_spline_surface_contours_and_boundaries_spot_mesh() -> None:
    """
    Part of testing for init_contour_network
    """
    # unimplemented("This should just rely on the results of the previous tests to pass.")
    filepath: str = "spot_control\\contour_network\\compute_contours\\compute_spline_surface_contours_and_boundaries\\"
    filepath_surface: str = os.path.abspath(
        f"src\\tests\\spot_control\\contour_network\\compute_contours\\compute_spline_surface_contours_and_boundaries\\spline_surface.txt")
    contour_domain_curve_segments: list[Conic]
    contour_segments: list[RationalFunction]
    contour_patch_indices: list[PatchIndex]
    contour_is_boundary: list[bool]
    contour_intersections: list[list[IntersectionData]]
    num_intersections: int

    spline_surface: QuadraticSplineSurface
    intersect_params: IntersectionParameters
    invisibility_params: InvisibilityParameters
    patch_boundary_edges: list[tuple[int, int]]
    frame: Matrix3x3f
    (spline_surface,
        intersect_params,
        invisibility_params,
        patch_boundary_edges,
        frame) = _initialize_contour_info_spot_mesh()

    (contour_domain_curve_segments,
     contour_segments,
     contour_patch_indices,
     contour_is_boundary,
     contour_intersections,
     num_intersections) = compute_spline_surface_contours_and_boundaries(

        # QuadraticSplineSurface.from_file(filepath_surface),
        spline_surface,
        np.identity(3),
        deserialize_eigen_matrix_csv_to_numpy(filepath+"patch_boundary_edges.csv").tolist())

    compare_conics_from_file(filepath+"contour_domain_curve_segments.json",
                             contour_domain_curve_segments)
    compare_rational_functions_from_file(filepath+"contour_segments.json",
                                         contour_segments)
    compare_eigen_numpy_matrix(filepath+"contour_patch_indices.csv",
                               np.array(contour_patch_indices))
    compare_eigen_numpy_matrix(filepath+"contour_is_boundary.csv",
                               np.array(contour_is_boundary))
    compare_list_list_intersection_data_from_file(filepath+"contour_intersections.json",
                                                  contour_intersections)
    assert num_intersections == 0
