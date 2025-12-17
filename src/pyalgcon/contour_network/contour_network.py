"""
contour_network.py
Methods to compute the contour curve network for a spline surface with view
frame.
"""

import logging
import pathlib
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np
import polyscope
import svg

from pyalgcon.contour_network.compute_closed_contours import \
    compute_closed_contours
from pyalgcon.contour_network.compute_contours import (
    compute_spline_surface_contours_and_boundaries, pad_contours)
from pyalgcon.contour_network.compute_cusps import compute_spline_surface_cusps
from pyalgcon.contour_network.compute_intersections import (
    IntersectionParameters, compute_intersections)
from pyalgcon.contour_network.compute_ray_intersections import (
    compute_spline_surface_ray_intersections, partition_ray_intersections)
from pyalgcon.contour_network.intersection_data import IntersectionData
from pyalgcon.contour_network.project_curves import project_curves
from pyalgcon.contour_network.projected_curve_network import (
    ProjectedCurveNetwork, SegmentChainIterator)
from pyalgcon.contour_network.write_output import \
    write_contours_with_annotations
from pyalgcon.core.common import (DISCRETIZATION_LEVEL,
                                  INLINE_TESTING_ENABLED_CONTOUR_NETWORK,
                                  INLINE_TESTING_ENABLED_QI, OFF_WHITE,
                                  TESTING_FOLDER_SOURCE,
                                  USE_DESERIALIZED_VALUES, Matrix2x3f,
                                  Matrix3x3f, MatrixNx3f, NodeIndex,
                                  PatchIndex, PlanarPoint1d, SegmentIndex,
                                  SpatialVector1d, Vector3f, Vector3i,
                                  compare_eigen_numpy_matrix,
                                  compare_list_list_varying_lengths,
                                  compare_list_list_varying_lengths_float,
                                  deserialize_eigen_matrix_csv_to_numpy,
                                  deserialize_list_list_varying_lengths,
                                  deserialize_list_list_varying_lengths_float,
                                  dot_product, nested_vector_size,
                                  vector_contains)
from pyalgcon.core.conic import Conic
from pyalgcon.core.rational_function import (CurveDiscretizationParameters,
                                             RationalFunction)
from pyalgcon.debug.debug import SPOT_FILEPATH
from pyalgcon.quadratic_spline_surface.quadratic_spline_surface import \
    QuadraticSplineSurface
from pyalgcon.utils.compute_intersections_testing_utils import \
    deserialize_list_list_intersection_data
from pyalgcon.utils.conic_testing_utils import (compare_conics_from_file,
                                                deserialize_conics_from_file)
from pyalgcon.utils.projected_curve_networks_utils import \
    deserialize_segment_labels
from pyalgcon.utils.rational_function_testing_utils import (
    compare_rational_functions_from_file,
    deserialize_rational_functions_from_file)

logger: logging.Logger = logging.getLogger(__name__)


def _build_contour_labels(contour_patch_indices: list[int],
                          contour_is_boundary: list[bool]) -> list[dict[str, int]]:
    """
    Build contour labels mapping label names to integer data per contour segment
    """
    num_segments: int = len(contour_patch_indices)
    contour_segment_labels: list[dict[str, int]] = []
    for i in range(num_segments):
        contour_segment_label: dict[str, int] = {"surface_patch": contour_patch_indices[i],
                                                 "is_boundary": 0 if contour_is_boundary[i] else 1}
        contour_segment_labels.append(contour_segment_label)
    return contour_segment_labels


class InvisibilityMethod(Enum):
    """
    Enum for Quantitative Invisibility calculation methods.
    """
    NONE = 0  # Set all QI to 0
    DIRECT = 1    # Ray test per segment
    CHAINING = 2    # Ray test per chain of segments between features
    PROPAGATION = 3    # Ray test for connected components with local propagation


@dataclass
class InvisibilityParameters():
    """
    Parameters for the invisibility computation
    """
    pad_amount: float = 1e-9  # Padding for contour domains
    write_contour_soup = False  # Option to write contours before graph construction for diagnostics

    # Method for computing quantitative visibility
    #
    # TODO: test with various QI modes
    #
    invisibility_method: InvisibilityMethod = InvisibilityMethod.CHAINING

    # Options to view each local propagation step during computation for debugging
    view_intersections = False
    view_cusps = False

    # Options for redundancy checks
    poll_chain_segments = True  # Sample and poll 3 segments for majority per chain QI
    poll_segment_points = False  # Sample and poll 3 points for majority per segment QI

    # Consistency checks
    check_chaining = False
    check_propagation = False


class ContourNetwork(ProjectedCurveNetwork):
    """
    Class to compute the projected contours of a quadratic spline surface
    and represent them as a curve network. Also computes the quantitative
    invisibility for the contours.
    """

    def __init__(self,
                 spline_surface: QuadraticSplineSurface,
                 intersect_params: IntersectionParameters,
                 invisibility_params: InvisibilityParameters,
                 patch_boundary_edges: list[tuple[int, int]]) -> None:
        """
        Constructor that takes a spline surface and computes the full projected
        contour curve network with standard viewing frame along the z axis.

        :param spline_surface:       [in] quadratic spline surface to build contours for
        :param intersect_params:     [in] parameters for the intersection methods
        :param intersect_params:     [in] parameters for the invisibility methods
        :param patch_boundary_edges: [in] patch boundary edge indices (default none)
        """
        # ***************************************
        # Contour Network public member variables
        # ***************************************
        self.ray_intersection_call: int = 0
        self.ray_bounding_box_call: int = 0
        self.ray_number: int = 0
        self.chain_number: int = 0
        self.segment_number: int = 0
        self.interior_cusp_number: int = 0
        self.boundary_cusp_number: int = 0
        self.intersection_call: int = 0

        self.surface_update_position_time: float = 0
        self.compute_contour_time: float = 0
        self.compute_cusp_time: float = 0
        self.compute_intersection_time: float = 0
        self.compute_visibility_time: float = 0
        self.compute_projected_time: float = 0

        # Build the curve network from the contours
        time_start = time.perf_counter()

        # Constructs parent class parameters
        if USE_DESERIALIZED_VALUES:
            # Skips the whole building contour calculation process, saving time when testing.
            filepath: str = (
                f"{TESTING_FOLDER_SOURCE}\\contour_network\\projected_curve_network\\init_projected_curve_network\\")
            super().__init__(deserialize_conics_from_file(filepath+"parameter_segments.json"),
                             deserialize_rational_functions_from_file(
                                 filepath+"spatial_segments.json"),
                             deserialize_rational_functions_from_file(
                                 filepath+"planar_segments.json"),
                             deserialize_segment_labels(filepath+"segment_labels.json"),
                             deserialize_list_list_varying_lengths(filepath+"chains.csv"),
                             deserialize_eigen_matrix_csv_to_numpy(
                                 filepath+"chain_labels.csv").tolist(),
                             deserialize_list_list_varying_lengths_float(
                                 filepath+"interior_cusps.csv"),
                             deserialize_eigen_matrix_csv_to_numpy(
                                 filepath+"has_cusp_at_base.csv").astype(bool).tolist(),
                             deserialize_list_list_varying_lengths_float(
                                 filepath+"intersections.csv"),
                             deserialize_list_list_varying_lengths(
                                 filepath+"intersection_indices.csv"),
                             deserialize_list_list_intersection_data(
                                 filepath+"intersection_data.json"),
                             num_intersections=176)
        else:
            # Standard constructor.
            super().__init__(*self.__build_projected_curve_network_params(spline_surface,
                                                                          intersect_params,
                                                                          invisibility_params,
                                                                          patch_boundary_edges))

        compute_projected_time: float = time.perf_counter() - time_start

        #
        # FIXME: quantitative invisibility calculation is a bit different
        # so, go about and fix this.
        #
        # Compute the quantitative invisibility
        time_start: float = time.perf_counter()
        self.__compute_quantitative_invisibility(spline_surface, invisibility_params)
        compute_visibility_time: float = time.perf_counter() - time_start

    @staticmethod
    def __build_projected_curve_network_params(
        spline_surface: QuadraticSplineSurface,
        intersect_params: IntersectionParameters,
        invisibility_params: InvisibilityParameters,
        patch_boundary_edges: list[tuple[int, int]]
    ) -> tuple[list[Conic],
               list[RationalFunction],
               list[RationalFunction],
               list[dict[str, int]],
               list[list[int]],
               list[int],
               list[list[float]],
               list[bool],
               list[list[float]],
               list[list[int]],
               list[list[IntersectionData]],
               int]:
        """
        Builds the parameters for the projected curve network parent class based on the
        parameters passed into contour network.
        Part of initializing the contour network.
        Originally init_contour_network()

        :return: contour_domain_curve_segments
        :return: contour_segments
        :return: planar_contour_segments
        :return: contour_segment_labels
        :return: contours
        :return: contour_labels
        :return: interior_cusps
        :return: has_cusp_at_base
        :return: intersection_knots
        :return: intersection_indices
        :return: contour_intersections
        :return: num_intersections
        """
        frame: Matrix3x3f = np.identity(3, dtype=np.float64)

        # Compute contours
        contour_domain_curve_segments: list[Conic]
        contour_segments: list[RationalFunction]  # <4, 3>
        contour_patch_indices: list[PatchIndex]
        contour_is_boundary: list[bool]
        contour_intersections: list[list[IntersectionData]]
        num_intersections: int

        time_start: float = time.perf_counter()

        #
        # FIXME: below method looks good
        #
        # NOTE: method below takes the most time.
        # TODO: but, try and speed up the testing by adding a statment to switch between these and
        # deserializing the code.

        # if USE_DESERIALIZED_VALUES or TESTING_ENABLED:
        # filepath: str = f"{TESTING_FOLDER_SOURCE}\\contour_network\\compute_contours\\compute_spline_surface_contours_and_boundaries\\"
        # contour_domain_curve_segments = deserialize_conics(
        #     filepath+"contour_domain_curve_segments.json")
        # contour_segments = deserialize_rational_functions(
        #     filepath+"contour_segments.json")
        # contour_patch_indices = np.array(deserialize_eigen_matrix_csv_to_numpy(
        #     filepath+"contour_patch_indices.csv"), dtype=int).tolist()
        # contour_is_boundary = np.array(deserialize_eigen_matrix_csv_to_numpy(
        #     filepath+"contour_is_boundary.csv"), dtype=bool).tolist()
        # contour_intersections = deserialize_list_list_intersection_data(
        #     filepath+"contour_intersections.json")
        # num_intersections = 0

        # else:
        (contour_domain_curve_segments,
            contour_segments,
            contour_patch_indices,
            contour_is_boundary,
            contour_intersections,
            num_intersections) = compute_spline_surface_contours_and_boundaries(
            spline_surface,
            frame,
            patch_boundary_edges)

        # Build contour labels for boundary countours and patch locations
        # FIXME: build contour labels looks GOOD
        contour_segment_labels: list[dict[str, int]] = _build_contour_labels(contour_patch_indices,
                                                                             contour_is_boundary)

        # Project contours to the plane
        # FIXME: project_curves also looks good
        planar_contour_segments: list[RationalFunction] = project_curves(contour_segments, frame)
        # lazy check
        assert (planar_contour_segments[0].degree, planar_contour_segments[0].dimension) == (4, 2)

        if INLINE_TESTING_ENABLED_CONTOUR_NETWORK:
            filepath: pathlib.Path = SPOT_FILEPATH / "contour_network" / "project_curves"
            compare_rational_functions_from_file(
                filepath / "planar_curves.json", planar_contour_segments)

        # Chain the contour segments into closed contours
        # FIXME: compute_closed_contours looks good
        contours: list[list[int]]
        contour_labels: list[int]
        contours, contour_labels = compute_closed_contours(contour_segments)

        if INLINE_TESTING_ENABLED_CONTOUR_NETWORK:
            filepath: pathlib.Path = SPOT_FILEPATH / "contour_network" / \
                "compute_closed_contours" / "compute_closed_contours"
            compare_list_list_varying_lengths(filepath / "contours.csv", contours)
            compare_eigen_numpy_matrix(filepath / "contour_labels.csv", np.array(contour_labels))

        # Pad contour domains by an epsilon
        # FIXME: pad_contours looks good
        pad_contours(contour_domain_curve_segments,
                     contour_segments,
                     planar_contour_segments,
                     invisibility_params.pad_amount)

        if INLINE_TESTING_ENABLED_CONTOUR_NETWORK:
            filepath: pathlib.Path = SPOT_FILEPATH / "contour_network" / "compute_contours" / "pad_contours"
            compare_conics_from_file(filepath / "contour_domain_curve_segments_PADDED.json",
                                     contour_domain_curve_segments)
            compare_rational_functions_from_file(
                filepath / "contour_segments_PADDED.json", contour_segments)
            compare_rational_functions_from_file(
                filepath / "planar_contour_segments_PADDED.json", planar_contour_segments)

        compute_contour_time: float = time.perf_counter() - time_start

        # Get cusp points and intersections if needed
        num_segments: int = len(contour_segments)
        interior_cusps: list[list[float]]
        boundary_cusps: list[list[float]]
        has_cusp_at_base: list[bool]
        has_cusp_at_tip: list[bool]

        time_start = time.perf_counter()

        #
        # FIXME: compute_spline_surface_cusps looks good.
        #
        (interior_cusps,
         boundary_cusps,
         has_cusp_at_base,
         has_cusp_at_tip) = compute_spline_surface_cusps(spline_surface,
                                                         contour_domain_curve_segments,
                                                         contour_segments,
                                                         contour_patch_indices,
                                                         contours)

        if INLINE_TESTING_ENABLED_CONTOUR_NETWORK:
            filepath: pathlib.Path = SPOT_FILEPATH / "contour_network" / \
                "compute_cusps" / "compute_spline_surface_cusps"
            compare_list_list_varying_lengths_float(filepath / "interior_cusps.csv", interior_cusps)
            compare_list_list_varying_lengths_float(filepath / "boundary_cusps.csv", boundary_cusps)
            compare_eigen_numpy_matrix(filepath / "has_cusp_at_base.csv",
                                       np.array(has_cusp_at_base))
            compare_eigen_numpy_matrix(filepath / "has_cusp_at_tip.csv", np.array(has_cusp_at_tip))

        compute_cusp_time: float = time.perf_counter() - time_start
        logger.debug("Found %s interior cusps", nested_vector_size(interior_cusps))
        logger.debug("Found %s boundary cusps", nested_vector_size(boundary_cusps))

        #
        # FIXME: compute_intersections has some problems. output is not 1-to-1 with the C++ code
        #
        # Compute planar contour intersections
        time_start = time.perf_counter()
        intersection_knots: list[list[float]]
        intersection_indices: list[list[int]]
        intersection_call: int
        (intersection_knots,
         intersection_indices,
         num_intersections,
         intersection_call) = compute_intersections(planar_contour_segments,
                                                    intersect_params,
                                                    contour_intersections,
                                                    num_intersections)

        compute_intersection_time: float = time.perf_counter() - time_start
        logger.debug("Found %s intersections", nested_vector_size(intersection_knots))

        # Optionally write contours before any graph construction
        if invisibility_params.write_contour_soup:
            viewport: tuple[int, int] = (800, 800)
            # TODO: include svgWriter of some sort
            svg_elements: list[svg.Element] = []
            write_contours_with_annotations(frame,
                                            contour_segments,
                                            interior_cusps,
                                            boundary_cusps,
                                            intersection_knots,
                                            CurveDiscretizationParameters(),
                                            svg_elements)

            print(svg.SVG(x=0, y=0,
                          width=viewport[0], height=viewport[0],
                          elements=svg_elements))

        # Now, can return values used for initializing projected network.
        return (contour_domain_curve_segments,
                contour_segments,
                planar_contour_segments,
                contour_segment_labels,
                contours,
                contour_labels,
                interior_cusps,
                has_cusp_at_base,
                intersection_knots,
                intersection_indices,
                contour_intersections,
                num_intersections)

    # **************
    # Public Methods
    # **************

    def add_contour_network_to_viewer(self) -> None:
        """
        Add the spatial network to the viewer
        """
        polyscope.init()
        # Add curve network to viewer
        self.add_spatial_network_to_viewer()

    def view(self,
             spline_surface: QuadraticSplineSurface) -> None:
        """
        View the contour network surface and spatial network

        :param spline_surface: [in] underlying quadratic spline surface
        """
        polyscope.init()
        polyscope.set_view_projection_mode("orthographic")
        spline_surface.add_surface_to_viewer(OFF_WHITE, DISCRETIZATION_LEVEL)
        self.add_contour_network_to_viewer()
        polyscope.show()
        polyscope.remove_all_structures()

    def view_contours(self) -> None:
        """
        View the contour network without the underlying surface
        """
        polyscope.init()
        polyscope.set_view_projection_mode("orthographic")
        self.add_spatial_network_to_viewer()
        polyscope.show()
        polyscope.remove_all_structures()

    def screenshot(self,
                   filename: str,
                   spline_surface: QuadraticSplineSurface,
                   camera_position: tuple[float, float, float] = (0, 0, 2),
                   camera_target: tuple[float, float, float] = (0, 0, 0),
                   use_orthographic: bool = False) -> None:
        """
        Save a screenshot of the contour network to file

        :param filename:         [in] file to save the screenshot to
        :param spline_surface:   [in] underlying quadratic spline surface
        :param camera_position:  [in] camera position for the screenshot
        :param camera_target:    [in] camera target for the screenshot
        :param use_orthographic: [in] use orthographic perspective if true
        """
        # Add the contour network to the surface
        polyscope.init()
        spline_surface.add_surface_to_viewer(OFF_WHITE, DISCRETIZATION_LEVEL)
        self.add_contour_network_to_viewer()

        # Set up the cameras
        polyscope.look_at(camera_position, camera_target)
        if use_orthographic:
            polyscope.set_view_projection_mode("orthographic")

        # Take the screenshot
        polyscope.screenshot(filename)
        logger.info("Screenshot saved to %s", filename)
        polyscope.remove_all_structures()

    def write_rasterized_contours(self,
                                  filename: str,
                                  camera_position: tuple[float, float, float] = (0, 0, 2),
                                  camera_target: tuple[float, float, float] = (0, 0, 0)) -> None:
        """
        Save a screenshot of just the contours to file

        :param filename:        [in] file to save the screenshot to
        :param camera_position: [in] camera position for the screenshot
        :param camera_target:   [in] camera target for the screenshot
        """
        polyscope.init()
        self.add_spatial_network_to_viewer()

        # Set up the cameras
        polyscope.look_at(camera_position, camera_target)
        polyscope.set_view_projection_mode("orthographic")

        # Take the screenshot
        polyscope.screenshot(filename)
        logger.info("Screenshot saved to %s", filename)
        polyscope.remove_all_structures()

    def reset_counter(self) -> None:
        """
        Reset timing information
        """
        self.ray_intersection_call = 0
        self.ray_bounding_box_call = 0
        self.ray_number = 0
        self.chain_number = 0
        self.segment_number = 0
        self.interior_cusp_number = 0
        self.boundary_cusp_number = 0
        self.intersection_call = 0

        self.surface_update_position_time = 0
        self.compute_contour_time = 0
        self.compute_cusp_time = 0
        self.compute_intersection_time = 0
        self.compute_visibility_time = 0
        self.compute_projected_time = 0

    # *********************
    # Direct QI Computation
    # *********************

    def __generate_ray_mapping_coeffs(self,
                                      sample_point: SpatialVector1d,
                                      ) -> Matrix2x3f:
        """
        Compute the ray mapping coeffs for a given sample point
        """
        assert sample_point.shape == (3, )
        view_direction: SpatialVector1d = np.array([0, 0, 1])
        ray_mapping_coeffs: Matrix2x3f = np.array([sample_point - 20 * view_direction,
                                                  40 * view_direction])
        assert ray_mapping_coeffs.shape == (2, 3)

        return ray_mapping_coeffs

    def __compute_quantitative_invisibility_from_ray_intersections(self,
                                                                   ray_mapping_coeffs: Matrix2x3f,
                                                                   point: SpatialVector1d,
                                                                   ray_intersections: list[float]
                                                                   ) -> int:
        """
        Compute the QI from a given point from the list of intersection parameters
        with the surface
        """
        assert ray_mapping_coeffs.shape == (2, 3)
        assert point.shape == (3, )

        # Partition intersections into points above and below the sample point
        if len(ray_intersections) == 0:
            return 0
        else:
            ray_intersections_below: list[float]
            ray_intersections_above: list[float]
            (ray_intersections_below,
             ray_intersections_above) = partition_ray_intersections(ray_mapping_coeffs,
                                                                    point,
                                                                    ray_intersections)
            #  Set QI as the number of intersection points occluding the sample point
            return len(ray_intersections_below)

    def __compute_segment_quantitative_invisibility(self,
                                                    spline_surface: QuadraticSplineSurface,
                                                    segment_index: SegmentIndex,
                                                    invisibility_params: InvisibilityParameters
                                                    ) -> int:
        """
        Compute the QI for a single segment with robust polling of three sample points
        """
        # Check segment validity
        if not self._is_valid_segment_index(segment_index):
            logger.debug("Attempting to propagate QI at invalid segment %s",
                         segment_index)
            return -1

        self.segment_number += 1

        ray_int_call: int = 0
        ray_bbox_call: int = 0
        qi_poll: Vector3i = np.zeros(shape=(3, ), dtype=np.int64)
        spatial_curve: RationalFunction = self.segment_spatial_curve(segment_index)
        assert (spatial_curve.degree, spatial_curve.dimension) == (4, 3)

        sample_parameters: list[float] = [0.5, 0.25, 0.75]
        for i, sample_parameter in enumerate(sample_parameters):

            sample_point: SpatialVector1d = spatial_curve.evaluate_normalized_coordinate(
                sample_parameter)
            assert sample_point.shape == (3, )
            logger.info("Sample point: %s", sample_point)

            # Build ray mapping coefficients

            ray_mapping_coeffs: Matrix2x3f = self.__generate_ray_mapping_coeffs(sample_point)
            logger.info("Ray mapping coefficients: %s", ray_mapping_coeffs)

            # Compute intersections of the ray with the surface
            patch_indices: list[PatchIndex]
            surface_intersections: list[PlanarPoint1d]
            ray_intersections: list[float]

            self.ray_number += 1

            # Compute the intersections of the ray with the spline surface

            # FIXME: the function below not giving the correct number of calls...
            # Also, ptch indices is wrong
            # Surface intersections is wrong as well
            (patch_indices,
             surface_intersections,
             ray_intersections,
             ray_int_call,
             ray_bbox_call) = compute_spline_surface_ray_intersections(spline_surface,
                                                                       ray_mapping_coeffs,
                                                                       ray_int_call,
                                                                       ray_bbox_call)
            # Compute the QI from the ray intersections
            qi_poll[i] = (self.__compute_quantitative_invisibility_from_ray_intersections(
                ray_mapping_coeffs, sample_point, ray_intersections))

            # If no polling, return after first computation
            if not invisibility_params.poll_segment_points:
                return qi_poll[i]
        logger.info("QI poll values: %s, %s, %s", qi_poll[0], qi_poll[1], qi_poll[2])

        self.ray_intersection_call += ray_int_call
        self.ray_bounding_box_call += ray_bbox_call

        # Poll for a majority
        if qi_poll[0] == qi_poll[1]:
            return qi_poll[0]
        if qi_poll[1] == qi_poll[2]:
            return qi_poll[1]
        if qi_poll[2] == qi_poll[0]:
            return qi_poll[2]

        # Arbitrarily choose the midpoint if no majority
        logger.warning("Could not compute consistent segment qi amongst %s, %s, %s",
                       qi_poll[0],
                       qi_poll[1],
                       qi_poll[2])
        return qi_poll[1]

    def __compute_chain_quantitative_invisibility(self,
                                                  spline_surface: QuadraticSplineSurface,
                                                  start_segment_index: SegmentIndex,
                                                  invisibility_params: InvisibilityParameters
                                                  ) -> int:
        """
        Compute the QI for a chain of segments with robust polling of three segments
        """
        # Check segment validity
        if not self._is_valid_segment_index(start_segment_index):
            logger.debug("Attempting to compute QI at invalid segment %s",
                         start_segment_index)
            return -1

        # Count number of segments
        counter_iter: SegmentChainIterator = self.get_segment_chain_iterator(start_segment_index)
        num_segments: int = 0
        while not counter_iter.at_end_of_chain:
            num_segments += 1
            counter_iter.increment()

        self.chain_number += 1

        #  Determine three segments to sample
        iteration: SegmentChainIterator = self.get_segment_chain_iterator(start_segment_index)
        jump_size: int = 0
        if num_segments >= 5:
            jump_size = num_segments // 5  # Warning: possibly slow

            # Skip forward one jump
            for j in range(jump_size):
                iteration.increment()
        # Small number of segment case
        elif num_segments >= 3:
            jump_size = 1
        # Less than 3 case (just sample one segment)
        else:
            jump_size = 0

        # FIXME: likely below causing trouble.
        qi_poll: Vector3f = np.zeros(shape=(3, ), dtype=np.int64)
        for i in range(3):
            segment_index: SegmentIndex = iteration.current_segment_index
            qi_poll[i] = self.__compute_segment_quantitative_invisibility(spline_surface,
                                                                          segment_index,
                                                                          invisibility_params)

            # If no polling or trivial jump size, return immediately
            if (not invisibility_params.poll_chain_segments) or (jump_size == 0):
                return qi_poll[i]

            # Jump to next query point otherwise
            for j in range(jump_size):
                iteration.increment()

        # Poll for majority
        if qi_poll[0] == qi_poll[1]:
            return qi_poll[0]
        if qi_poll[1] == qi_poll[2]:
            return qi_poll[1]
        if qi_poll[2] == qi_poll[0]:
            return qi_poll[2]

        # Choose the center segment if no majority found
        logger.warning("Could not compute consistent chain qi amongst %s, %s, %s",
                       qi_poll[0],
                       qi_poll[1],
                       qi_poll[2])
        return qi_poll[1]

    # ****************
    # Chain QI Methods
    # ****************
    def __chain_quantitative_invisibility_forward(self,
                                                  spline_surface: QuadraticSplineSurface,
                                                  start_segment_index: SegmentIndex,
                                                  invisibility_params: InvisibilityParameters
                                                  ) -> NodeIndex:
        """
        Propagate QI from the start segment forward to the next special node and
        return the final node index.

        :param self.segments: [out]
        :param self.nodes: [out]
        :param self.current_segment_index: [out]
        :param self.is_end_of_chain: [out]
        """
        # Check segment validity
        if not self._is_valid_segment_index(start_segment_index):
            logger.debug("Attempting to propagate QI at invalid segment %s",
                         start_segment_index)
            return -1

        # Get QI of the start segment
        quantitative_invisibility: int = self.get_segment_quantitative_invisibility(
            start_segment_index)
        if quantitative_invisibility < 0:
            logger.warning("Attempted to copy negative QI forward")
            return -1

        # Propigate the QI along the chain
        iteration: SegmentChainIterator = self.get_segment_chain_iterator(start_segment_index)
        from_node_index: NodeIndex = self.from_(start_segment_index)
        to_node_index: NodeIndex = self.to(start_segment_index)
        iteration.increment()
        while not iteration.at_end_of_chain:
            segment_index: SegmentIndex = iteration.current_segment_index
            from_node_index = self.from_(segment_index)
            to_node_index = self.to(segment_index)
            self.set_segment_quantitative_invisibility(segment_index,
                                                       quantitative_invisibility)
            self.set_node_quantitative_invisibility(from_node_index,
                                                    quantitative_invisibility)

            # Check against direct computation
            if invisibility_params.check_chaining:
                logger.debug("Checking QI chaining at knot node")
                self.__check_quantitative_invisibility_propagation(spline_surface,
                                                                   segment_index,
                                                                   invisibility_params)

            iteration.increment()

        return to_node_index

    def __chain_quantitative_invisibility_backward(self,
                                                   spline_surface: QuadraticSplineSurface,
                                                   start_segment_index: SegmentIndex,
                                                   invisibility_params: InvisibilityParameters
                                                   ) -> NodeIndex:
        """
        Propagate QI from the start segment backward to the next special node and
        return the final node index
        """
        # Check segment validity
        if not self._is_valid_segment_index(start_segment_index):
            logger.debug("Attempting to propagate QI at invalid segment %s",
                         start_segment_index)
            return -1

        # Get QI of the start segment
        quantitative_invisibility: int = self.get_segment_quantitative_invisibility(
            start_segment_index)
        if quantitative_invisibility < 0:
            logger.warning("Attempted to copy negative QI forward")
            return -1

        # Propagate the QI along the chain
        iteration: SegmentChainIterator = self.get_segment_chain_iterator(start_segment_index)
        from_node_index: NodeIndex = self.from_(start_segment_index)
        to_node_index: NodeIndex = self.to(start_segment_index)
        iteration.decrement()
        while not iteration.at_reverse_end_of_chain:
            segment_index: SegmentIndex = iteration.current_segment_index
            from_node_index = self.from_(segment_index)
            to_node_index = self.to(segment_index)
            self.set_segment_quantitative_invisibility(segment_index,
                                                       quantitative_invisibility)
            self.set_node_quantitative_invisibility(to_node_index,
                                                    quantitative_invisibility)

            # Check against direct computation
            if invisibility_params.check_chaining:
                logger.debug("Checking QI chaining at knot node")
                self.__check_quantitative_invisibility_propagation(spline_surface,
                                                                   segment_index,
                                                                   invisibility_params)
            iteration.decrement()

        return from_node_index

    # ****************************
    # Local QI Propagation Methods
    # ****************************

    def __propagate_quantitative_invisibility_forward(self,
                                                      spline_surface: QuadraticSplineSurface,
                                                      start_segment_index: SegmentIndex,
                                                      invisibility_params: InvisibilityParameters
                                                      ) -> None:
        """
        Chain QI from the start segment forward to the next special node and continue
        propagation
        """
        # Check segment validity
        if not self._is_valid_segment_index(start_segment_index):
            logger.debug("Attempting to propagate QI at invalid segment %s",
                         start_segment_index)
            return

        chain_end_node_index: NodeIndex = self.__chain_quantitative_invisibility_forward(
            spline_surface, start_segment_index, invisibility_params)
        self.__propagate_quantitative_invisibility_at_node(
            spline_surface, chain_end_node_index, invisibility_params)

    def __propagate_quantitative_invisibility_backward(self,
                                                       spline_surface: QuadraticSplineSurface,
                                                       start_segment_index: SegmentIndex,
                                                       invisibility_params: InvisibilityParameters
                                                       ):
        """
        Chain QI from the start segment backward to the next special node and
        continue propagation
        """
        # Check segment validity
        if not self._is_valid_segment_index(start_segment_index):
            logger.debug("Attempting to propagate QI at invalid segment %s",
                         start_segment_index)
            return

        chain_start_node_index: NodeIndex = self.__chain_quantitative_invisibility_backward(
            spline_surface, start_segment_index, invisibility_params)
        self.__propagate_quantitative_invisibility_at_node(
            spline_surface, chain_start_node_index, invisibility_params)

    def __propagate_quantitative_invisibility_at_node(self,
                                                      spline_surface: QuadraticSplineSurface,
                                                      node_index: NodeIndex,
                                                      invisibility_params: InvisibilityParameters
                                                      ):
        """
        Propagate QI at a node with case work for different node types
        """
        logger.debug("Propagating QI at node %s", node_index)

        # Check node validity
        if not self._is_valid_node_index(node_index):
            logger.debug("Attempting to propagate QI at invalid node %s", node_index)
            return

        # Stop processing the node if node QI already set, and mark the node as
        # processed otherwise
        if self.node_quantitative_invisibility_is_set(node_index):
            return
        self.mark_node_quantitative_invisibility_as_set(node_index)

        # Attempt to propagate QI at the node
        success: bool = True
        if self.is_intersection_node(node_index):
            logger.info("Propagating QI at intersection node")
            success = self.__propagate_quantitative_invisibility_at_intersection_node(
                spline_surface, node_index, invisibility_params)
        elif self.is_interior_cusp_node(node_index):
            logger.info("Propagating QI at interior cusp node")
            success = self.__propagate_quantitative_invisibility_at_cusp_node(
                spline_surface, node_index, invisibility_params)
        elif self.is_boundary_cusp_node(node_index):
            logger.info("Propagating QI at boundary cusp node")
            self.__propagate_quantitative_invisibility_at_boundary_cusp_node(
                spline_surface, node_index, invisibility_params)
        elif self.is_marked_knot_node(node_index):
            logger.info("Propagating QI at marked node")
            self.__propagate_quantitative_invisibility_at_marked_knot_node(
                spline_surface, node_index, invisibility_params)

        # Fallback to direct computation and chaining if not successful
        if not success:
            qi_out: int = self.__compute_chain_quantitative_invisibility(
                spline_surface, self.out(node_index), invisibility_params)
            qi_in: int = self.__compute_chain_quantitative_invisibility(
                spline_surface, self.in_(node_index), invisibility_params)
            self.set_segment_quantitative_invisibility(self.out(node_index), qi_out)
            self.set_segment_quantitative_invisibility(self.in_(node_index), qi_in)
            self.set_node_quantitative_invisibility(node_index, qi_out)
            self.__propagate_quantitative_invisibility_forward(
                spline_surface, self.out(node_index), invisibility_params)
            self.__propagate_quantitative_invisibility_backward(
                spline_surface, self.in_(node_index), invisibility_params)

    def __propagate_quantitative_invisibility_across_node(self,
                                                          node_index: NodeIndex,
                                                          change_in_quantitative_invisibility: int
                                                          ) -> int:
        """
        Propagate QI if it is known for the in or out segment at a node to the other
        node and return the QI of the out node
        """
        # Check node validity
        if not self._is_valid_node_index(node_index):
            logger.debug("Attempting to propagate QI at invalid node %s", node_index)
            return -1

        # Check if boundary node (no work can be done)
        in_segment: SegmentIndex = self.in_(node_index)
        out_segment: SegmentIndex = self.out(node_index)
        if (not self._is_valid_segment_index(in_segment) or
                not self._is_valid_segment_index(out_segment)):
            logger.debug("Cannot propagate QI across terminal boundary node")
            return -1
        qi_in: int = self.get_segment_quantitative_invisibility(in_segment)
        qi_out: int = self.get_segment_quantitative_invisibility(out_segment)

        # Do nothing if both segments already have QI
        if (qi_in >= 0) and (qi_out >= 0):
            new_qi: int = qi_in + change_in_quantitative_invisibility
            if qi_out != new_qi:
                logger.warning("Inconsistent QI propagation of %s = %s + %s to known value %s",
                               new_qi,
                               qi_in,
                               change_in_quantitative_invisibility,
                               qi_out)
                return -1
            return qi_out
        # Propagate to the out segment if the in segment has known QI
        elif qi_in >= 0:
            new_qi: int = qi_in + change_in_quantitative_invisibility
            if new_qi < 0:
                logger.warning("Attempted to propagate negative QI across node")
                return -1
            self.set_segment_quantitative_invisibility(out_segment, new_qi)
            self.set_node_quantitative_invisibility(node_index, new_qi)
            return new_qi
        # Propagate to the out segment if the in segment has known QI
        elif qi_out >= 0:
            new_qi: int = qi_out - change_in_quantitative_invisibility
            if new_qi < 0:
                logger.warning("Attempted to propagate negative QI across node")
                return -1
            self.set_segment_quantitative_invisibility(in_segment, new_qi)
            self.set_node_quantitative_invisibility(node_index, qi_out)
            return qi_out
        # Cannot propagate if both are unassigned
        else:
            logger.warning("Cannot propagate QI across node if both segments are unassigned")
            return -1

    def __propagate_quantitative_invisibility_at_intersection_node(
            self,
            spline_surface: QuadraticSplineSurface,
            node_index: NodeIndex,
            invisibility_params: InvisibilityParameters) -> bool:
        """
        Propagate the QI at an intersection node
        """
        logger.debug("Propagating QI at intersection node %s", node_index)
        tau: Vector3f = np.array([0, 0, 1], dtype=np.float64)

        # Check node validity
        if not self._is_valid_node_index(node_index):
            logger.debug("Attempting to propagate QI at invalid node %s", node_index)
            return False

        # Get the intersecting node and check it is valid
        intersecting_node_index: NodeIndex = self.intersection(node_index)
        if self._is_valid_node_index(intersecting_node_index):
            logger.error(
                "Attempting to propagate QI with intersection rule at invalid node %s",
                intersecting_node_index)
            return False

        # Handle boundary case separately
        if self.is_tnode(node_index):
            return self.__propagate_quantitative_invisibility_at_boundary_intersection_node(
                spline_surface, node_index, invisibility_params)

        # Determine if the node is above the intersection node
        node_point: SpatialVector1d = self.node_spatial_point(node_index)
        intersecting_node_point: SpatialVector1d = self.node_spatial_point(intersecting_node_index)
        node_tau_projection: float = node_point @ tau
        intersecting_node_tau_projection: float = intersecting_node_point @ tau
        is_above: bool = (node_tau_projection < intersecting_node_tau_projection)
        logger.debug("Node has view direction projection %s", node_tau_projection)
        logger.debug("Intersecting node has view direction projection %s",
                     intersecting_node_tau_projection)

        # Determine if the contour crosses the intersection with positive
        # orientation, meaning the intersecting tangent is ccw from the node tangent.
        # This means the intersecting tangent is outward normal to the projected
        # surface along the node contour.
        node_tangent: PlanarPoint1d = self.node_planar_tangent(node_index)
        intersecting_node_tangent: PlanarPoint1d = self.node_planar_tangent(intersecting_node_index)
        node_tangent_normal: PlanarPoint1d = np.array([-node_tangent[1], node_tangent[0]])
        orientation_ind: float = dot_product(node_tangent_normal, intersecting_node_tangent)
        is_positively_oriented: bool = orientation_ind > 0
        logger.debug("Node tangent %s with planar projection %s",
                     self.node_spatial_tangent(node_index),
                     node_tangent)
        logger.debug("Intersecting node tangent %s with planar projection %s",
                     self.node_spatial_tangent(intersecting_node_index),
                     intersecting_node_tangent)

        # Propagate QI at the intersecting according to casework
        new_qi: int
        # Above with positive orientation
        if is_above and is_positively_oriented:
            logger.debug("Input node is above, and the intersection is positively oriented")
            new_qi = self.__propagate_quantitative_invisibility_across_node(node_index, 0)
            if new_qi < 0:
                logger.error("Failed to propagate QI across intersection node")
                return False
            self.set_segment_quantitative_invisibility(self.out(intersecting_node_index),
                                                       new_qi)
            self.set_segment_quantitative_invisibility(self.in_(intersecting_node_index),
                                                       new_qi + 2)
            self.set_node_quantitative_invisibility(intersecting_node_index,
                                                    new_qi)
        # Above with negative orientation
        elif is_above and not is_positively_oriented:
            logger.debug("Input node is above, and the intersection is negatively oriented")
            new_qi = self.__propagate_quantitative_invisibility_across_node(node_index, 0)
            if new_qi < 0:
                logger.error("Failed to propagate QI across intersection node")
                return False
            self.set_segment_quantitative_invisibility(self.out(intersecting_node_index),
                                                       new_qi + 2)
            self.set_segment_quantitative_invisibility(self.in_(intersecting_node_index),
                                                       new_qi)
            self.set_node_quantitative_invisibility(intersecting_node_index,
                                                    new_qi + 2)
        # Below with positive orientation
        elif (not is_above) and is_positively_oriented:
            logger.debug("Input node is below, and the intersection is positively oriented")
            new_qi = self.__propagate_quantitative_invisibility_across_node(node_index, 2)
            if new_qi < 0:
                logger.error("Failed to propagate QI across intersection node")
                return False
            self.set_segment_quantitative_invisibility(self.out(intersecting_node_index),
                                                       new_qi - 2)
            self.set_segment_quantitative_invisibility(self.in_(intersecting_node_index),
                                                       new_qi - 2)
            self.set_node_quantitative_invisibility(intersecting_node_index, new_qi - 2)
        #  Below with negative orientation
        elif (not is_above) and (not is_positively_oriented):
            logger.debug("Input node is below, and the intersection is negatively oriented")
            new_qi = self.__propagate_quantitative_invisibility_across_node(node_index, -2)
            if new_qi < 0:
                logger.error("Failed to propagate QI across intersection node")
                return False
            self.set_segment_quantitative_invisibility(self.out(intersecting_node_index), new_qi)
            self.set_segment_quantitative_invisibility(self.in_(intersecting_node_index), new_qi)
            self.set_node_quantitative_invisibility(intersecting_node_index, new_qi)

        # Get the QI for the adjacent segments
        qi_in: int = self.get_segment_quantitative_invisibility(self.in_(node_index))
        qi_out: int = self. get_segment_quantitative_invisibility(self.out(node_index))
        qi_intersecting_in: int = self.get_segment_quantitative_invisibility(
            self.in_(intersecting_node_index))
        qi_intersecting_out: int = self. get_segment_quantitative_invisibility(
            self.out(intersecting_node_index))

        logger.debug("Propagated QI across intersection node %s -> %s with "
                     "intersecting node %s -> %s",
                     qi_in,
                     qi_out,
                     qi_intersecting_in,
                     qi_intersecting_out)

        # Check the QI against direct computation
        if invisibility_params.check_propagation:
            logger.debug("Checking QI propagation at intersection node")
            self.__check_quantitative_invisibility_propagation(
                spline_surface, self.in_(node_index), invisibility_params)
            self.__check_quantitative_invisibility_propagation(
                spline_surface, self.out(node_index), invisibility_params)
            self.__check_quantitative_invisibility_propagation(
                spline_surface, self.in_(intersecting_node_index), invisibility_params)
            self.__check_quantitative_invisibility_propagation(
                spline_surface, self.out(intersecting_node_index), invisibility_params)

        # View the intersection locally
        if invisibility_params.view_intersections:
            self.__view_intersection_node(spline_surface, node_index)

        # Propagate all nodes
        self.__propagate_quantitative_invisibility_forward(
            spline_surface, self.out(node_index), invisibility_params)
        self.__propagate_quantitative_invisibility_backward(
            spline_surface, self.in_(node_index), invisibility_params)
        self.__propagate_quantitative_invisibility_forward(
            spline_surface, self.out(intersecting_node_index), invisibility_params)
        self.__propagate_quantitative_invisibility_backward(
            spline_surface, self.in_(intersecting_node_index), invisibility_params)

        return True

    def __propagate_quantitative_invisibility_at_boundary_intersection_node(
            self,
            spline_surface: QuadraticSplineSurface,
            node_index: NodeIndex,
            invisibility_params: InvisibilityParameters) -> bool:
        """
        Propagate the QI at a boundary intersection node
        """
        logger.debug("Propagating QI at boundary intersection node %s", node_index)

        # Check node validity
        if not self._is_valid_node_index(node_index):
            logger.debug("Attempting to propagate QI at invalid node %s", node_index)
            return False

        # Get the intersecting node and check it is valid
        intersecting_node_index: NodeIndex = self.intersection(node_index)
        if self._is_valid_node_index(intersecting_node_index):
            logger.error(
                "Attempting to propagate QI with intersection rule at invalid node %s",
                intersecting_node_index)
            return False

        # Node out segment QI
        qi_out: int = self.__compute_chain_quantitative_invisibility(spline_surface,
                                                                     self.out(node_index),
                                                                     invisibility_params)
        if qi_out >= 0:
            self.set_segment_quantitative_invisibility(self.out(node_index), qi_out)

        # Node in segment QI
        qi_in: int = self.__compute_chain_quantitative_invisibility(spline_surface,
                                                                    self.in_(node_index),
                                                                    invisibility_params)
        if qi_in >= 0:
            self.set_segment_quantitative_invisibility(self.in_(node_index), qi_in)

        # Node QI
        self.set_node_quantitative_invisibility(node_index, max(qi_in, qi_out))

        # Intersection node out segment
        qi_intersection_out: int = self.__compute_chain_quantitative_invisibility(
            spline_surface, self.out(intersecting_node_index), invisibility_params)
        if qi_intersection_out >= 0:
            self.set_segment_quantitative_invisibility(self.out(intersecting_node_index),
                                                       qi_intersection_out)

        # Intersection node in segment QI
        qi_intersection_in: int = self.__compute_chain_quantitative_invisibility(
            spline_surface, self.in_(intersecting_node_index), invisibility_params)
        if qi_intersection_in >= 0:
            self.set_segment_quantitative_invisibility(self.in_(intersecting_node_index),
                                                       qi_intersection_in)

        # Intersection node QI
        self.set_node_quantitative_invisibility(intersecting_node_index,
                                                max(qi_intersection_in, qi_intersection_out))

        # Propagate QI directly at the boundary intersection. For the segment that
        # does not exist, nothing is done in these function calls
        logger.debug("Propagated QI across boundary intersection node %s -> %s with "
                     "intersecting node %s -> %s",
                     qi_in,
                     qi_out,
                     qi_intersection_in,
                     qi_intersection_out)

        self.__propagate_quantitative_invisibility_forward(
            spline_surface, self.out(node_index), invisibility_params)
        self.__propagate_quantitative_invisibility_backward(
            spline_surface, self.in_(node_index), invisibility_params)
        self.__propagate_quantitative_invisibility_forward(
            spline_surface, self.out(intersecting_node_index), invisibility_params)
        self.__propagate_quantitative_invisibility_backward(
            spline_surface, self.in_(intersecting_node_index), invisibility_params)

        return True

    def __propagate_quantitative_invisibility_at_cusp_node(
            self,
            spline_surface: QuadraticSplineSurface,
            node_index: NodeIndex,
            invisibility_params: InvisibilityParameters) -> bool:
        """
        Propagate QI at a cusp
        """
        # Check node validity
        if not self._is_valid_node_index(node_index):
            logger.debug("Attempting to propagate QI at invalid node %s", node_index)
            return False

        # Determine if the tangent is pointing toward or away from the camera
        tau: Vector3f = np.array([0, 0, 1], dtype=np.float64)
        node_tangent: SpatialVector1d = self.node_spatial_tangent(node_index)
        node_tangent_tau_projection = node_tangent * tau
        is_reversed: bool = node_tangent_tau_projection < 0
        logger.debug("Tangent at cusp node: %s", node_tangent)
        logger.debug("In segment midpoint is %s and out midpoint segment is %s",
                     self.segment_spatial_curve(self.in_(node_index)).mid_point(),
                     self.segment_spatial_curve(self.out(node_index)).mid_point())

        # Propigate the QI
        new_qi: int
        if is_reversed:
            new_qi = self.__propagate_quantitative_invisibility_across_node(node_index, -1)
        else:
            new_qi = self.__propagate_quantitative_invisibility_across_node(node_index, 1)

        if new_qi < 0:
            logger.error("Failed to propagate QI across cusp node")
            return False

        qi_in: int = self.get_segment_quantitative_invisibility(self.in_(node_index))
        qi_out: int = self.get_segment_quantitative_invisibility(self.out(node_index))
        logger.debug("Propagated QI across cusp node %s -> %s", qi_in, qi_out)

        # Check the QI against direct computation
        if invisibility_params.check_propagation:
            logger.debug("Checking QI propagation at cusp node")
            self.__check_quantitative_invisibility_propagation(
                spline_surface, self.in_(node_index), invisibility_params)
            self.__check_quantitative_invisibility_propagation(
                spline_surface, self.out(node_index), invisibility_params)

        # View the cusp locally
        if invisibility_params.view_cusps:
            self.__view_cusp_node(spline_surface, node_index)

        # Propagate the QI
        self.__propagate_quantitative_invisibility_forward(
            spline_surface, self.out(node_index), invisibility_params)
        self.__propagate_quantitative_invisibility_backward(
            spline_surface, self.in_(node_index), invisibility_params)

        return True

    def __propagate_quantitative_invisibility_at_boundary_cusp_node(
        self,
            spline_surface: QuadraticSplineSurface,
            node_index: NodeIndex,
            invisibility_params: InvisibilityParameters) -> None:
        """
        Propagate QI at a boundary cusp node by just computing QI directly and propagating
        """
        # Check node validity
        if not self._is_valid_node_index(node_index):
            logger.debug("Attempting to propagate QI at invalid node %s", node_index)
            return

        qi_out: int = self.__compute_chain_quantitative_invisibility(
            spline_surface, self.out(node_index), invisibility_params)
        qi_in: int = self.__compute_chain_quantitative_invisibility(
            spline_surface, self.in_(node_index), invisibility_params)
        self.set_segment_quantitative_invisibility(self.out(node_index), qi_out)
        self.set_segment_quantitative_invisibility(self.in_(node_index), qi_in)
        self.set_node_quantitative_invisibility(node_index, qi_out)
        self.__propagate_quantitative_invisibility_forward(
            spline_surface, self.out(node_index), invisibility_params)
        self.__propagate_quantitative_invisibility_backward(
            spline_surface, self.in_(node_index), invisibility_params)
        return

    def __propagate_quantitative_invisibility_at_marked_knot_node(
        self,
            spline_surface: QuadraticSplineSurface,
            node_index: NodeIndex,
            invisibility_params: InvisibilityParameters) -> None:
        """
        Propagate QI at a marked knot node by propagating known QI across it without change
        """
        new_qi: int = self.__propagate_quantitative_invisibility_across_node(node_index, 0)
        if new_qi < 0:
            logger.error("Failed to propagate QI across marked node")
            return
        self.__propagate_quantitative_invisibility_forward(
            spline_surface, self.out(node_index), invisibility_params)
        self.__propagate_quantitative_invisibility_backward(
            spline_surface, self.in_(node_index), invisibility_params)

    def __check_quantitative_invisibility_propagation(
        self,
            spline_surface: QuadraticSplineSurface,
            segment_index: SegmentIndex,
            invisibility_params: InvisibilityParameters) -> None:
        """
        Check QI propagation at segment with direct computation
        """
        qi: int = self.get_segment_quantitative_invisibility(segment_index)
        direct_qi: int = self.__compute_segment_quantitative_invisibility(
            spline_surface, segment_index, invisibility_params)
        if qi != direct_qi:
            logger.warning("Propagated QI %s differs from direct computation %s", qi, direct_qi)
        else:
            logger.debug("Propagated QI %s correctly", qi)

    # *****************
    # Global QI Methods
    # *****************
    def __compute_default_quantitative_invisibility(self) -> None:
        """
        Set QI to 0 for all segments
        """
        # Set QI to 0 for all segments independently
        for i in range(self.num_segments):
            self.set_segment_quantitative_invisibility(i, 0)

    def __compute_direct_quantitative_invisibility(self,
                                                   spline_surface: QuadraticSplineSurface,
                                                   invisibility_params: InvisibilityParameters
                                                   ) -> None:
        """
        Compute the quantitative invisibility for each segment of the contour
        network.
        Note: This method is extremely slow and should only be used for
        validation on small examples.
        """
        # Compute QI for all segments independently
        for i in range(self.num_segments):
            qi: int = self.__compute_segment_quantitative_invisibility(
                spline_surface, i, invisibility_params)
            self.set_segment_quantitative_invisibility(i, qi)

    def __compute_chained_quantitative_invisibility(self,
                                                    spline_surface: QuadraticSplineSurface,
                                                    invisibility_params: InvisibilityParameters
                                                    ) -> None:
        """
        Compute the quantitative invisibility for each chain of the contour network
        and propagate it to all segments
        """

        if INLINE_TESTING_ENABLED_QI:
            filepath: str = f"{TESTING_FOLDER_SOURCE}\\contour_network\\contour_network\\compute_chained_quantitative_invisibility\\"

            compare_eigen_numpy_matrix(filepath+"chain_start_nodes.csv",
                                       np.array(self.chain_start_nodes))

        # Compute QI for all segment chains independently
        for i, start_node_index in enumerate(self.chain_start_nodes):
            # Compute QI for the first segment
            start_segment_index: SegmentIndex = self.out(start_node_index)

            qi_start: int = self.__compute_chain_quantitative_invisibility(
                spline_surface, start_segment_index, invisibility_params)
            self.set_segment_quantitative_invisibility(start_segment_index, qi_start)

            if INLINE_TESTING_ENABLED_QI:
                filepath: str = f"{TESTING_FOLDER_SOURCE}\\contour_network\\contour_network\\compute_chained_quantitative_invisibility\\"
                # Start segment index is GOOD
                # compare_eigen_numpy_matrix(filepath+"start_segment_index\\" + f"{i}.csv",
                #                            np.array(start_segment_index))

                # only qi start is being picky right now.
                compare_eigen_numpy_matrix(filepath+"qi_start\\" + f"{i}.csv",
                                           np.array(qi_start))
            # Propagate the QI to the other chain segments
            self.__chain_quantitative_invisibility_forward(
                spline_surface, start_segment_index, invisibility_params)

    def __compute_propagated_quantitative_invisibility(self,
                                                       spline_surface: QuadraticSplineSurface,
                                                       invisibility_params: InvisibilityParameters
                                                       ) -> None:
        """
        Compute the quantitative invisibility by computing it per chain and propagating
        across nodes with local rules when possible
        """
        # Compute QI for all segment chains independently
        for start_node_index in self.chain_start_nodes:
            start_segment_index: SegmentIndex = self.out(start_node_index)
            logger.info("Attempting to continue QI propagation start node %s",
                        start_node_index)

            # Compute QI for the first segment and propagate it if it doesn't already exist
            if not self.node_quantitative_invisibility_is_set(start_node_index):
                qi_start: int = self.__compute_chain_quantitative_invisibility(
                    spline_surface, start_segment_index, invisibility_params)
                self.set_node_quantitative_invisibility(start_node_index, qi_start)
                self.set_segment_quantitative_invisibility(start_segment_index, qi_start)
                logger.info("Propagating QI of %s from start node %s",
                            self.get_segment_quantitative_invisibility(start_segment_index),
                            start_segment_index)
                self.__propagate_quantitative_invisibility_forward(
                    spline_surface, start_segment_index, invisibility_params)

    def __compute_quantitative_invisibility(self,
                                            spline_surface: QuadraticSplineSurface,
                                            invisibility_params: InvisibilityParameters
                                            ) -> None:
        """
        Compute the quantitative invisibility with a method of choice
        """
        # Compute hash table here
        # Perform chosen invisibility method
        # Optionally skip QI computation
        if invisibility_params.invisibility_method == InvisibilityMethod.NONE:
            self.__compute_default_quantitative_invisibility()
        # Compute QI per segment
        elif invisibility_params.invisibility_method == InvisibilityMethod.DIRECT:
            logger.info("Start compute direct QI")
            start: float = time.time()
            self.__compute_direct_quantitative_invisibility(spline_surface, invisibility_params)
            logger.info("Direct QI compute Time cost: %ss", time.time() - start)
        # Compute QI per chain
        elif invisibility_params.invisibility_method == InvisibilityMethod.CHAINING:
            logger.info("Start compute chaining QI")
            start: float = time.time()
            self.__compute_chained_quantitative_invisibility(spline_surface, invisibility_params)
            logger.info("Chaining QI compute Time cost: %ss", time.time() - start)
        elif invisibility_params.invisibility_method == InvisibilityMethod.PROPAGATION:
            logger.info("start compute propagation QI")
            start: float = time.time()
            self.__compute_propagated_quantitative_invisibility(spline_surface,
                                                                invisibility_params)
            logger.info("Propagation QI compute Time cost: %ss",
                        time.time() - start)

        # Check for errors in the QI
        #
        # TODO: quantitative invisibility is wrong, likely because the direction of the mesh is wrong... or whatnot.
        #
        quantitative_invisibility: list[int] = self.enumerate_quantitative_invisibility()

        logger.info(quantitative_invisibility)
        if vector_contains(quantitative_invisibility, -1):
            logger.error("Negative QI present in final values")
            raise ValueError("Negative QI present in final values")

    # *******
    # Viewers
    # *******
    def __view_local_features(self,
                              spline_surface: QuadraticSplineSurface,
                              patch_indices: list[int],
                              segment_indices: list[SegmentIndex],
                              node_indices: list[NodeIndex]) -> None:
        """
        View all local features in the contour network
        Warning: Should only be used for small meshes
        """
        # View patches
        for i, patch_index in enumerate(patch_indices):
            patch_name: str = "local_patch_" + str(i)
            spline_surface.get_patch(patch_index).add_patch_to_viewer(patch_name)

        # View segments
        for i, segment_index in enumerate(segment_indices):
            segment_name: str = "local_segment_" + str(i)
            self.segment_spatial_curve(segment_index).add_curve_to_viewer(segment_name)

        # View nodes
        node_points: MatrixNx3f = np.ndarray(shape=(len(node_indices), 3), dtype=np.float64)
        node_in_tangents: MatrixNx3f = np.ndarray(shape=(len(node_indices), 3), dtype=np.float64)
        node_out_tangents: MatrixNx3f = np.ndarray(shape=(len(node_indices), 3), dtype=np.float64)
        for i, node_index in enumerate(node_indices):
            node_points[i, :] = self.node_spatial_point(node_index)
            node_in_tangents[i, :] = self.node_spatial_in_tangent(node_index)
            node_out_tangents[i, :] = self.node_spatial_out_tangent(node_index)
        points: polyscope.PointCloud = polyscope.register_point_cloud("node_points", node_points)
        points.add_vector_quantity("node_in_tangents", node_in_tangents)
        points.add_vector_quantity("node_out_tangents", node_out_tangents)

        # Show viewer with orthographic projection
        polyscope.set_view_projection_mode("orthographic")

        # TODO: translate below
        #    auto viewer_callback = [&]() {
        #     if (ImGui::Button("Serialize")) {
        #       QuadraticSplineSurface local_surface =
        #         spline_surface.subsurface(patch_indices);
        #       local_surface.save_obj("local_features.obj");
        #       local_surface.write_spline("local_features");
        #     }
        #   };
        #   polyscope::state::userCallback = viewer_callback;

        polyscope.show()
        polyscope.remove_all_structures()

    def __view_intersection_node(self,
                                 spline_surface: QuadraticSplineSurface,
                                 node_index: NodeIndex) -> None:
        """
        View an intersection node in the contour network
        Warning: Should only be used for small meshes
        """
        # Get intersection nodes
        intersecting_node_index: NodeIndex = self. intersection(node_index)
        node_indices: list[NodeIndex] = [node_index, intersecting_node_index]

        # Get intersection segments
        in_segment: SegmentIndex = self. in_(node_index)
        out_segment: SegmentIndex = self.  out(node_index)
        intersecting_in_segment:  SegmentIndex = self. in_(intersecting_node_index)
        intersecting_out_segment: SegmentIndex = self.out(intersecting_node_index)
        segment_indices: list[SegmentIndex] = [
            in_segment, out_segment, intersecting_in_segment, intersecting_out_segment]

        # Get intersection patches
        in_patch: int = self. get_segment_label(in_segment, "surface_patch")
        out_patch: int = self.  get_segment_label(out_segment, "surface_patch")
        intersecting_in_patch: int = self.get_segment_label(intersecting_in_segment,
                                                            "surface_patch")
        intersecting_out_patch: int = self.get_segment_label(intersecting_out_segment,
                                                             "surface_patch")
        patch_indices: list[PatchIndex] = [
            in_patch, out_patch, intersecting_in_patch, intersecting_out_patch]

        spline_surface.add_surface_to_viewer()
        self.add_spatial_network_to_viewer()

        # View with local features
        self.__view_local_features(
            spline_surface, patch_indices, segment_indices, node_indices)

    def __view_cusp_node(self,
                         spline_surface: QuadraticSplineSurface,
                         node_index: NodeIndex) -> None:
        """
        View a cusp node in the contour network
        Warning: Should only be used for small meshes
        """
        # Get cusp nodes
        node_indices: list[NodeIndex] = [node_index]

        # Get cusp segments
        in_segment: SegmentIndex = self.in_(node_index)
        out_segment: SegmentIndex = self. out(node_index)
        segment_indices: list[SegmentIndex] = [in_segment, out_segment]

        # Get cusp patches
        in_patch: int = self.get_segment_label(in_segment, "surface_patch")
        out_patch: int = self.get_segment_label(out_segment, "surface_patch")
        patch_indices: list[PatchIndex] = [in_patch, out_patch]
        spline_surface.add_surface_to_viewer()  # FIXME
        self.add_spatial_network_to_viewer()   # FIXME

        self.__view_local_features(spline_surface, patch_indices, segment_indices, node_indices)
