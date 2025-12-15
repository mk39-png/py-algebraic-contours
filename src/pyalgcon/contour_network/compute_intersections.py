"""
Methods to compute intersections for quadratic surfaces.
"""

import logging
import pathlib
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from pyalgcon.contour_network.compute_rational_bezier_curve_intersection import (
    find_intersections_bezier_clipping,
    split_bezier_curve_no_self_intersection)
from pyalgcon.contour_network.intersection_data import IntersectionData
from pyalgcon.contour_network.intersection_heuristics import (
    IntersectionStats, are_nonintersecting_by_heuristic_bounding_box,
    compute_bezier_bounding_box, compute_bounding_box_hash_table,
    compute_homogeneous_bezier_points_over_interval)
from pyalgcon.core.common import (FIND_INTERSECTIONS_BEZIER_CLIPPING_PRECISION,
                                  INLINE_TESTING_ENABLED_CONTOUR_NETWORK,
                                  Matrix5x3f, PlanarPoint1d, SpatialVector1d,
                                  compare_eigen_numpy_matrix,
                                  compare_list_list_varying_lengths,
                                  compare_list_list_varying_lengths_float,
                                  float_equal_zero, interval_lerp, load_json)
from pyalgcon.core.rational_function import RationalFunction
from pyalgcon.debug.debug import SPOT_FILEPATH
from pyalgcon.utils.compute_intersections_testing_utils import (
    compare_intersection_stats, compare_list_list_intersection_data_from_file)
from pyalgcon.utils.rational_function_testing_utils import \
    compare_rational_functions_from_file

logger: logging.Logger = logging.getLogger(__name__)

# FIXME: remove when done testing inline
ROOT_FOLDER = "spot_control"


@dataclass
class IntersectionParameters():
    """
    Parameters for intersection computations.
    """
    # If true, use heuristics to check if there are no intersections
    use_heuristics: bool = True
    # Amount to trim ends of contour segments by; intersections that are trimmed are clamped
    # to the endpoint.
    trim_amount: float = 1e-5
    # trim_amount: float = 1.0000000000000001e-05


def _convert_spline_to_planar_curve_parameter(planar_curve: RationalFunction,
                                              t_spline: float,
                                              epsilon: float = 0.0) -> float:
    """
    Map from the uniform domain [0, 1] to the planar curve domain
    """
    assert (planar_curve.degree, planar_curve.dimension) == (4, 2)

    t_min: float = planar_curve.domain.lower_bound + epsilon
    t_max: float = planar_curve.domain.upper_bound - epsilon
    return interval_lerp(0, 1, t_max, t_min, t_spline)


def _compute_bezier_clipping_planar_curve_intersections(first_planar_curve: RationalFunction,
                                                        second_planar_curve: RationalFunction,
                                                        first_bezier_control_points: Matrix5x3f,
                                                        second_bezier_control_points: Matrix5x3f,
                                                        epsilon: float = 0.0
                                                        ) -> list[tuple[float, float]]:
    """
    Compute planar curve intersections with Bezier clipping
    """
    assert (first_planar_curve.degree, first_planar_curve.dimension) == (4, 2)
    assert (second_planar_curve.degree, second_planar_curve.dimension) == (4, 2)
    assert first_bezier_control_points.shape == (5, 3)
    assert second_bezier_control_points.shape == (5, 3)

    intersection_points: list[tuple[float, float]] = []
    # FIXME:
    # if not check_split_criteria(curve1):
    #     print("potential self intersection p1!")
    intersection_param_inkscope: list[tuple[float, float]]
    p1: list[SpatialVector1d] = []
    p2: list[SpatialVector1d] = []
    for i in range(5):
        assert first_bezier_control_points[i, :].shape == (3, )
        assert second_bezier_control_points[i, :].shape == (3, )
        p1.append(first_bezier_control_points[i, :])
        p2.append(second_bezier_control_points[i, :])

    # Below gives back the amount of intersections
    intersection_param_inkscope = find_intersections_bezier_clipping(
        p1,
        p2,
        FIND_INTERSECTIONS_BEZIER_CLIPPING_PRECISION)  # 1e-7

    #
    # FIXME: below seems OK? I think? I mean, the problem is with finding too many intersections
    # Rather than the actual value of those intersections.
    #
    for spline in intersection_param_inkscope:
        t_spline: float = spline[0]
        s_spline: float = spline[1]
        t: float = _convert_spline_to_planar_curve_parameter(
            first_planar_curve, t_spline, epsilon)
        s: float = _convert_spline_to_planar_curve_parameter(
            second_planar_curve, s_spline, epsilon)
        intersection_points.append((t, s))

    return intersection_points


def _prune_intersection_points(first_planar_curve: RationalFunction,
                               second_planar_curve: RationalFunction,
                               intersection_points:  list[tuple[float, float]],
                               first_curve_intersections_ref: list[float],
                               second_curve_intersections_ref: list[float]) -> None:
    """
    Prune curve intersection points to the proper domains

    :param first_curve_intersections_ref: [out]
    :param second_curve_intersections_ref: [out]
    """
    assert (first_planar_curve.degree, first_planar_curve.dimension) == (4, 2)
    assert (second_planar_curve.degree, second_planar_curve.dimension) == (4, 2)

    # FIXME: Potential this function that decides whether or not to add something...
    # Check for i == 76 of intersection_indices...

    for intersection_point in intersection_points:
        t: float = intersection_point[0]
        s: float = intersection_point[1]

        # Trim points entirely out of domain of one of the two curves
        if not first_planar_curve.is_in_domain_interior(t):
            continue
        if not second_planar_curve.is_in_domain_interior(s):
            continue

        # Add points otherwise
        first_curve_intersections_ref.append(t)
        second_curve_intersections_ref.append(s)


# counter_compute_planar_curve_testing: int = 0


def _compute_planar_curve_intersections_from_bounding_box(
        first_planar_curve: RationalFunction,
        second_planar_curve: RationalFunction,
        intersect_params: IntersectionParameters,
        first_curve_intersections_ref: list[float],
        second_curve_intersections_ref: list[float],
        intersection_stats_ref: IntersectionStats,
        first_bounding_box: tuple[PlanarPoint1d, PlanarPoint1d],
        second_bounding_box: tuple[PlanarPoint1d, PlanarPoint1d],
        first_bezier_control_points: Matrix5x3f,
        second_bezier_control_points: Matrix5x3f) -> None:
    """
    Compute the intersections between the two planar curve segments using existing
    bounding box and control point data.

    :param first_planar_curve:  [in] first planar curve segment
    :param second_planar_curve: [in] second planar curve segment
    :param intersect_params:    [in] parameters for the intersection computation
    :param first_curve_intersections:  [out] list of intersection knots in the first curve
    :param second_curve_intersections: [out] list of intersection knots in the second curve
    :param intersection_stats:         [out] diagnostic information about the intersection
    :param first_bounding_box:  [in] bounding box for the first curve
    :param second_bounding_box: [in] bounding box for the second curve
    :param first_bezier_control_points:  [in] control points for the first curve
    :param second_bezier_control_points: [in] control points for the second curve
    """
    assert (first_planar_curve.degree, first_planar_curve.dimension) == (4, 2)
    assert (second_planar_curve.degree, second_planar_curve.dimension) == (4, 2)
    assert first_bezier_control_points.shape == (5, 3)
    assert second_bezier_control_points.shape == (5, 3)

    intersection_stats_ref.num_intersection_tests += 1
    t1: datetime = datetime.now()
    logger.debug("Finding intersections for %s and %s",
                 first_planar_curve,
                 second_planar_curve)

    intersection_stats_ref.bounding_box_call += 1

    if intersect_params.use_heuristics and are_nonintersecting_by_heuristic_bounding_box(
            first_bounding_box, second_bounding_box, intersection_stats_ref):
        return

    intersection_stats_ref.intersection_call += 1

    # Compute intersection points by Bezier clipping
    intersection_points: list[tuple[float, float]]
    try:
        #
        # FIXME: below is not giving values wanted... for some intersections
        #
        intersection_points = _compute_bezier_clipping_planar_curve_intersections(
            first_planar_curve,
            second_planar_curve,
            first_bezier_control_points,
            second_bezier_control_points)
    except Exception as e:
        logger.error("Failed to find intersection points %s", e)
        # FIXME: raising another error to just end the program whenever theres a big failure
        # But change back to exception for the final release build
        raise ValueError(e)
        return

    # Prune the computed intersections to ensure they are in the correct domain
    #
    # FIXME: this appears to work for image_segment_index == 9 in compute_intersections
    #
    _prune_intersection_points(first_planar_curve,
                               second_planar_curve,
                               intersection_points,
                               first_curve_intersections_ref,
                               second_curve_intersections_ref)

    # TODO: test this function for every call of this as well...
    # Which means... yeah... testing both pieces together since there could be something wrong with
    # how they interact?
    # Because both of them work fine on their own...

    t2: datetime = datetime.now()
    total_time: int = (t2 - t1).microseconds
    logger.debug("Finding intersections took %s ms", total_time)


def compute_planar_curve_intersections(first_planar_curve: RationalFunction,
                                       second_planar_curve: RationalFunction,
                                       intersect_params: IntersectionParameters,
                                       first_curve_intersections: list[float],
                                       second_curve_intersections: list[float],
                                       intersection_stats: IntersectionStats) -> None:
    """
    Compute the intersections between the two planar curve segments.

    :param first_planar_curve:  [in] first planar curve segment
    :param second_planar_curve: [in] second planar curve segment
    :param intersect_params:    [in] parameters for the intersection computation
    :param first_curve_intersections:  [out] list of intersection knots in the first curve
    :param second_curve_intersections: [out] list of intersection knots in the second curve
    :param intersection_stats:         [out] diagnostic information about the intersection
    """
    assert (first_planar_curve.degree, first_planar_curve.dimension) == (4, 2)
    assert (second_planar_curve.degree, second_planar_curve.dimension) == (4, 2)

    lower_left_point: PlanarPoint1d
    upper_right_point: PlanarPoint1d
    first_bounding_box: tuple[PlanarPoint1d, PlanarPoint1d]
    second_bounding_box: tuple[PlanarPoint1d, PlanarPoint1d]
    lower_left_point, upper_right_point = compute_bezier_bounding_box(first_planar_curve)
    first_bounding_box = (lower_left_point, upper_right_point)
    lower_left_point, upper_right_point = compute_bezier_bounding_box(second_planar_curve)
    second_bounding_box = (lower_left_point, upper_right_point)

    # Compute bezier points
    first_bezier_control_points: Matrix5x3f
    second_bezier_control_points: Matrix5x3f
    first_bezier_control_points = compute_homogeneous_bezier_points_over_interval(
        first_planar_curve,
        first_planar_curve.domain.lower_bound,
        first_planar_curve.domain.upper_bound)
    second_bezier_control_points = compute_homogeneous_bezier_points_over_interval(
        second_planar_curve,
        second_planar_curve.domain.lower_bound,
        second_planar_curve.domain.upper_bound)

    # Compute intersections with computed bounding boxes and Bezier points
    _compute_planar_curve_intersections_from_bounding_box(first_planar_curve,
                                                          second_planar_curve,
                                                          intersect_params,
                                                          first_curve_intersections,
                                                          second_curve_intersections,
                                                          intersection_stats,
                                                          first_bounding_box,
                                                          second_bounding_box,
                                                          first_bezier_control_points,
                                                          second_bezier_control_points)


def compute_intersections(image_segments: list[RationalFunction],
                          intersect_params: IntersectionParameters,
                          contour_intersections_ref: list[list[IntersectionData]],
                          num_intersections: int
                          ) -> tuple[list[list[float]],
                                     list[list[int]],
                                     int,
                                     int]:
    """
    Compute all intersections between the planar curve image segments.

    For each intersection, the knot in the curve and the other curve it
    corresponds to are recorded.

    :param image_segments:   [in] list of planar curve segments
    :param intersect_params: [in] parameters for the intersection computation
    :param contour_intersections: [out] list of lists of full intersection data
    :param num_intersections: [in] total number of intersections to increment
    TODO: some of the below need to be modified by reference...

    :return intersections: list of lists of intersection knots
    :return intersection_indices: list of lists of intersection indices
    :return num_intersections: total number of intersections to increment
    :return intersection_call: TODO description
    """
    # Lazy check
    assert (image_segments[0].degree, image_segments[0].dimension) == (4, 2)
    intersections: list[list[float]] = [[] for _ in enumerate(image_segments)]
    intersection_indices: list[list[int]] = [[] for _ in enumerate(image_segments)]

    # Setup intersection diagnostic tools
    intersection_stats: IntersectionStats = IntersectionStats()

    #
    # FIXME: method below looks good
    #
    # Compute all rational bezier control points
    image_segments_bezier_control_points: list[Matrix5x3f] = []
    for image_segment in image_segments:
        assert (image_segment.degree, image_segment.dimension) == (4, 2)
        bezier_control_points: Matrix5x3f = compute_homogeneous_bezier_points_over_interval(
            image_segment,
            image_segment.domain.lower_bound,
            image_segment.domain.upper_bound)
        image_segments_bezier_control_points.append(bezier_control_points)

    # if INLINE_TESTING_ENABLED_CONTOUR_NETWORK:
    #     filepath: pathlib.Path = SPOT_FILEPATH / "contour_network" / "intersection_heuristics" / \
    #         "compute_homogeneous_bezier_points_over_interval"
    #     compare_eigen_numpy_matrix(filepath / "image_segments_bezier_control_points.csv",
    #                                np.array(image_segments_bezier_control_points),
    #                                make_3d=True)

    #
    # FIXME: method below looks good
    #
    # Compute all bounding boxes
    image_segments_bounding_box: list[tuple[PlanarPoint1d, PlanarPoint1d]] = []
    for image_segment in image_segments:
        lower_left_point: PlanarPoint1d
        upper_right_point: PlanarPoint1d
        lower_left_point, upper_right_point = compute_bezier_bounding_box(image_segment)
        image_segments_bounding_box.append((lower_left_point, upper_right_point))

    # if INLINE_TESTING_ENABLED_CONTOUR_NETWORK:
    #     filepath: str = "spot_control\\contour_network\\intersection_heuristics\\compute_bezier_bounding_box\\"
    #     compare_eigen_numpy_matrix(filepath+"image_segments_bounding_box.csv",
    #                                np.array(image_segments_bounding_box),
    #                                make_3d=True)

    #
    # FIXME: method below looks good
    #
    # Hash by uv
    num_interval: int = 50
    # FIXME Make global: change both here and num_interval
    hash_table: dict[int, dict[int, list[int]]]
    reverse_hash_table: list[list[int]]
    hash_table, reverse_hash_table = compute_bounding_box_hash_table(image_segments_bounding_box)

    if INLINE_TESTING_ENABLED_CONTOUR_NETWORK:
        filepath: pathlib.Path = SPOT_FILEPATH / "contour_network" / \
            "intersection_heuristics" / "compute_bounding_box_hash_table"

        hash_table_control: list[list[list[int]]] = load_json(filepath / "hash_table_out.json")
        # Hash table is 50x50 with each element holding an array of int.
        for i in range(50):
            outer_table: list[list[int]] = hash_table_control[i]
            for j in range(50):
                inner_table: list[int] = hash_table_control[i][j]
                for k, data_control in enumerate(inner_table):
                    assert hash_table[i][j][k] == data_control
        compare_list_list_varying_lengths(filepath / "reverse_hash_table.csv", reverse_hash_table)

    # COUNTER FOR TESTING
    counter: int = 0
    counter_planar_curve_calls: int = 0
    # counter_compute_planar_curve_testing

    # Compute intersections
    # FIXME: check this loop itself to see if its correct or not
    num_segments: int = len(image_segments)
    for image_segment_index in range(num_segments):
        cells: list[int] = reverse_hash_table[image_segment_index]
        visited: list[bool] = [False] * image_segment_index

        for cell in cells:
            j: int = cell // num_interval
            k: int = cell % num_interval

            # FIXME: maybe switch data strucutre for the hash table
            # Wait, the ordering of the hash table is a bit weird...

            for i in hash_table[j][k]:
                if i >= image_segment_index or visited[i]:
                    continue
                visited[i] = True

                # Iterate over image segments with lower indices
                logger.debug("Computing segments %s, %s out of %s",
                             image_segment_index,
                             i,
                             len(image_segments))

                # Compute intersections between the two image segments
                current_segment_intersections: list[float] = []
                other_segment_intersections: list[float] = []

                # Testing to see if parameters into the function below are correct
                if INLINE_TESTING_ENABLED_CONTOUR_NETWORK:
                    filepath: pathlib.Path = SPOT_FILEPATH / "contour_network" / \
                        "compute_intersections" / "compute_planar_curve_intersections"
                    compare_rational_functions_from_file(
                        filepath / "first_planar_curve" / f"{counter}.json",
                        [image_segments[image_segment_index]])
                    compare_rational_functions_from_file(
                        filepath / "second_planar_curve" / f"{counter}.json", [image_segments[i]])
                    compare_intersection_stats(filepath / "intersection_stats_in" / f"{counter}.json",
                                               intersection_stats)
                    compare_eigen_numpy_matrix(filepath / "first_bounding_box" / f"{counter}.csv",
                                               np.array(image_segments_bounding_box[image_segment_index]))
                    compare_eigen_numpy_matrix(filepath / "second_bounding_box" / f"{counter}.csv",
                                               np.array(image_segments_bounding_box[i]))
                    compare_eigen_numpy_matrix(
                        filepath / "first_bezier_control_points" / f"{counter}.csv",
                        image_segments_bezier_control_points[image_segment_index])
                    compare_eigen_numpy_matrix(
                        filepath / "second_bezier_control_points" / f"{counter}.csv",
                        image_segments_bezier_control_points[i])

                #
                # FIXME: method below is providing more segment intersections than we would like
                #
                _compute_planar_curve_intersections_from_bounding_box(
                    image_segments[image_segment_index],
                    image_segments[i],
                    intersect_params,
                    current_segment_intersections,  # first_curve_intersections_ref
                    other_segment_intersections,  # second_curve_intersections_ref
                    intersection_stats,
                    image_segments_bounding_box[image_segment_index],
                    image_segments_bounding_box[i],
                    image_segments_bezier_control_points[image_segment_index],
                    image_segments_bezier_control_points[i])
                counter_planar_curve_calls += 1
                # print(counter_planar_curve_calls)
                # logger.error(counter_planar_curve_calls)

                # TODO: test this loop below...
                # Because uhhh, I haven't tested it yet for every iteration.
                intersections[image_segment_index].extend(current_segment_intersections)
                intersections[i].extend(other_segment_intersections)

                # Record the respective indices corresponding to the intersections
                for _ in other_segment_intersections:
                    intersection_indices[image_segment_index].append(i)
                    intersection_indices[i].append(image_segment_index)

                # Build full intersection data
                for k, _ in enumerate(other_segment_intersections):
                    current_intersection_data: IntersectionData = IntersectionData(
                        knot=current_segment_intersections[k],
                        intersection_index=i,
                        intersection_knot=other_segment_intersections[k],
                        id_=num_intersections)
                    current_intersection_data.check_if_tip(
                        image_segments[image_segment_index].domain,
                        intersect_params.trim_amount)
                    current_intersection_data.check_if_base(
                        image_segments[image_segment_index].domain,
                        intersect_params.trim_amount)
                    contour_intersections_ref[image_segment_index].append(
                        current_intersection_data)

                    # Build complementary boundary intersection data
                    other_intersection_data: IntersectionData = IntersectionData(
                        knot=other_segment_intersections[k],
                        intersection_index=image_segment_index,
                        intersection_knot=current_segment_intersections[k],
                        id_=num_intersections)
                    other_intersection_data.check_if_tip(image_segments[i].domain,
                                                         intersect_params.trim_amount)
                    other_intersection_data.check_if_base(image_segments[i].domain,
                                                          intersect_params.trim_amount)
                    contour_intersections_ref[i].append(other_intersection_data)
                    num_intersections += 1

                # Comparing intermediate results
                if INLINE_TESTING_ENABLED_CONTOUR_NETWORK:
                    filepath: pathlib.Path = SPOT_FILEPATH / \
                        "contour_network" / "compute_intersections" / \
                        "compute_intersections" / "after_compute_curve_intersections"
                    compare_list_list_varying_lengths_float(
                        filepath / "intersections" / f"{counter}.csv",
                        intersections,
                        1e-3)
                    compare_list_list_varying_lengths(
                        filepath / "intersection_indices" / f"{counter}.csv",
                        intersection_indices)
                    compare_list_list_intersection_data_from_file(
                        filepath / "contour_intersections" / f"{counter}.json",
                        contour_intersections_ref)
                    compare_eigen_numpy_matrix(
                        filepath / "num_intersections" / f"{counter}.csv",
                        np.array(num_intersections))
                    counter += 1

    # Record intersection information
    intersection_call: int = intersection_stats.intersection_call
    logger.info("Number of intersection tests: %s",
                intersection_stats.num_intersection_tests)
    logger.info("Number of nonoverlapping Bezier boxes: %s",
                intersection_stats.num_bezier_nonoverlaps)

    return (intersections,
            intersection_indices,
            num_intersections,
            intersection_call)


def split_planar_curves_no_self_intersection(planar_curves: list[RationalFunction]
                                             ) -> list[list[float]]:
    """
    Get planar curve split parameters until the refined curves cannot have self
    intersections

    TODO: this is not used anywhere else besides testing.

    :param planar_curves: [in] list of planar curve segments
    :return split_points: points to split curves at to avoid self intersections
    """
    split_points: list[list[float]] = [[] for _ in enumerate(planar_curves)]

    for i, planar_curve in enumerate(planar_curves):
        assert (planar_curve.degree, planar_curve.dimension) == (4, 2)
        if float_equal_zero(planar_curve.domain.get_length(), 1e-6):
            raise ValueError(f"Splitting curve of length {planar_curve.domain.get_length()}")

        # Get Bezier points
        bezier_control_points: Matrix5x3f = compute_homogeneous_bezier_points_over_interval(
            planar_curve,
            planar_curve.domain.lower_bound,
            planar_curve.domain.upper_bound)
        curve: list[SpatialVector1d] = [bezier_control_points[i, :] for i in range(5)]
        assert curve[0].shape == (3, )  # lazy shape check

        # Get splits in Bezier domain
        split_points_bezier: list[float] = []
        split_bezier_curve_no_self_intersection(curve, 0, 1, split_points_bezier)

        # Get splits in the planar curve domain
        for j, split_point_bezier in enumerate(split_points_bezier):
            split_points[i].append(_convert_spline_to_planar_curve_parameter(
                planar_curve,  # i
                split_point_bezier))  # j

    return split_points
