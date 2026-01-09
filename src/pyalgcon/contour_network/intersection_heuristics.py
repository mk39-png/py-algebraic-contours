"""
intersection_heuristics.py
Methods to quickly determine if two planar curves do not intersect.
"""


import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from pyalgcon.core.common import (
    PLANAR_BOUNDING_BOX_PRECISION, Matrix4x2f, Matrix5x2f, Matrix5x3f,
    Matrix5x5f, PlanarPoint1d, Vector5f, compute_point_cloud_bounding_box,
    float_equal)
from pyalgcon.core.rational_function import RationalFunction

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class IntersectionStats():
    """
    Statistics regarding Intersections
    """
    num_intersection_tests: int = 0
    num_bezier_nonoverlaps: int = 0
    bounding_box_call: int = 0
    intersection_call: int = 0


# **************************************************
# Helper methods (only for use inside intersection_heursitics.py)
# **************************************************

def _are_disjoint_bounding_boxes(first_box_lower_left_point: PlanarPoint1d,
                                 first_box_upper_right_point: PlanarPoint1d,
                                 second_box_lower_left_point: PlanarPoint1d,
                                 second_box_upper_right_point: PlanarPoint1d) -> bool:
    """
    Return true iff the two boxes defined by the extreme points are disjoint
    """
    assert first_box_lower_left_point.shape == (2, )
    assert first_box_upper_right_point.shape == (2, )
    assert second_box_lower_left_point.shape == (2, )
    assert second_box_upper_right_point.shape == (2, )

    # Extract points
    min_x_0: float = first_box_lower_left_point[0]
    min_y_0: float = first_box_lower_left_point[1]
    max_x_0: float = first_box_upper_right_point[0]
    max_y_0: float = first_box_upper_right_point[1]
    min_x_1: float = second_box_lower_left_point[0]
    min_y_1: float = second_box_lower_left_point[1]
    max_x_1: float = second_box_upper_right_point[0]
    max_y_1: float = second_box_upper_right_point[1]

    # Check if boxes are disjoint
    if min_x_0 > max_x_1:
        return True
    if min_x_1 > max_x_0:
        return True
    if min_y_0 > max_y_1:
        return True
    if min_y_1 > max_y_0:
        return True

    # Otherwise, there is overlap
    return False


def _is_valid_bounding_box(planar_curve: RationalFunction,
                           t_min: float,
                           t_max: float,
                           lower_left_point: PlanarPoint1d,
                           upper_right_point: PlanarPoint1d) -> bool:
    """
    Check if the bounding box contains the curve over interval
    WARNING: May have false positives
    """
    assert (planar_curve.degree, planar_curve.dimension) == (4, 2)
    assert lower_left_point.shape == (2, )
    assert upper_right_point.shape == (2, )

    t_avg: float = (t_min + t_max) / 2.0

    # Build test points
    test_point_1: PlanarPoint1d = planar_curve(t_min + 1e-6)
    test_point_2: PlanarPoint1d = planar_curve(t_avg)
    test_point_3: PlanarPoint1d = planar_curve(t_max - 1e-6)

    logger.debug("Testing points on curve at %s, %s, %s: %s, %s, %s",
                 t_min,
                 t_avg,
                 t_max,
                 test_point_1,
                 test_point_2,
                 test_point_3)

    # Check all points
    if not is_in_bounding_box(test_point_1, lower_left_point, upper_right_point):
        logger.warning("Point %s at %s not in bounding box %s, %s",
                       test_point_1,
                       t_min,
                       lower_left_point,
                       upper_right_point)
    if not is_in_bounding_box(test_point_2, lower_left_point, upper_right_point):
        logger.warning("Point %s at %s not in bounding box %s, %s",
                       test_point_2,
                       t_avg,
                       lower_left_point,
                       upper_right_point)
    if not is_in_bounding_box(test_point_3, lower_left_point, upper_right_point):
        logger.warning("Point %s at %s not in bounding box %s, %s",
                       test_point_3,
                       t_max,
                       lower_left_point,
                       upper_right_point)

    # Valid otherwise
    return True


def _pad_bounding_box(lower_left_point_ref: PlanarPoint1d,
                      upper_right_point_ref: PlanarPoint1d,
                      padding: float = 0.0) -> None:
    """
    Pad the bounding box by some epsilon
    """
    assert lower_left_point_ref.shape == (2, )
    assert upper_right_point_ref.shape == (2, )

    lower_left_point_ref[0] -= padding
    lower_left_point_ref[1] -= padding
    upper_right_point_ref[0] += padding
    upper_right_point_ref[1] += padding


def _compute_homogeneous_bezier_points_from_matrix(planar_curve: RationalFunction,
                                                   monomial_to_bezier_matrix: Matrix5x5f
                                                   ) -> Matrix5x3f:
    """
    Compute bezier control points for a rational curve from a given change
    of coordinates matrix
    """
    assert (planar_curve.degree, planar_curve.dimension) == (4, 2)
    assert monomial_to_bezier_matrix.shape == (5, 5)

    P_coeffs: Matrix5x2f = planar_curve.numerators
    Q_coeffs: Vector5f = planar_curve.denominator
    assert P_coeffs.shape == (5, 2)
    assert Q_coeffs.shape == (5, )

    x_coeffs: Vector5f = P_coeffs[:, 0]
    y_coeffs: Vector5f = P_coeffs[:, 1]
    w_coeffs: Vector5f = np.copy(Q_coeffs)  # deep copy
    assert x_coeffs.shape == (5, )
    assert y_coeffs.shape == (5, )
    assert w_coeffs.shape == (5, )

    # Compute bezier homogeneous points
    bezier_points: Matrix5x3f = np.column_stack([monomial_to_bezier_matrix @ x_coeffs,
                                                 monomial_to_bezier_matrix @ y_coeffs,
                                                 monomial_to_bezier_matrix @ w_coeffs])
    assert bezier_points.shape == (5, 3)
    return bezier_points


def _compute_homogeneous_bezier_points(planar_curve: RationalFunction) -> Matrix5x3f:
    """
    Compute the bezier points for a planar curve over the full domain [-1, 1]
    """
    assert planar_curve.degree == 4
    assert planar_curve.dimension == 2

    logger.info("Computing Bezier coefficients")

    # Compute matrix to go from monomial coefficients to Bezier coefficients
    # TODO: double check order of matrix creation
    monomial_to_bezier_matrix: Matrix5x5f = np.array([[1.0, 1.0, 1.0, 1.0, 1.0],
                                                      [1.0, 0.5, 0.0, -0.5, -1.0],
                                                      [1.0, 0.0, -(1.0 / 3.0), 0.0, 1.0],
                                                      [1.0, -0.5, 0.0, 0.5, -1.0],
                                                      [1.0, -1.0, 1.0, -1.0, 1.0]],
                                                     dtype=np.float64)
    assert monomial_to_bezier_matrix.shape == (5, 5)

    # Compute bezier points from the matrix
    bezier_points: Matrix5x3f = _compute_homogeneous_bezier_points_from_matrix(
        planar_curve, monomial_to_bezier_matrix)
    return bezier_points

# ***************
# Primary methods
# ***************


def compute_homogeneous_bezier_points_over_interval(planar_curve: RationalFunction,
                                                    t_min: float,
                                                    t_max: float) -> Matrix5x3f:
    """
    Compute the bezier control points for a curve over a domain

    The domain may differ from the domain of the planar curve itself.

    :param planar_curve: [in] planar curve segment. planar_curve degree = 4, dimension = 2
    :param t_min: [in] lower bound of the domain
    :param t_max: [in] upper bound of the domain
    :return bezier_points: homogeneous control points of the curve
    """
    assert (planar_curve.degree, planar_curve.dimension) == (4, 2)

    logger.debug("Computing Bezier coefficients for interval [%s, %s]", t_min, t_max)
    r: float = t_max - t_min

    # Compute matrix to go from monomial coefficients to Bezier coefficients
    monomial_to_bezier_matrix: Matrix5x5f = np.ndarray(shape=(5, 5), dtype=np.float64)

    # First row
    monomial_to_bezier_matrix[0, 0] = pow(r + t_min, 0)
    monomial_to_bezier_matrix[0, 1] = pow(r + t_min, 1)
    monomial_to_bezier_matrix[0, 2] = pow(r + t_min, 2)
    monomial_to_bezier_matrix[0, 3] = pow(r + t_min, 3)
    monomial_to_bezier_matrix[0, 4] = pow(r + t_min, 4)

    # Second row
    monomial_to_bezier_matrix[1, 0] = 1.0
    monomial_to_bezier_matrix[1, 1] = 0.75 * r + t_min
    monomial_to_bezier_matrix[1, 2] = 0.5 * ((r + 2 * t_min) * (r + t_min))
    monomial_to_bezier_matrix[1, 3] = 0.25 * ((r + 4 * t_min) * pow(r + t_min, 2))
    monomial_to_bezier_matrix[1, 4] = t_min * pow(r + t_min, 3)

    # Third row
    monomial_to_bezier_matrix[2, 0] = 1.0
    monomial_to_bezier_matrix[2, 1] = 0.5 * r + t_min
    monomial_to_bezier_matrix[2, 2] = pow(r, 2) / 6.0 + r * t_min + pow(t_min, 2)
    monomial_to_bezier_matrix[2, 3] = 0.5 * (t_min * (r + 2 * t_min) * (r + t_min))
    monomial_to_bezier_matrix[2, 4] = pow(t_min, 2) * pow(r + t_min, 2)

    # Fourth row
    monomial_to_bezier_matrix[3, 0] = 1.0
    monomial_to_bezier_matrix[3, 1] = 0.25 * r + t_min
    monomial_to_bezier_matrix[3, 2] = 0.5 * (t_min * (r + 2 * t_min))
    monomial_to_bezier_matrix[3, 3] = 0.25 * (pow(t_min, 2) * (3 * r + 4 * t_min))
    monomial_to_bezier_matrix[3, 4] = pow(t_min, 3) * (r + t_min)

    # Fifth row
    monomial_to_bezier_matrix[4, 0] = pow(t_min, 0)
    monomial_to_bezier_matrix[4, 1] = pow(t_min, 1)
    monomial_to_bezier_matrix[4, 2] = pow(t_min, 2)
    monomial_to_bezier_matrix[4, 3] = pow(t_min, 3)
    monomial_to_bezier_matrix[4, 4] = pow(t_min, 4)

    # Compute bezier points from the matrix
    bezier_points: Matrix5x3f = _compute_homogeneous_bezier_points_from_matrix(
        planar_curve, monomial_to_bezier_matrix)
    assert bezier_points.shape == (5, 3)
    return bezier_points


def _compute_bezier_bounding_box_over_domain(planar_curve: RationalFunction,
                                             t_min: float,
                                             t_max: float) -> tuple[PlanarPoint1d, PlanarPoint1d]:
    """
    Compute a bounding box for a planar curve over a domain using Bezier control
    points
    """
    assert (planar_curve.degree,  planar_curve.dimension) == (4, 2)
    logger.debug("Computing bezier bounding box for %s over [%s, %s]",
                 planar_curve,
                 t_min,
                 t_max)

    # Convert to bezier coordinates
    bezier_points: Matrix5x3f = compute_homogeneous_bezier_points_over_interval(
        planar_curve, t_min, t_max)
    assert bezier_points.shape == (5, 3)

    # Normalize homogeneous coordinates
    bezier_x_coords: Vector5f = bezier_points[:, 0] / bezier_points[:, 2]
    bezier_y_coords: Vector5f = bezier_points[:, 1] / bezier_points[:, 2]

    # Bezier points should interpolate the endpoints
    assert float_equal(planar_curve(t_min)[0], bezier_x_coords[4])
    assert float_equal(planar_curve(t_min)[1], bezier_y_coords[4])
    assert float_equal(planar_curve(t_max)[0], bezier_x_coords[0])
    assert float_equal(planar_curve(t_max)[1], bezier_y_coords[0])

    # Get the max and min x values from the points
    x_min: float = np.min(bezier_x_coords)
    x_max: float = np.max(bezier_x_coords)

    # Get the max and min y values from the points
    y_min: float = np.min(bezier_y_coords)
    y_max: float = np.max(bezier_y_coords)

    # Build lower left point
    lower_left_point: PlanarPoint1d = np.array([x_min, y_min])
    logger.debug("Lower left point: %s", lower_left_point)

    # Build upper right point
    upper_right_point: PlanarPoint1d = np.array([x_max, y_max])
    logger.debug("Upper right point: %s", upper_right_point)

    # Check validity
    assert _is_valid_bounding_box(planar_curve, t_min, t_max, lower_left_point,
                                  upper_right_point)

    return lower_left_point, upper_right_point


def compute_bezier_bounding_box(planar_curve: RationalFunction) -> tuple[PlanarPoint1d,
                                                                         PlanarPoint1d]:
    """
    Compute a bounding box for a planar curve using Bezier control points for the
    two subcurves split at the middle

    :param planar_curve: [in] planar curve segment. planar_curve degree = 4, dimension = 2
    :return lower_left_point: lower left point of the bounding box
    :return upper_right_point: upper right point of the bounding box
    """
    assert (planar_curve.degree, planar_curve.dimension) == (4, 2)

    # Get domain and domain midpoint
    t_min: float = planar_curve.domain.lower_bound
    t_max: float = planar_curve.domain.upper_bound
    t_avg: float = (t_max + t_min) / 2.0

    # Get first bezier bounding box
    first_box_lower_left_point: PlanarPoint1d
    first_box_upper_right_point: PlanarPoint1d
    (first_box_lower_left_point,
     first_box_upper_right_point) = _compute_bezier_bounding_box_over_domain(planar_curve,
                                                                             t_min,
                                                                             t_avg)

    # Get second bezier bounding box
    second_box_lower_left_point: PlanarPoint1d
    second_box_upper_right_point: PlanarPoint1d
    (second_box_lower_left_point,
     second_box_upper_right_point) = _compute_bezier_bounding_box_over_domain(planar_curve,
                                                                              t_avg,
                                                                              t_max)
    # Combine two bounding boxes
    lower_left_point: PlanarPoint1d
    upper_right_point: PlanarPoint1d
    lower_left_point, upper_right_point = combine_bounding_boxes(first_box_lower_left_point,
                                                                 first_box_upper_right_point,
                                                                 second_box_lower_left_point,
                                                                 second_box_upper_right_point)

    # Inversely pad (that is, trim) the bounding box slightly to prevent
    # common endpoint intersections
    _pad_bounding_box(lower_left_point, upper_right_point, -1e-10)

    # Check validity
    assert _is_valid_bounding_box(planar_curve, t_min, t_max, lower_left_point, upper_right_point)

    return lower_left_point, upper_right_point


def compute_bezier_bounding_boxes(planar_curves: list[RationalFunction]
                                  ) -> tuple[list[PlanarPoint1d], list[PlanarPoint1d]]:
    """
    Compute a bounding boxes for a planar curve using Bezier control points.

    :param planar_curves: [in] planar curve segment. planar_curve degree = 4, dimension = 2
    :return lower_left_points: lower left point of the bounding box
    :return upper_right_points: upper right point of the bounding box
    """
    # lazy checking
    assert (planar_curves[0].degree, planar_curves[0].dimension) == (4, 2)

    lower_left_points: list[PlanarPoint1d] = []
    upper_right_points: list[PlanarPoint1d] = []
    for _, planar_curve in enumerate(planar_curves):
        lower_left_point: PlanarPoint1d
        upper_right_point: PlanarPoint1d
        lower_left_point, upper_right_point = compute_bezier_bounding_box(planar_curve)
        assert lower_left_point.shape == (2, )
        assert upper_right_point.shape == (2, )

        lower_left_points.append(lower_left_point)
        upper_right_points.append(upper_right_point)

    return lower_left_points, upper_right_points


def combine_bounding_boxes(first_box_lower_left_point: PlanarPoint1d,
                           first_box_upper_right_point: PlanarPoint1d,
                           second_box_lower_left_point: PlanarPoint1d,
                           second_box_upper_right_point: PlanarPoint1d) -> tuple[PlanarPoint1d,
                                                                                 PlanarPoint1d]:
    """
    Get the bounding box containing two bounding boxes

    :param first_box_lower_left_point: [in] lower left point of the first bounding box
    :param first_box_upper_right_point: [in] upper right point of the first bounding box
    :param second_box_lower_left_point: [in] lower left point of the second bounding box
    :param second_box_upper_right_point: [in] upper right point of the second bounding box

    :return lower_left_point: lower left point of the combined bounding box
    :return upper_right_point: upper right point of the combined bounding box
    """
    # Build point cloud from the input bounding box points
    points: Matrix4x2f = np.array([first_box_lower_left_point,  # row 0
                                   first_box_upper_right_point,  # row 1
                                   second_box_lower_left_point,  # row 2
                                   second_box_upper_right_point],  # row 3
                                  dtype=np.float64)
    assert points.shape == (4, 2)

    # Get the bounding box of the point cloud
    lower_left_point: PlanarPoint1d
    upper_right_point: PlanarPoint1d
    lower_left_point, upper_right_point = compute_point_cloud_bounding_box(points)
    assert lower_left_point.shape == (2, )
    assert upper_right_point.shape == (2, )

    return lower_left_point, upper_right_point


def is_in_bounding_box(test_point: PlanarPoint1d,
                       lower_left_point: PlanarPoint1d,
                       upper_right_point: PlanarPoint1d) -> bool:
    """
    Check if a test point is in a bounding box with extreme corners lower left
    point and upper right point

    :param test_point: [in] point to test for containment
    :param lower_left_point: [in] lower left point of the bounding box
    :param upper_right_point: [in] upper right point of the bounding box
    :return: true iff the test point is in the bounding box
    """
    assert test_point.shape == (2, )
    assert lower_left_point.shape == (2, )
    assert upper_right_point.shape == (2, )

    # Equivalent to checking if the trivial bounding box at the test point
    # overlaps the bounding box
    return not _are_disjoint_bounding_boxes(
        test_point, test_point, lower_left_point, upper_right_point)


def _are_disjoint_bezier_bounding_boxes_planar_curves(
        first_planar_curve: RationalFunction,
        second_planar_curve: RationalFunction) -> bool:
    """
    Check if two curves have disjoint Bezier bounding boxes
    """
    assert first_planar_curve.degree == 4
    assert first_planar_curve.dimension == 2
    assert second_planar_curve.degree == 4
    assert second_planar_curve.dimension == 2

    # Compute first bezier bounding box
    first_box_lower_left_point: PlanarPoint1d
    first_box_upper_right_point: PlanarPoint1d
    (first_box_lower_left_point,
     first_box_upper_right_point) = compute_bezier_bounding_box(first_planar_curve)

    # Compute second bezier bounding box
    second_box_lower_left_point: PlanarPoint1d
    second_box_upper_right_point: PlanarPoint1d
    (second_box_lower_left_point,
     second_box_upper_right_point) = compute_bezier_bounding_box(second_planar_curve)

    # Compare bezier bounding boxes
    return _are_disjoint_bounding_boxes(first_box_lower_left_point,
                                        first_box_upper_right_point,
                                        second_box_lower_left_point,
                                        second_box_upper_right_point)


def _are_disjoint_bezier_bounding_boxes_bounding_boxes(
        first_bounding_box: tuple[PlanarPoint1d, PlanarPoint1d],
        second_bounding_box: tuple[PlanarPoint1d, PlanarPoint1d]) -> bool:
    """
    Check if two bounding boxes are disjoint
    """
    first_box_lower_left_point: PlanarPoint1d = first_bounding_box[0]
    first_box_upper_right_point: PlanarPoint1d = first_bounding_box[1]
    second_box_lower_left_point: PlanarPoint1d = second_bounding_box[0]
    second_box_upper_right_point: PlanarPoint1d = second_bounding_box[1]

    # Compare bezier bounding boxes
    return _are_disjoint_bounding_boxes(first_box_lower_left_point,
                                        first_box_upper_right_point,
                                        second_box_lower_left_point,
                                        second_box_upper_right_point)


def are_nonintersecting_by_heuristic_planar_curve(first_planar_curve: RationalFunction,
                                                  second_planar_curve: RationalFunction,
                                                  intersection_stats_ref: IntersectionStats
                                                  ) -> bool:
    """
    Determine if two planar curves cannot intersect by heuristics

    This method can only determine if two curves do not intersect and may have false negatives.

    :param first_planar_curve:  [in] first planar curve segment. degree = 4, dimension = 2
    :param second_planar_curve: [in] second planar curve segment. degree = 4, dimension = 2
    :param intersection_stats_ref: [in, out] statistics for intersections computation
    :return: true if the two curves do not intersect
    """
    # Check bezier bounding boxes
    if _are_disjoint_bezier_bounding_boxes_planar_curves(first_planar_curve,
                                                         second_planar_curve):
        logger.info("Bezier bounding boxes do not overlap")
        intersection_stats_ref.num_bezier_nonoverlaps += 1
        return True

    # Otherwise, we cannot determine if there are intersections
    return False


def are_nonintersecting_by_heuristic_bounding_box(
        first_bounding_box: tuple[PlanarPoint1d, PlanarPoint1d],
        second_bounding_box: tuple[PlanarPoint1d, PlanarPoint1d],
        intersection_stats_ref: IntersectionStats) -> bool:
    """
    Determine if two planar curves cannot intersect by heuristics using just  the bounding boxes
    of the curve

    This method can only determine if two curves do not intersect and may have false negatives.

    :param first_bounding_box: [in] bounding box of the first planar curve segment
    :param second_bounding_box: [in] bounding box of the second planar curve segment
    :param intersection_stats_ref: [in, out] statistics for intersections computation
    :return: true if the two curves do not intersect
    """
    # Check bezier bounding boxes
    if _are_disjoint_bezier_bounding_boxes_bounding_boxes(first_bounding_box,
                                                          second_bounding_box):
        logger.info("Bezier bounding boxes do not overlap")
        intersection_stats_ref.num_bezier_nonoverlaps += 1
        return True

    # Otherwise, we cannot determine if there are intersections
    return False


def compute_bounding_box_hash_table(bounding_boxes: list[tuple[PlanarPoint1d, PlanarPoint1d]]
                                    ) -> tuple[dict[int, dict[int, list[int]]],
                                               list[list[int]]]:
    """
    Hash bounding boxes by x and y coordinates.

    :param bounding_boxes: [in] bounding boxes to hash

    :return hash_table: lists of bounding boxes per hash region
    :return reverse_hash_table: map from bounding boxes to ids of hash regions containing them
    """
    num_interval: int = 50
    num_segments: int = len(bounding_boxes)

    # So, hash_table[49][49] will give a list[int], equivalent to a vector.
    hash_table: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    reverse_hash_table: list[list[int]] = [[] for _ in range(num_segments)]
    if num_segments == 0:
        return hash_table, reverse_hash_table

    # Below defined for readability in terms of reading from tuple
    FIRST = 0
    SECOND = 1
    segments_bbox_x_min: float = bounding_boxes[0][FIRST][0]
    segments_bbox_y_min: float = bounding_boxes[0][FIRST][1]
    segments_bbox_x_max: float = bounding_boxes[0][SECOND][0]
    segments_bbox_y_max: float = bounding_boxes[0][SECOND][1]

    for i in range(1, num_segments):
        if segments_bbox_x_min > bounding_boxes[i][FIRST][0]:
            segments_bbox_x_min = bounding_boxes[i][FIRST][0]
        if segments_bbox_y_min > bounding_boxes[i][FIRST][1]:
            segments_bbox_y_min = bounding_boxes[i][FIRST][1]
        if segments_bbox_x_max < bounding_boxes[i][SECOND][0]:
            segments_bbox_x_max = bounding_boxes[i][SECOND][0]
        if segments_bbox_y_max < bounding_boxes[i][SECOND][1]:
            segments_bbox_y_max = bounding_boxes[i][SECOND][1]

    x_interval: float = (segments_bbox_x_max - segments_bbox_x_min) / num_interval
    y_interval: float = (segments_bbox_y_max - segments_bbox_y_min) / num_interval
    eps: float = PLANAR_BOUNDING_BOX_PRECISION

    for i in range(num_segments):
        left_x: int = int(
            (bounding_boxes[i][FIRST][0] - eps - segments_bbox_x_min) / x_interval)
        right_x: int = num_interval - int(
            (segments_bbox_x_max - eps - bounding_boxes[i][SECOND][0]) / x_interval) - 1
        left_y: int = int(
            (bounding_boxes[i][FIRST][1] - eps - segments_bbox_y_min) / y_interval)
        right_y: int = num_interval - int(
            (segments_bbox_y_max - eps - bounding_boxes[i][SECOND][1]) / y_interval) - 1

        # TODO: check if below is equivalent to (int j = left_x; j <= right_x; j++)
        for j in range(left_x, right_x + 1):

            # TODO: check if below equivalent to  (int k = left_y; k <= right_y; k++)
            for k in range(left_y, right_y + 1):
                hash_table[j][k].append(i)
                reverse_hash_table[i].append(j * num_interval + k)

    return hash_table, reverse_hash_table


def _ccw(p0: PlanarPoint1d, p1: PlanarPoint1d, p2: PlanarPoint1d) -> bool:
    """
    Helper for has_linear_intersection()
    """
    assert p0.shape == (2, )
    assert p1.shape == (2, )
    assert p2.shape == (2, )
    return ((p2[1] - p0[1]) * (p1[0] - p0[0])) > ((p1[1] - p0[1]) * (p2[0] - p0[0]))


def has_linear_intersection(first_planar_curve: RationalFunction,
                            second_planar_curve: RationalFunction) -> bool:
    """
    Determine if the line between the endpoints of two curves intersect.

    Note that linear intersection does not imply curve intersection, and the
    the absence of linear intersection does not imply the curves do not intersect.
    It just roughly indicates potential intersections for simple curves.

    :param first_planar_curve: [in] first planar curve segment. degree = 4, dimension = 2
    :param second_planar_curve: [in] second planar curve segment. degree = 4, dimension = 2
    :return: true iff the lines between the respective endpoints of the curves intersect
    """
    l1_0: PlanarPoint1d = first_planar_curve.start_point()
    l1_1: PlanarPoint1d = first_planar_curve.end_point()
    l2_0: PlanarPoint1d = second_planar_curve.start_point()
    l2_1: PlanarPoint1d = second_planar_curve.end_point()
    assert l1_0.shape == (2, )
    assert l1_1.shape == (2, )
    assert l2_0.shape == (2, )
    assert l2_1.shape == (2, )

    return ((_ccw(l1_0, l2_0, l2_1) != _ccw(l1_1, l2_0, l2_1)) and
            (_ccw(l1_0, l1_1, l2_0) != _ccw(l1_0, l1_1, l2_1)))
