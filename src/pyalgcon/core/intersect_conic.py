"""
Conic intersections.
"""

import copy
import logging

from pyalgcon.core.common import (Matrix3x1r, Matrix3x2f,
                                  PlanarPoint1d, Vector2f,
                                  Vector3f, todo)
from pyalgcon.core.conic import Conic
from pyalgcon.core.convex_polygon import ConvexPolygon
from pyalgcon.core.polynomial_function import \
    quadratic_real_roots

logger: logging.Logger = logging.getLogger(__name__)


def intersect_conic_with_line(C_param: Conic,
                              L_coeffs: Vector3f,
                              intersections_ref: list[float]) -> bool:
    """
    Intersect a parametrized conic segment with an implicit line.

    :param C_param:  [in]  parametrized conic segment
    :param L_coeffs: [in] implicit line function
    :param intersections_ref: [out] list of intersection points in the domain of the conic
    :return: true iff TODO
    """
    P_coeffs: Matrix3x2f = C_param.numerators
    assert P_coeffs.shape == (3, 2)
    # HACK: flattening here when denominator should just be 1D by default
    Q_coeffs: Vector3f = C_param.denominator.flatten()

    # Get equation for intersection
    X_coeffs: Vector3f = P_coeffs[:, 0]
    Y_coeffs: Vector3f = P_coeffs[:, 1]
    a: float = L_coeffs[0]
    b: float = L_coeffs[1]
    c: float = L_coeffs[2]
    I_coeffs: Vector3f = a * Q_coeffs + b * X_coeffs + c * Y_coeffs

    use_stable_quadratic: bool = True
    if use_stable_quadratic:
        solutions: Vector2f
        num_solutions: int
        solutions, num_solutions = quadratic_real_roots(I_coeffs)
        if num_solutions == 0:
            return False

        elif num_solutions == 1:
            t0: float = solutions[0]
            if C_param.is_in_domain(t0):
                intersections_ref.append(t0)
            return True
        elif num_solutions == 2:
            t0: float = min(solutions[0], solutions[1])
            t1: float = max(solutions[0], solutions[1])
            assert t0 <= t1
            if C_param.is_in_domain(t0):
                intersections_ref.append(t0)
            if C_param.is_in_domain(t1):
                intersections_ref.append(t1)
            return True
    return False


def _are_in_polygon(conic_segments: list[Conic],  convex_polygon: ConvexPolygon) -> bool:
    """
    Checks if conic segments are in convex polygon
    """
    for conic_segment in conic_segments:
        if not convex_polygon.contains(conic_segment.mid_point()):
            return False

    return True


def intersect_conic_with_convex_polygon(conic: Conic,
                                        convex_polygon: ConvexPolygon
                                        ) -> tuple[list[Conic],
                                                   list[tuple[int, int]]]:
    """
    Intersect a parametrized conic segment with a convex polygon and
    split the conic into segments between intersection points.
    The indices of the line segments of the convex polygon bounding the conic
    segments (or -1 for open segments) are also recorded

    :param conic: [in] parametrized conic segment
    :param convex_polygon: [in] convex polygon to intersect

    :return conic_segments: parametrized conic split at intersection
    :return line_intersection_indices: indices of the bounding polygon edges
    """
    intersections_poly: list[float] = []
    indexed_intersections: list[tuple[float, int]] = []
    polygon_boundary_coeffs: list[Vector3f] = convex_polygon.boundary_segments

    conic_segments: list[Conic] = []
    line_intersection_indices: list[tuple[int, int]] = []
    t0: float
    t1: float
    t_sample: float
    p_sample: PlanarPoint1d

    # Get all intersections of the conic with the lines bounding the polygon
    for i, L_coeffs in enumerate(polygon_boundary_coeffs):
        intersect_conic_with_line(conic, L_coeffs, intersections_poly)

        # Record intersections with the line
        for j in range(len(indexed_intersections), len(intersections_poly)):
            indexed_intersections.append((intersections_poly[j], i))
        assert len(indexed_intersections) == len(intersections_poly)

    logger.debug("Intersections: %s", intersections_poly)
    indexed_intersections.sort()

    if len(indexed_intersections) == 0:
        t0 = conic.domain.lower_bound
        t1 = conic.domain.upper_bound
        if conic.domain.is_bounded_above() and conic.domain.is_bounded_below():
            t_sample = 0.5 * (t0 + t1)
        elif conic.domain.is_bounded_below():
            t_sample = t0 + 1.0
        elif conic.domain.is_bounded_above():
            t_sample = t1 - 1.0
        else:
            t_sample = 0.0

        p_sample = conic(t_sample).flatten()
        if convex_polygon.contains(p_sample):
            conic_segments.append(conic)
            line_intersection_indices.append((-1, -1))

        return conic_segments, line_intersection_indices

    # First segment
    t0 = conic.domain.lower_bound
    t1 = indexed_intersections[0][0]
    t_sample = max(0.5 * (t0 + t1), t1 - 1)
    p_sample = conic(t_sample).flatten()
    logger.debug("Sampling at %s", t_sample)
    if convex_polygon.contains(p_sample):
        conic_segment: Conic = copy.deepcopy(conic)
        conic_segment.domain.set_upper_bound(t1, False)
        conic_segments.append(conic_segment)
        line_intersection_indices.append((-1, indexed_intersections[0][1]))

    # Determine whether conic segments are contained in the polygon
    for i in range(len(indexed_intersections) - 1):
        t0 = indexed_intersections[i][0]
        t1 = indexed_intersections[i + 1][0]
        t_sample = 0.5 * (t0 + t1)
        p_sample = conic(t_sample)
        logger.debug("Sampling at %s", t_sample)
        if convex_polygon.contains(p_sample):
            # FIXME: double check that this does not modify original conic...
            # i.e. actually makes a deep copy
            conic_segment: Conic = copy.deepcopy(conic)
            conic_segment.domain.set_lower_bound(t0, False)
            conic_segment.domain.set_upper_bound(t1, False)
            conic_segments.append(conic_segment)
            line_intersection_indices.append(
                (indexed_intersections[i][1], indexed_intersections[i + 1][1]))

    # Last segment
    t0 = indexed_intersections[-1][0]
    t1 = conic.domain.upper_bound
    t_sample = min(0.5 * (t0 + t1), t0 + 1)
    logger.debug("Sampling at %s", t_sample)
    p_sample = conic(t_sample)
    if convex_polygon.contains(p_sample):
        conic_segment: Conic = copy.deepcopy(conic)
        conic_segment.domain.set_lower_bound(t0, False)
        conic_segments.append(conic_segment)
        line_intersection_indices.append((indexed_intersections[-1][1], -1))

    assert _are_in_polygon(conic_segments, convex_polygon)

    return conic_segments, line_intersection_indices


def intersect_conic_in_cone_patch(conic: Conic,
                                  convex_polygon: ConvexPolygon,
                                  cone_corner_index: int) -> tuple[bool,
                                                                   Conic | None,
                                                                   tuple[int, int] | None]:
    """"
    Intersect a parametrized conic segment in a cone patch with a convex
    polygon and split the conic into segments between the intersection points.
    The indices of the line segments of the convex polygon bounding the conic
    segments (or -1 for open segments) are also recorded

    NOTE: used in compute_contours.py

    :param conic:             [in] parametrized conic segment
    :param convex_polygon:    [in] convex polygon to intersect
    :param cone_corner_index: [in] index of the corner where the cone is located

    :return conic_segment: parametrized conic split at intersection
    :return line_intersection_indices: indices of the bounding polygon edges
    :return: true iff an intersection is found
    """
    # Get boundary for the edge opposite the cone corner
    polygon_boundaries_2d: list[Matrix3x1r] = convex_polygon.boundary_segments  # length 3
    assert len(polygon_boundaries_2d) == 3
    polygon_boundaries: list[Vector3f] = []
    # HACK: flattening elements shape (3, 1) list of boundary_segments to (3, )
    # TODO: remove this once transition fully away from 2D Vectors and towards 1D vectors of ndim == 1
    for boundary in polygon_boundaries_2d:
        polygon_boundaries.append(boundary.flatten())

    polygon_boundary_index: int = cone_corner_index
    L_coeffs: Vector3f = polygon_boundaries[polygon_boundary_index]

    # Get the intersection with the line
    intersections: list[float] = []
    intersect_conic_with_line(conic, L_coeffs, intersections)

    # Check that there are precisely two intersections
    if len(intersections) > 1:
        logger.error("More than two intersections found in cone patch")
        return False, None, None

    if len(intersections) == 0:
        logger.error("No intersection of the ray with the opposing line")
        return False, None, None

    # Split the conic at the intersection (depending on what kind of ray it is)
    # FIXME: potentially incompatible C++ translation below...
    # TODO: instead of making a deep copy... actually nvm porbably need a deep copy
    conic_segment: Conic = copy.deepcopy(conic)
    line_intersection_indices: tuple[int, int] = (-1, -1)
    t: float = intersections[0]
    if conic_segment.domain.is_bounded_above():
        conic_segment.domain.set_lower_bound(t, False)
        line_intersection_indices = (polygon_boundary_index, -1)
    elif conic_segment.domain.is_bounded_below():
        conic_segment.domain.set_upper_bound(t, False)
        line_intersection_indices = (-1, polygon_boundary_index)
    else:
        logger.error("Cone conic is a line, not the expected ray")
        return False, None, None

    return True, conic_segment, line_intersection_indices


def check_if_conic_intersects_cone_patch_domain(conic: Conic,
                                                convex_polygon: ConvexPolygon,
                                                cone_corner_index: int) -> bool:
    """
    Check if a parameterized line conic segment that passes through a cone
    corner in a cone patch intersects the interior of the triangle

    NOTE: used in compute_contours.py

    :param conic:             [in] parametrized conic segment
    :param convex_polygon:    [in] convex polygon to intersect
    :param cone_corner_index: [in] index of the corner where the cone is located

    :return: true iff the line through the conic stably intersects the interior
    of the domain
    """
    # Get implicit line equation L0 + Lx x + Ly y for the conic with 0 constant term
    P_coeffs: Matrix3x2f = conic.numerators
    assert P_coeffs.shape == (3, 2)

    Lx: float = P_coeffs[1, 1]
    Ly: float = -P_coeffs[1, 0]
    L0: float = -(Lx * P_coeffs[0, 0] + Ly * P_coeffs[0, 1])

    # Get the two domain points that the conic does not pass through
    uv: Matrix3x2f = convex_polygon.vertices
    assert uv.shape == (3, 2)
    domain_point_0: PlanarPoint1d = uv[((cone_corner_index + 1) % 3), :]
    domain_point_1: PlanarPoint1d = uv[((cone_corner_index + 2) % 3), :]
    assert domain_point_0.shape == (2, )
    assert domain_point_1.shape == (2, )

    # Get sign of the two points relative to the line
    domain_point_0_sign: float = L0 + Lx * domain_point_0[0] + Ly * domain_point_0[1]
    domain_point_1_sign: float = L0 + Lx * domain_point_1[0] + Ly * domain_point_1[1]

    # Check for sign difference
    return (domain_point_0_sign * domain_point_1_sign) < 0.0
