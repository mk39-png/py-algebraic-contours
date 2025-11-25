"""
Methods to compute a contours for quadratic surfaces.
"""

import logging

import numpy as np

from pyalgcon.contour_network.intersection_data import \
    IntersectionData
from pyalgcon.contour_network.validity import (
    is_valid_frame, is_valid_spatial_mapping)
from pyalgcon.core.common import (Matrix3x2f, Matrix3x3f,
                                  Matrix6x3f, PatchIndex,
                                  PlanarPoint1d, Vector3f,
                                  Vector6f)
from pyalgcon.core.conic import Conic
from pyalgcon.core.convex_polygon import ConvexPolygon
from pyalgcon.core.intersect_conic import (
    check_if_conic_intersects_cone_patch_domain, intersect_conic_in_cone_patch,
    intersect_conic_with_convex_polygon)
from pyalgcon.core.line_segment import LineSegment
from pyalgcon.core.parametrize_conic import (
    parametrize_cone_patch_conic, parametrize_conic)
from pyalgcon.core.rational_function import RationalFunction
from pyalgcon.quadratic_spline_surface.quadratic_spline_surface import \
    QuadraticSplineSurface
from pyalgcon.quadratic_spline_surface.quadratic_spline_surface_patch import \
    QuadraticSplineSurfacePatch

logger: logging.Logger = logging.getLogger(__name__)


# **************
# Helper Methods
# **************
def are_contained_in_patch_heuristic(contour_domain_curve_segments: list[Conic],
                                     spline_surface_patch: QuadraticSplineSurfacePatch) -> bool:
    """
    Return false if the contour domain curve segments are not contained in the
    patch domain based on a simple sampling heuristic
    """
    for _, contour_domain_curve_segment in enumerate(contour_domain_curve_segments):
        # Get curve sample points at the midpoint and near endpoints
        start_point: PlanarPoint1d
        mid_point: PlanarPoint1d
        end_point: PlanarPoint1d
        start_point = contour_domain_curve_segment.evaluate_normalized_coordinate(0.01)
        mid_point = contour_domain_curve_segment.evaluate_normalized_coordinate(0.5)
        end_point = contour_domain_curve_segment.evaluate_normalized_coordinate(0.99)
        assert start_point.ndim == 1
        assert mid_point.ndim == 1
        assert end_point.ndim == 1

        # Check domain containment
        domain: ConvexPolygon = spline_surface_patch.domain
        if not domain.contains(start_point):
            return False
        if not domain.contains(mid_point):
            return False
        if not domain.contains(end_point):
            return False

    return True


def compute_contour_equation(normal_mapping_coeffs: Matrix6x3f,
                             frame: Matrix3x3f) -> Vector6f:
    """
    Given the coefficients for the normal vector function and a projection
    frame, compute the coefficients for the implicit contour function.

    :param normal_mapping_coeffs: [in] coefficients for the surface normal function
    :param frame: [in] 3x3 matrix defining the projection
    :return contour_equation_coeffs: implicit contour function coefficients
    """
    assert is_valid_spatial_mapping(normal_mapping_coeffs)
    assert is_valid_frame(frame)

    tau: Vector3f = frame[:, 2]
    contour_equation_coeffs: Vector6f = normal_mapping_coeffs @ tau  # row dot product
    assert contour_equation_coeffs.shape == (6, )
    assert contour_equation_coeffs.ndim == 1

    # Normalize equation coefficients for stability
    contour_equation_coeffs /= np.linalg.norm(contour_equation_coeffs)

    return contour_equation_coeffs


def _compute_spline_surface_patch_contours(spline_surface_patch: QuadraticSplineSurfacePatch,
                                           frame: Matrix3x3f) -> tuple[list[Conic],
                                                                       list[RationalFunction],
                                                                       list[tuple[int, int]]]:
    """
    Get all contour segments for a spline surface patch

    :param spline_surface_patch: [in] quadratic spline surface patch
    :param frame: [in] 3x3 matrix defining the projection

    :return contour_domain_curve_segments: local parametric domain contour segments
    :return contour_segments: surface contour segments
    :return line_intersection_indices: indices of the bounding polygon edges
    """
    logger.debug("Computing contours for spline surface patch")
    contour_domain_curve_segments: list[Conic] = []
    contour_segments: list[RationalFunction] = []
    line_intersection_indices: list[tuple[int, int]] = []

    # Get surface mapping
    surface_mapping_coeffs: Matrix6x3f = spline_surface_patch.surface_mapping
    logger.debug("Patch surface mapping coefficients: %s", surface_mapping_coeffs)

    # Get surface normal mapping
    normal_mapping_coeffs: Matrix6x3f = spline_surface_patch.normal_mapping
    logger.debug("Patch normal mapping coefficients: %s",
                 normal_mapping_coeffs)

    # Get implicit contour equation
    contour_equation_coeffs: Vector6f
    contour_equation_coeffs = compute_contour_equation(normal_mapping_coeffs, frame)
    logger.debug("Patch contour equation coefficients: %s", contour_equation_coeffs)

    # Get full quadratic contours
    logger.debug("Parametrizing patch contour domain curves")
    contour_domain_curves: list[Conic]
    contour_domain_curves = parametrize_conic(contour_equation_coeffs)
    logger.debug("Domain curves: %s", contour_domain_curves)

    # Intersect contour domain curves with patch boundaries
    logger.debug("Intersecting domain curves with patch domain boundary")
    domain: ConvexPolygon = spline_surface_patch.domain
    for current_contour_domain_curve in contour_domain_curves:
        current_contour_domain_curve_segments: list[Conic]
        current_contour_line_intersection_indices: list[tuple[int, int]]
        (current_contour_domain_curve_segments,
         current_contour_line_intersection_indices) = intersect_conic_with_convex_polygon(
            current_contour_domain_curve,
            domain)
        contour_domain_curve_segments.extend(current_contour_domain_curve_segments)
        line_intersection_indices.extend(current_contour_line_intersection_indices)
        logger.debug("Contour domain curve split into %s segments",
                     len(current_contour_domain_curve_segments))

    # Lift contour domain curves to the surface
    logger.debug("Lifting domain curves to the surface")
    for contour_domain_curve_segment in contour_domain_curve_segments:
        contour_segment: RationalFunction = (
            contour_domain_curve_segment.pullback_quadratic_function(3, surface_mapping_coeffs))
        assert (contour_segment.degree, contour_segment.dimension) == (4, 3)
        contour_segments.append(contour_segment)
        logger.debug("Domain curve %s lifted to %s",
                     contour_domain_curve_segment,
                     contour_segment)

    assert are_contained_in_patch_heuristic(contour_domain_curve_segments, spline_surface_patch)
    return contour_domain_curve_segments, contour_segments, line_intersection_indices


def _compute_spline_surface_cone_patch_contours(spline_surface_patch: QuadraticSplineSurfacePatch,
                                                frame: Matrix3x3f
                                                ) -> tuple[list[Conic],
                                                           list[RationalFunction],  # <4, 3>
                                                           list[tuple[int, int]]]:
    """
    Get all contour segments for a spline surface patch

    :return contour_domain_curve_segments: list[Conic]
    :return contour_segments: list[RationalFunction]
    :return line_intersection_indices: list[tuple[int, int]]
    """
    logger.debug("Computing contours for spline surface patch")
    contour_domain_curve_segments: list[Conic] = []
    contour_segments: list[RationalFunction] = []  # <4, 3>
    line_intersection_indices: list[tuple[int, int]] = []

    # Check for cone
    if not spline_surface_patch.has_cone():
        raise ValueError("Tried to compute cone patch contours in a patch without a cone")
    cone_corner_index: int = spline_surface_patch.cone_index

    # Get surface mapping
    surface_mapping_coeffs: Matrix6x3f = spline_surface_patch.surface_mapping
    logger.debug("Patch surface mapping coefficients: %s", surface_mapping_coeffs)

    # Get surface normal mapping
    normal_mapping_coeffs: Matrix6x3f = spline_surface_patch.normal_mapping
    logger.debug("Patch normal mapping coefficients: %s", normal_mapping_coeffs)

    # Get implicit contour equation
    contour_equation_coeffs: Vector6f = compute_contour_equation(normal_mapping_coeffs, frame)
    logger.debug("Patch contour equation coefficients: %s", contour_equation_coeffs)

    # Get full quadratic contours
    logger.debug("Parametrizing cone patch contour domain curves")
    contour_domain_curves: list[Conic] = parametrize_cone_patch_conic(contour_equation_coeffs)
    logger.debug("Domain curves: %s", contour_domain_curves)

    # Intersect contour domain curves with patch boundaries
    domain: ConvexPolygon = spline_surface_patch.domain
    logger.debug("Intersecting domain curves with patch domain %s", domain.vertices)

    for current_contour_domain_curve in contour_domain_curves:
        # Check if the contour is stable
        conic_intersects: bool = check_if_conic_intersects_cone_patch_domain(
            current_contour_domain_curve, domain, cone_corner_index)
        if not conic_intersects:
            logger.debug("Line through conic does not intersect the domain")
            continue

        # Intersect the contour domain with the domain robustly
        current_contour_domain_curve_segment: Conic | None
        current_contour_line_intersection_indices: tuple[int, int] | None
        # TODO: fix typing, potentially None
        (conic_intersects,
         current_contour_domain_curve_segment,
         current_contour_line_intersection_indices) = intersect_conic_in_cone_patch(
            current_contour_domain_curve,
            domain,
            cone_corner_index)
        if not conic_intersects:
            logger.debug("Conic ray does not intersect the domain")
            continue

        # Add the contour if it exists
        contour_domain_curve_segments.append(current_contour_domain_curve_segment)
        line_intersection_indices.append(current_contour_line_intersection_indices)

    # Lift contour domain curves to the surface
    logger.debug("Lifting cone patch domain curves to the surface")
    for i, _ in enumerate(contour_domain_curve_segments):
        contour_domain_curve_segment: Conic
        contour_segment: RationalFunction  # <4, 3>
        contour_domain_curve_segment = contour_domain_curve_segments[i]
        contour_segment = contour_domain_curve_segment.pullback_quadratic_function(
            3,
            surface_mapping_coeffs)
        contour_segments.append(contour_segment)
        logger.debug("Cone patch domain curve %s lifted to %s",
                     contour_domain_curve_segment,
                     contour_segment)

    assert are_contained_in_patch_heuristic(contour_domain_curve_segments,
                                            spline_surface_patch)

    return contour_domain_curve_segments, contour_segments, line_intersection_indices


# ***************
# Primary Methods
# ***************

def compute_spline_surface_contours(spline_surface: QuadraticSplineSurface,
                                    frame: Matrix3x3f) -> tuple[list[Conic],
                                                                list[RationalFunction],
                                                                list[int],
                                                                list[tuple[int, int]]]:
    """
    Given a quadratic spline surface and a projection frame, compute the
    rational functions defining the contour segments.

    TODO: optimize this method as it has to iterate through over 9000 patches 

    Both (quadratic) parametric domain contour functions and (higher order)
    surface contour functions are computed. The indices of the corresponding
    patches for each segment are also extracted.

    :param spline_surface: [in] quadratic spline surface
    :param frame:          [in] 3x3 matrix defining the projection

    :return contour_domain_curve_segments: local parametric domain contour segments
    :return contour_segments: surface contour segments
    :return contour_patch_indices: spline surface patch indices for the contour segments
    :return line_intersection_indices: TODO description
    """
    assert is_valid_frame(frame)

    contour_domain_curve_segments: list[Conic] = []
    contour_segments:  list[RationalFunction] = []  # <4, 3>
    contour_patch_indices: list[PatchIndex] = []
    line_intersection_indices: list[tuple[int, int]] = []

    # Compute contours for each quadratic patch
    for patch_index in range(spline_surface.num_patches):
        spline_surface_patch: QuadraticSplineSurfacePatch = spline_surface.get_patch(patch_index)

        patch_contour_domain_curve_segments: list[Conic]
        patch_contour_segments: list[RationalFunction]  # <4, 3>
        patch_line_intersection_indices: list[tuple[int, int]]
        if spline_surface_patch.has_cone():
            logger.info("Parametrizing cone patch %s", patch_index)
            (patch_contour_domain_curve_segments,
             patch_contour_segments,
             patch_line_intersection_indices) = _compute_spline_surface_cone_patch_contours(
                spline_surface_patch,
                frame)
        else:
            (patch_contour_domain_curve_segments,
             patch_contour_segments,
             patch_line_intersection_indices) = _compute_spline_surface_patch_contours(
                spline_surface_patch,
                frame)

        # Append patch indices
        num_patch_contours: int = len(patch_contour_segments)
        patch_index_list: list[PatchIndex] = [patch_index] * num_patch_contours
        contour_domain_curve_segments.extend(patch_contour_domain_curve_segments)
        contour_segments.extend(patch_contour_segments)
        contour_patch_indices.extend(patch_index_list)
        line_intersection_indices.extend(patch_line_intersection_indices)
        assert len(contour_domain_curve_segments) == len(contour_segments)
        assert len(contour_patch_indices) == len(contour_segments)

    logger.info("Found %s contour segments", len(contour_segments))

    return (contour_domain_curve_segments,
            contour_segments,
            contour_patch_indices,
            line_intersection_indices)


def compute_spline_surface_boundaries(spline_surface: QuadraticSplineSurface,
                                      patch_boundary_edges: list[tuple[int, int]]
                                      ) -> tuple[list[Conic],
                                                 list[RationalFunction],  # <4, 3>
                                                 list[PatchIndex]]:
    """
    Given a quadratic spline surface, compute the rational functions defining
    the surface boundary.

    Both (quadratic) parametric domain boundary functions and (higher order)
    surface boundary functions are computed. The indices of the corresponding
    patches for each segment are also extracted.

    :param spline_surface:       [in] quadratic spline surface
    :param patch_boundary_edges: [in] edges of the patch triangle domains that
    are boundaries
    :return boundary_domain_curve_segments: local parametric domain boundary
    segments
    :return boundary_segments: surface boundary segments
    :return boundary_patch_indices: spline surface patch indices for the
    boundary segments
    """
    num_boundary_edges: int = len(patch_boundary_edges)
    boundary_domain_curve_segments: list[Conic] = []
    boundary_segments: list[RationalFunction] = []
    boundary_patch_indices: list[PatchIndex] = []

    # Get patch boundary
    for i in range(num_boundary_edges):
        # Get spline surface patch
        patch_index: int = patch_boundary_edges[i][0]
        patch_edge_index: int = patch_boundary_edges[i][1]
        # FIXME: potentailly incompatible C++ implementation of appending reference of same
        # object rather than apppending deepcopy of object
        spline_surface_patch: QuadraticSplineSurfacePatch = spline_surface.get_patch(patch_index)

        # Get patch domain boundaries
        patch_domain_boundaries: list[LineSegment]
        patch_domain_boundaries = spline_surface_patch.domain.parametrize_patch_boundaries()
        boundary_domain_curve_segments.append(patch_domain_boundaries[patch_edge_index])
        assert len(patch_domain_boundaries) == 3

        # Get patch boundaries
        patch_boundaries: list[RationalFunction]  # elements <4, 3>, list length 3
        patch_boundaries = spline_surface_patch.get_patch_boundaries()
        boundary_segments.append(patch_boundaries[patch_edge_index])
        assert len(patch_boundaries) == 3
        assert patch_boundaries[0]

        # Record patch index
        boundary_patch_indices.append(patch_index)

    return boundary_domain_curve_segments, boundary_segments, boundary_patch_indices


def compute_boundary_intersection_parameter(boundary_domain_curve_segment: Conic,
                                            intersection_point: PlanarPoint1d) -> float:
    """
    Computes boundary intersection parameter.
    """
    # The boundary domain curve is always a line
    P_coeffs: Matrix3x2f = boundary_domain_curve_segment.numerators
    x0: PlanarPoint1d = P_coeffs[0, :] - intersection_point
    d: PlanarPoint1d = P_coeffs[1, :]
    return -x0.dot(d) / d.dot(d)


def compute_spline_surface_boundary_intersections(spline_surface: QuadraticSplineSurface,
                                                  contour_domain_curve_segments: list[Conic],
                                                  contour_patch_indices: list[PatchIndex],
                                                  line_intersection_indices: list[tuple[int, int]],
                                                  patch_boundary_edges: list[tuple[int, int]],
                                                  boundary_domain_curve_segments: list[Conic],
                                                  ) -> tuple[list[list[IntersectionData]],
                                                             #   dict[int, list[IntersectionData]],
                                                             int]:
    """
    Computes spline surface boundary intersection.

    :return contour_intersections: [out]
    :return num_intersections: [out]
    """
    num_patches: int = spline_surface.num_patches
    num_interior_contours: int = len(contour_domain_curve_segments)

    # contour_intersections: dict[int, list[IntersectionData]] = defaultdict(list)
    contour_intersections: list[list[IntersectionData]] = [
        [] for _ in range(len(contour_domain_curve_segments) + len(boundary_domain_curve_segments))]
    num_intersections: int = 0

    # Build map from spine surface edges to boundary edge indices (or -1)
    patch_boundary_contour_map: list[list[int]] = [[-1, -1, -1] for _ in range(num_patches)]
    is_boundary_patch: list[bool] = [False] * num_patches
    logger.debug("Intersecting with %s patch boundaries",
                 len(patch_boundary_edges))
    for i, _ in enumerate(patch_boundary_edges):
        patch_index: int = patch_boundary_edges[i][0]
        patch_edge_index: int = patch_boundary_edges[i][1]
        logger.debug("Boundary %s: %s, %s", i, patch_index, patch_edge_index)
        patch_boundary_contour_map[patch_index][patch_edge_index] = i
        is_boundary_patch[patch_index] = True

    # Compute intersections for contours with intersections on the boundary
    for i, _ in enumerate(contour_domain_curve_segments):
        patch_index: int = contour_patch_indices[i]
        # FIXME (ASOC) This +1 is from an indexing inconsistency somewhere that should be fixed
        start_line_intersection_index: PatchIndex = (line_intersection_indices[i][0] + 1) % 3
        end_line_intersection_index: PatchIndex = (line_intersection_indices[i][1] + 1) % 3
        if is_boundary_patch[patch_index]:
            logger.debug("Contour %s in boundary patch %s", i, patch_index)
            logger.debug("Contour %s has line intersection indices %s and %s",
                         i,
                         start_line_intersection_index,
                         end_line_intersection_index)
            if (start_line_intersection_index >= 0) and (end_line_intersection_index >= 0):
                logger.debug("Contour %s has boundary contours %s and %s",
                             i,
                             patch_boundary_contour_map[patch_index][start_line_intersection_index],
                             patch_boundary_contour_map[patch_index][end_line_intersection_index])

        # Handle start point
        if start_line_intersection_index >= 0:
            start_boundary_contour: int = (
                patch_boundary_contour_map[patch_index][start_line_intersection_index])
            if start_boundary_contour >= 0:
                logger.debug("Start point intersection for interior contour %s in "
                             "patch %s with boundary contour %s on patch line %s",
                             i,
                             patch_index,
                             start_boundary_contour,
                             start_line_intersection_index)
                logger.debug("Interior domain contour: %s",
                             contour_domain_curve_segments[i])
                logger.debug("Boundary domain contour: %s",
                             boundary_domain_curve_segments[start_boundary_contour])

                # Build interior intersection data
                interior_intersection_data: IntersectionData
                interior_knot: float = contour_domain_curve_segments[i].domain.lower_bound
                interior_intersection_index: int = (
                    num_interior_contours + start_boundary_contour)
                interior_intersection_knot: float = compute_boundary_intersection_parameter(
                    boundary_domain_curve_segments[start_boundary_contour],
                    contour_domain_curve_segments[i](interior_knot))
                interior_id: int = num_intersections
                interior_is_base = True
                interior_intersection_data = IntersectionData(interior_knot,
                                                              interior_intersection_index,
                                                              interior_intersection_knot,
                                                              interior_id,
                                                              is_base=interior_is_base)
                contour_intersections[i].append(interior_intersection_data)

                # Build complementary boundary intersection data
                boundary_intersection_data: IntersectionData
                boundary_knot: float = interior_intersection_data.intersection_knot
                boundary_intersection_index: int = i
                boundary_intersection_knot: float = interior_intersection_data.knot
                boundary_id: int = num_intersections
                boundary_intersection_data = IntersectionData(boundary_knot,
                                                              boundary_intersection_index,
                                                              boundary_intersection_knot,
                                                              boundary_id)
                contour_intersections[num_interior_contours +
                                      start_boundary_contour].append(boundary_intersection_data)
                num_intersections += 1
        else:
            logger.debug("Contour %s start in patch %s does not intersect an edge",
                         i,
                         patch_index)

        # Handle endpoint
        if end_line_intersection_index >= 0:
            end_boundary_contour: int = patch_boundary_contour_map[patch_index
                                                                   ][end_line_intersection_index]
            if end_boundary_contour >= 0:
                logger.debug("End point intersection for interior contour %s in patch "
                             "%s with boundary contour %s on patch line %s",
                             i,
                             patch_index,
                             end_boundary_contour,
                             end_line_intersection_index)
                logger.debug("Interior domain contour: %s",
                             contour_domain_curve_segments[i])
                logger.debug("Boundary domain contour: %s",
                             boundary_domain_curve_segments[end_boundary_contour])

                # Build interior intersection data
                interior_intersection_data: IntersectionData
                interior_knot = contour_domain_curve_segments[i].domain.upper_bound
                interior_intersection_index: int = num_interior_contours + end_boundary_contour
                interior_intersection_knot = compute_boundary_intersection_parameter(
                    boundary_domain_curve_segments[end_boundary_contour],
                    contour_domain_curve_segments[i](interior_knot))
                interior_is_tip = True
                interior_id: int = num_intersections
                interior_intersection_data = IntersectionData(interior_knot,
                                                              interior_intersection_index,
                                                              interior_intersection_knot,
                                                              interior_id,
                                                              is_tip=interior_is_tip)
                contour_intersections[i].append(interior_intersection_data)

                # Build complementary boundary intersection data
                boundary_intersection_data: IntersectionData
                boundary_knot = interior_intersection_data.intersection_knot
                boundary_intersection_index = i
                boundary_intersection_knot = interior_intersection_data.knot
                boundary_id: int = num_intersections
                boundary_intersection_data = IntersectionData(boundary_knot,
                                                              boundary_intersection_index,
                                                              boundary_intersection_knot,
                                                              boundary_id)
                contour_intersections[num_interior_contours +
                                      end_boundary_contour].append(boundary_intersection_data)
                num_intersections += 1
        else:
            logger.debug("Contour %s end in patch %s does not intersect an edge",
                         i,
                         patch_index)

    logger.info("%s interior and boundary intersections found", num_intersections)

    return contour_intersections, num_intersections


def compute_spline_surface_contours_and_boundaries(spline_surface: QuadraticSplineSurface,
                                                   frame: Matrix3x3f,
                                                   patch_boundary_edges: list[tuple[int, int]]
                                                   ) -> tuple[list[Conic],
                                                              list[RationalFunction],
                                                              list[PatchIndex],
                                                              list[bool],
                                                              list[list[IntersectionData]],
                                                              int]:
    """
    Given a quadratic spline surface and a projection frame, compute the
    rational functions defining both the contour segments and the surface
    boundaries

    Both (quadratic) parametric domain contour functions and (higher order)
    surface contour functions are computed. The indices of the corresponding
    patches for each segment are also extracted and a flag for whether they are
    boundaries.

    An initial collection of intersections of the boundary and the interior
    contours are also computed, which exist for any planar projection.

    :param spline_surface: [in] quadratic spline surface
    :param frame: [in] 3x3 matrix defining the projection
    :param patch_boundary_edges: [in] edges of the patch triangle domains that are boundaries

    :return contour_domain_curve_segments: local parametric domain contour segments
    :return contour_segments: surface contour segments
    :return contour_patch_indices: spline surface patch indices for the contour segments
    :return contour_is_boundary: true iff the patch is a boundary contour
    :return intersection_data: data for the intersections of the contours and boundaries
    :return num_intersections: number of intersections
    """
    num_intersections: int = 0

    # Compute contours
    contour_domain_curve_segments: list[Conic]
    contour_segments: list[RationalFunction]  # <degree 4, dimension 3>
    contour_patch_indices: list[PatchIndex]
    line_intersection_indices: list[tuple[int, int]]

    # FIXME: below method looks good
    (contour_domain_curve_segments,
     contour_segments,
     contour_patch_indices,
     line_intersection_indices) = compute_spline_surface_contours(spline_surface,
                                                                  frame)
    contour_is_boundary: list[bool] = [False] * len(contour_segments)

    # Compute boundaries
    # FIXME: below method looks good
    boundary_domain_curve_segments: list[Conic]
    boundary_segments: list[RationalFunction]  # <degree 4, dimension 3>
    boundary_patch_indices: list[PatchIndex]
    (boundary_domain_curve_segments,
     boundary_segments,
     boundary_patch_indices) = compute_spline_surface_boundaries(spline_surface,
                                                                 patch_boundary_edges)
    boundary_is_boundary: list[bool] = [True] * len(boundary_segments)

    # Compute intersections of the contours and the boundaries
    #
    # FIXME: below method looks good
    #
    contour_intersections: list[list[IntersectionData]]
    contour_intersections, num_intersections = compute_spline_surface_boundary_intersections(
        spline_surface,
        contour_domain_curve_segments,
        contour_patch_indices,
        line_intersection_indices,
        patch_boundary_edges,
        boundary_domain_curve_segments)

    contour_domain_curve_segments.extend(boundary_domain_curve_segments)
    contour_segments.extend(boundary_segments)
    contour_patch_indices.extend(boundary_patch_indices)
    contour_is_boundary.extend(boundary_is_boundary)

    return (contour_domain_curve_segments,
            contour_segments,
            contour_patch_indices,
            contour_is_boundary,
            contour_intersections,
            num_intersections)


def pad_contours(contour_domain_curve_segments_ref: list[Conic],
                 contour_segments_ref: list[RationalFunction],  # <4, 3>
                 planar_contour_segments_ref: list[RationalFunction],  # <4, 2>
                 pad_amount: float = 0.0) -> None:
    """
    Pad contour domains by some given amount in the parametric domain

    :param contour_domain_curve_segments_ref: [out] local parametric domain contour segments
    :param contour_segments_ref: [out] surface contour segments
    :param planar_contour_segments_ref:  [out] projected planar contour segments
    :param pad_amount: [in] amount to pad the domain ends by
    """
    if pad_amount <= 0.0:
        return

    for i, _ in enumerate(contour_domain_curve_segments_ref):
        contour_domain_curve_segments_ref[i].domain.pad_lower_bound(pad_amount)
        contour_domain_curve_segments_ref[i].domain.pad_upper_bound(pad_amount)
        contour_segments_ref[i].domain.pad_lower_bound(pad_amount)
        contour_segments_ref[i].domain.pad_upper_bound(pad_amount)
        planar_contour_segments_ref[i].domain.pad_lower_bound(pad_amount)
        planar_contour_segments_ref[i].domain.pad_upper_bound(pad_amount)
