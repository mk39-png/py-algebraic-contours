"""
discretize.py
Discretization methods.
"""

from pyalgcon.contour_network.compute_curve_frame import \
    compute_spline_surface_patch_curve_frame
from pyalgcon.core.common import (PatchIndex, SpatialVector1d,
                                  Vector1D, unimplemented)
from pyalgcon.core.conic import Conic
from pyalgcon.core.rational_function import (
    CurveDiscretizationParameters, RationalFunction)
from pyalgcon.quadratic_spline_surface.quadratic_spline_surface import \
    QuadraticSplineSurface


def discretize_curve_segments(degree: int,
                              dimension: int,
                              curve_segments: list[RationalFunction],  # <degree, dimension>
                              curve_disc_params: CurveDiscretizationParameters
                              ) -> tuple[list[Vector1D],
                                         list[list[int]]]:
    """
    Discretize the given curve segments as a polyline curve network

    :param degree: [in]
    :param dimension: [in] either 2 or 3
    :param contour_segments: [in] curve segments
    :param curve_disc_params: [in] parameters for the contour discretization

    :return points: points of the curve network.
    :return polylines: polyline indices of the curve network.
    """

    # lazy checking
    assert curve_segments[0].degree == degree
    assert curve_segments[0].dimension == dimension
    points: list[Vector1D] = []  # elements shape (dimension, )
    polylines: list[list[int]] = []

    for curve_segment in curve_segments:
        # Write curves
        num_samples: int = curve_disc_params.num_samples
        # TODO: rename points_k
        points_k: list[Vector1D] = curve_segment.sample_points(num_samples)

        # Build polyline for the given curve
        polyline: list[int] = []
        for l, _ in enumerate(points_k):
            polyline.append(len(points) + l)

        points.extend(points_k)
        polylines.append(polyline)

    return points, polylines


# def discretize_patch_boundaries() -> None:
#     """
#     TODO: go to QuadraticSplineSurface and make its function static... or just...
#     Move the function outside of the class to make life easier for us.
#     """
#     unimplemented("Move function to outside of class")


# def sample_contour_frames(spline_surface: QuadraticSplineSurface,
#                           contour_domain_curve_segment: list[Conic],
#                           contour_segments: list[RationalFunction],  # 4, 3
#                           contour_patch_indices: list[PatchIndex],
#                           curve_disc_params: CurveDiscretizationParameters
#                           ) -> tuple[list[SpatialVector1d],
#                                      list[SpatialVector1d],
#                                      list[SpatialVector1d],
#                                      list[SpatialVector1d]]:
#     """
#     Sample the Darboux frames of the contours on the surface.

#     :param spline_surface: [in] quadratic spline surface
#     :param contour_domain_curve_segments: [in] local parametric domain contour segments
#     :param contour_segments:      [in] surface contour segments
#     :param contour_patch_indices: [in] spline surface patch indices for the contour segments
#     :param curve_disc_params:     [in] parameters for the contour discretization

#     :return base_points: sampled points on the contours
#     :return tangents: tangents to the contours at the sampled points
#     :return normals: surface normals at the sampled points
#     :return tangent_normals: cross products of normals and tangents at the sampled points
#     """
#     base_points: list[SpatialVector1d] = []
#     tangents: list[SpatialVector1d] = []
#     normals: list[SpatialVector1d] = []
#     tangent_normals: list[SpatialVector1d] = []

#     # Sample a given number of frames from each contour segment
#     num_points: int = curve_disc_params.num_tangents_per_segment
#     for k, contour_segment in enumerate(contour_segments):
#         parameter_contour_segment: Conic = contour_domain_curve_segment[k]
#         patch_indices: PatchIndex = contour_patch_indices[k]

#         # Compute the frame functions for the given segment
#         contour_segment_tangent: RationalFunction
#         contour_segment_normal: RationalFunction
#         contour_segment_tangent_normal: RationalFunction
#         (contour_segment_tangent,
#          contour_segment_normal,
#          contour_segment_tangent_normal) = compute_spline_surface_patch_curve_frame(
#             spline_surface.get_patch(patch_indices),
#             parameter_contour_segment)
#         assert (contour_segment_tangent.degree,
#                 contour_segment_tangent.dimension) == (8, 3)
#         assert (contour_segment_normal.degree,
#                 contour_segment_normal.dimension) == (4, 3)
#         assert (contour_segment_tangent_normal.degree,
#                 contour_segment_tangent_normal.dimension) == (12, 3)

#         # Add the frame samples for the contour segment to the lists
#         segment_base_points: list[SpatialVector1d]
#         segment_tangents: list[SpatialVector1d]
#         segment_normals: list[SpatialVector1d]
#         segment_tangent_normals: list[SpatialVector1d]
#         segment_base_points = contour_segment.sample_points(num_points, )
#         segment_tangents = contour_segment_tangent.sample_points(num_points, )
#         segment_normals = contour_segment_normal.sample_points(num_points, )
#         segment_tangent_normals = contour_segment_tangent_normal.sample_points(num_points,)

#         base_points.extend(segment_base_points)
#         tangents.extend(segment_tangents)
#         normals.extend(segment_normals)
#         tangent_normals.extend(segment_tangent_normals)

#     return (base_points, tangents, normals, tangent_normals)
