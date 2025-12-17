"""
compute_ray_intersections.py

Methods to compute intersections of a ray and a quadratic surface.
"""


import logging

import numpy as np

from pyalgcon.contour_network.compute_ray_intersections_pencil_method import \
    compute_spline_surface_patch_ray_intersections_pencil_method
from pyalgcon.core.common import (MAX_PATCH_RAY_INTERSECTIONS, Matrix2x3f,
                                  PatchIndex, PlanarPoint1d, SpatialVector1d,
                                  float_equal)
from pyalgcon.quadratic_spline_surface.quadratic_spline_surface import \
    QuadraticSplineSurface

logger: logging.Logger = logging.getLogger(__name__)


def compute_spline_surface_ray_intersections(spline_surface: QuadraticSplineSurface,
                                             ray_mapping_coeffs: Matrix2x3f,
                                             ray_intersections_call: int,
                                             ray_bbox_call: int,
                                             ) -> tuple[list[PatchIndex],
                                                        list[PlanarPoint1d],
                                                        list[float],
                                                        int,
                                                        int]:
    """
    Given a quadratic spline surface and a ray, compute the intersections of the
    ray with each patch.

    :param spline_surface:     [in] quadratic spline surface
    :param ray_mapping_coeffs: [in] coefficients for the linear ray
    :param ray_int_call:    [in] number of ray intersections called
    :param ray_bbox_call:   [in] number of ray bounding box called

    :return patch_indices: indices of the patches with intersections
    :return surface_intersections: parameters of the intersections on the surface
    :return ray_intersections: parameters of the intersections on the ray
    :return ray_int_call: increment the number of ray intersection called
    :return ray_bbox_call: increment the number of ray bounding box called
    """
    assert ray_mapping_coeffs.shape == (2, 3)
    patch_indices: list[PatchIndex] = []
    surface_intersections: list[PlanarPoint1d] = []
    ray_intersections: list[float] = []

    logger.info("Computing intersections for spline surface with %s patches and ray %s",
                spline_surface.num_patches,
                ray_mapping_coeffs)

    # FIXME: check c++ translation of .block()
    ray_plane_point: PlanarPoint1d = ray_mapping_coeffs[0:1, 0:2].flatten()
    assert ray_plane_point.shape == (2, )
    hash_indices: tuple[int, int] = spline_surface.compute_hash_indices(ray_plane_point)
    # TODO: CHECK HASH TABLE AND SEE IF THEYRE THE SAME

    for i in spline_surface.hash_table[hash_indices[0]][hash_indices[1]]:
        num_intersections: int
        patch_surface_intersections: list[PlanarPoint1d]  # length MAX_PATCH_RAY_INTERSECTIONS
        patch_ray_intersections: list[float]  # length MAX_PATCH_RAY_INTERSECTIONS

        # FIXME: the below method not giving any patch_sruface_intersections
        # nor any ray_intersections... which is bad for us
        # But how do I even test this?
        # Is there an accurate was to deserialize QuadraticSplineSurface?
        (num_intersections,
         patch_surface_intersections,
         patch_ray_intersections,
         ray_intersections_call,
         ray_bbox_call
         ) = compute_spline_surface_patch_ray_intersections_pencil_method(
            spline_surface.get_patch(i),
            ray_mapping_coeffs,
            ray_intersections_call,
            ray_bbox_call)

        # Add patch intersections to surface intersections arrays
        if num_intersections > MAX_PATCH_RAY_INTERSECTIONS:
            # TODO: raise value error here?
            logger.error("More than four intersections found of a ray with a patch")
            raise ValueError("More than four intersections found of a ray with a patch")

        for j in range(num_intersections):
            patch_indices.append(i)
            surface_intersections.append(patch_surface_intersections[j])
            ray_intersections.append(patch_ray_intersections[j])
            logger.info("Patch ray intersection at t=%s found, out of %s",
                        patch_ray_intersections[j],
                        num_intersections)

    logger.info("%s surface ray intersections found", len(surface_intersections))
    logger.info("Spline surface intersection points: %s", surface_intersections)
    logger.info("Ray intersection points: %s", ray_intersections)

    return (patch_indices,
            surface_intersections,
            ray_intersections,
            ray_intersections_call,
            ray_bbox_call)


def partition_ray_intersections(ray_mapping_coeffs: Matrix2x3f,
                                comparison_point: SpatialVector1d,
                                ray_intersections: list[float]) -> tuple[list[float],
                                                                         list[float]]:
    """
    Given a ray with intersection points and a point on the ray, partition the points into
    those above and below the given point

    :param ray_mapping_coeffs: [in] coefficients for the linear ray
    :param comparison_point:   [in] point on the ray to compare other intersections against
    :param ray_intersections:  [in] parameters for intersection points on the ray

    :return ray_intersections_below: intersections below the comparison point
    :return ray_intersections_above: intersections above the comparison point
    """
    assert ray_mapping_coeffs.shape == (2, 3)
    assert comparison_point.shape == (3, )

    ray_intersections_above: list[float] = []
    ray_intersections_below: list[float] = []
    num_intersections: PatchIndex = len(ray_intersections)
    logger.info("Partitioning intersections %s on ray %s around point %s",
                ray_intersections,
                ray_mapping_coeffs,
                comparison_point)

    # Compute parameter for point
    point_0: SpatialVector1d = ray_mapping_coeffs[0, :]
    direction: SpatialVector1d = ray_mapping_coeffs[1, :]
    difference: SpatialVector1d = comparison_point - point_0
    assert point_0.shape == (3, )
    assert direction.shape == (3, )
    assert difference.shape == (3, )
    point_parameter: float = np.linalg.norm(difference) / np.linalg.norm(direction)
    logger.info("Point parameter: %s", point_parameter)

    # Partition intersections into points above and below the comparison point
    for i in range(num_intersections):
        ray_intersection: float = ray_intersections[i]
        if float_equal(ray_intersection, point_parameter, 1e-7):
            continue
        elif ray_intersection < point_parameter:
            ray_intersections_below.append(ray_intersection)
        else:
            ray_intersections_above.append(ray_intersection)
    logger.info("Intersections below: %s", ray_intersections_below)
    logger.info("Intersections above: %s", ray_intersections_above)

    return ray_intersections_below, ray_intersections_above
