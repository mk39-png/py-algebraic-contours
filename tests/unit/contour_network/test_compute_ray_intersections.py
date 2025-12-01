"""
Ray Intersection Tests
"""

import numpy as np
import numpy.testing as npt

from pyalgcon.contour_network.compute_ray_intersections import \
    compute_spline_surface_ray_intersections
from pyalgcon.contour_network.compute_ray_intersections_pencil_method import \
    compute_spline_surface_patch_ray_intersections_pencil_method
from pyalgcon.core.common import (Matrix2x2f, Matrix2x3f, Matrix3x2f,
                                  Matrix6x3f, PatchIndex, PlanarPoint1d,
                                  float_equal, todo)
from pyalgcon.core.convex_polygon import ConvexPolygon
from pyalgcon.quadratic_spline_surface.quadratic_spline_surface_patch import \
    QuadraticSplineSurfacePatch
from pyalgcon.quadratic_spline_surface.twelve_split_spline import \
    TwelveSplitSplineSurface


def test_compute_spline_surface_ray_intersections_spot_control(
        twelve_split_spline_transformed: TwelveSplitSplineSurface) -> None:
    """

    """
    spline_surface: TwelveSplitSplineSurface = twelve_split_spline_transformed

    patch_indices: list[PatchIndex]
    surface_intersections: list[PlanarPoint1d]
    ray_intersections: list[float]
    ray_int_call: int = 0
    ray_bbox_call: int = 0
    ray_mapping_coeffs: Matrix2x3f

    # for i in
    (patch_indices,
     surface_intersections,
     ray_intersections,
     ray_int_call,
     ray_bbox_call) = compute_spline_surface_ray_intersections(spline_surface,
                                                               ray_mapping_coeffs,
                                                               ray_int_call,
                                                               ray_bbox_call)

    todo("finish implementation")


def test_intersection_of_ray_and_plane_nonstandard_domain() -> None:
    """
    "The intersection of a line and a plane can be found",
          "[compute_ray_intersections]"
    From original C++ code.
    """

    surface_mapping_coeffs: Matrix6x3f
    ray_mapping_coeffs: Matrix2x2f
    surface_intersections: list[PlanarPoint1d]
    ray_intersections: list[float]

    # surface_mapping_coeffs = np.array([
    #     0.1, 0.2, 0,     # 1
    #     1, 0.1, 0.2,     # u
    #     0.3, 0.2, 0.6,   # v
    #     -0.2, 0.3, 2,    # uv
    #     -0.2, 0.1, 1.2,  # uu
    #     1, 0.2, 2       # vv
    # ])

    surface_mapping_coeffs = np.array([
        [-1.7644781234132338, 2.3739354387396845, 151.40962125112011],
        [0.064700996167022434, 0.060355200875465254, 0.11004004230059464],
        [0.096340254611671616, 0.20934622175013734, -0.037910458133162371],
        [-0.046780436914869238, 0.03653445519442261, 0.0067183808714837631],
        [-0.019222335822841597, -0.0027165803799707611, 0.0060945333950915663],
        [-0.01945302670205272, 0.012961148925535924, -0.0050044769478544589],
    ])
    assert surface_mapping_coeffs.shape == (6, 3)

    uv: Matrix3x2f = np.array([[0, 0],
                               [1, 0],
                               [0, 1]])
    assert uv.shape == (3, 2)

    normalized_domain: ConvexPolygon = ConvexPolygon.init_from_vertices(uv)
    spline_surface_patch: QuadraticSplineSurfacePatch = QuadraticSplineSurfacePatch(
        surface_mapping_coeffs,
        normalized_domain)
    ray_mapping_coeffs: Matrix2x3f = np.array(
        [[-1.69671806538393, 2.5609728762486372, 51.377018112355586], [0, 0, 200]])
    assert ray_mapping_coeffs.shape == (2, 3)

    num_intersections: int = 0
    ray_intersections_call: int = 0
    ray_bounding_box_call: int = 0
    (num_intersections,
     surface_intersections,
     ray_intersections,
     ray_intersections_call,
     ray_bounding_box_call) = compute_spline_surface_patch_ray_intersections_pencil_method(
        spline_surface_patch, ray_mapping_coeffs, ray_intersections_call, ray_bounding_box_call)

    assert len(surface_intersections) == 1
    assert len(ray_intersections) == 1
    assert npt.assert_allclose(surface_intersections[0], np.array([0.75, 0.6]))
    assert float_equal(ray_intersections[0], 0.5)
