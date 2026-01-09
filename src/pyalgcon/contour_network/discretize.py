"""
discretize.py
Discretization methods.
"""

from pyalgcon.core.common import Vector1D
from pyalgcon.core.rational_function import (CurveDiscretizationParameters,
                                             RationalFunction)


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
