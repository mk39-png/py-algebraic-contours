"""
Test compute self intersections
"""

import numpy as np

from pyalgcon.contour_network.compute_intersections import \
    split_planar_curves_no_self_intersection
from pyalgcon.core.common import Matrix5x2f, Vector5f
from pyalgcon.core.interval import Interval
from pyalgcon.core.rational_function import RationalFunction


def test_self_intersecting_curve_circle() -> None:
    """
    From original C++ code
    """
    split_points: list[list[float]]
    P_coeffs: Matrix5x2f
    Q_coeffs: Vector5f

    P_coeffs = np.array([[1, 0],
                         [0, 1],
                         [-1, 0],
                         [0, 0],
                         [0, 0]])
    Q_coeffs = np.array([1, 0, 1, 0, 0])
    domain: Interval = Interval(-2, 2)

    planar_segment = RationalFunction(4, 2, P_coeffs, Q_coeffs, domain)
    planar_curves: list[RationalFunction] = [planar_segment]
    split_points = split_planar_curves_no_self_intersection(planar_curves)
    assert len(split_points) == 1
    assert len(split_points[0]) == 1


def test_self_intersecting_curve_spot_bug() -> None:
    """
    From original C++ code
    """
    split_points: list[list[float]]
    P_coeffs: Matrix5x2f
    Q_coeffs: Vector5f

    P_coeffs = np.array([[-0.73120161816068385, -2.25202965564466018],
                         [-0.40059603916677894,  0.12008492684645028],
                         [-1.39905702192626835,  4.47876303596620318],
                         [-4.50593422038637392, -5.30152851610740328],
                         [-0.57491127825807053, -3.18194386122352757]])
    domain: Interval = Interval(-0.09280896795608366, -0.07905005296651063)
    Q_coeffs = np.array([1, 0, -2, 0, 1])

    # Observed to split at 32272 knots due to potential self intersections

    planar_segment = RationalFunction(4, 2, P_coeffs, Q_coeffs, domain)
    planar_curves: list[RationalFunction] = [planar_segment]
    split_points = split_planar_curves_no_self_intersection(planar_curves)
    assert len(split_points) == 1
    # assert len(split_points[0]) == -1
