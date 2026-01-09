"""
compute_ray_intersections_pencil_method.py

Methodds to assist with computiong cusps.
"""

import ctypes
import logging
import math
import sys

import numpy as np

from pyalgcon.contour_network.intersection_heuristics import is_in_bounding_box
from pyalgcon.core.common import (MAX_PATCH_RAY_INTERSECTIONS, Matrix2x2f,
                                  Matrix2x3f, Matrix6x3f, PlanarPoint1d,
                                  SpatialVector1d, Vector2f, Vector3f,
                                  Vector4f, Vector6f)
from pyalgcon.core.convex_polygon import ConvexPolygon
from pyalgcon.quadratic_spline_surface.quadratic_spline_surface_patch import \
    QuadraticSplineSurfacePatch

logger: logging.Logger = logging.getLogger(__name__)


def _solve_quadratic(q: Vector3f, threshold: float) -> tuple[int, Vector2f]:
    """
    Solves quadratic.
    Returns number of solutions and solution.
    """
    discr: float
    num_solution: int = 0
    solution: Vector2f = np.zeros((2, ), dtype=np.float64)
    if threshold <= abs(q[0]):
        discr = -0.4e1 * q[0] * q[2] + q[1] * q[1]
        if threshold * threshold <= discr:
            if 0.0e0 < q[1]:
                solution[0] = 0.2e1 * q[2] / (-q[1] - math.sqrt(discr))
                solution[1] = (-q[1] - math.sqrt(discr)) / q[0] / 0.2e1
            else:
                solution[0] = (-q[1] + math.sqrt(discr)) / q[0] / 0.2e1
                solution[1] = 0.2e1 * q[2] / (-q[1] + math.sqrt(discr))

            num_solution = 2
        elif 0.0e0 <= discr:
            solution[0] = -q[1] / q[0] / 0.2e1
            num_solution = 1
        else:
            num_solution = 0
    elif threshold <= abs(q[1]):
        solution[0] = -q[2] / q[1]
        num_solution = 1
    else:
        num_solution = 0

    return num_solution, solution


def _solve_linear_quadratic(a_in: Vector6f, b_in: Vector6f, threshold: float) -> tuple[int,
                                                                                       Matrix2x2f]:
    """
    uu vv uv 1 u v

    # FIXME: isn't there a numpy solution for this?

    :return: num_solution
    :return: solution
    """
    assert a_in.shape == (6, )
    assert b_in.shape == (6, )

    num_solution: int = 0
    solution: Matrix2x2f = np.zeros(shape=(2, 2), dtype=np.float64)
    a: Vector6f = np.zeros(shape=(6, ), dtype=np.float64)
    b: Vector6f = np.zeros(shape=(6, ), dtype=np.float64)
    ma: float = a_in[0]
    mb: float = b_in[0]

    # FIXME: optimize below with vectorization (AFTER DONE IMPLEMENTING EVERYTHING)
    for i in range(2, a_in.size + 1):
        ma = max(abs(a_in[i - 1]), ma)
        mb = max(abs(b_in[i - 1]), mb)
    if ma < threshold or mb < threshold:
        return num_solution, solution
    else:
        for i in range(1, a_in.size + 1):
            a[i - 1] = a_in[i - 1] / ma
            b[i - 1] = b_in[i - 1] / mb

    if abs(b[5]) <= abs(b[4]) and threshold <= abs(b[4]):
        quadratic_coefficients: Vector3f = np.array(
            [a[0] * b[5] * b[5] - a[2] * b[5] * b[4] + a[1] * b[4] * b[4],  # element 1
             (0.2e1 * a[0] * b[3] * b[5] - a[2] * b[3] * b[4] -  # element 2
              a[4] * b[5] * b[4] + a[5] * b[4] * b[4]),
             b[3] * b[3] * a[0] - a[4] * b[3] * b[4] + a[3] * b[4] * b[4]],  # element 3
            dtype=np.float64)
        v_solution: Vector2f
        num_solution, v_solution = _solve_quadratic(quadratic_coefficients, threshold)
        assert quadratic_coefficients.shape == (3, )
        assert v_solution.shape == (2, )

        if num_solution == 2:
            solution[0][0] = -(v_solution[0] * b[5] + b[3]) / b[4]
            solution[0][1] = v_solution[0]
            solution[1][0] = -(v_solution[1] * b[5] + b[3]) / b[4]
            solution[1][1] = v_solution[1]
        elif num_solution == 1:
            solution[0][0] = -(v_solution[0] * b[5] + b[3]) / b[4]
            solution[0][1] = v_solution[0]
    elif abs(b[4]) < abs(b[5]) and threshold <= abs(b[5]):
        quadratic_coefficients: Vector3f = np.array(
            [(a[0] * b[5] * b[5] - a[2] * b[5] * b[4] + a[1] * b[4] * b[4]),  # element 1
             (-a[2] * b[3] * b[5] + 0.2e1 * a[1] * b[3] * b[4] +  # element 2
              a[4] * b[5] * b[5] - a[5] * b[4] * b[5]),
             a[1] * b[3] * b[3] - a[5] * b[3] * b[5] + a[3] * b[5] * b[5]],  # element 3
            dtype=np.float64)
        u_solution: Vector2f
        num_solution, u_solution = _solve_quadratic(quadratic_coefficients, threshold)
        assert quadratic_coefficients.shape == (3, )
        assert u_solution.shape == (2, )

        if num_solution == 2:
            solution[0][0] = u_solution[0]
            solution[0][1] = -(u_solution[0] * b[4] + b[3]) / b[5]
            solution[1][0] = u_solution[1]
            solution[1][1] = -(u_solution[1] * b[4] + b[3]) / b[5]
        elif num_solution == 1:
            solution[0][0] = u_solution[0]
            solution[0][1] = -(u_solution[0] * b[4] + b[3]) / b[5]
    else:
        num_solution = 0

    return num_solution, solution


def pencil_first_part(coeff_F: Vector6f,
                      coeff_G: Vector6f) -> tuple[bool, int, list[PlanarPoint1d]]:
    """
    FIXME: this method is likeliest to fail
    FIXME: test with ASOC code first! As in, go through ASOC debugger to grab values for comparison
    FIXME: do this by grabbing each part to compare and whatnot.
    FIXME: just like I said.
    TODO: Check the outputs of this method with others.

    :return: intersection_points
    """
    assert coeff_F.shape == (6, )
    assert coeff_G.shape == (6, )
    intersection_points: list[PlanarPoint1d] = [np.zeros((2, ), dtype=np.float64)
                                                for _ in range(MAX_PATCH_RAY_INTERSECTIONS)]

    assert len(intersection_points) == MAX_PATCH_RAY_INTERSECTIONS

    num_intersections: int = 0
    coeff_threshold: float = 1e-10

    F_linear: bool = False
    G_linear: bool = False
    if (abs(coeff_F[0]) < coeff_threshold and abs(coeff_F[1]) < coeff_threshold and
            abs(coeff_F[3]) < coeff_threshold):
        F_linear = True
    if (abs(coeff_G[0]) < coeff_threshold and abs(coeff_G[1]) < coeff_threshold and
            abs(coeff_G[3]) < coeff_threshold):
        G_linear = True

    # Different cases
    if F_linear and G_linear:
        # Case both linear
        d: float = coeff_F[4] * coeff_G[5] - coeff_F[5] * coeff_G[4]
        if abs(d) > coeff_threshold:
            u: float = -(coeff_F[2] * coeff_G[5] - coeff_F[5] * coeff_G[2]) / d
            v: float = (coeff_F[2] * coeff_G[4] - coeff_F[4] * coeff_G[2]) / d

            if 0.0 <= u and 0.0 <= v and u + v <= 1.0:
                intersection_points[num_intersections] = np.array([u, v], dtype=np.float64)
                num_intersections += 1
                return True, num_intersections, intersection_points
            else:
                return False, num_intersections, intersection_points
        else:
            return False, num_intersections, intersection_points

    elif F_linear and not G_linear:
        # Case F linear, G quadratic
        coeff_F_solution: Vector6f = np.array([coeff_F[0], coeff_F[1], coeff_F[3],
                                               coeff_F[2], coeff_F[4], coeff_F[5]],
                                              dtype=np.float64)
        coeff_G_solution: Vector6f = np.array([coeff_G[0], coeff_G[1], coeff_G[3],
                                               coeff_G[2], coeff_G[4], coeff_G[5]],
                                              dtype=np.float64)
        num_solution: int
        solution: Matrix2x2f
        num_solution, solution = _solve_linear_quadratic(coeff_G_solution,
                                                         coeff_F_solution,
                                                         coeff_threshold)
        assert solution.shape == (2, 2)

        for i in range(num_solution):
            if (0.0 <= solution[i][0] and
                0.0 <= solution[i][1] and
                    solution[i][0] + solution[i][1] <= 1.0):
                intersection_points[num_intersections] = np.array([solution[i][0], solution[i][1]],
                                                                  dtype=np.float64)
                num_intersections += 1
        if num_intersections > 0:
            return True, num_intersections, intersection_points
        else:
            return False, num_intersections, intersection_points
    elif not F_linear and G_linear:
        # Case F quadratic, G linear
        coeff_F_solution: Vector6f = np.array([coeff_F[0], coeff_F[1], coeff_F[3],
                                               coeff_F[2], coeff_F[4], coeff_F[5]],
                                              dtype=np.float64)
        coeff_G_solution: Vector6f = np.array([coeff_G[0], coeff_G[1], coeff_G[3],
                                               coeff_G[2], coeff_G[4], coeff_G[5]],
                                              dtype=np.float64)
        num_solution: int
        solution: Matrix2x2f
        num_solution, solution = _solve_linear_quadratic(coeff_G_solution,
                                                         coeff_F_solution,
                                                         coeff_threshold)
        for i in range(num_solution):
            if (0.0 <= solution[i][0] and
                0.0 <= solution[i][1] and
                    solution[i][0] + solution[i][1] <= 1.0):
                intersection_points[num_intersections] = np.array([solution[i][0], solution[i][1]],
                                                                  dtype=np.float64)
                num_intersections += 1
        if num_intersections > 0:
            return True, num_intersections, intersection_points
        else:
            return False, num_intersections, intersection_points

    # else, F and G both quadratic

    #  input F = a uu + b vv + c + d uv + e u + f v
    #  input G = l uu + m vv + n + o uv + p u + q v

    #  code convention
    #  F = a uu + b vv + c + 2d uv + 2e u + 2f v
    #  G = l uu + m vv + n + 2o uv + 2p u + 2q v

    #  so the each last 3 inputs should be divided by 2
    intersection_flag: bool = False
    a: float = coeff_F[0]
    b: float = coeff_F[1]
    c: float = coeff_F[2]
    d: float = coeff_F[3] / 2.0
    e: float = coeff_F[4] / 2.0
    f: float = coeff_F[5] / 2.0
    l: float = coeff_G[0]
    m: float = coeff_G[1]
    n: float = coeff_G[2]
    o: float = coeff_G[3] / 2.0
    p: float = coeff_G[4] / 2.0
    q: float = coeff_G[5] / 2.0

    # Cubic equation
    a0: float = (l * m * n + 2.0 * o * p * q) - (l * q * q + m * p * p + n * o * o)
    a1: float = ((a * m * n + l * b * n + l * m * c +
                  2.0 * (d * p * q + o * e * q + o * p * f)) -
                 (a * q * q + b * p * p + c * o * o +
                  2.0 * (l * f * q + m * e * p + n * d * o)))
    a2: float = ((a * b * n + a * m * c + l * b * c +
                  2.0 * (o * e * f + d * e * q + d * p * f)) -
                 (l * f * f + m * e * e + n * d * d +
                  2.0 * (a * f * q + b * e * p + c * d * o)))
    a3: float = (a * b * c + 2.0 * d * e * f) - (a * f * f + b * e * e + c * d * d)

    num_cubic_real_roots: int = 0
    cubic_real_roots: Vector3f = np.zeros((3, ), dtype=np.float64)

    # FIXME: check behavior with C++ code below since dealing with imaginary numbers...
    # AS in, make a test case for this.
    # Check if a3 = 0
    if abs(a3) < coeff_threshold:
        # Quadratic equation
        quadratic: Vector3f = np.array([a2, a1, a0], dtype=np.float64)
        solution: Vector2f
        num_solution, solution = _solve_quadratic(quadratic, coeff_threshold)
        assert quadratic.shape == (3, )
        assert solution.shape == (2, )
        for i in range(num_solution):
            cubic_real_roots[num_cubic_real_roots] = solution[i]
            num_cubic_real_roots += 1
    else:
        # Solve cubic
        # FIXME: cubic_coeffs order and values looks good for the first iteration of
        #        pencil_first_part
        cubic_coeffs: Vector4f = np.array([a0, a1, a2, a3], dtype=np.float64)
        assert cubic_coeffs.shape == (4, )

        # XXX:  NumPy polynomial solver expects roots in ascending order, so lowest degree is first
        cubic_solver = np.polynomial.Polynomial(cubic_coeffs)
        # NOTE: indexing roots [::-1] since the roots must be in ascending order
        # cubic_roots: np.ndarray = cubic_solver.roots()[::-1]
        cubic_roots: Vector3f = np.roll(cubic_solver.roots(), -1)
        #
        assert cubic_roots.shape == (3, )

        # Check real roots
        imag_threshold: float = 1e-12
        for i in range(3):
            if abs(cubic_roots.imag[i]) < imag_threshold:
                cubic_real_roots[num_cubic_real_roots] = cubic_roots.real[i]
                num_cubic_real_roots += 1

    if num_cubic_real_roots == 0:
        # No real root
        return False, num_intersections, intersection_points

    x: float = 0.0
    determinant: float = math.inf
    for i in range(num_cubic_real_roots):
        A: float = a * cubic_real_roots[i] + l
        B: float = b * cubic_real_roots[i] + m
        D: float = d * cubic_real_roots[i] + o
        if determinant > (D * D - A * B):
            determinant = D * D - A * B
            x = cubic_real_roots[i]

    A = a * x + l
    B = b * x + m
    D = d * x + o
    C: float
    E: float
    F: float

    if determinant > 1e-10:
        C = c * x + n
        E = e * x + p
        F = f * x + q

        # FIXME: MISSING BIG ELSE STATEMENT WHAT
        if abs(A) < abs(B):
            A /= B
            C /= B
            D /= B
            E /= B
            F /= B
            B = 1.0
            sqrtA: float = math.sqrt(D * D - A)
            sqrtC: float = math.sqrt(F * F - C)
            la1: float = D + sqrtA
            la2: float = D - sqrtA
            lc1: float = F + sqrtC
            lc2: float = F - sqrtC

            if (abs((2.0 * E) - (la1 * lc1 + la2 * lc2)) <
                    abs((2.0 * E) - (la1 * lc2 + la2 * lc1))):
                tmp: float = lc1
                lc1 = lc2
                lc2 = tmp

            # quadratic equation: c0 uu + c1 u + c2 = 0  c0 c1 c2 g h
            for i in range(2):
                g: float = -la1 if (i == 0) else -la2
                h: float = -lc1 if (i == 0) else -lc2
                # c0: float = ctypes.c_double(a + (2.0 * d + b * g) * g)
                c0: float = (a + (2.0 * d + b * g) * g)

                c1: float = 2.0 * ((d + b * g) * h + e + f * g)
                c2: float = (b * h + 2.0 * f) * h + c

                if ((0.0 < c0) and (((0.0 < c1) and (0.0 < c2)) or
                                    ((0.0 > c2) and (0.0 > c0 + c1 + c2)))):
                    continue
                if ((0.0 > c0) and (((0.0 > c1) and (0.0 > c2)) or
                                    ((0.0 < c2) and (0.0 < c0 + c1 + c2)))):
                    continue

                if 0.00000000001 > abs(c0):
                    if 0.00000000001 < abs(c1):
                        u = -c2 / c1
                        v = g * u + h
                        w: float = 1.0 - (u + v)

                        if (0.0 <= u and 0.0 <= v and 0.0 <= w):
                            intersection_points[num_intersections] = np.array([u, v],
                                                                              dtype=np.float64)
                            num_intersections += 1
                            intersection_flag = True
                else:
                    discriminant: float = (c1 * c1) - 4.0 * (c0 * c2)
                    if 0.0 <= discriminant:
                        quadratic_coeffs: Vector3f = np.array([c2, c1, c0], dtype=np.float64)

                        # XXX:  NumPy polynomial solver expects coefficients in ascending order.
                        #  so lowest degree is first.
                        # But, flipping the order of the roots to be matching the C++ code's Eigen
                        #  implementation.
                        quadratic_solver = np.polynomial.Polynomial(quadratic_coeffs)
                        quadratic_roots: np.ndarray = quadratic_solver.roots()[::-1]
                        assert quadratic_roots.shape == (2, )

                        # with line i
                        for i in range(2):
                            u = quadratic_roots.real[i]
                            v = g * u + h
                            w = 1.0 - (u + v)

                            if 0.0 <= u and 0.0 <= v and 0.0 <= w:
                                intersection_points[num_intersections] = np.array([u, v],
                                                                                  dtype=np.float64)
                                num_intersections += 1
                                intersection_flag = True
        else:
            B /= A
            C /= A
            D /= A
            E /= A
            F /= A
            A = 1.0
            # sqrtB: float = math.sqrt(D * D - B)
            sqrtB: float = np.sqrt(D * D - B)

            # sqrtC = math.sqrt(E * E - C)
            sqrtC = np.sqrt(E * E - C)
            lb1: float = D + sqrtB
            lb2: float = D - sqrtB
            lc1 = E + sqrtC
            lc2 = E - sqrtC

            if (abs((2.0 * F) - (lb1 * lc1 + lb2 * lc2)) <
                    abs((2.0 * F) - (lb1 * lc2 + lb2 * lc1))):
                tmp = lc1
                lc1 = lc2
                lc2 = tmp

            # quadratic equation: c0 vv + c1 v + c2 = 0  c0 c1 c2 g h
            for i in range(2):
                g = -lb1 if (i == 0) else -lb2
                h = -lc1 if (i == 0) else -lc2
                c0 = b + (2.0 * d + a * g) * g
                c0 = float(c0)
                c1 = 2.0 * ((d + a * g) * h + f + e * g)
                c1 = float(c1)
                c2 = (a * h + 2.0 * e) * h + c
                c2 = float(c2)

                if ((0.0 < c0) and (((0.0 < c1) and (0.0 < c2)) or
                                    ((0.0 > c2) and (0.0 > c0 + c1 + c2)))):
                    continue
                if ((0.0 > c0) and (((0.0 > c1) and (0.0 > c2)) or
                                    ((0.0 < c2) and (0.0 < c0 + c1 + c2)))):
                    continue

                if (0.00000000001 > abs(c0)):
                    if (0.00000000001 < abs(c1)):
                        v: float = -c2 / c1
                        u: float = g * v + h
                        w: float = 1.0 - (u + v)
                        if (0.0 <= u and 0.0 <= v and 0.0 <= w):
                            intersection_points[num_intersections] = np.array([u, v],
                                                                              dtype=np.float64)
                            num_intersections += 1
                            intersection_flag = True
                else:
                    # FIXME: precision problem with floating points
                    # Call C pow()
                    # Autogen
                    # a = ctypes.c_double(c1)
                    # b = ctypes.c_double(c1)
                    # res = libc.pow(c1, c1)

                    # discriminant = math.pow(c1, 2) - (4.0 * (c0 * c2))  # FIXME: discriminant is wrong for 1046
                    # FIXME: discriminant is wrong for 1046
                    discriminant = (c1 * c1) - (4.0 * (c0 * c2))

                    # discriminant = math.pow(c1, 2) - (4.0 * (c0 * c2))  # FIXME: discriminant is wrong for 1046
                    # discriminant = ((ctypes.c_double(c1) * ctypes.c_double(c1)) -
                    # (ctypes.c_double4.0 * (c0 * c2)))  # FIXME: discriminant is wrong for 1046
                    # if

                    if 0.0 <= discriminant:
                        # if 1 <= discriminant:
                        quadratic_coeffs: Vector3f = np.array([c2, c1, c0], dtype=np.float64)
                        quadratic_solver = np.polynomial.Polynomial(quadratic_coeffs)

                        # FIXME: is the problem with the ordering of the roots?
                        # quadratic_roots: np.ndarray = quadratic_solver.roots()[::-1]
                        quadratic_roots: np.ndarray = quadratic_solver.roots()

                        # with line i
                        for i in range(2):
                            v = quadratic_roots.real[i]
                            u = g * v + h
                            w = 1.0 - (u + v)

                            if (0.0 <= u and 0.0 <= v and 0.0 <= w):
                                intersection_points[num_intersections] = np.array([u, v],
                                                                                  dtype=np.float64)
                                num_intersections += 1
                                intersection_flag = True

    else:
        # Ellipsoid with zero area
        return False, num_intersections, intersection_points
    # FIXME: jumped to asset at  i == 285.... cuz I didnt have the else statement after if absA < absB
    assert len(intersection_points) == MAX_PATCH_RAY_INTERSECTIONS
    return intersection_flag, num_intersections, intersection_points


def solve_quadratic_quadratic_equation_pencil_method(a: Vector6f,
                                                     b: Vector6f
                                                     ) -> tuple[int, list[PlanarPoint1d]]:
    """
    Solves quadratic equation pencil method.
    1 u v uv uu uv call
    """
    assert a.shape == (6, )
    assert b.shape == (6, )

    # Divide by max coefficient to get better precision
    max_: float = abs(a[0])
    for i in range(6):
        if max_ < abs(a[i]):
            max_ = abs(a[i])
        if max_ < abs(b[i]):
            max_ = abs(b[i])

    F: Vector6f = np.array([a[4] / max_, a[5] / max_, a[0] / max_,
                            a[3] / max_, a[1] / max_, a[2] / max_], dtype=np.float64)
    G: Vector6f = np.array([b[4] / max_, b[5] / max_, b[0] / max_,
                            b[3] / max_, b[1] / max_, b[2] / max_], dtype=np.float64)
    intersection_flag: bool
    num_intersections: int
    intersection_points: list[PlanarPoint1d]
    intersection_flag, num_intersections, intersection_points = pencil_first_part(F, G)
    assert len(intersection_points) == MAX_PATCH_RAY_INTERSECTIONS

    return num_intersections, intersection_points


def compute_spline_surface_patch_ray_intersections_pencil_method(
        spline_surface_patch: QuadraticSplineSurfacePatch,
        ray_mapping_coeffs: Matrix2x3f,
        ray_int_call: int,
        ray_bbox_call: int
) -> tuple[int, list[PlanarPoint1d], list[float], int, int]:
    """
    :param spline_surface:     [in] quadratic spline surface
    :param ray_mapping_coeffs: [in] coefficients for the linear ray
    :param ray_int_call:    [in] number of ray intersections called
    :param ray_bbox_call:   [in] number of ray bounding box called

    :return num_intersections: number of intersections
    :return surface_intersections: parameters of the intersections on the surface
    :return ray_intersections: parameters of the intersections on the ray
    :return ray_int_call: increment the number of ray intersection called
    :return ray_bbox_call: increment the number of ray bounding box called
    """
    num_intersections = 0
    surface_intersections: list[PlanarPoint1d] = []
    ray_intersections: list[float] = []

    domain: ConvexPolygon = spline_surface_patch.domain
    logger.debug("Domain: %s", domain.vertices)

    ray_origin: SpatialVector1d = ray_mapping_coeffs[0, :]  # row 0
    ray_plane_point: PlanarPoint1d = np.array([ray_origin[0], ray_origin[1]])
    assert ray_origin.shape == (3, )
    assert ray_plane_point.shape == (2, )
    logger.debug("Computing intersections for ray origin %s with planar projection %s",
                 ray_origin,
                 ray_plane_point)

    # Check if the bounding box of the projected patch boundaries contains the ray FIXME:
    min_point: SpatialVector1d = spline_surface_patch.get_bounding_box_min_point()
    max_point: SpatialVector1d = spline_surface_patch.get_bounding_box_max_point()
    assert min_point.shape == (3, )
    assert max_point.shape == (3, )
    lower_left_point: PlanarPoint1d = np.array([min_point[0], min_point[1]])
    upper_right_point: PlanarPoint1d = np.array([max_point[0], max_point[1]])
    assert lower_left_point.shape == (2, )
    assert upper_right_point.shape == (2, )
    ray_bbox_call += 1
    if not is_in_bounding_box(ray_plane_point, lower_left_point, upper_right_point):
        logger.info("Skipping intersection test for patch %s and ray %s with "
                    "bounding box (%s, %s)",
                    spline_surface_patch,
                    ray_mapping_coeffs,
                    lower_left_point,
                    upper_right_point)
        return (num_intersections,
                surface_intersections,
                ray_intersections,
                ray_int_call,
                ray_bbox_call)

    ray_int_call += 1

    logger.info("Computing intersections for patch %s and ray %s",
                spline_surface_patch,
                ray_mapping_coeffs)

    # Normalize the spline surface patch to have domain triangle u + v <= 1 in [0, 1]^2
    normalized_surface_mapping_coeffs: Matrix6x3f = (
        spline_surface_patch.get_normalized_surface_mapping())
    logger.info("Coefficients for the surface mapping with normalized domain: %s",
                normalized_surface_mapping_coeffs)
    assert normalized_surface_mapping_coeffs.shape == (6, 3)

    #  1 u v uv u2 v2
    #  to
    #   a uu + b vv + c + d uv + e u + f v
    coeff_F: Vector6f = np.array([normalized_surface_mapping_coeffs[4, 0],
                                  normalized_surface_mapping_coeffs[5, 0],
                                  normalized_surface_mapping_coeffs[0, 0] - ray_origin[0],
                                  normalized_surface_mapping_coeffs[3, 0],
                                  normalized_surface_mapping_coeffs[1, 0],
                                  normalized_surface_mapping_coeffs[2, 0]])
    coeff_G: Vector6f = np.array([normalized_surface_mapping_coeffs[4, 1],
                                  normalized_surface_mapping_coeffs[5, 1],
                                  normalized_surface_mapping_coeffs[0, 1] - ray_origin[1],
                                  normalized_surface_mapping_coeffs[3, 1],
                                  normalized_surface_mapping_coeffs[1, 1],
                                  normalized_surface_mapping_coeffs[2, 1]])

    # Renormalize coeffs
    max_c: float = abs(coeff_F[0])
    for i in range(6):
        if max_c < abs(coeff_F[i]):
            max_c = abs(coeff_F[i])
        if max_c < abs(coeff_G[i]):
            max_c = abs(coeff_G[i])
    coeff_F = coeff_F / max_c
    coeff_G = coeff_G / max_c

    # Get all intersections of the quadratic normalized surface and the ray
    num_intersections_all: int
    # length MAX_PATCH_RAY_INTERSECTIONS
    normalized_surface_intersections_all: list[PlanarPoint1d] = []
    normalized_surface_intersections: list[PlanarPoint1d] = []  # length MAX_PATCH_RAY_INTERSECTIONS
    normalized_ray_intersections: list[float] = []  # length MAX_PATCH_RAY_INTERSECTIONS
    _, num_intersections_all, normalized_surface_intersections_all = pencil_first_part(
        coeff_F,
        coeff_G)
    if num_intersections_all > MAX_PATCH_RAY_INTERSECTIONS:
        logger.error("More than the maximum possible number of patch ray intersections found")
    assert num_intersections_all <= MAX_PATCH_RAY_INTERSECTIONS
    logger.debug("%s intersections found before pruning", num_intersections_all)

    # Get intersections that are in the domain
    num_intersections_normalized: int = 0
    for i in range(num_intersections_all):
        t: float = ((normalized_surface_mapping_coeffs[0, 2] +
                     normalized_surface_mapping_coeffs[1, 2] *
                     normalized_surface_intersections_all[i][0] +
                     normalized_surface_mapping_coeffs[2, 2] *
                     normalized_surface_intersections_all[i][1] +
                     normalized_surface_mapping_coeffs[3, 2] *
                     normalized_surface_intersections_all[i][0] *
                     normalized_surface_intersections_all[i][1] +
                     normalized_surface_mapping_coeffs[4, 2] *
                     normalized_surface_intersections_all[i][0] *
                     normalized_surface_intersections_all[i][0] +
                     normalized_surface_mapping_coeffs[5, 2] *
                     normalized_surface_intersections_all[i][1] *
                     normalized_surface_intersections_all[i][1] -
                     ray_origin[2]) /
                    ray_mapping_coeffs[1, 2])
        if t > 0 and t <= 1:
            normalized_ray_intersections.append(t)
            normalized_surface_intersections.append(normalized_surface_intersections_all[i])
            num_intersections_normalized += 1

    # Invert the normalization and prune intersections outside of the triangle domain
    for i in range(num_intersections_normalized):
        normalized_domain_point: PlanarPoint1d = normalized_surface_intersections[i]
        surface_intersection: PlanarPoint1d = (
            spline_surface_patch.denormalize_domain_point(normalized_domain_point))
        assert normalized_domain_point.shape == (2, )
        assert surface_intersection.shape == (2, )

        if domain.contains(surface_intersection):
            surface_intersections.append(surface_intersection)
            ray_intersections.append(normalized_ray_intersections[i])
            num_intersections += 1
        logger.info("Normalized domain point: %s", normalized_domain_point)
        logger.info("Domain point: %s", surface_intersection)

    # Log intersections
    if num_intersections >= 0:
        logger.info("Intersections for patch %s and ray %s",
                    spline_surface_patch,
                    ray_mapping_coeffs)
        logger.info("Coefficients for the surface mapping with normalized domain: %s",
                    normalized_surface_mapping_coeffs)

    return (num_intersections,
            surface_intersections,
            ray_intersections,
            ray_int_call,
            ray_bbox_call)
