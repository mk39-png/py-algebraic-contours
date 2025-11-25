
import logging
import math

import numpy as np

from pyalgcon.core.common import *
from pyalgcon.core.conic import Conic

logger: logging.Logger = logging.getLogger(__name__)


def generate_circle(radius: float) -> Conic:
    """
    Generate circle of given radius

    :param radius: radius of the circle
    :return: parametrized circle (missing point at bottom)
    """
    unimplemented()


def generate_torus_point(major_radius: float,
                         minor_radius: float,
                         i: int,
                         j: int,
                         resolution: int,
                         angle_offset: float) -> SpatialVector:
    """
    Returns np array of shape (1, 3)
    """
    theta: float = generate_angle(i, resolution, angle_offset)
    phi: float = generate_angle(j, resolution, angle_offset)

    return np.array([
        [(major_radius + minor_radius * math.cos(theta)) * math.cos(phi)],
        [(major_radius + minor_radius * math.cos(theta)) * math.sin(phi)],
        [minor_radius * math.sin(theta)]
    ])


def generate_angle(i: float, resolution: int, angle_offset: float) -> float:
    return angle_offset + 2 * math.pi * i / resolution


def generate_angle_derivative(resolution: int) -> float:
    """
    Used in generate_position_data.py
    """
    unimplemented()


def generate_elliptic_contour_quadratic_surface():
    """
    Generate a quadratic surface with an ellipse as the parametric contour.
    :return: surface_mapping_coeffs: Coefficients for the quadratic surface
    :return: normal_mapping_coeffs: Coefficients for the quadratic surface normal
    """
    # surface_mapping_coeffs: Matrix6x3r
    # normal_mapping_coeffs: Matrix6x3r
    # return surface_mapping_coeffs, normal_mapping_coeffs
    unimplemented()


# ***************
# VF construction
# ***************

def generate_equilateral_triangle_VF(length: float = 1) -> tuple[np.ndarray, np.ndarray]:
    # V: np.ndarray
    # F: np.ndarray
    # return V, F
    unimplemented()


def generate_right_triangle_VF(width: float = 1.0,
                               height: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    # V: np.ndarray
    # F: np.ndarray
    # return V, F
    unimplemented()


def generate_rectangle_VF(width: float = 1.0,
                          height: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    # V: np.ndarray
    # F: np.ndarray
    # return V, F
    unimplemented()


def generate_square_VF(length: float = 1) -> tuple[np.ndarray, np.ndarray]:
    # V: np.ndarray
    # F: np.ndarray
    # return V, F
    unimplemented()


def generate_tetrahedron_VF() -> tuple[np.ndarray, np.ndarray]:
    # TODO: how to include typing in np.ndarray? like, int64 and whatnot?
    # np.ndarray[np.dtype[np.float64]]?
    """
    Generate simple tetrahedron mesh.

    :return: tuple of (tetrahedron vertices (V), tetrahedron faces (F))
    :rtype: tuple[np.ndarray[dtype=np.float64], np.ndarray[dtype=np.int64]]
    """

    V = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=float)

    F = np.array([
        [0, 2, 1],
        [0, 1, 3],
        [0, 3, 2],
        [1, 2, 3]
    ], dtype=int)

    return V, F


def generate_minimal_torus_VF(
        major_radius: float = 3.0, minor_radius: float = 1.0) -> tuple[
        np.ndarray, np.ndarray]:
    """
    Generate simple torus mesh.

    :param major_radius:
    :type major_radius: float
    :param minor_radius:
    :type minor_radius: float

    :return: tuple of ( V: Torus vertices, F: Torus facets )
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    V: np.ndarray = np.ndarray(shape=(9, 3))
    F: np.ndarray = np.array([[0, 1, 3],
                              [1, 4, 3],
                              [1, 2, 4],
                              [2, 5, 4],
                              [2, 0, 5],
                              [0, 3, 5],
                              [3, 4, 6],
                              [4, 7, 6],
                              [4, 5, 7],
                              [5, 8, 7],
                              [5, 3, 8],
                              [3, 6, 8],
                              [6, 7, 0],
                              [7, 1, 0],
                              [7, 8, 1],
                              [8, 2, 1],
                              [8, 6, 2],
                              [6, 0, 2]])
    resolution: int = 3

    # NOTE: need to flatten below for proper broadcasting
    # NOTE: generate_torus_point returns SpatialVector (which is shape (1, 3))
    V[0, :] = generate_torus_point(
        major_radius, minor_radius, 0, 0, resolution, 0.1).flatten()
    V[1, :] = generate_torus_point(
        major_radius, minor_radius, 0, 1, resolution, 0.1).flatten()
    V[2, :] = generate_torus_point(
        major_radius, minor_radius, 0, 2, resolution, 0.1).flatten()
    V[3, :] = generate_torus_point(
        major_radius, minor_radius, 1, 0, resolution, 0.1).flatten()
    V[4, :] = generate_torus_point(
        major_radius, minor_radius, 1, 1, resolution, 0.1).flatten()
    V[5, :] = generate_torus_point(
        major_radius, minor_radius, 1, 2, resolution, 0.1).flatten()
    V[6, :] = generate_torus_point(
        major_radius, minor_radius, 2, 0, resolution, 0.1).flatten()
    V[7, :] = generate_torus_point(
        major_radius, minor_radius, 2, 1, resolution, 0.1).flatten()
    V[8, :] = generate_torus_point(
        major_radius, minor_radius, 2, 2, resolution, 0.1).flatten()

    return V, F


# ********************
# Polygon construction
# ********************

def generate_rectangle():
    unimplemented()


def generate_square():
    unimplemented()


# ***********************
# Point grid construction
# ***********************

def generate_plane_grid(resolution: int,
                        u_slope: float,
                        v_slope: float) -> list[list[SpatialVector]]:
    """ 
    NOTE: In ASOC code, there is another function of the same name with different parameters.
    But this version of the function is the one that is used.

     Generate a grid of uniformly spaced points in the xy plane.
    param[in] resolution: number of points in the x and y directions
    param[in] delta: spacing between points

    TODO: change the below docstring
    param[in] x0: x coordinate of lower left corner of grid
    param[in] y0: y coordinate of lower left corner of grid
    param[out] point_grid: output point grid
    """
    center: float = resolution / 2.0

    point_grid: list[list[SpatialVector]] = []
    for i in range(resolution):
        point_grid.append([])
        for j in (range(resolution)):
            u: float = i - center
            v: float = j - center
            point_grid[i].append(np.array([
                [u],
                [v],
                [u_slope * u + v_slope * v],
            ]))
            assert point_grid[i][j].shape == (1, 3)

    # redundant checks
    assert len(point_grid) == resolution
    assert len(point_grid[0]) == resolution

    return point_grid


def generate_torus_grid(resolution: int,
                        major_radius: float,
                        minor_radius: float) -> list[list[SpatialVector]]:
    """
    Generate a grid of points on a torus. The spacing is uniform in the usual
    angular parametrization of the torus.
    param[in] resolution: number of points in the x and y directions
    param[in] major_radius: radius of the major circle of the torus
    param[in] minor_radius: radius of the minor circle of the torus
    param[out] point_grid: output point grid
    """
    angle_offset: float = 0.1

    point_grid: list[list[SpatialVector]] = []
    for i in range(resolution):
        point_grid.append([])
        for j in (range(resolution)):
            point_grid[i].append(
                generate_torus_point(
                    major_radius,
                    minor_radius,
                    i,
                    j,
                    resolution,
                    angle_offset
                ))
            assert point_grid[i][j].shape == (1, 3)

    # redundant checks
    assert len(point_grid) == resolution
    assert len(point_grid[0]) == resolution

    return point_grid


def generate_quadratic_grid(layout_grid: list[list[PlanarPoint]],
                            u_curvature: float,
                            v_curvature: float,
                            uv_curvature: float) -> list[list[SpatialVector]]:
    """
    Generate a grid of points on a quadratic surface. 

    param[in] resolution: number of points in the x and y directions
    param[in] u_curvature: curvature in the u direction
    param[in] v_curvature: curvature in the v direction
    param[out] point_grid: output point grid
    """
    point_grid: list[list[SpatialVector]] = []
    for i, _ in enumerate(layout_grid):
        point_grid.append([])
        for j, _ in enumerate(layout_grid[i]):
            u: float = layout_grid[i][j].flatten()[0]
            v: float = layout_grid[i][j].flatten()[1]

            # Appending (1, 3) SpatialVector shape
            point_grid[i].append(np.array([
                [u],
                [v],
                [0.5 * u_curvature * u * u + 0.5 * v_curvature * v * v + uv_curvature * u * v]
            ]))

            assert point_grid[i][j].shape == (1, 3)

    assert len(point_grid) == len(layout_grid)
    assert len(point_grid[0]) == len(layout_grid[0])

    return point_grid


def generate_sinusoidal_torus_grid():
    unimplemented()


def generate_bumpy_torus_grid():
    unimplemented()


def generate_ellpise_contour_quadratic_grid():
    unimplemented()


def generate_hyperbola_contour_quadratic_grid():
    unimplemented()


def generate_perturbed_quadratic_grid():
    unimplemented()


# ***********************
# General utility methods
# ***********************
def generate_mesh_from_grid(point_grid: list[list[SpatialVector]],
                            closed_surface: bool) -> tuple[np.ndarray,
                                                           np.ndarray,
                                                           list[list[float]]]:
    """
    Generate a VF mesh for a point grid for visualization.
    param[out] V: vertices of the output mesh (same as the point grid)
    param[out] F: triangulation of the point grid
    param[out] l: uv parametrizatioin lengths for the grid
    """
    n: int = len(point_grid)
    assert n > 0
    assert n == len(point_grid[0])
    V: np.ndarray = np.ndarray(shape=(n * n, len(point_grid[0][0])))

    # Flatten vertices in the point grid to a standard vector V of vertices
    for i in range(n):
        for j in range(n):
            V[flatten(i, j, n), :] = point_grid[i][j].flatten()

    # Use periodic boundary triangulation if closed surface
    N: int
    if (closed_surface):
        N = n
    else:
        N = n - 1

    # Create triangulation F for the grid
    F: np.ndarray = np.ndarray(shape=(2 * N * N, 3))
    PLACEHOLDER_FLOAT = -1

    # l: list[list[float]] = [[0.0 for _ in range(3)] for _ in range(N)]
    l: list[list[float]] = [[] for _ in range(2 * N * N)]

    for i in range(N):
        for j in range(N):
            # Create first face
            F[2*flatten(i, j, N), 0] = flatten(i, j, n)
            F[2*flatten(i, j, N), 1] = flatten((i + 1) % n, j, n)
            F[2*flatten(i, j, N), 2] = flatten(i, (j + 1) % n, n)
            l[2*flatten(i, j, N)].extend([PLACEHOLDER_FLOAT,
                                          PLACEHOLDER_FLOAT,
                                          PLACEHOLDER_FLOAT])
            l[2*flatten(i, j, N)][0] = math.sqrt(2)
            l[2*flatten(i, j, N)][1] = 1.0
            l[2*flatten(i, j, N)][2] = 1.0

            # Create second face
            F[2*flatten(i, j, N) + 1, 0] = flatten((i + 1) % n, j, n)
            F[2*flatten(i, j, N) + 1, 1] = flatten((i + 1) % n, (j + 1) % n, n)
            F[2*flatten(i, j, N) + 1, 2] = flatten(i, (j + 1) % n, n)
            l[2*flatten(i, j, N) + 1].extend([PLACEHOLDER_FLOAT,
                                              PLACEHOLDER_FLOAT,
                                              PLACEHOLDER_FLOAT])
            l[2*flatten(i, j, N) + 1][0] = 1.0
            l[2*flatten(i, j, N) + 1][1] = math.sqrt(2)
            l[2*flatten(i, j, N) + 1][2] = 1.0

            assert len(l[2*flatten(i, j, N)]) == 3
            assert len(l[2*flatten(i, j, N) + 1]) == 3

    logger.debug("Faces:\n%s", F)

    assert len(l) == 2 * N * N

    return V, F, l


def generate_global_layout_grid(resolution: int) -> list[list[PlanarPoint]]:
    """
    Used in quadratic_spline_surface and optimize_spline_surface test cases.
    """
    assert resolution != 0

    center: float = resolution / 2.0
    layout_grid: list[list[PlanarPoint]] = []
    # TODO: have planarpoint check that shape is (1, 2)

    for i in range(resolution):
        layout_grid.append([])
        for j in range(resolution):
            layout_grid[i].append(np.array([[i - center], [j - center]]))

    return layout_grid


def flatten(i: int, j: int, n: int) -> int:
    """
    Only used in generate_mesh_from_grid()
    """
    return j * n + i
