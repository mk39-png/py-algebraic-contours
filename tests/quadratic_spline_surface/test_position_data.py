"""
File simply tests that  gradients can be found through constant and linear vertices.
So, dependent on the assertions made in methods like compute_least_squares_vertex_gradient()

"""

import numpy as np

from pyalgcon.core.common import *
from pyalgcon.quadratic_spline_surface.position_data import *
from pyalgcon.quadratic_spline_surface.position_data import (
    TriangleCornerData, TriangleMidpointData)

# def test_gradients_find_constant() -> None:
#     V = np.array([[1.0, 0.0, 0.0],
#                   [1.0, 0.0, 0.0],
#                   [1.0, 0.0, 0.0],
#                   [1.0, 0.0, 0.0]])

#     F = np.array(
#         [[0, 1, 2],
#          [0, 2, 3],
#          [0, 3, 1]])

#     vertex_index: int = 0
#     vertex_one_ring: list[int] = [1, 2, 3]
#     face_one_ring: list[int] = [0, 1, 2]

#     one_ring_uv_positions: np.ndarray = np.array(
#         [[1.0, 0.0],
#          [(-math.sqrt(3) / 2.0), 0.5],
#          [(-math.sqrt(3) / 2.0), -0.5]])

#     assert V.shape == (4, 3)
#     assert F.shape == (3, 3)
#     assert one_ring_uv_positions.shape == (3, 2)

#     todo("There are no assert statements in the ASOC code for this test")


def test_generate_corner_data_matrices() -> None:
    """
    Test to see if the method goes through ALL elements in the arrays 
    position_matrix, first_derivative_matrix, and second_derivative_matrix.
    """
    function_value = np.ones(shape=(3, ), dtype=np.float64)
    first_edge_derivative = np.ones(shape=(3, ), dtype=np.float64)
    second_edge_derivative = np.ones(shape=(3, ), dtype=np.float64)

    corner_data: list[list[TriangleCornerData]] = [[
        TriangleCornerData(function_value,
                           first_edge_derivative,
                           second_edge_derivative),
        TriangleCornerData(function_value,
                           first_edge_derivative,
                           second_edge_derivative),
        TriangleCornerData(function_value,
                           first_edge_derivative,
                           second_edge_derivative)], [
        TriangleCornerData(function_value,
                           first_edge_derivative,
                           second_edge_derivative),
        TriangleCornerData(function_value,
                           first_edge_derivative,
                           second_edge_derivative),
        TriangleCornerData(function_value,
                           first_edge_derivative,
                           second_edge_derivative)]]

    position_matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    first_derivative_matrix:  np.ndarray[tuple[int, int], np.dtype[np.float64]]
    second_derivative_matrix:  np.ndarray[tuple[int, int], np.dtype[np.float64]]
    position_matrix, first_derivative_matrix, second_derivative_matrix = generate_corner_data_matrices(
        corner_data)

    # print("man")
    np.testing.assert_equal(position_matrix, np.ones(shape=(6, 3)))
    np.testing.assert_equal(first_derivative_matrix, np.ones(shape=(6, 3)))
    np.testing.assert_equal(second_derivative_matrix, np.ones(shape=(6, 3)))


def test_generate_midpoint_data_matrices() -> None:
    function_value = np.ones(shape=(3, ), dtype=np.float64)
    first_edge_derivative = np.ones(shape=(3, ), dtype=np.float64)
    second_edge_derivative = np.ones(shape=(3, ), dtype=np.float64)
    normal_derivative = np.ones(shape=(3, ), dtype=np.float64)

    corner_data: list[list[TriangleCornerData]] = [
        [TriangleCornerData(function_value,
                            first_edge_derivative,
                            second_edge_derivative),
         TriangleCornerData(function_value,
                            first_edge_derivative,
                            second_edge_derivative),
         TriangleCornerData(function_value,
                            first_edge_derivative,
                            second_edge_derivative)],
        [TriangleCornerData(function_value,
                            first_edge_derivative,
                            second_edge_derivative),
         TriangleCornerData(function_value,
                            first_edge_derivative,
                            second_edge_derivative),
            TriangleCornerData(function_value,
                               first_edge_derivative,
                               second_edge_derivative)]
    ]

    midpoint_data: list[list[TriangleMidpointData]] = [
        [TriangleMidpointData(normal_derivative),
         TriangleMidpointData(normal_derivative),
         TriangleMidpointData(normal_derivative)],
        [TriangleMidpointData(normal_derivative),
         TriangleMidpointData(normal_derivative),
         TriangleMidpointData(normal_derivative)]
    ]

    position_matrix, tangent_derivative_matrix, normal_derivative_matrix = generate_midpoint_data_matrices(
        corner_data, midpoint_data)

    np.testing.assert_equal(position_matrix, np.full(shape=(6, 3),
                                                     fill_value=1.25,
                                                     dtype=np.float64))

    # TODO: is the tangent_derivative_matrix supposed to be full of 0s? I'm not too sure.
    np.testing.assert_equal(tangent_derivative_matrix, np.full(shape=(6, 3),
                                                               fill_value=0.0,
                                                               dtype=np.float64))

    # TODO: is normal_derivative_matrix supposed to be full of 1.0s? I'm not too sure.
    np.testing.assert_equal(normal_derivative_matrix, np.full(shape=(6, 3),
                                                              fill_value=1.0,
                                                              dtype=np.float64))
