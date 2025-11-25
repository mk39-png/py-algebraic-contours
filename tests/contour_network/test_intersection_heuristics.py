""""
test_intersection_heuristics.py

Testing intersection_heuristics methods.
"""


import numpy as np

from pyalgcon.contour_network.intersection_heuristics import (
    compute_bezier_bounding_box, compute_bounding_box_hash_table,
    compute_homogeneous_bezier_points_over_interval)
from pyalgcon.core.common import (
    Matrix5x3f, PlanarPoint1d, compare_eigen_numpy_matrix,
    compare_list_list_varying_lengths, deserialize_eigen_matrix_csv_to_numpy,
    load_json)
from pyalgcon.core.rational_function import RationalFunction

from pyalgcon.utils.rational_function_testing_utils import \
    deserialize_rational_functions


def test_compute_homogeneous_bezier_points_over_interval_spot_control() -> None:
    """
    Tests compute_homogeneous_bezier_points_over_interval() method according to its usage in 
    compute_intersections()
    """
    filepath: str = "spot_control\\contour_network\\intersection_heuristics\\compute_homogeneous_bezier_points_over_interval\\"
    image_segments: list[RationalFunction] = deserialize_rational_functions(
        filepath+"image_segments.json")

    # Compute all rational bezier control points
    image_segments_bezier_control_points: list[Matrix5x3f] = []
    for image_segment in image_segments:
        assert (image_segment.degree, image_segment.dimension) == (4, 2)
        bezier_control_points: Matrix5x3f = compute_homogeneous_bezier_points_over_interval(
            image_segment,
            image_segment.domain.lower_bound,
            image_segment.domain.upper_bound)
        image_segments_bezier_control_points.append(bezier_control_points)

    compare_eigen_numpy_matrix(filepath+"image_segments_bezier_control_points.csv",
                               np.array(image_segments_bezier_control_points),
                               make_3d=True)


def test_compute_bezier_bounding_box_spot_control() -> None:
    """ 
    Tests compute_bezier_bounding_box() according to its usage in compute_intersections.
    """
    filepath: str = "spot_control\\contour_network\\intersection_heuristics\\compute_bezier_bounding_box\\"
    image_segments: list[RationalFunction] = deserialize_rational_functions(
        filepath+"image_segments.json")

    # Compute all bounding boxes
    image_segments_bounding_box: list[tuple[PlanarPoint1d, PlanarPoint1d]] = []
    for image_segment in image_segments:
        lower_left_point: PlanarPoint1d
        upper_right_point: PlanarPoint1d
        lower_left_point, upper_right_point = compute_bezier_bounding_box(image_segment)
        image_segments_bounding_box.append((lower_left_point, upper_right_point))

    compare_eigen_numpy_matrix(filepath+"image_segments_bounding_box.csv",
                               np.array(image_segments_bounding_box),
                               make_3d=True)


def test_compute_bounding_box_hash_table_spot_control() -> None:
    """
    Tests compute_bounding_box_hash_table() according to its usage in compute_intersections.
    """
    filepath: str = "spot_control\\contour_network\\intersection_heuristics\\compute_bounding_box_hash_table\\"
    image_segments_bounding_box_intermediate: np.ndarray = (
        deserialize_eigen_matrix_csv_to_numpy(filepath+"bounding_boxes.csv",
                                              make_3d=True))

    image_segments_bounding_boxes: list[tuple[PlanarPoint1d, PlanarPoint1d]] = []
    for i in range(image_segments_bounding_box_intermediate.shape[0]):
        image_segments_bounding_boxes.append((image_segments_bounding_box_intermediate[i][0],
                                              image_segments_bounding_box_intermediate[i][1]))

    # Hash by uv
    num_interval: int = 50
    # FIXME Make global: change both here and num_interval
    hash_table: dict[int, dict[int, list[int]]]
    reverse_hash_table: list[list[int]]
    hash_table, reverse_hash_table = compute_bounding_box_hash_table(image_segments_bounding_boxes)

    hash_table_control: list[list[list[int]]] = load_json(filepath+"hash_table_out.json")
    # Hash table is 50x50 with each element holding an array of int.
    for i in range(50):
        outer_table: list[list[int]] = hash_table_control[i]
        for j in range(50):
            inner_table: list[int] = hash_table_control[i][j]
            for k, data_control in enumerate(inner_table):
                assert hash_table[i][j][k] == data_control

    compare_list_list_varying_lengths(filepath+"reverse_hash_table.csv", reverse_hash_table)
