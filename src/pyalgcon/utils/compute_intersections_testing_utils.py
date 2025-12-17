"""
compute_intersections_testing_utils.py

Utility methods for testing compute_intersections.
"""


# **************
# Helper Methods
# **************


import pathlib

import numpy.testing as npt

from pyalgcon.contour_network.intersection_data import IntersectionData
from pyalgcon.contour_network.intersection_heuristics import IntersectionStats
from pyalgcon.core.common import float_equal, load_json


def compare_intersection_stats(filepath: pathlib.Path, intersection_stats_test: IntersectionStats) -> None:
    """
    Compares IntersectionStats from file
    """
    intersection_stats_control: IntersectionStats = deserialize_intersection_stats(filepath)
    assert intersection_stats_control == intersection_stats_test


def deserialize_intersection_stats(filepath: pathlib.Path) -> IntersectionStats:
    """
    Deserializes IntersectionStats json file
    """
    intersection_stats_intermediate: dict = load_json(filepath)
    return IntersectionStats(intersection_stats_intermediate.get("num_intersection_tests"),
                             intersection_stats_intermediate.get("num_bezier_nonoverlaps"),
                             intersection_stats_intermediate.get("bounding_box_call"),
                             intersection_stats_intermediate.get("intersection_call"))


def compare_list_list_intersection_data_from_file(
        filepath: pathlib.Path,
        contour_intersections_test: list[list[IntersectionData]]) -> None:
    """
    Reads in contour intersections file to compare to.
    """
    contour_intersections_control: list[list[IntersectionData]] = (
        deserialize_list_list_intersection_data(filepath)
    )
    __compare_list_list_intersection_data(contour_intersections_test,
                                          contour_intersections_control)


def __compare_list_list_intersection_data(contour_intersections_test: list[list[IntersectionData]],
                                          contour_intersections_control: list[list[IntersectionData]],
                                          ) -> None:
    """
    Compares list[list[IntersectionData]]
    """
    # FIXME: accept the control as the first parameter and the test as the second parameter
    assert len(contour_intersections_control) == len(contour_intersections_test), \
        f"Expected {len(contour_intersections_control)}, got {len(contour_intersections_test)}"
    num_outer_list: int = len(contour_intersections_control)

    for i in range(num_outer_list):
        inner_list_control: list[IntersectionData] = contour_intersections_control[i]
        inner_list_test: list[IntersectionData] = contour_intersections_test[i]
        assert len(inner_list_control) == len(inner_list_test)
        num_inner_list: int = len(inner_list_control)

        for j in range(num_inner_list):
            data_control: IntersectionData = inner_list_control[j]
            data_test: IntersectionData = inner_list_test[j]

            assert float_equal(data_control.knot, data_test.knot)
            assert data_control.intersection_index == data_test.intersection_index
            assert float_equal(data_control.intersection_knot, data_test.intersection_knot)
            assert data_control.id == data_test.id
            assert data_control.is_base == data_test.is_base
            assert data_control.is_tip == data_test.is_tip
            assert data_control.is_redundant == data_test.is_redundant


def deserialize_list_list_intersection_data(filepath: pathlib.Path) -> list[list[IntersectionData]]:
    """
    Deserializes contour_intersections.json into list[list[IntersectionData]]
    """

    intersection_data_intermediate: list[list[dict]] = load_json(filepath)
    intersection_data_final: list[list[IntersectionData]] = []

    for inner_list in intersection_data_intermediate:
        inner_list_intersection_data: list[IntersectionData] = []
        for data in inner_list:
            knot: float = data.get("knot")
            intersection_index: int = data.get("intersection_index")
            intersection_knot: float = data.get("intersection_knot")
            id_: int = data.get("id")
            is_base: bool = data.get("is_base")
            is_tip: bool = data.get("is_tip")
            is_redundant: bool = data.get("is_redundant")

            inner_list_intersection_data.append(IntersectionData(knot,
                                                                 intersection_index,
                                                                 intersection_knot,
                                                                 id_,
                                                                 is_base,
                                                                 is_tip,
                                                                 is_redundant))
        intersection_data_final.append(inner_list_intersection_data)

    return intersection_data_final
