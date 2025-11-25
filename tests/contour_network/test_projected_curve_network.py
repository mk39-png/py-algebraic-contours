"""
Projected Curve Network testing
"""

import numpy as np

from pyalgcon.contour_network.intersection_data import \
    IntersectionData
from pyalgcon.contour_network.projected_curve_network import \
    ProjectedCurveNetwork
from pyalgcon.core.abstract_curve_network import \
    AbstractCurveNetwork
from pyalgcon.core.common import (
    NodeIndex, SegmentIndex, compare_eigen_numpy_matrix,
    compare_list_list_varying_lengths, deserialize_eigen_matrix_csv_to_numpy,
    deserialize_list_list_varying_lengths,
    deserialize_list_list_varying_lengths_float)
from pyalgcon.core.conic import Conic
from pyalgcon.core.rational_function import RationalFunction

from pyalgcon.utils.compute_intersections_testing_utils import (
    compare_list_list_intersection_data,
    deserialize_list_list_intersection_data)
from pyalgcon.utils.conic_testing_utils import deserialize_conics
from pyalgcon.utils.projected_curve_networks_utils import (
    NodeGeometry, SegmentGeometry, _deserialize_list_node_geometry,
    _deserialize_list_segment_geometry,
    build_projected_curve_network_without_intersections,
    compare_list_node_geometry, compare_list_segment_geometry,
    connect_segment_intersections, deserialize_segment_labels,
    remove_redundant_intersections, split_segments_at_cusps,
    split_segments_at_intersections)
from pyalgcon.utils.rational_function_testing_utils import \
    deserialize_rational_functions

# ************************
# Main Test Cases
# ************************

# Name of folder of mesh to test
ROOTNAME: str = "spot_control"


def test_init_chain_start_nodes() -> None:
    """
    This essentially tests all of the methods in the ProjectedCurveNetwork() constructor.
    Which in turn is used to test init_chain_start_nodes().
    """
    filepath: str = (
        f"{ROOTNAME}\\contour_network\\projected_curve_network\\init_projected_curve_network\\")

    # Deserialize arguments (and executes init_chain_start_nodes)
    projected_curve_network_test = ProjectedCurveNetwork(
        deserialize_conics(filepath+"parameter_segments.json"),
        deserialize_rational_functions(filepath+"spatial_segments.json"),
        deserialize_rational_functions(filepath+"planar_segments.json"),
        deserialize_segment_labels(filepath+"segment_labels.json"),
        deserialize_list_list_varying_lengths(filepath+"chains.csv"),
        deserialize_eigen_matrix_csv_to_numpy(filepath+"chain_labels.csv").tolist(),
        deserialize_list_list_varying_lengths_float(filepath+"interior_cusps.csv"),
        deserialize_eigen_matrix_csv_to_numpy(
            filepath+"has_cusp_at_base.csv").astype(bool).tolist(),
        deserialize_list_list_varying_lengths_float(filepath+"intersections.csv"),
        deserialize_list_list_varying_lengths(filepath+"intersection_indices.csv"),
        deserialize_list_list_intersection_data(filepath+"intersection_data.json"),
        num_intersections=176)

    # Compare results
    filepath_2: str = (
        f"{ROOTNAME}\\contour_network\\projected_curve_network\\init_chain_start_nodes\\")
    compare_list_node_geometry(filepath_2+"nodes_out.json", projected_curve_network_test.nodes)
    compare_eigen_numpy_matrix(filepath_2+"chain_start_nodes.csv",
                               np.array(projected_curve_network_test.chain_start_nodes, dtype=int))


def test_split_segments_at_cusps() -> None:
    """
    Testing split_segments_at_cusps() with spot_control mesh
    """
    filepath: str = (
        f"{ROOTNAME}\\contour_network\\projected_curve_network\\split_segments_at_cusps\\")

    # Deserialize arguments
    interior_cusps: list[list[float]] = deserialize_list_list_varying_lengths_float(
        filepath+"interior_cusps.csv")
    original_segment_indices_test: list[SegmentIndex] = deserialize_eigen_matrix_csv_to_numpy(
        filepath+"original_segment_indices_in.csv").astype(int).tolist()
    split_segment_indices_test: list[list[SegmentIndex]] = deserialize_list_list_varying_lengths(
        filepath+"split_segment_indices_in.csv")
    to_array_test: list[NodeIndex] = deserialize_eigen_matrix_csv_to_numpy(
        filepath+"to_array_in.csv").astype(int).tolist()
    out_array_test: list[SegmentIndex] = deserialize_eigen_matrix_csv_to_numpy(
        filepath+"out_array_in.csv").astype(int).tolist()
    intersection_array_test: list[NodeIndex] = deserialize_eigen_matrix_csv_to_numpy(
        filepath+"intersection_array_in.csv").astype(int).tolist()
    segments_test: list[SegmentGeometry] = _deserialize_list_segment_geometry(
        filepath+"segments_in.json")
    nodes_test: list[NodeGeometry] = _deserialize_list_node_geometry(
        filepath+"nodes_in.json")

    # Execute method
    split_segments_at_cusps(interior_cusps,
                            original_segment_indices_test,
                            split_segment_indices_test,
                            to_array_test,
                            out_array_test,
                            intersection_array_test,
                            segments_test,
                            nodes_test)

    # Compare results
    compare_eigen_numpy_matrix(filepath+"original_segment_indices_out.csv",
                               np.array(original_segment_indices_test))
    compare_list_list_varying_lengths(filepath+"split_segment_indices_out.csv",
                                      split_segment_indices_test)
    compare_eigen_numpy_matrix(filepath+"to_array_out.csv",
                               np.array(to_array_test))
    compare_eigen_numpy_matrix(filepath+"out_array_out.csv",
                               np.array(out_array_test))
    compare_eigen_numpy_matrix(filepath+"intersection_array_out.csv",
                               np.array(intersection_array_test))
    compare_list_segment_geometry(filepath+"segments_out.json", segments_test)
    compare_list_node_geometry(filepath+"nodes_out.json", nodes_test)


def test_connect_segment_intersections_spot_control() -> None:
    """
    Testing connect_segment_intersections() with spot_control mesh
    """
    filepath: str = (
        f"{ROOTNAME}\\contour_network\\projected_curve_network\\connect_segment_intersections\\")

    # Deserialize arguments
    intersection_nodes: list[list[NodeIndex]] = deserialize_list_list_varying_lengths(
        filepath+"intersection_nodes.csv")
    intersection_array_test: list[NodeIndex] = deserialize_eigen_matrix_csv_to_numpy(
        filepath+"intersection_array_in.csv").astype(int).tolist()
    nodes_test: list[NodeGeometry] = _deserialize_list_node_geometry(
        filepath+"nodes_in.json")

    # Execute method
    connect_segment_intersections(
        intersection_nodes,
        intersection_array_test,
        nodes_test)

    # Compare results
    compare_eigen_numpy_matrix(filepath+"intersection_array_out.csv",
                               np.array(intersection_array_test))
    compare_list_node_geometry(filepath+"nodes_out.json", nodes_test)


def test_split_segments_at_intersections_spot_control() -> None:
    """
    Testing split_segments_at_intersections() with spot_control mesh
    """

    filepath: str = (
        f"{ROOTNAME}\\contour_network\\projected_curve_network\\split_segments_at_intersections\\")

    # Deserialize arguments

    intersection_data_ref: list[list[IntersectionData]] = deserialize_list_list_intersection_data(
        filepath+"intersection_data.json")
    num_intersections: int = 176
    to_array_test: list[NodeIndex] = deserialize_eigen_matrix_csv_to_numpy(
        filepath+"to_array_in.csv").astype(int).tolist()
    out_array_test: list[SegmentIndex] = deserialize_eigen_matrix_csv_to_numpy(
        filepath+"out_array_in.csv").astype(int).tolist()
    segments_test: list[SegmentGeometry] = _deserialize_list_segment_geometry(
        filepath+"segments_in.json")
    nodes_test: list[NodeGeometry] = _deserialize_list_node_geometry(
        filepath+"nodes_in.json")

    original_segment_indices: list[SegmentIndex]
    split_segment_indices: list[list[SegmentIndex]]
    intersection_nodes: list[list[NodeIndex]]

    # Execute method for testing
    (original_segment_indices,
     split_segment_indices,
     intersection_nodes) = split_segments_at_intersections(
        intersection_data_ref,
        num_intersections,
        to_array_test,
        out_array_test,
        segments_test,
        nodes_test)

    # Comparing results
    compare_eigen_numpy_matrix(filepath+"original_segment_indices.csv",
                               np.array(original_segment_indices))
    compare_list_list_varying_lengths(filepath+"intersection_nodes.csv", intersection_nodes)
    compare_list_list_varying_lengths(filepath+"split_segment_indices.csv", split_segment_indices)

    compare_eigen_numpy_matrix(filepath+"to_array_out.csv", np.array(to_array_test))
    compare_eigen_numpy_matrix(filepath+"out_array_out.csv", np.array(out_array_test))
    compare_list_segment_geometry(filepath+"segments_out.json", segments_test)
    compare_list_node_geometry(filepath+"nodes_out.json", nodes_test)


def test_remove_redundant_intersections_spot_control() -> None:
    """
    Testing remove_redundant_intersections() with spot_control mesh
    """
    filepath: str = (
        f"{ROOTNAME}\\contour_network\\projected_curve_network\\remove_redundant_intersections\\")

    # Deserialize arguments
    to_array: list[NodeIndex] = deserialize_eigen_matrix_csv_to_numpy(
        filepath+"to_array.csv").astype(int).tolist()
    out_array: list[SegmentIndex] = deserialize_eigen_matrix_csv_to_numpy(
        filepath+"out_array.csv").astype(int).tolist()
    num_intersections: int = 176  # value from C++ code

    #
    # TODO: fix the deserialize_list_list_intersection_data deserializer
    # Because
    # num_intersections should be 176, meaning that intersection_segments within
    # deserialize_list_list_intersection_data() will also be 176 elements long.
    #
    intersection_data_test: list[list[IntersectionData]] = deserialize_list_list_intersection_data(
        filepath+"intersection_data_in.json")

    # Actual function to test
    remove_redundant_intersections(to_array, out_array, num_intersections, intersection_data_test)
    intersection_data_control: list[list[IntersectionData]] = (
        deserialize_list_list_intersection_data(filepath+"intersection_data_out.json"))

    compare_list_list_intersection_data(intersection_data_test, intersection_data_control)


def test_mark_open_chain_endpoints_spot_control() -> None:
    """
    Testing mark_open_chain_endpoints() with spot control mesh
    """
    filepath: str = (
        f"{ROOTNAME}\\contour_network\\projected_curve_network\\mark_open_chain_endpoints\\")

    # Deserialize arguments
    to_array: list[NodeIndex] = deserialize_eigen_matrix_csv_to_numpy(
        filepath+"to_array.csv").astype(int).tolist()
    out_array: list[SegmentIndex] = deserialize_eigen_matrix_csv_to_numpy(
        filepath+"out_array.csv").astype(int).tolist()
    chains: list[list[SegmentIndex]] = deserialize_list_list_varying_lengths(
        filepath+"chains.csv")
    nodes_in: list[NodeGeometry] = _deserialize_list_node_geometry(filepath+"nodes_in.json")

    # Actual execution
    ProjectedCurveNetwork.testing_mark_open_chain_endpoints(
        to_array, out_array, chains, nodes_in)

    # Testing output values
    compare_list_node_geometry(filepath+"nodes_out.json", nodes_in)


def test_build_projected_curve_network_without_intersections_spot_control() -> None:
    """
    Test build_projected_curve_network_without_intersections() for its usage in 
    init_projected_curve_network() with the spot control mesh.
    """
    filepath: str = "spot_control\\contour_network\\projected_curve_network\\build_projected_curve_network_without_intersections\\"

    parameter_segments: list[Conic] = deserialize_conics(filepath+"parameter_segments.json")
    spatial_segments: list[RationalFunction] = deserialize_rational_functions(
        filepath+"spatial_segments.json")
    planar_segments: list[RationalFunction] = deserialize_rational_functions(
        filepath+"planar_segments.json")
    segment_labels: list[dict[str, int]] = deserialize_segment_labels(
        filepath+"segment_labels.json")
    chains: list[list[int]] = deserialize_list_list_varying_lengths(
        filepath+"chains.csv")
    has_cusp_at_base: list[bool] = np.array(deserialize_eigen_matrix_csv_to_numpy(
        filepath+"has_cusp_at_base.csv"), dtype=bool).tolist()

    # Connect segments into chains before splitting at intersections
    to_array: list[NodeIndex]
    out_array: list[SegmentIndex]
    segments: list[SegmentGeometry]
    nodes: list[NodeGeometry]

    (to_array,
     out_array,
     segments,
     nodes) = build_projected_curve_network_without_intersections(
        parameter_segments,
        spatial_segments,
        planar_segments,
        segment_labels,
        chains,
        has_cusp_at_base)

    compare_eigen_numpy_matrix(filepath+"to_array.csv", np.array(to_array))
    compare_eigen_numpy_matrix(filepath+"out_array.csv", np.array(out_array))
    compare_list_node_geometry(filepath+"nodes.json", nodes)
    compare_list_segment_geometry(filepath+"segments.json", segments)

# ************************
# Original C++ test cases
# ************************


def test_closed_loop() -> None:
    """
    "A curve network can be built from topology information", "[projected_curve_network]"
    """
    out_array: list[SegmentIndex] = [0, 1, 2]
    to_array: list[NodeIndex] = [1, 2, 0]
    intersections: list[NodeIndex] = [-1, -1, -1]
    curve_network = AbstractCurveNetwork(to_array, out_array, intersections)

    assert curve_network.next(0) == 1
    assert curve_network.next(1) == 2
    assert curve_network.next(2) == 0
    assert curve_network.prev(0) == 2
    assert curve_network.prev(1) == 0
    assert curve_network.prev(2) == 1
    assert curve_network.to(0) == 1
    assert curve_network.to(1) == 2
    assert curve_network.to(2) == 0
    assert curve_network.from_(0) == 0
    assert curve_network.from_(1) == 1
    assert curve_network.from_(2) == 2
    assert curve_network.out(0) == 0
    assert curve_network.out(1) == 1
    assert curve_network.out(2) == 2
    assert curve_network.in_(0) == 2
    assert curve_network.in_(1) == 0
    assert curve_network.in_(2) == 1
