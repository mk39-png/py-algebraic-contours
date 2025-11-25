
from typing import Any

import numpy as np
import numpy.testing as npt

from pyalgcon.core.affine_manifold import (
    AffineManifold, EdgeManifoldChart, FaceManifoldChart,
    ParametricAffineManifold, VertexManifoldChart)
from pyalgcon.core.common import (
    FLOAT_EQUAL_PRECISION, Index, Matrix2x2f, MatrixXf, MatrixXi, PlanarPoint,
    Vector1D, compare_eigen_numpy_matrix, float_equal,
    initialize_spot_control_mesh, load_json, vector_equal)

# ********************
# Helper Methods
# ********************

# below was generated for testing!


def compare_vertex_charts(
        vertex_charts_control: list[dict[str, Any]],
        vertex_charts_test: list['VertexManifoldChart']) -> None:
    """
    TEMPORARY: used to test the vertex charts inputs...
    """
    assert len(vertex_charts_control) == len(vertex_charts_test), "Mismatch in chart count"

    for i, chart_control in enumerate(vertex_charts_control):
        control_vertex_index: int = chart_control.get("vertex_index")
        control_vertex_one_ring: list[int] = chart_control.get("vertex_one_ring")
        control_face_one_ring: list[int] = chart_control.get("face_one_ring")
        control_uv_positions: Vector1D = np.array(chart_control.get("one_ring_uv_positions"))
        control_is_boundary: bool = chart_control.get("is_boundary")
        control_is_cone: bool = chart_control.get("is_cone")
        control_is_cone_adjacent: bool = chart_control.get("is_cone_adjacent")

        vertex_chart_test: VertexManifoldChart = vertex_charts_test[i]
        test_uv_positions: np.ndarray = vertex_chart_test.one_ring_uv_positions

        assert control_vertex_index == vertex_chart_test.vertex_index
        assert control_vertex_one_ring == vertex_chart_test.vertex_one_ring
        assert control_face_one_ring == vertex_chart_test.face_one_ring
        npt.assert_allclose(test_uv_positions, control_uv_positions, atol=1e-6)
        assert control_is_boundary == vertex_chart_test.is_boundary
        assert control_is_cone == vertex_chart_test.is_cone
        assert control_is_cone_adjacent == vertex_chart_test.is_cone_adjacent


def compare_edge_charts(edge_charts_control: list[dict],
                        edge_charts_test: list[EdgeManifoldChart]) -> None:
    """
    TEMPORARY: used to test the edge charts inputs...
    """
    assert len(edge_charts_control) == len(edge_charts_test)
    for i, chart_control in enumerate(edge_charts_control):
        control_top_face_idx: int = chart_control.get("top_face_index")
        control_bot_face_idx: int = chart_control.get("bottom_face_index")
        control_left_vert_idx: int = chart_control.get("left_vertex_index")
        control_right_vert_idx: int = chart_control.get("right_vertex_index")
        control_top_vert_idx: int = chart_control.get("top_vertex_index")
        control_bot_vert_idx: int = chart_control.get("bottom_vertex_index")
        control_left_vert_uv_posn: np.ndarray = np.array(
            chart_control.get("left_vertex_uv_position"))
        control_right_vert_uv_posn: np.ndarray = np.array(
            chart_control.get("right_vertex_uv_position"))
        control_top_vert_uv_posn: np.ndarray = np.array(chart_control.get("top_vertex_uv_position"))
        control_bottom_vert_uv_posn: np.ndarray = np.array(
            chart_control.get("bottom_vertex_uv_position"))
        control_is_boundary_control: bool = chart_control.get("is_boundary")

        edge_chart_test: EdgeManifoldChart = edge_charts_test[i]
        test_top_face_idx: int = edge_chart_test.top_face_index
        test_bot_face_idx: int = edge_chart_test.bottom_face_index
        test_left_vert_idx: int = edge_chart_test.left_vertex_index
        test_right_vert_idx: int = edge_chart_test.right_vertex_index
        test_top_vert_idx: int = edge_chart_test.top_vertex_index
        test_bot_vert_idx: int = edge_chart_test.bottom_vertex_index
        test_left_vert_uv_posn: np.ndarray = edge_chart_test.left_vertex_uv_position.flatten()
        test_right_vert_uv_posn: np.ndarray = edge_chart_test.right_vertex_uv_position.flatten()
        test_top_vert_uv_posn: np.ndarray = edge_chart_test.top_vertex_uv_position.flatten()
        test_bottom_vert_uv_posn: np.ndarray = edge_chart_test.bottom_vertex_uv_position.flatten()
        test_is_boundary_control: bool = edge_chart_test.is_boundary

        assert control_top_face_idx == test_top_face_idx
        assert control_bot_face_idx == test_bot_face_idx
        assert control_left_vert_idx == test_left_vert_idx
        assert control_right_vert_idx == test_right_vert_idx
        assert control_top_vert_idx == test_top_vert_idx
        assert control_bot_vert_idx == test_bot_vert_idx
        npt.assert_allclose(test_left_vert_uv_posn, control_left_vert_uv_posn,
                            atol=FLOAT_EQUAL_PRECISION, verbose=True)
        npt.assert_allclose(test_right_vert_uv_posn, control_right_vert_uv_posn, atol=0.001)
        npt.assert_allclose(test_top_vert_uv_posn, control_top_vert_uv_posn, atol=0.001)
        npt.assert_allclose(test_bottom_vert_uv_posn, control_bottom_vert_uv_posn, atol=0.001)
        assert control_is_boundary_control == test_is_boundary_control


def compare_face_charts(face_charts_control: list[dict],
                        face_charts_test: list['FaceManifoldChart']) -> None:
    """
    Compare control dicts with actual FaceManifoldChart objects.
    """
    assert len(face_charts_control) == len(face_charts_test), "Mismatch in number of face charts"

    for i, chart_control in enumerate(face_charts_control):
        control_face_index: int = chart_control.get("face_index")
        control_uv_positions: np.ndarray = np.array(chart_control.get(
            "face_uv_positions"))  # list of PlanarPoint... so Nx2 shape
        # np.array(p) for p in chart_control.get("face_uv_positions")
        control_is_boundary: bool = chart_control.get("is_boundary")
        control_is_cone_adjacent: bool = chart_control.get("is_cone_adjacent")
        control_is_cone_corner: list[bool] = np.array(chart_control.get("is_cone_corner"))

        chart_test: FaceManifoldChart = face_charts_test[i]

        assert control_face_index == chart_test.face_index, f"Mismatch at face index {i}"

        # Compare UV positions one by one

        npt.assert_allclose(control_uv_positions,
                            np.array(chart_test.face_uv_positions).squeeze(),
                            atol=1e-5,
                            )

        assert control_is_boundary == chart_test.is_boundary, f"Boundary flag mismatch at face {i}"
        assert control_is_cone_adjacent == chart_test.is_cone_adjacent, f"Cone-adjacent flag mismatch at face {i}"
        npt.assert_array_equal(
            np.array(chart_test.is_cone_corner),
            np.array(control_is_cone_corner))


def initialize_affine_manifold_from_spot_control() -> AffineManifold:
    """
    Helper function to initialize AffineManifold from spot_control mesh.
    """
    # Get input mesh
    V, uv, F, FT = initialize_spot_control_mesh()
    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)
    affine_manifold_filepath: str = "spot_control\\affine_manifold\\"

    return affine_manifold


# ********************
# Test Methods
# ********************
def test_compute_cone_corners_from_spot_control() -> None:
    """
    Tests compute_cone_corner() of AffineManifold.
    compute_cone_corner() used in TwelveSplitSplineSurface constructor
    """
    affine_manifold: AffineManifold = initialize_affine_manifold_from_spot_control()

    # list[bool] of length 3
    is_cone_corner: list[list[bool]] = affine_manifold.compute_cone_corners()
    compare_eigen_numpy_matrix(
        "spot_control\\12_split_spline\\is_cone_corner.csv", np.array(is_cone_corner))


def test_compute_cones_from_spot_control() -> None:
    """
    This tests the entire constructor of AffineManifold.
    """
    affine_manifold: AffineManifold = initialize_affine_manifold_from_spot_control()
    cones: list[Index] = affine_manifold.compute_cones()
    # Testing --
    # XXX: magic numbers from ASOC 12 Split Spline with control spot model with UV
    cones_control = np.array([0, 23, 43, 52, 62, 76, 88, 115])
    assert len(cones) == len(cones_control)  # should be 8
    npt.assert_allclose(cones_control, np.array(cones))


def test_affine_manifold_from_spot_control() -> None:
    """
    This tests the entire constructor of AffineManifold.
    """
    affine_manifold: AffineManifold = initialize_affine_manifold_from_spot_control()
    affine_manifold_filepath: str = "spot_control\\affine_manifold\\"

    compare_eigen_numpy_matrix(f"{affine_manifold_filepath}l.csv",
                               np.array(affine_manifold.l))
    compare_eigen_numpy_matrix(f"{affine_manifold_filepath}corner_to_edge.csv",
                               np.array(affine_manifold.corner_to_edge))
    compare_eigen_numpy_matrix(f"{affine_manifold_filepath}corner_to_he.csv",
                               np.array(affine_manifold.corner_to_he))
    compare_eigen_numpy_matrix(f"{affine_manifold_filepath}F.csv",
                               affine_manifold.faces)
    compare_eigen_numpy_matrix(f"{affine_manifold_filepath}F_uv.csv",
                               affine_manifold.F_uv)
    compare_eigen_numpy_matrix(f"{affine_manifold_filepath}global_uv.csv",
                               affine_manifold.global_uv)
    compare_eigen_numpy_matrix(f"{affine_manifold_filepath}he_to_corner.csv",
                               np.array(affine_manifold.he_to_corner))
    # TODO: find a way to test he_to_edge
    # compare_eigen_numpy_matrix(
    #     f"{filepath}he_to_edge.csv", np.array(affine_manifold.he_to_edge))
    vertex_charts_control: list[dict] = load_json(
        f"{affine_manifold_filepath}vertex_charts_aligned.json")
    compare_vertex_charts(vertex_charts_control, affine_manifold.vertex_charts)

    edge_charts_control: list[dict] = load_json(
        f"{affine_manifold_filepath}edge_charts_aligned.json")
    compare_edge_charts(edge_charts_control, affine_manifold.edge_charts)

    face_charts_control: list[dict] = load_json(
        f"{affine_manifold_filepath}face_charts_aligned.json")
    compare_face_charts(face_charts_control, affine_manifold.face_charts)


def test_affine_manifold_from_global_uvs() -> None:
    V: MatrixXf = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]], dtype=np.float64)
    F: MatrixXi = np.array([[0, 1, 2]], dtype=np.int64)

    assert V.shape == (3, 2)
    assert F.shape == (1, 3)

    affine_manifold = ParametricAffineManifold(F, V)

    # Check basic manifold information
    assert affine_manifold.num_faces == 1
    assert affine_manifold.num_vertices == 3

    # Check vertex chart at vertex 0
    chart: VertexManifoldChart = affine_manifold.get_vertex_chart(0)
    assert chart.vertex_index == 0
    assert len(chart.vertex_one_ring) == 2
    assert chart.vertex_one_ring[0] == 1
    assert chart.vertex_one_ring[1] == 2
    assert len(chart.face_one_ring) == 1
    assert chart.face_one_ring[0] == 0
    assert float_equal(chart.one_ring_uv_positions[0, 0], 3.0)
    assert float_equal(chart.one_ring_uv_positions[0, 1], 0.0)
    assert float_equal(chart.one_ring_uv_positions[1, 0], 0.0)
    assert float_equal(chart.one_ring_uv_positions[1, 1], 4.0)

    # Check face corner charts for face 0
    corner_uv_positions: Matrix2x2f = affine_manifold.get_face_corner_charts(0)

    assert float_equal(corner_uv_positions[0][0, 0],  3.0)
    assert float_equal(corner_uv_positions[0][0, 1],  0.0)
    assert float_equal(corner_uv_positions[0][1, 0],  0.0)
    assert float_equal(corner_uv_positions[0][1, 1],  4.0)
    assert float_equal(corner_uv_positions[1][0, 0], -3.0)
    assert float_equal(corner_uv_positions[1][0, 1],  4.0)
    assert float_equal(corner_uv_positions[1][1, 0], -3.0)
    assert float_equal(corner_uv_positions[1][1, 1],  0.0)
    assert float_equal(corner_uv_positions[2][0, 0],  0.0)
    assert float_equal(corner_uv_positions[2][0, 1], -4.0)
    assert float_equal(corner_uv_positions[2][1, 0],  3.0)
    assert float_equal(corner_uv_positions[2][1, 1], -4.0)

    # Check global uv
    uv0: PlanarPoint = affine_manifold.get_vertex_global_uv(0)
    uv1: PlanarPoint = affine_manifold.get_vertex_global_uv(1)
    uv2: PlanarPoint = affine_manifold.get_vertex_global_uv(2)
    assert np.array_equal(affine_manifold.get_global_uv, V)
    assert vector_equal(uv0, V[[0], :])
    assert vector_equal(uv1, V[[1], :])
    assert vector_equal(uv2, V[[2], :])
