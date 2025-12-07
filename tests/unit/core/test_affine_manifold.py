"""
Test affine manifold
"""
import pathlib

import numpy as np
import numpy.testing as npt

from pyalgcon.core.affine_manifold import (AffineManifold,
                                           ParametricAffineManifold,
                                           VertexManifoldChart)
from pyalgcon.core.common import (Index, Matrix2x2f, MatrixXf, MatrixXi,
                                  PlanarPoint, compare_eigen_numpy_matrix,
                                  float_equal, vector_equal)
from pyalgcon.debug.debug import (compare_edge_charts_from_file,
                                  compare_face_charts_from_file,
                                  compare_vertex_charts_from_file)


# ********************
# Test Methods
# ********************
def test_compute_cone_corners(testing_fileinfo,
                              initialize_affine_manifold) -> None:
    """
    Tests compute_cone_corner() of AffineManifold.
    compute_cone_corner() used in TwelveSplitSplineSurface constructor
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / "quadratic_spline_surface" / "12_split_spline"
    affine_manifold: AffineManifold = initialize_affine_manifold

    # Execute method
    # list[bool] of length 3
    is_cone_corner: list[list[bool]] = affine_manifold.compute_cone_corners()

    # Compare results
    compare_eigen_numpy_matrix(filepath / "is_cone_corner.csv", np.array(is_cone_corner))


def test_compute_cones(initialize_affine_manifold) -> None:
    """
    This tests the entire constructor of AffineManifold.
    """
    # Retrieve parameters
    affine_manifold: AffineManifold = initialize_affine_manifold

    # Execute method
    cones: list[Index] = affine_manifold.compute_cones()

    # Compare results
    # XXX: magic numbers from ASOC 12 Split Spline with control spot model with UV
    cones_control = np.array([0, 23, 43, 52, 62, 76, 88, 115])
    assert len(cones) == len(cones_control)  # should be 8
    npt.assert_allclose(cones_control, np.array(cones))


def test_affine_manifold(testing_fileinfo,
                         initialize_affine_manifold) -> None:
    """
    This tests the entire constructor of AffineManifold.
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / "core" / "affine_manifold"

    # "Execute" method
    affine_manifold: AffineManifold = initialize_affine_manifold

    # Compare results
    compare_eigen_numpy_matrix(filepath / "l.csv",
                               np.array(affine_manifold.l))
    compare_eigen_numpy_matrix(filepath / "corner_to_edge.csv",
                               np.array(affine_manifold.corner_to_edge))
    compare_eigen_numpy_matrix(filepath / "corner_to_he.csv",
                               np.array(affine_manifold.corner_to_he))
    compare_eigen_numpy_matrix(filepath / "F.csv",
                               affine_manifold.faces)
    compare_eigen_numpy_matrix(filepath / "F_uv.csv",
                               affine_manifold.F_uv)
    compare_eigen_numpy_matrix(filepath / "global_uv.csv",
                               affine_manifold.global_uv)
    compare_eigen_numpy_matrix(filepath / "he_to_corner.csv",
                               np.array(affine_manifold.he_to_corner))

    # TODO: find a way to test he_to_edge
    # compare_eigen_numpy_matrix(
    #     f"{filepath}he_to_edge.csv", np.array(affine_manifold.he_to_edge))

    compare_vertex_charts_from_file(filepath / "vertex_charts_aligned.json",
                                    affine_manifold.vertex_charts)
    compare_edge_charts_from_file(filepath / "edge_charts_aligned.json",
                                  affine_manifold.edge_charts)
    compare_face_charts_from_file(filepath / "face_charts_aligned.json",
                                  affine_manifold.face_charts)


def test_affine_manifold_from_global_uvs() -> None:
    """
    From original Algebraic Contours test case.
    """
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
