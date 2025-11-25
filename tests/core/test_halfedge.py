"""
File to test Halfedge class.
Also includes test to compare with original implementation.
"""

import os

import igl
import numpy as np
import pytest

from pyalgcon.core.common import (Index, MatrixNx3i, MatrixXf,
                                  MatrixXi,
                                  compare_eigen_numpy_matrix)
from pyalgcon.core.halfedge import Halfedge


def test_halfedge_spot_mesh() -> None:
    """
    Tests constructor of halfedge from spot control mesh.
    """
    # Get input mesh
    V: MatrixXf = np.ndarray(shape=(0, 0), dtype=np.float64)
    uv: MatrixXf = np.ndarray(shape=(0, 0), dtype=np.float64)
    N: MatrixXf = np.ndarray(shape=(0, 0), dtype=np.float64)
    F: MatrixXi = np.ndarray(shape=(0, 0), dtype=np.int64)
    FT: MatrixXi = np.ndarray(shape=(0, 0), dtype=np.int64)
    FN: MatrixXi = np.ndarray(shape=(0, 0), dtype=np.int64)

    filename: str = "spot_control_mesh-cleaned_conf_simplified_with_uv.obj"
    filepath: str = os.path.abspath(f"src\\tests\\spot_control\\{filename}")
    V, uv, N, F, FT, FN = igl.readOBJ(filepath)

    halfedge = Halfedge(F)

    filepath: str = "spot_control\\halfedge\\"
    # TODO: maybe test num_edges, num_he, num_faces, and num_vertices members.
    compare_eigen_numpy_matrix(f"{filepath}corner_to_he.csv",
                               np.array(halfedge.corner_to_he))
    compare_eigen_numpy_matrix(f"{filepath}he_to_corner.csv",
                               np.array(halfedge.he_to_corner))
    compare_eigen_numpy_matrix(f"{filepath}e2he.csv",
                               np.array(halfedge.edge_to_halfedge_map))
    compare_eigen_numpy_matrix(f"{filepath}f2he.csv",
                               np.array(halfedge.f2he))
    compare_eigen_numpy_matrix(f"{filepath}face.csv",
                               np.array(halfedge.face))
    compare_eigen_numpy_matrix(f"{filepath}from.csv",
                               np.array(halfedge._from))
    compare_eigen_numpy_matrix(f"{filepath}he2e.csv",
                               np.array(halfedge.halfedge_to_edge_map))
    compare_eigen_numpy_matrix(f"{filepath}next.csv",
                               np.array(halfedge.next))
    compare_eigen_numpy_matrix(f"{filepath}opp.csv",
                               np.array(halfedge.opp))
    compare_eigen_numpy_matrix(f"{filepath}out.csv",
                               np.array(halfedge.out))
    compare_eigen_numpy_matrix(f"{filepath}to.csv",
                               np.array(halfedge.to))


def test_halfedge_one_triangle() -> None:
    F: MatrixNx3i = np.array([[0, 1, 2]], dtype=np.int64)

    mesh = Halfedge(F)
    corner_to_he: list[list[Index]] = mesh.corner_to_he
    he_to_corner: list[tuple[Index, Index]] = mesh.he_to_corner

    # Check size information
    assert len(corner_to_he) == 1
    assert len(corner_to_he[0]) == 3
    assert len(he_to_corner) == 3
    assert mesh.num_halfedges == 3
    assert mesh.num_faces == 1
    assert mesh.num_vertices == 3
    assert mesh.num_edges == 3

    # Get mesh elements
    he0: int = corner_to_he[0][0]
    he1: int = corner_to_he[0][1]
    he2: int = corner_to_he[0][2]
    f0 = 0
    v0 = 0
    v1 = 1
    v2 = 2

    # Check next
    assert mesh.next_halfedge(he0) == he1
    assert mesh.next_halfedge(he1) == he2
    assert mesh.next_halfedge(he2) == he0

    # Check face
    assert mesh.halfedge_to_face(he0) == f0
    assert mesh.halfedge_to_face(he1) == f0
    assert mesh.halfedge_to_face(he2) == f0

    # Check vertex
    assert mesh.halfedge_to_tail_vertex(he0) == v1
    assert mesh.halfedge_to_head_vertex(he0) == v2
    assert mesh.halfedge_to_tail_vertex(he1) == v2
    assert mesh.halfedge_to_head_vertex(he1) == v0
    assert mesh.halfedge_to_tail_vertex(he2) == v0
    assert mesh.halfedge_to_head_vertex(he2) == v1


def test_halfedge_two_closed_triangles() -> None:
    F: MatrixNx3i = np.array([[0, 1, 2], [0, 2, 1]], dtype=np.int64)
    mesh = Halfedge(F)
    corner_to_he: list[list[Index]] = mesh.corner_to_he
    he_to_corner: list[tuple[Index, Index]] = mesh.he_to_corner

    # Check size information
    assert len(corner_to_he) == 2
    assert len(corner_to_he[0]) == 3
    assert len(corner_to_he[1]) == 3
    assert len(he_to_corner) == 6
    assert mesh.num_halfedges == 6
    assert mesh.num_faces == 2
    assert mesh.num_vertices == 3
    assert mesh.num_edges == 3

    # Get mesh elements
    # Halfedges are indexed by face and global vertex index
    he00: int = corner_to_he[0][0]
    he01: int = corner_to_he[0][1]
    he02: int = corner_to_he[0][2]
    he10: int = corner_to_he[1][0]
    he11: int = corner_to_he[1][2]
    he12: int = corner_to_he[1][1]
    f0 = 0
    f1 = 1
    v0 = 0
    v1 = 1
    v2 = 2

    # Check next
    assert mesh.next_halfedge(he00) == he01
    assert mesh.next_halfedge(he01) == he02
    assert mesh.next_halfedge(he02) == he00
    assert mesh.next_halfedge(he10) == he12
    assert mesh.next_halfedge(he11) == he10
    assert mesh.next_halfedge(he12) == he11

    # Check opposite
    assert mesh.opposite_halfedge(he00) == he10
    assert mesh.opposite_halfedge(he01) == he11
    assert mesh.opposite_halfedge(he02) == he12
    assert mesh.opposite_halfedge(he10) == he00
    assert mesh.opposite_halfedge(he11) == he01
    assert mesh.opposite_halfedge(he12) == he02

    # Check face
    assert mesh.halfedge_to_face(he00) == f0
    assert mesh.halfedge_to_face(he01) == f0
    assert mesh.halfedge_to_face(he02) == f0
    assert mesh.halfedge_to_face(he10) == f1
    assert mesh.halfedge_to_face(he11) == f1
    assert mesh.halfedge_to_face(he12) == f1

    # Check vertex
    assert mesh.halfedge_to_tail_vertex(he00) == v1
    assert mesh.halfedge_to_head_vertex(he00) == v2
    assert mesh.halfedge_to_tail_vertex(he01) == v2
    assert mesh.halfedge_to_head_vertex(he01) == v0
    assert mesh.halfedge_to_tail_vertex(he02) == v0
    assert mesh.halfedge_to_head_vertex(he02) == v1

    assert mesh.halfedge_to_tail_vertex(he10) == v2
    assert mesh.halfedge_to_head_vertex(he10) == v1
    assert mesh.halfedge_to_tail_vertex(he11) == v0
    assert mesh.halfedge_to_head_vertex(he11) == v2
    assert mesh.halfedge_to_tail_vertex(he12) == v1
    assert mesh.halfedge_to_head_vertex(he12) == v0


def test_halfedge_two_open_triangles() -> None:
    F: MatrixNx3i = np.array([[0, 1, 2], [0, 3, 1]], dtype=np.int64)
    mesh = Halfedge(F)
    corner_to_he: list[list[Index]] = mesh.corner_to_he
    he_to_corner: list[tuple[Index, Index]] = mesh.he_to_corner

    # Check size information
    assert len(corner_to_he) == 2
    assert len(corner_to_he[0]) == 3
    assert len(corner_to_he[1]) == 3
    assert len(he_to_corner) == 6
    assert mesh.num_halfedges == 6
    assert mesh.num_faces == 2
    assert mesh.num_vertices == 4
    assert mesh.num_edges == 5

    # Get mesh elements
    # Halfedges are indexed by face and global vertex index
    he00: int = corner_to_he[0][0]
    he01: int = corner_to_he[0][1]
    he02: int = corner_to_he[0][2]
    he10: int = corner_to_he[1][0]
    he11: int = corner_to_he[1][2]
    he13: int = corner_to_he[1][1]
    f0 = 0
    f1 = 1
    v0 = 0
    v1 = 1
    v2 = 2
    v3 = 3

    # Check next
    assert mesh.next_halfedge(he00) == he01
    assert mesh.next_halfedge(he01) == he02
    assert mesh.next_halfedge(he02) == he00
    assert mesh.next_halfedge(he10) == he13
    assert mesh.next_halfedge(he11) == he10
    assert mesh.next_halfedge(he13) == he11

    # Check opposite
    assert mesh.opposite_halfedge(he02) == he13
    assert mesh.opposite_halfedge(he13) == he02

    # Check face
    assert mesh.halfedge_to_face(he00) == f0
    assert mesh.halfedge_to_face(he01) == f0
    assert mesh.halfedge_to_face(he02) == f0
    assert mesh.halfedge_to_face(he10) == f1
    assert mesh.halfedge_to_face(he11) == f1
    assert mesh.halfedge_to_face(he13) == f1

    # Check vertex
    assert mesh.halfedge_to_tail_vertex(he00) == v1
    assert mesh.halfedge_to_head_vertex(he00) == v2
    assert mesh.halfedge_to_tail_vertex(he01) == v2
    assert mesh.halfedge_to_head_vertex(he01) == v0
    assert mesh.halfedge_to_tail_vertex(he02) == v0
    assert mesh.halfedge_to_head_vertex(he02) == v1

    assert mesh.halfedge_to_tail_vertex(he10) == v3
    assert mesh.halfedge_to_head_vertex(he10) == v1
    assert mesh.halfedge_to_tail_vertex(he11) == v0
    assert mesh.halfedge_to_head_vertex(he11) == v3
    assert mesh.halfedge_to_tail_vertex(he13) == v1
    assert mesh.halfedge_to_head_vertex(he13) == v0
