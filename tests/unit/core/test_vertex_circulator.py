"""
Test vertex circulator
"""

import os
import pathlib

import igl
import numpy as np

from pyalgcon.core.common import (MatrixNx3f, MatrixNx3i, MatrixXf, MatrixXi,
                                  compare_eigen_numpy_matrix,
                                  compare_list_list_varying_lengths,
                                  vector_contains)
from pyalgcon.core.vertex_circulator import VertexCirculator
from pyalgcon.utils.generate_shapes import (generate_minimal_torus_VF,
                                            generate_tetrahedron_VF)


def test_vertex_circulator_from_spot_control(testing_fileinfo,
                                             parsed_control_mesh) -> None:
    """
    Testing vertex circulator constructor from spot control mesh.
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / "core" / "vertex_circulator"
    V: np.ndarray
    uv: np.ndarray
    F: np.ndarray
    FT: np.ndarray
    V, uv, F, FT = parsed_control_mesh

    # Execute method
    vertex_circulator = VertexCirculator(F)

    # Compare results
    compare_eigen_numpy_matrix(filepath / "F.csv",
                               vertex_circulator.m_F)
    compare_list_list_varying_lengths(filepath / "all_adjacent_faces.csv",
                                      vertex_circulator.m_all_adjacent_faces)
    compare_list_list_varying_lengths(filepath / "all_face_one_rings.csv",
                                      vertex_circulator.m_all_face_one_rings)
    compare_list_list_varying_lengths(filepath / "all_vertex_one_rings.csv",
                                      vertex_circulator.m_all_vertex_one_rings)


def test_tetrahedron() -> None:
    """
    From original C++ code.
    """
    V: MatrixNx3f
    F: MatrixNx3i
    V, F = generate_tetrahedron_VF()

    vertex_circulator = VertexCirculator(F)
    vertex_one_ring: list[int]
    face_one_ring: list[int]
    vertex_one_ring, face_one_ring = vertex_circulator.get_one_ring(0)

    assert len(vertex_one_ring) == 4
    assert len(face_one_ring) == 3
    assert vector_contains(vertex_one_ring, 1)
    assert vector_contains(vertex_one_ring, 2)
    assert vector_contains(vertex_one_ring, 3)
    assert vector_contains(face_one_ring, 0)
    assert vector_contains(face_one_ring, 1)
    assert vector_contains(face_one_ring, 2)


def test_torus() -> None:
    """
    From original C++ code.
    """

    V: MatrixNx3f
    F: MatrixNx3i
    V, F = generate_minimal_torus_VF()

    vertex_circulator = VertexCirculator(F)
    vertex_one_ring: list[int]
    face_one_ring: list[int]
    vertex_one_ring, face_one_ring = vertex_circulator.get_one_ring(4)

    assert len(vertex_one_ring) == 7
    assert len(face_one_ring) == 6
    assert vector_contains(vertex_one_ring, 1)
    assert vector_contains(vertex_one_ring, 2)
    assert vector_contains(vertex_one_ring, 3)
    assert vector_contains(vertex_one_ring, 5)
    assert vector_contains(vertex_one_ring, 6)
    assert vector_contains(vertex_one_ring, 7)
    assert vector_contains(face_one_ring, 1)
    assert vector_contains(face_one_ring, 2)
    assert vector_contains(face_one_ring, 3)
    assert vector_contains(face_one_ring, 6)
    assert vector_contains(face_one_ring, 7)
    assert vector_contains(face_one_ring, 8)


def test_triangle() -> None:
    """
    From original C++ code.
    """

    F: MatrixNx3i = np.array([[0, 1, 2]], dtype=np.int64)

    # NOTE: problem with sizing of F.... an it being (3,) rather than (1, 3) or something.
    vertex_circulator = VertexCirculator(F)
    vertex_one_ring: list[int]
    face_one_ring: list[int]

    #  First vertex
    vertex_one_ring, face_one_ring = vertex_circulator.get_one_ring(0)
    assert len(vertex_one_ring) == 2
    assert len(face_one_ring) == 1
    assert vertex_one_ring[0] == 1
    assert vertex_one_ring[1] == 2
    assert face_one_ring[0] == 0

    #  Second vertex
    vertex_one_ring, face_one_ring = vertex_circulator.get_one_ring(1)
    assert len(vertex_one_ring) == 2
    assert len(face_one_ring) == 1
    assert (vertex_one_ring[0] == 2)
    assert (vertex_one_ring[1] == 0)
    assert (face_one_ring[0] == 0)

    #  Third vertex
    vertex_one_ring, face_one_ring = vertex_circulator.get_one_ring(2)
    assert len(vertex_one_ring) == 2
    assert len(face_one_ring) == 1
    assert vertex_one_ring[0] == 0
    assert vertex_one_ring[1] == 1
    assert face_one_ring[0] == 0


def test_square() -> None:
    """
    From original C++ code.
    """

    F: MatrixNx3i = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)

    vertex_circulator = VertexCirculator(F)
    vertex_one_ring: list[int]
    face_one_ring: list[int]

    # First vertex
    vertex_one_ring, face_one_ring = vertex_circulator.get_one_ring(0)
    assert len(vertex_one_ring) == 2
    assert len(face_one_ring) == 1
    assert vertex_one_ring[0] == 1
    assert vertex_one_ring[1] == 2
    assert face_one_ring[0] == 0

    # Second vertex
    vertex_one_ring, face_one_ring = vertex_circulator.get_one_ring(1)
    assert len(vertex_one_ring) == 3
    assert len(face_one_ring) == 2
    assert vertex_one_ring[0] == 3
    assert vertex_one_ring[1] == 2
    assert vertex_one_ring[2] == 0
    assert face_one_ring[0] == 1
    assert face_one_ring[1] == 0

    # Third vertex
    vertex_one_ring, face_one_ring = vertex_circulator.get_one_ring(2)
    assert len(vertex_one_ring) == 3
    assert len(face_one_ring) == 2
    assert vertex_one_ring[0] == 0
    assert vertex_one_ring[1] == 1
    assert vertex_one_ring[2] == 3
    assert face_one_ring[0] == 0
    assert face_one_ring[1] == 1

    # Third vertex
    vertex_one_ring, face_one_ring = vertex_circulator.get_one_ring(3)
    assert len(vertex_one_ring) == 2
    assert len(face_one_ring) == 1
    assert vertex_one_ring[0] == 2
    assert vertex_one_ring[1] == 1
    assert face_one_ring[0] == 1
