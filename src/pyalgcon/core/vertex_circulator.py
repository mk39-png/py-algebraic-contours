"""
Class to build circulators around vertices in VF representation
"""

import logging

import numpy as np

from pyalgcon.core.common import (PLACEHOLDER_VALUE,
                                  MatrixNx3i, Vector1D,
                                  contains_vertex)

logger: logging.Logger = logging.getLogger(__name__)


def contains_edge(face: Vector1D, vertex_0: int, vertex_1: int) -> bool:
    """
    Return true iff the face contains the edge { vertex_0, vertex_1 }
    """
    assert face.ndim == 1

    return ((contains_vertex(face, vertex_0)) and
            (contains_vertex(face, vertex_1)))


def is_left_face(face: Vector1D, vertex_0: int, vertex_1: int) -> bool:
    """
    Return true iff the face is to the left of the given edge
    """
    assert face.ndim == 1

    if ((face[0] == vertex_0) and (face[1] == vertex_1)):
        return True
    if ((face[1] == vertex_0) and (face[2] == vertex_1)):
        return True
    if ((face[2] == vertex_0) and (face[0] == vertex_1)):
        return True

    return False


def is_right_face(face: Vector1D, vertex_0: int, vertex_1: int) -> bool:
    """
    Return true iff the face is to the right of the given edge
    """
    assert face.ndim == 1

    if ((face[1] == vertex_0) and (face[0] == vertex_1)):
        return True
    if ((face[2] == vertex_0) and (face[1] == vertex_1)):
        return True
    if ((face[0] == vertex_0) and (face[2] == vertex_1)):
        return True

    return False


def find_next_vertex(face: Vector1D, vertex: int) -> int:
    """
    Get the index of the vertex in the face ccw from the given vertex
    """
    assert face.ndim == 1

    if face[0] == vertex:
        return face[1]
    if face[1] == vertex:
        return face[2]
    if face[2] == vertex:
        return face[0]

    return -1


def find_prev_vertex(face: Vector1D, vertex: int) -> int:
    """
    Get the index of the vertex in the face clockwise from the given vertex
    """
    assert face.ndim == 1

    if face[0] == vertex:
        return face[2]
    if face[1] == vertex:
        return face[0]
    if face[2] == vertex:
        return face[1]

    return -1


def are_adjacent(face_0: Vector1D, face_1: Vector1D) -> bool:
    """
    Return true iff the two faces are adjacent
    """
    assert face_0.ndim == 1
    assert face_1.ndim == 1

    if contains_edge(face_0, face_1[0], face_1[1]):
        return True
    if contains_edge(face_0, face_1[1], face_1[2]):
        return True
    if contains_edge(face_0, face_1[2], face_1[0]):
        return True

    return False


def compute_adjacent_faces(F: np.ndarray) -> list[list[int]]:
    """
    Get list of all faces adjacent to each vertex
    """
    num_vertices: int = F.max() + 1
    all_adjacent_faces: list[list[int]] = [[] for _ in range(num_vertices)]

    # Initialize adjacent faces list
    for i, face in enumerate(all_adjacent_faces):
        face.clear()

    for i in range(F.shape[0]):  # rows
        for j in range(F.shape[1]):  # cols
            all_adjacent_faces[F[i, j]].append(i)

    return all_adjacent_faces


def compute_vertex_one_ring_first_face(
        F: np.ndarray, vertex_index: int, adjacent_faces: list[int]) -> int:
    """
    Compute the first face of the vertex one ring, which should be right
    boundary face for a boundary vertex.
    """
    if len(adjacent_faces) == 0:
        return -1

    #  Get arbitrary adjacent face to start and vertex on the face
    current_face: int = adjacent_faces[0]
    current_vertex: int = find_next_vertex(F[current_face, :], vertex_index)
    logger.info("Starting search for first face from vertex %s on face %s",
                current_vertex, F[current_face, :])

    # Cycle clockwise to a starting face
    for i in range(1, len(adjacent_faces)):
        # Get previous face or return if none exists
        prev_face: int = -1

        for j, face in enumerate(adjacent_faces):
            f: int = adjacent_faces[j]

            # Grabs row f
            if is_right_face(F[f, :], vertex_index, current_vertex):
                prev_face = f
                break

        # Return current face if no previous face found
        if prev_face == -1:
            return current_face

        # Get previous face and vertex
        current_face = prev_face
        current_vertex = find_prev_vertex(F[current_face, :], current_vertex)  # row retrieval

    # If we have not returned yet, this is an interior vertex, and we return
    # the current face as an arbitrary choice
    return current_face


def compute_vertex_one_ring(F: np.ndarray,
                            vertex_index: int,
                            adjacent_faces: list[int]) -> tuple[list[int], list[int]]:
    """
    Compute the vertex one ring for a vertex index using adjacent faces.

    Args:
        F: [in].
        vertex_index: [in].
        adjacent_faces: [in].

    Returns:
        vertex_one_ring: [out].
        face_one_ring: [out].
    """
    # TODO: double check logic of this method and if it makes sense to return a new value
    num_faces:       int = len(adjacent_faces)
    vertex_one_ring: list[int] = [PLACEHOLDER_VALUE] * (num_faces + 1)
    face_one_ring:   list[int] = [PLACEHOLDER_VALUE] * num_faces

    if len(adjacent_faces) == 0:
        return [], []

    # Get first face and vertex
    face_one_ring[0] = compute_vertex_one_ring_first_face(F, vertex_index, adjacent_faces)
    vertex_one_ring[0] = find_next_vertex(F[face_one_ring[0], :], vertex_index)

    # Get remaining one ring faces and vertices
    for i in range(1, num_faces):
        # Get next vertex
        vertex_one_ring[i] = find_next_vertex(F[face_one_ring[i - 1], :],
                                              vertex_one_ring[i - 1])
        # Get next face
        for j in range(num_faces):
            f: int = adjacent_faces[j]
            if is_left_face(F[f, :], vertex_index, vertex_one_ring[i]):
                face_one_ring[i] = f

    # Get final vertex(same as first for closed loop)
    logger.info("Adding last vertex for face %s from vertex %s",
                F[face_one_ring[num_faces - 1], :], vertex_one_ring[num_faces - 1])
    vertex_one_ring[num_faces] = find_next_vertex(
        F[face_one_ring[num_faces - 1], :], vertex_one_ring[num_faces - 1])
    logger.info("Last vertex: %s", vertex_one_ring[num_faces])

    return vertex_one_ring, face_one_ring


class VertexCirculator:
    """
    Class to build circulators around vertices in VF representation
    """

    def __init__(self, F: MatrixNx3i) -> None:
        """
        Constructor for the vertex circulator from the faces of the mesh.

        :param F: [in] input mesh faces
        :return: None
        """
        # ***************
        # Private Members
        # ***************
        self.m_F: MatrixNx3i = F

        # Initialize adjacent faces list
        num_vertices: int = F.max() + 1
        self.m_all_adjacent_faces: list[list[int]] = compute_adjacent_faces(F)

        # Compute face and vertex one rings
        self.m_all_vertex_one_rings: list[list[int]] = [[] for _ in range(num_vertices)]
        self.m_all_face_one_rings: list[list[int]] = [[] for _ in range(num_vertices)]

        # TODO: there must be a better way of doing this.
        for i in range(num_vertices):
            # TODO: the function below did NOT modify a particular part of the list by reference...
            # which is bad for us.
            # TODO: meaning, we'll have to return something to store into m_all_vertex_one_rings and
            # m_all_face_one_rings
            self.m_all_vertex_one_rings[i], self.m_all_face_one_rings[i] = (
                compute_vertex_one_ring(F, i, self.m_all_adjacent_faces[i]))

    # ***************
    # Public Members
    # ***************

    def get_one_ring(self, vertex_index: int) -> tuple[list[int], list[int]]:
        """
        Get the one ring of a vertex.

        The one ring of both faces and vertices counter clockwise around the
        vertex are returned. For boundary vertices, the faces and vertices start
        at the right boundary and traverse the faces in order to the left
        boundary. For interior vertices, an arbitrary start face is chosen, and
        the vertex one ring is closed so that v_0 = v_n.

        Args:
            vertex_index:    [in]  index of the vertex to get the one ring for

        Returns:
            vertex_one_ring: [out] vertices ccw around the one ring
            face_one_ring:   [out] faces ccw around the one ring
        """

        # TODO: maybe just use a list of list of ints rather than a 2D NumPy array...
        vertex_one_ring: list[int] = self.m_all_vertex_one_rings[vertex_index]
        face_one_ring: list[int] = self.m_all_face_one_rings[vertex_index]
        return vertex_one_ring, face_one_ring
