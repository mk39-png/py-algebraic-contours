"""
Methods to compute the boundaries of a mesh
"""
import numpy as np

from pyalgcon.core.common import (
    Index, convert_boolean_array_to_index_vector)
from pyalgcon.core.halfedge import Halfedge


def compute_face_boundary_edges(F: np.ndarray) -> list[tuple[int, int]]:
    """
    Given a mesh, compute the edges on the boundary.

    :param F: [in] mesh faces

    :return face_boundary_edges: edges of the triangles that are boundaries, indexed by face 
    and opposite corner local face vertex index
    :rtype: (list[tuple[int, int]])
    """
    assert F.dtype == np.int64

    # Build halfedge for the mesh
    halfedge: Halfedge = Halfedge(F)
    he_to_corner: list[tuple[Index, Index]] = halfedge.he_to_corner

    # Get boundary halfedges
    boundary_halfedges: list[Index] = halfedge.build_boundary_halfedge_list()

    # Get boundary face corners opposite halfedge
    face_boundary_edges: list[tuple[int, int]] = []
    for boundary_halfedge in boundary_halfedges:
        face_boundary_edges.append(he_to_corner[boundary_halfedge])

    return face_boundary_edges


# def compute_boundary_vertices(F: np.ndarray) -> list[int]:
#     """
#     Given a mesh, compute the vertices on the boundary.

#     Args:
#         F: mesh_faces

#     Returns:
#         boundary_vertices: vertices of the mesh on the boundary

#     """
#     assert F.dtype == np.int64

#     # Get face boundary edges
#     face_boundary_edges: list[tuple[int, int]] = compute_face_boundary_edges(F)

#     # Get boolean array of boundary indices
#     num_vertices: int = F.max() + 1
#     is_boundary_vertex: list[bool] = [False] * num_vertices

#     for i, boundary_edge in enumerate(face_boundary_edges):
#         # Mark boundary edge endpoints as boundary vertices
#         face_index: int = face_boundary_edges[i][0]
#         face_vertex_index: int = face_boundary_edges[i][1]
#         is_boundary_vertex[F[face_index, (face_vertex_index + 1) % 3]] = True
#         is_boundary_vertex[F[face_index, (face_vertex_index + 2) % 3]] = True

#     # Convert boolean array to index vector
#     boundary_vertices: list[int] = convert_boolean_array_to_index_vector(
#         is_boundary_vertex)

#     # TODO: maybe get rid of the function below? Aren't boundaries already signed? At least how the implementation is in Python?
#     # boundary_vertices = convert_unsigned_vector_to_signed(
#     #     unsigned_boundary_vertices)

#     return boundary_vertices
