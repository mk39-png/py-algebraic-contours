"""
Methods to compute the boundaries of a mesh
"""
import numpy as np

from pyalgcon.core.common import Index
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
