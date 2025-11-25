"""
Class to build halfedge from VF

TODO The cleanest way to handle this is to fill boundaries with faces
and handle boundary cases with these to avoid invalid operations. The
interface should be chosen with care to balance elegance and versatility.

Mesh halfedge representation. Supports meshes with boundary and basic
topological information. Can be initialized from face topology information.
"""

import logging

from pyalgcon.core.common import (CHECK_VALIDITY,
                                  PLACEHOLDER_INDEX, Index,
                                  MatrixNx3i,
                                  find_face_vertex_index,
                                  is_manifold)
from pyalgcon.core.vertex_circulator import VertexCirculator

logger: logging.Logger = logging.getLogger(__name__)


def build_halfedge_to_edge_maps(opp: list[Index]
                                ) -> tuple[list[Index], list[tuple[Index, Index]]]:
    """
    Builds lists of halfedges to edges maps.

    Args:
        opp (list[Index]): in.

    Returns:
        he2e (list[Index]): halfedges to edge.
        e2he (list[tuple[Index, Index]]): edges to halfedges.
    """

    num_he: int = len(opp)
    he2e: list[Index] = [PLACEHOLDER_INDEX] * num_he
    e2he: list[tuple[Index, Index]] = []

    # Iterate over halfedges to build maps between halfedges and edges
    e_index: Index = 0
    for he in range(num_he):
        # Check if the halfedge is on the boundary
        is_boundary: bool = ((opp[he] < 0) or (opp[he] >= num_he))

        # Skip interior halfedges with lower index, but always process a boundary
        # halfedge
        if ((he >= opp[he]) or (is_boundary)):
            e2he.append((he, opp[he]))
            he2e[he] = e_index

            # Only valid for interior edges
            if not is_boundary:
                he2e[opp[he]] = e_index

            # Update current edge index
            e_index += 1

    return he2e, e2he


class Halfedge:
    """
    Boilerplate halfedge mesh representation.
    """
    # ************
    # CONSTRUCTORS
    # ************

    def __init__(self, F: MatrixNx3i) -> None:
        """
        TODO: deal with actually default constructor where Default trivial halfedge is made.
        Build halfedge mesh from mesh faces F with.

        :param F: faces to build halfedge mesh from
        :type F: MatrixNx3i
        """
        # **************************
        # Invalid index constructors (used across all Halfedge objects)
        # **************************

        self.INVALID_HALFEDGE_INDEX = -1
        self.INVALID_VERTEX_INDEX = -1
        self.INVALID_FACE_INDEX = -1
        self.INVALID_EDGE_INDEX = -1

        # Why does this start off with clear when nothing exists yet?
        # NOTE: ensuring that F is a matrix rather than a vector...
        assert F.ndim > 1

        # TODO: store F into m_F
        self.__F = F
        num_faces: int = F.shape[0]
        num_vertices: int = F.max() + 1
        num_halfedges: int = 3 * num_faces

        if CHECK_VALIDITY:
            if not is_manifold(F):
                logger.error("Input mesh is not manifold")
                self.clear()
                return

        # Build maps between corners and halfedges
        self.__corner_to_he: list[list[Index]]
        self.__he_to_corner: list[tuple[int, int]]
        self.__corner_to_he, self.__he_to_corner = self.build_corner_to_he_maps(num_faces)

        # Iterate over faces to build next, face, and to arrays
        self.__next: list[int] = [self.INVALID_HALFEDGE_INDEX] * num_halfedges
        self.__face: list[int] = [self.INVALID_FACE_INDEX] * num_halfedges
        self.__to: list[int] = [self.INVALID_VERTEX_INDEX] * num_halfedges
        self.__from: list[int] = [self.INVALID_VERTEX_INDEX] * num_halfedges
        for face_index in range(num_faces):
            for i in range(3):
                current_he: Index = self.__corner_to_he[face_index][i]
                next_he: Index = self.__corner_to_he[face_index][(i + 1) % 3]
                self.__next[current_he] = next_he
                self.__face[current_he] = face_index
                self.__to[current_he] = F[face_index, (i + 2) % 3]
                self.__from[current_he] = F[face_index, (i + 1) % 3]

        # Build out and f2he arrays
        self.__out: list[int] = [-1] * num_vertices
        self.__f2he: list[int] = [-1] * num_faces
        for he_index in range(num_halfedges):
            self.__out[self.__to[he_index]] = self.__next[he_index]
            self.__f2he[self.__face[he_index]] = he_index

        # Iterate over vertices to build opp using a vertex circulator
        # Note that this is the main difficulty in constructing halfedge from VF
        vertex_circulator = VertexCirculator(F)
        self.__opp: list[int] = [-1] * (3 * num_faces)
        for vertex_index in range(num_vertices):
            # Get vertex one ring
            vertex_one_ring: list[int]
            face_one_ring: list[int]
            vertex_one_ring, face_one_ring = vertex_circulator.get_one_ring(vertex_index)

            # Determine if we are in a boundary case
            is_boundary: bool = (vertex_one_ring[0] != vertex_one_ring[-1])  # Front != Back
            num_adjacent_faces: Index = len(face_one_ring)
            num_interior_edges: Index
            if is_boundary:
                num_interior_edges = num_adjacent_faces - 1
            else:
                num_interior_edges = num_adjacent_faces

            # Build opposite arrays
            for i in range(num_interior_edges):
                # Get current face prev (cw) halfedge from the vertex
                fi: Index = face_one_ring[i]
                # TODO: confirm I'm doing correct slicing like Eigen .row()
                fi_vertex_index: Index = find_face_vertex_index(F[fi, :], vertex_index)
                current_he: Index = self.__corner_to_he[fi][(fi_vertex_index + 1) % 3]

                # Get next (ccw) face next (ccw) halfedge from the vertex
                fj: Index = face_one_ring[(i + 1) % num_adjacent_faces]
                fj_vertex_index: Index = find_face_vertex_index(F[fj, :], vertex_index)
                opposite_he: Index = self.__corner_to_he[fj][(fj_vertex_index + 2) % 3]

                # Assign opposite halfedge
                self.__opp[current_he] = opposite_he

        # Build maps between edges and halfedges
        self.__he2e: list[Index]
        self.__e2he: list[tuple[Index, Index]]
        self.__he2e, self.__e2he = build_halfedge_to_edge_maps(self.__opp)

        # Set sizes
        self.__num_halfedges: int = num_halfedges
        self.__num_faces: int = num_faces
        self.__num_vertices: int = num_vertices
        self.__num_edges: int = len(self.__e2he)

        #  Check validity
        if logger.getEffectiveLevel() == logging.DEBUG:
            if not self.is_valid():
                logger.error("Could not build halfedge")
                self.clear()
                return

    # **************
    # Element counts
    # **************

    @property
    def num_halfedges(self) -> Index:
        """Retrieves num_halfedges."""
        return self.__num_halfedges

    @property
    def num_faces(self) -> Index:
        """Retrieves num_faces."""
        return self.__num_faces

    @property
    def num_vertices(self) -> Index:
        """Retrieves num_vertices."""
        return self.__num_vertices

    @property
    def num_edges(self) -> Index:
        """Retrieves num_edges."""
        return self.__num_edges

    # *********
    # Adjacency
    # *********
    def next_halfedge(self, he: Index) -> Index:
        """
        Retrieves self.next at index he.

        :param he: [in] index to retrieve next Halfedge
        :return: index of next Halfedge
        """
        if not self.is_valid_halfedge_index(he):
            return self.INVALID_HALFEDGE_INDEX
        return self.__next[he]

    def opposite_halfedge(self, he: Index) -> Index:
        if not self.is_valid_halfedge_index(he):
            return self.INVALID_HALFEDGE_INDEX
        return self.__opp[he]

    def halfedge_to_face(self, he: Index) -> Index:
        if not self.is_valid_halfedge_index(he):
            return self.INVALID_FACE_INDEX
        return self.__face[he]

    def halfedge_to_head_vertex(self, he: Index) -> Index:
        if not self.is_valid_halfedge_index(he):
            return self.INVALID_VERTEX_INDEX
        return self.__to[he]

    def halfedge_to_tail_vertex(self, he: Index) -> Index:
        if not self.is_valid_halfedge_index(he):
            return self.INVALID_VERTEX_INDEX
        return self.__from[he]

    # ********************
    # Edge Representations
    # ********************

    def halfedge_to_edge(self, he: Index) -> Index:
        """Gets halfedge to edge at index"""
        if not self.is_valid_halfedge_index(he):
            return self.INVALID_EDGE_INDEX
        return self.__he2e[he]

    def edge_to_halfedge(self, e: Index) -> tuple[Index, Index]:
        """Gets edge to halfedge at index"""
        return self.__e2he[e]

    def edge_to_first_halfedge(self, e: Index) -> Index:
        """Gets edge to first halfedge at index"""
        # NOTE: equivalent to C++ pair .first
        return self.__e2he[e][0]

    def edge_to_second_halfedge(self, e: Index) -> Index:
        """Gets edge to second halfedge at index"""
        # NOTE: equivalent to C++ pair .second
        return self.__e2he[e][1]

    @property
    def halfedge_to_edge_map(self) -> list[Index]:
        """Gets halfedge to edge"""
        return self.__he2e

    @property
    def edge_to_halfedge_map(self) -> list[tuple[Index, Index]]:
        """Gets edge to halfedge"""
        return self.__e2he

    # **********************
    # Attributes for Testing
    # **********************
    @property
    def f2he(self) -> list[int]:
        """Gets face to halfedge"""
        return self.__f2he

    @property
    def face(self) -> list[int]:
        """Gets face"""
        return self.__face

    @property
    def _from(self) -> list[int]:
        """Gets halfedge from"""
        return self.__from

    @property
    def next(self) -> list[int]:
        """Gets halfedge next"""
        return self.__next

    @property
    def opp(self) -> list[int]:
        """Gets halfedge opposite"""
        return self.__opp

    @property
    def out(self) -> list[int]:
        """Gets halfedge out"""
        return self.__out

    @property
    def to(self) -> list[int]:
        """Gets halfedge to"""
        return self.__to

    # ******************
    # Element predicates
    # ******************

    def is_boundary_edge(self, e: Index) -> bool:
        """
        Checks if halfedge is boundary edge
        """
        if not self.is_valid_halfedge_index(self.edge_to_first_halfedge(e)):
            return True
        if not self.is_valid_halfedge_index(self.edge_to_second_halfedge(e)):
            return True
        return False

    def is_boundary_halfedge(self, he: Index) -> bool:
        """
        Checks if halfedge is boundary halfedge
        """
        return self.is_boundary_edge(self.halfedge_to_edge(he))

    def build_boundary_edge_list(self) -> list[Index]:
        """
        :return boundary_halfedges: [out]
        """
        boundary_edges: list[Index] = []

        for ei in range(self.__num_edges):
            if self.is_boundary_edge(ei):
                boundary_edges.append(ei)

        return boundary_edges

    def build_boundary_halfedge_list(self) -> list[Index]:
        """
        :return boundary_halfedges: [out]
        """
        boundary_halfedges: list[Index] = []

        for hi in range(self.__num_halfedges):
            if self.is_boundary_halfedge(hi):
                boundary_halfedges.append(hi)

        return boundary_halfedges

    @property
    def corner_to_he(self) -> list[list[int]]:
        """
        Gets corner to halfedge map
        :return: corner_to_he
        """
        return self.__corner_to_he

    @property
    def he_to_corner(self) -> list[tuple[int, int]]:
        """
        Gets halfedge to corner map
        :return: he_to_corner
        """
        return self.__he_to_corner

    def clear(self) -> None:
        self.__next.clear()
        self.__opp.clear()
        self.__he2e.clear()
        self.__e2he.clear()
        self.__to.clear()
        self.__from.clear()
        self.__face.clear()
        self.__out.clear()
        self.__f2he.clear()
        self.__F.resize(0, 0)

    def build_corner_to_he_maps(self, num_faces: Index) -> tuple[list[list[Index]],
                                                                 list[tuple[Index, Index]]]:
        """
        Builds list for corner to he and he to corner maps.

        :param num_faces: [in]

        :returns:
            - corner_to_he (list[list[Index]]): [out]

            - he_to_corner (list[tuple[Index, Index]]): out
        """

        # FIXME: resizing probably all works, but find a way that's neater than whatever is below
        corner_to_he: list[list[Index]] = [[] for _ in range(num_faces)]
        he_to_corner: list[tuple[Index, Index]] = [(PLACEHOLDER_INDEX, PLACEHOLDER_INDEX)
                                                   for _ in range(3 * num_faces)]

        # Iterate over faces to build corner to he maps
        he_index: Index = 0

        for face_index in range(num_faces):
            corner_to_he[face_index] = [PLACEHOLDER_INDEX] * 3

            for i in range(3):
                # Assign indices
                corner_to_he[face_index][i] = he_index
                he_to_corner[he_index] = (face_index, i)

                # Update current face index
                he_index += 1

        return corner_to_he, he_to_corner

    # *********************
    # Index validity checks
    # *********************
    def is_valid_halfedge_index(self, he: Index) -> bool:
        if (he < 0):
            return False
        if (he >= self.num_halfedges):
            return False
        return True

    def is_valid_vertex_index(self, vertex_index: Index) -> bool:
        if (vertex_index < 0):
            return False
        if (vertex_index >= self.num_vertices):
            return False
        return True

    def is_valid_face_index(self, face_index: Index) -> bool:
        if face_index < 0:
            return False
        if face_index >= self.num_faces:
            return False
        return True

    def is_valid_edge_index(self, edge_index: Index) -> bool:
        if edge_index < 0:
            return False
        if edge_index >= self.num_edges:
            return False
        return True

    def is_valid(self) -> bool:
        if len(self.__next) != self.num_halfedges:
            logger.error("next domain not in bijection with halfedges")
            return False
        if len(self.__opp) != self.num_halfedges:
            logger.error("opp domain not in bijection with halfedges")
            return False
        if len(self.__he2e) != self.num_halfedges:
            logger.error("he2e domain not in bijection with halfedges")
            return False
        if len(self.__to) != self.num_halfedges:
            logger.error("to domain not in bijection with halfedges")
            return False
        if len(self.__from) != self.num_halfedges:
            logger.error("from domain not in bijection with halfedges")
            return False
        if len(self.__face) != self.num_halfedges:
            logger.error("face domain not in bijection with halfedges")
            return False
        if len(self.__e2he) != self.num_edges:
            logger.error("e2he domain not in bijection with edges")
            return False
        if len(self.__out) != self.num_vertices:
            logger.error("out domain not in bijection with vertices")
            return False
        if len(self.__f2he) != self.num_faces:
            logger.error("f2he domain not in bijection with faces")
            return False
        if self.__F.shape[0] != self.num_faces:
            logger.error("F rows not in bijection with faces")
            return False

        return True
