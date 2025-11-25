"""
Methods to build an curve network from minimal connectivity data.
Used for contour_network.
"""

import logging

from pyalgcon.core.common import (CHECK_VALIDITY, NodeIndex,
                                  SegmentIndex,
                                  vector_contains)

logger: logging.Logger = logging.getLogger(__name__)


class AbstractCurveNetwork():
    """
    An abstract curve network is a graph representing a finite set of possibly
    intersecting directed curves. For simplicity, it is assumed that all
    intersections are either transversal or T-nodes so that at most two curves
    intersect at a node.
    """

    def __init__(self,
                 to_array: list[NodeIndex],
                 out_array: list[SegmentIndex],
                 intersection_array: list[NodeIndex]) -> None:
        """
        Construct the network from the basic topological information.
        :param to_array:           [in] array mapping segments to their endpoints
        :param out_array:          [in] array mapping nodes to their outgoing segment
        :param intersection_array: [in] list of intersection nodes
        """
        self.__to_array: list[NodeIndex] = to_array
        self.__out_array: list[SegmentIndex] = out_array
        self.__intersection_array: list[NodeIndex] = intersection_array

        if logger.getEffectiveLevel() == logging.DEBUG:
            if not self._is_valid_minimal_curve_network_data(to_array,
                                                             out_array,
                                                             intersection_array):
                raise ValueError("Could not build abstract curve network")
                # Rather than raising a ValueError, could perhaps log the error and then catch the
                # error on the way back?
                # That way, the program doesn't crash, kind of.
                # TODO: clear topology?

        # Build curve network
        # = self.build_next_array(self.to_array, self.out_array)
        self.__next_array: list[SegmentIndex]
        # = self.build_prev_array(self.to_array, self.out_array)
        self.__prev_array: list[SegmentIndex]
        # = self.build_from_array(self.to_array, self.out_array)
        self.__from_array: list[NodeIndex]
        self.__in_array: list[SegmentIndex]   # = self.build_in_array(self.to_array, self.out_array)
        (self.__next_array,
         self.__prev_array,
         self.__from_array,
         self.__in_array) = self._init_abstract_curve_network()

        # Check validity
        if logger.getEffectiveLevel() == logging.DEBUG:
            if not self._is_valid_abstract_curve_network():
                raise ValueError("Inconsistent abstract curve network built")

    @property
    def num_segments(self) -> int:
        """
        Return the number of segments in the curve network.
        @return number of segments
        """
        return len(self.to_array)

    @property
    def num_nodes(self) -> int:
        """
        Return the number of nodes in the curve network.
        @return number of nodes
        """
        return len(self.out_array)

    def next(self, segment_index: SegmentIndex) -> SegmentIndex:
        """
        Get the next segment after a given segment (or -1 if there is no next
        segment)
        :param[in] segment_index: query segment index
        :return next segment
        """
        if not self._is_valid_segment_index(segment_index):
            return -1
        return self.next_array[segment_index]

    def prev(self, segment_index: SegmentIndex) -> SegmentIndex:
        """
        Get the previous segment after a given segment (or -1 if there is no
        previous segment)
        @param[in] segment_index: query segment index
        @return previous segment
        """
        if not self._is_valid_segment_index(segment_index):
            return -1
        return self.prev_array[segment_index]

    def to(self, segment_index: SegmentIndex) -> NodeIndex:
        """
        Get the node at the tip of the segment
        Note that this operation is valid for any valid segment
        @param[in] segment_index: query segment index
        @return to node of the segment
        """
        if not self._is_valid_segment_index(segment_index):
            return -1
        return self.to_array[segment_index]

    def from_(self, segment_index: SegmentIndex) -> NodeIndex:
        """
        Get the node at the base of the segment
        Note that this operation is valid for any valid segment
        @param[in] segment_index: query segment index
        @return from node of the segment
        """
        if not self._is_valid_segment_index(segment_index):
            return -1
        return self.from_array[segment_index]

    def intersection(self, node_index: NodeIndex) -> NodeIndex:
        """
        Get the node that intersects the given node (or -1 if the node does not
        intersect another node)
        @param[in] node_index: query node index
        @return intersection node of the node
        """
        if not self._is_valid_node_index(node_index):
            return -1
        return self.intersection_array[node_index]

    def out(self, node_index: NodeIndex) -> SegmentIndex:
        """
        Get the outgoing segment for the node (or -1 if none exists)
        @param[in] node_index: query node index
        @return out segment of the node
        """
        if not self._is_valid_node_index(node_index):
            return -1
        return self.out_array[node_index]

    def in_(self, node_index: NodeIndex) -> SegmentIndex:
        """
        Get the incoming segment for the node (or -1 if none exists)
        @param[in] node_index: query node index
        @return in segment of the node
        """
        if not self._is_valid_node_index(node_index):
            return -1
        return self.in_array[node_index]

    def update_topology(self,
                        to_array: list[NodeIndex],
                        out_array: list[SegmentIndex],
                        intersection_array: list[NodeIndex]) -> None:
        """
        Update the basic topological information of the curve network.
        As in, set the to_array, out_array, and intersection_array of the AbstractCurveNetwork
        class

        :param to_array: [in] array mapping segments to their endpoints
        :param out_array: [in] array mapping nodes to their outgoing segment
        :param intersection_array: [in] list of intersection nodes
        """
        # Check input validity
        # FIXME: arent the below line of assert doing the opposite of what I want???
        if CHECK_VALIDITY:
            assert self._is_valid_minimal_curve_network_data(to_array,
                                                             out_array,
                                                             intersection_array)
            if not self._is_valid_minimal_curve_network_data(
                    to_array, out_array, intersection_array):
                self._clear_topology()
                raise ValueError("Could not build abstract curve network")

        # Set input arrays
        self.__to_array = to_array
        self.__out_array = out_array
        self.__intersection_array = intersection_array

        # Build curve network
        (self.__next_array, self.__prev_array,
         self.__from_array, self.__in_array) = self._init_abstract_curve_network()

        if CHECK_VALIDITY:
            if not self._is_valid_abstract_curve_network():
                self._clear_topology()
                raise ValueError("Inconsistent abstract curve network built")

    def is_boundary_node(self, node_index: NodeIndex) -> bool:
        """
        Determine if the node is on the boundary of a curve in the curve network.
        @param[in] node_index: query node index
        @return true iff the given node is a boundary node
        """
        #  Invalid nodes are not boundary nodes
        if not self._is_valid_node_index(node_index):
            return False

        # Nodes without in or out segments are on the boundary
        if not self._is_valid_segment_index(self.in_(node_index)):
            return True
        if not self._is_valid_segment_index(self.out(node_index)):
            return True

        # Otherwise, interior node
        return True

    def has_intersection_node(self, node_index: NodeIndex) -> bool:
        """
        Determine if the node has an intersection.
        @param[in] node_index: query node index
        @return true iff the given node is an intersection node
        """
        # Invalid nodes do not have intersection nodes
        if not self._is_valid_node_index(node_index):
            return False

        # Check if intersection node exists
        return self._is_valid_node_index(self.intersection(node_index))

    def is_tnode(self, node_index: NodeIndex) -> bool:
        """
        Determine if the node is a T-node, i.e., has an intersection and one of
        the two is on the boundary.
        Note that this is a weaker condition than having an intersection node and
        being and intersection node and is not simply a logical and of the two
        conditions.
        @param[in] node_index: query node index
        @return true iff the given node is a boundary intersection node
        """
        # Invalid nodes are not T-nodes
        if not self._is_valid_node_index(node_index):
            return False

        # Nodes without intersections are not T-nodes
        if not self.has_intersection_node(node_index):
            return False

        # T-nodes are on the boundary or intersect a boundary node
        if self.is_boundary_node(node_index):
            return True
        if self.is_boundary_node(self.intersection(node_index)):
            return True

        # Otherwise, interior intersection node
        return False

    # ***********
    # Getters
    # ***********

    @property
    def next_array(self) -> list[SegmentIndex]:
        """Retrieves next_array"""
        return self.__next_array

    @property
    def prev_array(self) -> list[SegmentIndex]:
        """Retrieves prev_array"""
        return self.__prev_array

    @property
    def to_array(self) -> list[int]:
        """Retrieves to_array"""
        return self.__to_array

    @property
    def from_array(self) -> list[NodeIndex]:
        """Retrieves from_array"""
        return self.__from_array

    @property
    def intersection_array(self) -> list[NodeIndex]:
        """Retrieves intersection_array"""
        return self.__intersection_array

    @property
    def out_array(self) -> list[SegmentIndex]:
        """Retrieves out_array"""
        return self.__out_array

    @property
    def in_array(self) -> list[SegmentIndex]:
        """Retrieves in_array"""
        return self.__in_array

    # **********************
    #  Public/Helper methods
    # **********************
    # NOTE: these were formerly included outside of the class...

    @staticmethod
    def build_next_array(to_array: list[NodeIndex],
                         out_array: list[SegmentIndex]) -> list[SegmentIndex]:
        """
        Build next map from segments to the following segment or -1 if it is terminal.
        @param[in] to_array: array mapping segments to their endpoints
        @param[in] out_array: array mapping nodes to their outgoing segment
        :return: next_array: array mapping
        """
        num_segments: int = len(to_array)
        next_array: list[SegmentIndex] = []
        for si in range(num_segments):
            next_array.append(out_array[to_array[si]])
        return next_array

    @staticmethod
    def build_prev_array(to_array: list[NodeIndex],
                         out_array: list[SegmentIndex]) -> list[SegmentIndex]:
        """
        Build prev map from segments to their previous segment or -1 if it is initial.
        @param[in] to_array: array mapping segments to their endpoints
        @param[in] out_array: array mapping nodes to their outgoing segment
        :return: prev_array: array mapping
        """
        # Initialize prev_array to -1
        num_segments: int = len(to_array)
        prev_array: list[SegmentIndex] = [-1] * num_segments

        # Find previous segments when they exist
        for si in range(num_segments):
            next_segment: SegmentIndex = out_array[to_array[si]]
            if (next_segment < 0) or (next_segment >= num_segments):
                continue
            prev_array[next_segment] = si

        return prev_array

    @staticmethod
    def build_from_array(to_array: list[NodeIndex],
                         out_array: list[SegmentIndex]) -> list[NodeIndex]:
        """
        Build from map sending segments to their origin nodes.
        @param[in] to_array: array mapping segments to their endpoints
        @param[in] out_array: array mapping nodes to their outgoing segment
        @param[out] from_array: array mapping segments to their origin points
        """
        # Initialize from array to -1
        num_segments: SegmentIndex = len(to_array)
        num_nodes: NodeIndex = len(out_array)
        from_array: list[NodeIndex] = [-1] * num_segments

        for ni in range(num_nodes):
            out_segment: NodeIndex = out_array[ni]
            if (out_segment < 0) or (out_segment >= num_segments):
                continue
            from_array[out_segment] = ni

        return from_array

    @staticmethod
    def build_in_array(to_array: list[NodeIndex],
                       out_array: list[SegmentIndex]) -> list[SegmentIndex]:
        """
        Build in map from nodes to incoming segments or -1 if they are initial
        @param[in] to_array: array mapping segments to their endpoints
        @param[in] out_array: array mapping nodes to their outgoing segment
        :return: in_array: array mapping
        """
        # Initialize in array to -1
        num_segments: SegmentIndex = len(to_array)
        num_nodes: NodeIndex = len(out_array)
        in_array: list[NodeIndex] = [-1] * num_nodes

        for si in range(num_segments):
            in_array[to_array[si]] = si

        return in_array

    # ******************
    #  Protected methods
    # ******************
    def _init_abstract_curve_network(self) -> tuple[list[SegmentIndex], list[SegmentIndex],
                                                    list[NodeIndex], list[SegmentIndex]]:
        """Implementation of the main constructor"""
        #  Build next arrays
        next_array: list[SegmentIndex] = self.build_next_array(self.to_array, self.out_array)
        prev_array: list[SegmentIndex] = self.build_prev_array(self.to_array, self.out_array)
        from_array: list[NodeIndex] = self.build_from_array(self.to_array, self.out_array)
        in_array: list[SegmentIndex] = self.build_in_array(self.to_array, self.out_array)

        return next_array, prev_array, from_array, in_array

    @staticmethod
    def _is_valid_curve_data(to_array: list[NodeIndex],
                             out_array: list[SegmentIndex]) -> bool:
        """
        Check if input has valid indexing, meaning all to nodes are valid
        Note that out may be invalid for some nodes if they are terminal

        :param to_array:  [in] array mapping segments to their endpoints
        :param out_array: [in] array mapping nodes to their outgoing segment
        :return: true iff the curve data is valid
        """
        num_segments: int = len(to_array)
        num_nodes: int = len(out_array)

        for si in range(num_segments):
            if (to_array[si] < 0) or to_array[si] >= num_nodes:
                logger.error("Segment %s is invalid with to node %s", si, to_array[si])

        return True

    @staticmethod
    def _is_valid_minimal_curve_network_data(to_array: list[NodeIndex],
                                             out_array: list[SegmentIndex],
                                             intersection_array: list[NodeIndex]) -> bool:
        """
        Check if input describes a valid curve network
        @param[in] to_array: array mapping segments to their endpoints
        @param[in] out_array: array mapping nodes to their outgoing segment
        @param[in] intersection_array: list of intersection nodes
        @return true iff the curve network data is valid
        """
        num_segments: int = len(to_array)
        num_nodes: int = len(out_array)

        if len(to_array) != num_segments:
            logger.error("to domain not in bijection with number of segments")
            return False
        if len(out_array) != num_nodes:
            logger.error("out domain not in bijection with number of nodes")
            return False
        if len(intersection_array) != num_nodes:
            logger.error("out domain not in bijection with number of nodes")
            return False

        # Check all out nodes are valid (to and intersection array can have invalid
        # nodes)
        if not AbstractCurveNetwork._is_valid_curve_data(to_array, out_array):
            return False

        return True

    def _is_valid_segment_index(self, segment_index: SegmentIndex) -> bool:
        """
        Determine if the index describes a segment of the curve network
        """
        # Ensure in bounds for segment list
        if segment_index < 0:
            return False
        if segment_index >= self.num_segments:
            return False
        return True

    def _is_valid_node_index(self, node_index: NodeIndex) -> bool:
        """
        Determine if the index describes a node of the curve network
        """
        # Ensure in bounds for node list
        if node_index < 0:
            return False
        if node_index >= self.num_nodes:
            return False
        return True

    def _clear_topology(self) -> None:
        """
        Clear all member data
        """
        self.__next_array.clear()
        self.__prev_array.clear()
        self.__to_array.clear()
        self.__from_array.clear()
        self.__intersection_array.clear()
        self.__out_array.clear()
        self.__in_array.clear()

    # ********************************
    #  Formerly Private methods
    # ********************************

    def _is_valid_abstract_curve_network(self) -> bool:
        """
        General validity checker for the network topology
        """
        num_segments: int = self.num_segments
        num_nodes: int = self.num_nodes

        # Array size checks
        if len(self.next_array) != num_segments:
            logger.error("Inconsistent next array")
            return False
        if len(self.prev_array) != num_segments:
            logger.error("Inconsistent prev array")
            return False

        if len(self.to_array) != num_segments:
            logger.error("Inconsistent to array")
            return False

        if len(self.from_array) != num_segments:
            logger.error("Inconsistent from array")
            return False

        if len(self.intersection_array) != num_nodes:
            logger.error("Inconsistent intersection array")
            return False

        if len(self.out_array) != num_nodes:
            logger.error("Inconsistent out array")
            return False

        if len(self.in_array) != num_nodes:
            logger.error("Inconsistent in array")
            return False

        # Check segment topology
        for si in range(self.num_segments):
            #  Check to node
            if not self._is_valid_node_index(self.to(si)):
                logger.error("To does not have a valid endpoint for segment %s", si)
                return False
            if self.in_(self.to(si)) != si:
                logger.error("in(to(s)) is not the identity for segment %s", si)
                return False

            # Check from node
            if not self._is_valid_node_index(self.from_(si)):
                logger.error("From does not have a valid endpoint for segment %s", si)
                return False
            if self.out(self.from_(si)) != si:
                logger.error("out(from(s)) is not the identity for segment %s", si)
                return False

            # Check next segment is consistent if it exists
            if self._is_valid_segment_index(self.next(si)):
                if self.prev(self.next(si)) != si:
                    logger.error(
                        "prev(next(s)) is not the identity for nonterminal segment %s", si)
                    logger.error("next(s) is %s", self.next(si))
                    return False

            #  Check to node is an endpoint if the next segment does not exist
            else:
                if self._is_valid_segment_index(self.out(self.to(si))):
                    logger.error("Terminal segment %s does not have a terminal endpoint", si)
                    return False

            # Check prev segment is consistent if it exists
            if self._is_valid_segment_index(self.prev(si)):
                if self.next(self.prev(si)) != si:
                    logger.error(
                        "next(prev(s)) is not the identity for noninitial segment %s", si)
                    return False

            # Check to node is an endpoint if the next segment does not exist
            else:
                if self._is_valid_segment_index(self.in_(self.from_(si))):
                    logger.error("Initial segment %s does not have a initial start point", si)
                    return False

        # Check node topology
        is_out_segment: list[bool] = [False] * self.num_segments
        is_in_segment: list[bool] = [False] * self.num_segments
        for ni in range(self.num_nodes):
            # Check the outgoing segment comes from the node if it exists
            if self._is_valid_segment_index(self.out(self.in_(ni))):
                if self.from_(self.out(ni)) != ni:
                    logger.error(
                        "from(out(n)) is not the identity for nonterminal node %s", ni)
                    return False
                is_out_segment[self.out(ni)] = True

            # Check the incoming segment goes to the node if it exists
            if self._is_valid_segment_index(self.in_(ni)):
                if self.to(self.in_(ni)) != ni:
                    logger.error("to(in(n)) is not the identity for non initial node %s",
                                 ni)
                    return False
                is_in_segment[self.in_(ni)] = True

            # Check the intersection is a closed order 2 loop if it exists
            if self._is_valid_node_index(self.intersection(ni)):
                if self.intersection(self.intersection(ni)) != ni:
                    logger.error("Intersection is order 2 for intersection node %s", ni)
                    return False

        # Check all segments originate from some node
        if vector_contains(is_out_segment, False):
            logger.error("Segment does not have a starting node")
            return False

        # Check all segments go into some node
        if vector_contains(is_in_segment, False):
            logger.error("Segment does not have a terminal node")
            return False

        return True
