"""
Methods to compute a simple planar curve network from annotated plane curve
soup.
"""
import logging
import pathlib

import numpy as np
import polyscope
import svg

from pyalgcon.contour_network.discretize import discretize_curve_segments
from pyalgcon.contour_network.intersection_data import IntersectionData
from pyalgcon.contour_network.write_output import (add_curve_to_svg,
                                                   write_planar_curve_segment,
                                                   write_planar_point)
from pyalgcon.core.abstract_curve_network import AbstractCurveNetwork
from pyalgcon.core.common import (CHECK_VALIDITY,
                                  INLINE_TESTING_ENABLED_CONTOUR_NETWORK,
                                  PLACEHOLDER_VALUE, Color, MatrixXf,
                                  NodeIndex, PlanarPoint1d, SegmentIndex,
                                  SpatialVector1d, Vector3f,
                                  compare_eigen_numpy_matrix,
                                  convert_nested_vector_to_matrix,
                                  convert_polylines_to_edges, todo,
                                  unimplemented)
from pyalgcon.core.conic import Conic
from pyalgcon.core.generate_colormap import generate_random_category_colormap
from pyalgcon.core.rational_function import (CurveDiscretizationParameters,
                                             RationalFunction)
from pyalgcon.debug.debug import SPOT_FILEPATH
from pyalgcon.utils.projected_curve_networks_utils import (
    NodeGeometry, SegmentGeometry, SVGOutputMode,
    build_projected_curve_network_without_intersections,
    compare_list_node_geometry, connect_segment_intersections,
    is_valid_next_prev_pair, remove_redundant_intersections,
    split_segments_at_cusps, split_segments_at_intersections)

logger: logging.Logger = logging.getLogger(__name__)


# ***********************
# Helper Methods
# ***********************

class SegmentChainIterator():
    """
    Iterator over contour segment chains until FIXME is found.
    """

    def __init__(self,
                 parent: "ProjectedCurveNetwork",
                 segment_index: SegmentIndex) -> None:
        """
        Constructor for SegmentChainIterator class.
        """
        self.__parent: ProjectedCurveNetwork = parent
        self.__current_segment_index: SegmentIndex = segment_index
        self.__is_end_of_chain: bool = False
        self.__is_reverse_end_of_chain: bool = False
        logger.debug("Iterator initialized to segment %s", self.__current_segment_index)

    def increment(self) -> None:
        """
        Replaces C++ operator++().
        Updates current_segment_index and is_end_of_chain.
        """
        # TODO: could reimplemnt as an iterator

        # Check if off end of chain
        if self.__is_end_of_chain or self.__is_reverse_end_of_chain:
            return

        # Check if next is end of chain (i.e. the next node is not a knot node)
        to_node: NodeIndex = self.__parent.to(self.__current_segment_index)
        if not self.__parent.is_knot_node(to_node):
            self.__current_segment_index = -1
            self.__is_end_of_chain = True
        else:
            self.__current_segment_index = self.__parent.next(self.__current_segment_index)

        logger.debug("Iterator moved to segment %s", self.__current_segment_index)

    def decrement(self) -> None:
        """
        Replaces C++ operator--()
        Updates current_segment_index and is_reverse_end_of_chain.
        """
        # Check if off end of chain
        if self.__is_end_of_chain or self.__is_reverse_end_of_chain:
            return

        # Check if prev is reverse end of chain (i.e. the prev node is not a knot node)
        from_node: NodeIndex = self.__parent.from_(self.__current_segment_index)

        if not self.__parent.is_knot_node(from_node):
            self.__current_segment_index = -1
            self.__is_reverse_end_of_chain = True
        else:
            self.__current_segment_index = self.__parent.prev(self.__current_segment_index)

        logger.debug("Iterator moved to segment %s", self.__current_segment_index)

    @property
    def at_end_of_chain(self) -> bool:
        """
        :returns: is_end_of_chain
        """
        return self.__is_end_of_chain

    @property
    def at_reverse_end_of_chain(self) -> bool:
        """
        :returns: is_reverse_end_of_chain
        """
        return self.__is_reverse_end_of_chain

    @property
    def current_segment_index(self) -> SegmentIndex:
        """
        Replaces C++ operator*()

        :returns: current_segment_index
        """
        return self.__current_segment_index


# ***********************
# Projected Curve Network
# ***********************

class ProjectedCurveNetwork(AbstractCurveNetwork):
    """
    A projected curve network is a curve network of intersecting planar curves
    arising from the projection of spatial curves to the xy plane.
    :ivar segments: list[SegmentGeometry]
    :ivar nodes: list[NodeGeometry]
    :ivar chain_start_nodes: list[NodeIndex]
    """

    def __init__(self,
                 parameter_segments: list[Conic],
                 spatial_segments: list[RationalFunction],  # RationalFunction<4, 3>
                 planar_segments: list[RationalFunction],  # RationalFunction<4, 2>
                 segment_labels: list[dict[str, int]],
                 chains: list[list[int]],
                 chain_labels: list[int],
                 interior_cusps: list[list[float]],
                 has_cusp_at_base: list[bool],
                 intersections: list[list[float]],
                 intersection_indices: list[list[int]],
                 intersection_data: list[list[IntersectionData]],
                 num_intersections: int
                 ) -> None:
        """
        Construct the curve network from the relevant annotated geometric information.
        :param parameter_segments:   [in] uv domain quadratic curves parametrizing the other curves
        :param spatial_segments:     [in] spatial rational curves before projection
        :param planar_segments:      [in] planar rational curves after projection
        :param chain_labels:         [in] list of maps of labels for each segment 
                                          (e.g., patch label)
        :param chains:               [in] list of lists of chained curve indices
        :param chain_labels:         [in] list of chain labels for each segment
        :param interior_cusps:       [in] list of lists of cusp points per segment
        :param has_cusp_at_base:     [in] list of bools per segment indicating if the segment base 
                                          node is a cusp
        :param intersections:        [in] list of lists of intersection points per segment
        :param intersection_indices: [in] list of lists of indices of curves corresponding to 
                                          intersection points per segment
        """
        # Member variables:
        self.__segments: list[SegmentGeometry]
        self.__nodes: list[NodeGeometry]
        self.__chain_start_nodes: list[NodeIndex]

        # Rebuild topology with intersection and cusp splits
        #
        # FIXME: below super should be equivalent
        #
        # self.update_topology(to_array, out_array, intersection_array)
        super().__init__(*self._init_projected_curve_network(parameter_segments,
                                                             spatial_segments,
                                                             planar_segments,
                                                             segment_labels,
                                                             chains,
                                                             chain_labels,
                                                             interior_cusps,
                                                             has_cusp_at_base,
                                                             intersections,
                                                             intersection_indices,
                                                             intersection_data,
                                                             num_intersections))

        # Record chain start points
        #
        # FIXME: test the function below...
        # TODO: change the method to take in arguments?
        #
        self.__init_chain_start_nodes()

        # TODO: testing with chain start nodes

        if INLINE_TESTING_ENABLED_CONTOUR_NETWORK:
            filepath: pathlib.Path = SPOT_FILEPATH / "contour_network" / \
                "projected_curve_network" / "init_chain_start_nodes"
            compare_list_node_geometry(filepath / "nodes_out.json", self.__nodes)
            compare_eigen_numpy_matrix(filepath / "chain_start_nodes.csv",
                                       np.array(self.__chain_start_nodes, dtype=int))

        # Check the validity of the topological graph structure
        if CHECK_VALIDITY:
            for node_index in range(self.num_nodes):
                if (self.is_intersection_node(node_index) and
                        not self._is_valid_node_index(self.intersection(node_index))):
                    raise ValueError(f"Intersection node {node_index} does not \
                                    have a valid intersection")

            # Check the validity of the geometric graph structure
            if not self._is_valid_projected_curve_network():
                logger.error("Invalid projected curve network made")
                raise RuntimeError("Invalid projected curve network made")

    # ******
    # Counts (inherited from parent class AbstractCurveNetworks)
    # ******
    # num_segments
    # num_nodes

    # ******
    # Geometry (setters and getters)
    # ******

    @property
    def nodes(self) -> list[NodeGeometry]:
        """Retrieves nodes of projected curve network"""
        return self.__nodes

    # @nodes.setter
    # def nodes(self, nodes: list[NodeGeometry]) -> None:
    #     self.__nodes = nodes

    @property
    def segments(self) -> list[SegmentGeometry]:
        """Retrieves segments of projected curve network"""
        return self.__segments

    # @segments.setter
    # def segments(self, segments: list[SegmentGeometry]) -> None:
    #     self.__segments = segments

    @property
    def chain_start_nodes(self) -> list[NodeIndex]:
        """
        Get list of all start nodes in the network

        :return: list of all chain start nodes
        """
        return self.__chain_start_nodes

    # ****************
    # Segment geometry
    # ****************

    # def segment_parameter_curve(self) -> None:
    #     """
    #     """
    #     unimplemented()

    def segment_planar_curve(self, segment_index: SegmentIndex) -> RationalFunction:
        """
        Returns RationalFunction with degree 4, dimension 2
        """
        assert self._is_valid_segment_index(segment_index)
        return self.segments[segment_index].planar_curve

    def segment_spatial_curve(self, segment_index: SegmentIndex) -> RationalFunction:
        """
        Returns RationalFunction with degree 4, dimension 3
        """
        assert self._is_valid_segment_index(segment_index)
        return self.segments[segment_index].spatial_curve

    def get_segment_label(self,
                          segment_index: SegmentIndex,
                          label_name: str) -> int:
        """
        Returns segment label of segment at segment_index
        """
        assert self._is_valid_segment_index(segment_index)
        return self.segments[segment_index].get_segment_label(label_name)

    def get_segment_quantitative_invisibility(self, segment_index: SegmentIndex) -> int:
        """
        Returns quantitative invisibility of segment at segment_index
        """
        assert self._is_valid_segment_index(segment_index)
        return self.segments[segment_index].quantitative_invisibility

    def set_segment_quantitative_invisibility(self,
                                              segment_index: SegmentIndex,
                                              new_quantitative_invisibility: int) -> None:
        """
        Sets quantitative invisibility of segment
        """
        if new_quantitative_invisibility < 0:
            logger.error("Cannot set negative segment QI")
            return
        self.segments[segment_index].quantitative_invisibility = new_quantitative_invisibility

    # def enumerate_parameter_curves(self) -> None:
    #     """
    #     """
    #     unimplemented()

    def enumerate_spatial_curves(self) -> list[RationalFunction]:
        """
        Get list of all spatial curves in the network
        FIXME: potentially flawed deep copy of RationalFunction spatial curve.
        FIXME: may just want a list of references to the spatial of segments

        :return spatial_curves_ref: list of all spatial curves reference
        """
        spatial_curves_ref: list[RationalFunction] = []
        for i in range(self.num_segments):
            assert (self.__segments[i].spatial_curve.degree,
                    self.__segments[i].spatial_curve.dimension) == (4, 3)

            spatial_curves_ref.append(self.__segments[i].spatial_curve)

        return spatial_curves_ref

    # def enumerate_planar_curves(self) -> list[RationalFunction]:
    #     """
    #     Returns new list of planar curves.
    #     FIXME: potentially flawed deep copy of RationalFunction planar curve.
    #     FIXME: may just want a list of references to the planar_curve of segments

    #     :return planar_curves: list of all planar curves
    #     """
    #     planar_curves: list[RationalFunction] = []

    #     for i in range(self.num_segments):
    #         planar_curves.append(copy.deepcopy(self.__segments[i].planar_curve))
    #         assert (self.__segments[i].planar_curve.degree,
    #                 self.__segments[i].planar_curve.dimension) == (4, 2)

    #     return planar_curves

    def enumerate_planar_nodes(self) -> tuple[list[PlanarPoint1d],
                                              list[PlanarPoint1d],
                                              list[PlanarPoint1d],
                                              list[PlanarPoint1d],
                                              list[PlanarPoint1d],
                                              list[PlanarPoint1d],
                                              list[PlanarPoint1d]]:
        """
        Get list of all annotated planar nodes in the network

        :return planar_knot_nodes: list of all planar knot nodes
        :return planar_marked_knot_nodes: list of all planar marked knot nodes
        :return planar_intersection_nodes: list of all planar intersection nodes
        :return planar_interior_cusp_nodes: list of all planar interior cusp nodes
        :return planar_boundary_cusp_nodes: list of all planar boundary cusp nodes
        :return planar_path_start_nodes: list of all planar path start nodes
        :return planar_path_end_nodes: list of all planar path end nodes
        """
        planar_knot_nodes: list[PlanarPoint1d] = []
        planar_marked_knot_nodes: list[PlanarPoint1d] = []
        planar_intersection_nodes: list[PlanarPoint1d] = []
        planar_interior_cusp_nodes: list[PlanarPoint1d] = []
        planar_boundary_cusp_nodes: list[PlanarPoint1d] = []
        planar_path_start_nodes: list[PlanarPoint1d] = []
        planar_path_end_nodes: list[PlanarPoint1d] = []

        # Iterate over nodes, sorting into the appropriate categories
        for ni in range(self.num_nodes):
            planar_point: PlanarPoint1d = self.node_planar_point(ni)
            assert planar_point.shape == (2, )

            if self.is_knot_node(ni):
                planar_knot_nodes.append(planar_point)
            elif self.is_marked_knot_node(ni):
                planar_marked_knot_nodes.append(planar_point)
            elif self.is_intersection_node(ni):
                planar_intersection_nodes.append(planar_point)
            elif self.is_interior_cusp_node(ni):
                planar_interior_cusp_nodes.append(planar_point)
            elif self.is_boundary_cusp_node(ni):
                planar_boundary_cusp_nodes.append(planar_point)
            elif self.is_path_start_node(ni):
                planar_path_start_nodes.append(planar_point)
            elif self.is_path_end_node(ni):
                planar_path_end_nodes.append(planar_point)

        return (planar_knot_nodes,
                planar_marked_knot_nodes,
                planar_intersection_nodes,
                planar_interior_cusp_nodes,
                planar_boundary_cusp_nodes,
                planar_path_start_nodes,
                planar_path_end_nodes)

    def enumerate_spatial_nodes(self) -> tuple[list[SpatialVector1d],
                                               list[SpatialVector1d],
                                               list[SpatialVector1d],
                                               list[SpatialVector1d],
                                               list[SpatialVector1d],
                                               list[SpatialVector1d],
                                               list[SpatialVector1d]]:
        """
        Get list of all annotated spatial nodes in the network

        :return spatial_knot_nodes: list of all spatial knot nodes
        :return spatial_marked_knot_nodes: list of all spatial marked knot nodes
        :return spatial_intersection_nodes: list of all spatial intersection nodes
        :return spatial_interior_cusp_nodes: list of all spatial interior cusp nodes
        :return spatial_boundary_cusp_nodes: list of all spatial boundary cusp nodes
        :return spatial_path_start_nodes: list of all spatial path start nodes
        :return spatial_path_end_nodes: list of all spatial path end nodes
        """
        spatial_knot_nodes: list[SpatialVector1d] = []
        spatial_marked_knot_nodes: list[SpatialVector1d] = []
        spatial_intersection_nodes: list[SpatialVector1d] = []
        spatial_interior_cusp_nodes: list[SpatialVector1d] = []
        spatial_boundary_cusp_nodes: list[SpatialVector1d] = []
        spatial_path_start_nodes: list[SpatialVector1d] = []
        spatial_path_end_nodes: list[SpatialVector1d] = []

        # Iterate over nodes, sorting into the appropriate categories
        for ni in range(self.num_nodes):
            spatial_point: SpatialVector1d = self.node_spatial_point(ni)
            if self.is_knot_node(ni):
                spatial_knot_nodes.append(spatial_point)
            elif self.is_marked_knot_node(ni):
                spatial_marked_knot_nodes.append(spatial_point)
            elif self.is_intersection_node(ni):
                spatial_intersection_nodes.append(spatial_point)
            elif self.is_interior_cusp_node(ni):
                spatial_interior_cusp_nodes.append(spatial_point)
            elif self.is_boundary_cusp_node(ni):
                spatial_boundary_cusp_nodes.append(spatial_point)
            elif self.is_path_start_node(ni):
                spatial_path_start_nodes.append(spatial_point)
            elif self.is_path_end_node(ni):
                spatial_path_end_nodes.append(spatial_point)

        logger.debug("Enumerated %s spatial intersection nodes",
                     len(spatial_intersection_nodes))

        return (spatial_knot_nodes,
                spatial_marked_knot_nodes,
                spatial_intersection_nodes,
                spatial_interior_cusp_nodes,
                spatial_boundary_cusp_nodes,
                spatial_path_start_nodes,
                spatial_path_end_nodes)

    def enumerate_cusp_spatial_tangents(self) -> tuple[list[SpatialVector1d],
                                                       list[SpatialVector1d],
                                                       list[SpatialVector1d],
                                                       list[SpatialVector1d]]:
        """
        Returns cusp spatial tangents.

        :return spatial_interior_cusp_in_tangents:  cusp spatial tangents
        :return spatial_interior_cusp_out_tangents: cusp spatial tangents
        :return spatial_boundary_cusp_in_tangents:  cusp spatial tangents
        :return spatial_boundary_cusp_out_tangents: cusp spatial tangents
        """
        spatial_interior_cusp_in_tangents:  list[SpatialVector1d] = []
        spatial_interior_cusp_out_tangents: list[SpatialVector1d] = []
        spatial_boundary_cusp_in_tangents:  list[SpatialVector1d] = []
        spatial_boundary_cusp_out_tangents: list[SpatialVector1d] = []

        # Iterate over nodes, sorting into the appropriate categories
        for ni in range(self.num_nodes):
            if self.is_interior_cusp_node(ni):
                spatial_in_tangent: SpatialVector1d = self.node_spatial_in_tangent(ni)
                spatial_out_tangent: SpatialVector1d = self.node_spatial_out_tangent(ni)
                spatial_interior_cusp_in_tangents.append(spatial_in_tangent)
                spatial_interior_cusp_out_tangents.append(spatial_out_tangent)
            elif self.is_boundary_cusp_node(ni):
                spatial_in_tangent: SpatialVector1d = self.node_spatial_in_tangent(ni)
                spatial_out_tangent: SpatialVector1d = self.node_spatial_out_tangent(ni)
                spatial_boundary_cusp_in_tangents.append(spatial_in_tangent)
                spatial_boundary_cusp_out_tangents.append(spatial_out_tangent)

        return (spatial_interior_cusp_in_tangents,
                spatial_interior_cusp_out_tangents,
                spatial_boundary_cusp_in_tangents,
                spatial_boundary_cusp_out_tangents)

    def enumerate_quantitative_invisibility(self) -> list[int]:
        """
        Get list of all quantitative invisibility values in the network

        :param quantitative_invisibility: [out] list of all QI values
        """
        quantitative_invisibility: list[int] = []
        for i in range(self.num_segments):
            quantitative_invisibility.append(self.segments[i].quantitative_invisibility)
        return quantitative_invisibility

    # *************
    # Node geometry
    # *************

    def is_knot_node(self, node_index: NodeIndex) -> bool:
        """
        Used in SegmentChainIterator
        Return is_knot at node_index
        """
        assert self._is_valid_node_index(node_index)
        if not self._is_valid_node_index(node_index):
            logger.error("Invalid node query")
            return False

        return self.nodes[node_index].is_knot()

    def is_marked_knot_node(self, node_index: NodeIndex) -> bool:
        """
        Return is_marked_knot at node_index
        """
        assert self._is_valid_node_index(node_index)
        if not self._is_valid_node_index(node_index):
            logger.error("Invalid node query")
            return False

        return self.nodes[node_index].is_marked_knot()

    def is_intersection_node(self, node_index) -> bool:
        """
        Checks self.__nodes at node_index if .is_intersection()
        """
        assert self._is_valid_node_index(node_index)
        if not self._is_valid_node_index(node_index):
            logger.error("Invalid node query")
            return False
        return self.__nodes[node_index].is_intersection()

    def is_interior_cusp_node(self, node_index: NodeIndex) -> bool:
        """
        Returns is_interior_cusp at node_index
        """
        assert self._is_valid_node_index(node_index)
        if not self._is_valid_node_index(node_index):
            logger.error("Invalid node query")
            return False
        return self.__nodes[node_index].is_interior_cusp()

    def is_boundary_cusp_node(self, node_index: NodeIndex) -> bool:
        """
        Returns is_boundary_cusp at node_index
        """
        assert self._is_valid_node_index(node_index)
        if not self._is_valid_node_index(node_index):
            logger.error("Invalid node query")
            return False
        return self.__nodes[node_index].is_boundary_cusp()

    def is_path_start_node(self, node_index: NodeIndex) -> bool:
        """
        Returns is_path_start_node at node_index
        """
        assert self._is_valid_node_index(node_index)
        if not self._is_valid_node_index(node_index):
            logger.error("Invalid node query")
            return False
        return self.__nodes[node_index].is_path_start_node()

    def is_path_end_node(self, node_index: NodeIndex) -> bool:
        """
        Returns is_path_end_node at node_index
        """
        assert self._is_valid_node_index(node_index)
        if not self._is_valid_node_index(node_index):
            logger.error("Invalid node query")
            return False
        return self.__nodes[node_index].is_path_end_node()

    def node_planar_point(self, node_index: NodeIndex) -> PlanarPoint1d:
        """
        Returns planar point at node index
        """
        assert self._is_valid_node_index(node_index)

        if self._is_valid_segment_index(self.in_(node_index)):
            planar_point: PlanarPoint1d = (
                self.segment_planar_curve(self.in_(node_index)).end_point())
            assert planar_point.shape == (2, )
            return planar_point
        elif self._is_valid_segment_index(self.out(node_index)):
            planar_point: PlanarPoint1d = (
                self.segment_planar_curve(self.out(node_index)).start_point())
            assert planar_point.shape == (2, )
            return planar_point
        else:
            return np.zeros(shape=(2, ), dtype=np.float64)

    def node_spatial_point(self, node_index: NodeIndex) -> SpatialVector1d:
        """
        Returns spatial vector at node index
        """
        assert self._is_valid_node_index(node_index)

        if self._is_valid_segment_index(self.in_(node_index)):
            spatial_point: SpatialVector1d = (
                self.segment_spatial_curve(self.in_(node_index)).end_point())
            assert spatial_point.shape == (3, )
            return spatial_point
        elif self._is_valid_segment_index(self.out(node_index)):
            spatial_point: SpatialVector1d = (
                self.segment_spatial_curve(self.out(node_index)).start_point())
            assert spatial_point.shape == (3, )
            return spatial_point
        else:
            return np.zeros(shape=(3, ), dtype=np.float64)

    def node_spatial_in_tangent(self, node_index: NodeIndex) -> SpatialVector1d:
        """
        Return end point of tangent curve at node index
        tangent curve degree 8, dimension 3
        """
        tangent_curve: RationalFunction
        tangent_curve = self.segment_spatial_curve(self.in_(node_index)).compute_derivative()
        assert (tangent_curve.degree, tangent_curve.dimension) == (8, 3)

        spatial_in_tangent: SpatialVector1d = tangent_curve.end_point()
        assert spatial_in_tangent.shape == (3, )
        return spatial_in_tangent

    def node_spatial_out_tangent(self, node_index: NodeIndex) -> SpatialVector1d:
        """
        Return start point of tangent curve at node index
        tangent curve degree 8, dimension 3
        """
        tangent_curve: RationalFunction
        tangent_curve = self.segment_spatial_curve(self.out(node_index)).compute_derivative()
        assert (tangent_curve.degree, tangent_curve.dimension) == (8, 3)

        spatial_out_tangent: SpatialVector1d = tangent_curve.start_point()
        assert spatial_out_tangent.shape == (3, )
        return spatial_out_tangent

    def node_spatial_tangent(self, node_index: NodeIndex) -> SpatialVector1d:
        """
        Return spatial tangent of curve at node index
        """
        if self._is_valid_segment_index(self.in_(node_index)):
            return self.node_spatial_in_tangent(node_index)
        elif self._is_valid_segment_index(self.out(node_index)):
            return self.node_spatial_out_tangent(node_index)
        else:
            logger.error("Isolated node")
            return np.zeros(shape=(3, ), dtype=np.float64)

    def node_planar_in_tangent(self, node_index: NodeIndex) -> PlanarPoint1d:
        """
        Return end point of tangent curve at node index
        tangent curve degree 8, dimension 2
        """
        tangent_curve: RationalFunction
        tangent_curve = self.segment_planar_curve(self.in_(node_index)).compute_derivative()
        assert (tangent_curve.degree, tangent_curve.dimension) == (8, 2)

        return tangent_curve.end_point()

    def node_planar_out_tangent(self, node_index: NodeIndex) -> PlanarPoint1d:
        """
        Return start point of tangent curve at node index
        tangent curve degree 8, dimension 2
        """
        tangent_curve: RationalFunction
        tangent_curve = self.segment_planar_curve(self.out(node_index)).compute_derivative()
        assert (tangent_curve.degree, tangent_curve.dimension) == (8, 2)

        return tangent_curve.start_point()

    def node_planar_tangent(self, node_index: NodeIndex) -> PlanarPoint1d:
        """
        Return either node planar in tangent or node planar out tangent
        depending on whether segment is invalid for node index in or node index out.
        """
        if self._is_valid_segment_index(self.in_(node_index)):
            planar_tangent: PlanarPoint1d = self.node_planar_in_tangent(node_index)
            assert planar_tangent.shape == (2, )
            return planar_tangent
        elif self._is_valid_segment_index(self.out(node_index)):
            planar_tangent: PlanarPoint1d = self.node_planar_out_tangent(node_index)
            assert planar_tangent.shape == (2, )
            return planar_tangent
        else:
            logger.error("Isolated node")
            return np.zeros(shape=(2, ), dtype=np.float64)

    def get_node_quantitative_invisibility(self, node_index: NodeIndex) -> int:
        """
        Return quantitative invisibility at node_index.
        """
        return self.nodes[node_index].quantitative_invisibility

    def set_node_quantitative_invisibility(self,
                                           node_index: NodeIndex,
                                           new_quantitative_invisibility: int) -> None:
        """
        Sets quantitative invisibility at node_index.
        """
        if new_quantitative_invisibility < 0:
            logger.error("Cannot set negative node QI")
            return
        self.nodes[node_index].quantitative_invisibility = new_quantitative_invisibility

    def node_quantitative_invisibility_is_set(self, node_index: NodeIndex) -> bool:
        """
        Checks if quantitative invisibilty is set at node index
        """
        return self.__nodes[node_index].quantitative_invisibility_is_set()

    def mark_node_quantitative_invisibility_as_set(self, node_index: NodeIndex) -> None:
        """
        Marks quantitative invisibility as set at node index
        """
        self.__nodes[node_index].mark_quantitative_invisibility_as_set()

    # *************
    # *************

    def get_segment_chain_iterator(self, segment_index: SegmentIndex) -> "SegmentChainIterator":
        """
        Build a segment chain iterator from some starting segment index.

        :param segment_index: [in] index to start the chain iterator at
        :return chain iterator for the given starting segment
        """
        assert self._is_valid_segment_index(segment_index)
        return SegmentChainIterator(self, segment_index)

    def write(self,
              output_path: str | pathlib.Path,
              color_mode: SVGOutputMode,
              show_nodes: bool = False) -> None:
        """
        Output visible planar curves as an SVG file, with visibility determined by
        having a non-positive quantitative invisibility.

        :param output_path: [in] filepath for the output SVG file
        :param color_mode:  [in] choice of segment coloring
        :param show_nodes:  [in] show nodes iff true
        """
        viewport = svg.ViewBoxSpec(0, 0, 800, 800)
        svg_elements: list[svg.Element] = []

        # Write curves
        if color_mode == SVGOutputMode.UNIFORM_SEGMENTS:
            self.__write_uniform_segments(svg_elements)
        elif color_mode == SVGOutputMode.UNIFORM_VISIBLE_SEGMENTS:
            self.__write_uniform_visible_segments(svg_elements)
        elif color_mode == SVGOutputMode.CONTRAST_INVISIBLE_SEGMENTS:
            self.__write_contrast_invisible_segments(svg_elements)
        elif color_mode == SVGOutputMode.RANDOM_CHAINS:
            self.__write_random_chains(svg_elements)
        elif color_mode == SVGOutputMode.UNIFORM_CHAINS:
            self.__write_uniform_chains(svg_elements)
        elif color_mode == SVGOutputMode.UNIFORM_VISIBLE_CHAINS:
            self.__write_uniform_visible_chains(svg_elements)
        elif color_mode == SVGOutputMode.UNIFORM_CLOSED_CURVES:
            self.__write_uniform_closed_curves(svg_elements)
        elif color_mode == SVGOutputMode.UNIFORM_VISIBLE_CURVES:
            self.__write_uniform_visible_curves(svg_elements)
        elif color_mode == SVGOutputMode.UNIFORM_SIMPLIFIED_VISIBLE_CURVES:
            self.__write_uniform_simplified_visible_curves(svg_elements)

        # Write nodes
        if show_nodes:
            for i in range(self.num_nodes):
                node_point: PlanarPoint1d = self.node_planar_point(i)
                if self.is_boundary_cusp_node(i):
                    logger.info("Writing boundary cusp node at %s", node_point)
                    write_planar_point(
                        node_point, svg_elements, 800, 400, (0, 0, 0.545, 1))  # Blue
                elif self.is_interior_cusp_node(i):
                    logger.info("Writing interior cusp node at %s", node_point)
                    write_planar_point(
                        node_point, svg_elements, 800, 400, (0.537, 0.671, 0.890, 1))  # Light blue
                elif self.is_intersection_node(i):
                    logger.info("Writing intersection node at %s", node_point)
                    write_planar_point(
                        node_point, svg_elements, 800, 400, (0.227, 0.420, 0.208, 1))  # Green

        # Write SVG
        svg_writer = svg.SVG(viewBox=viewport, elements=svg_elements)
        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(svg_writer.as_str())
            output_file.close()

    def serialize_closed_curves(self, filename: str) -> None:
        """
        Write closed curves to file as polygons with cusps indicated in a parallel
        list as 1 for cusp and 0 for no cusp.

        :param filename: [in] file to write the serialized contours to
        """
        precision: int = 17

        # TODO: maybe rename filename to filepath?
        with open(filename, 'w', encoding='utf-8') as output_file:
            curve_disc_params: CurveDiscretizationParameters = CurveDiscretizationParameters(
                num_samples=25)

            # Get closed curve nodes
            closed_curve_start_nodes: list[NodeIndex]
            closed_curve_end_nodes: list[NodeIndex]
            closed_curve_start_nodes, closed_curve_end_nodes = self.get_closed_curves()

            # Serialize closed curves
            for i, _ in enumerate(closed_curve_start_nodes):
                # Discretize the curve
                points: list[PlanarPoint1d]
                polyline: list[int]
                is_cusp: list[bool]
                (points,
                 polyline,
                 is_cusp) = self.__discretize_curve(closed_curve_start_nodes[i],
                                                    closed_curve_end_nodes[i],
                                                    curve_disc_params)

                # Write the number of points in the curve
                output_file.write(f"n {len(points)}\n")

                # Write the curve points with cusp annotation to file
                for j, _ in enumerate(points):
                    output_file.write(f"{1 if is_cusp[j] else 0}")
                    output_file.write(
                        f"{points[j][0]:.{precision}f} {points[j][1]:.{precision}f}\n")

    def clear(self) -> None:
        """
        Clear all topology and geometry data
        FIXME: overrriding declaration of clear in AbstractCurveNetwork?
        """
        todo()

    def add_spatial_network_to_viewer(self) -> None:
        """
        Add spatial curve network segments and annotated nodes to polyscope viewer.
        """
        self.__add_spatial_segments_to_viewer()
        self.__add_spatial_nodes_to_viewer()

    def spatial_network_viewer(self) -> None:
        """
        View spatial curve network segments and annotated nodes.
        """
        polyscope.init()
        self.add_spatial_network_to_viewer()
        polyscope.show()
        polyscope.remove_all_structures()

    def get_closed_curves(self) -> tuple[list[NodeIndex], list[NodeIndex]]:
        """
        Get start and end nodes for the closed curves of the surface. For open
        curves, this is the start node of the curve.
        :return closed_curve_start_nodes: closed curve start node indices
        :return closed_curve_end_nodes: closed curve end node indices
        """
        closed_curve_start_nodes: list[NodeIndex] = []
        closed_curve_end_nodes: list[NodeIndex] = []

        # Maintain record of nodes that have been covered by a closed curve
        num_nodes: NodeIndex = self.num_nodes
        is_covered_node: list[bool] = [False] * num_nodes

        # Find closed curve start and end nodes
        for start_node_index in self.chain_start_nodes:
            if is_covered_node[start_node_index]:
                continue
            is_covered_node[start_node_index] = True

            # Go backward along the chain until the start is found or a closed loop is obtained
            node_index: NodeIndex = start_node_index
            while True:
                # Get incoming segment
                in_segment_index: SegmentIndex = self.in_(node_index)

                # Add node if it is an initial endpoint
                if not self._is_valid_segment_index(in_segment_index):
                    closed_curve_start_nodes.append(node_index)
                    break

                # Get previous node (always valid)
                node_index = self.from_(in_segment_index)
                is_covered_node[node_index] = True

                # Add node if it is the start node and thus a closed loop
                if node_index == start_node_index:
                    closed_curve_start_nodes.append(node_index)
                    break

            # Go forward along the curve until an endpoint is reached
            node_index = start_node_index
            while True:
                # Get outgoing segment
                out_segment_index: SegmentIndex = self.out(node_index)

                # Add node if it is a terminal endpoint
                if not self._is_valid_segment_index(out_segment_index):
                    closed_curve_end_nodes.append(node_index)
                    break

                # Get next node (always valid)
                node_index = self.to(out_segment_index)
                is_covered_node[node_index] = True

                # Add node if it is the start node and thus a closed loop
                if node_index == start_node_index:
                    closed_curve_end_nodes.append(node_index)
                    break

        return closed_curve_start_nodes, closed_curve_end_nodes

    def get_visible_curves(self) -> tuple[list[NodeIndex], list[NodeIndex]]:
        """
        Get start and end nodes for the visible curves of the surface. For open
        curves, this is the start node of the curve.

        :return visible_curve_start_nodes: visible curve start node indices
        :return visible_curve_end_nodes: visible curve end node indices
        """
        visible_curve_start_nodes: list[NodeIndex] = []
        visible_curve_end_nodes: list[NodeIndex] = []

        # Maintain record of segments that have been covered by a visible curve
        num_segments: NodeIndex = self.num_segments
        is_covered_segment: list[bool] = [False] * num_segments

        # Find visible curve start and end nodes
        for start_node_index in self.chain_start_nodes:
            # Skip if the outgoing segment is not visible
            if ((not self._is_valid_segment_index(self.out(start_node_index))) or
                    (self.get_segment_quantitative_invisibility(self.out(start_node_index)) != 0)):
                continue

            # Skip already covered segments
            if is_covered_segment[self.out(start_node_index)]:
                continue

            # Go backward along the chain until an invalid/invisible segment is found
            # or a closed loop is obtained
            node_index: NodeIndex = start_node_index
            while True:
                # Get incoming segment
                in_segment_index: SegmentIndex = self.in_(node_index)

                # Add node if it is an initial endpoint or the previous segment is
                # invisible
                if ((not self._is_valid_segment_index(in_segment_index)) or
                        (self.get_segment_quantitative_invisibility(in_segment_index) != 0)):
                    visible_curve_start_nodes.append(node_index)
                    break

                # Get previous node (always valid)
                is_covered_segment[in_segment_index] = True
                node_index = self.from_(in_segment_index)

                # Add node if it is the start node and thus a closed loop
                if node_index == start_node_index:
                    visible_curve_start_nodes.append(node_index)
                    break

            # Go forward along the curve until an endpoint is reached
            node_index = start_node_index
            while True:
                # Get outgoing segment
                out_segment_index: SegmentIndex = self.out(node_index)

                # Add node if it is a terminal endpoint
                if ((not self._is_valid_segment_index(out_segment_index)) or
                        (self.get_segment_quantitative_invisibility(out_segment_index) != 0)):
                    visible_curve_end_nodes.append(node_index)
                    break

                # Get next node (always valid)
                is_covered_segment[out_segment_index] = True
                node_index = self.to(out_segment_index)

                # Add node if it is the start node and thus a closed loop
                if node_index == start_node_index:
                    visible_curve_end_nodes.append(node_index)
                    break

        return visible_curve_start_nodes, visible_curve_end_nodes

    # *****************
    # Protected Helpers
    # *****************

    def _init_projected_curve_network(self,
                                      parameter_segments: list[Conic],
                                      spatial_segments: list[RationalFunction],
                                      planar_segments: list[RationalFunction],
                                      segment_labels: list[dict[str, int]],
                                      chains: list[list[int]],
                                      chain_labels: list[int],
                                      interior_cusps: list[list[float]],
                                      has_cusp_at_base: list[bool],
                                      intersections: list[list[float]],
                                      intersection_indices: list[list[int]],
                                      intersection_data_ref: list[list[IntersectionData]],
                                      num_intersections: int) -> tuple[list[NodeIndex],
                                                                       list[SegmentIndex],
                                                                       list[NodeIndex]]:
        """
        Main constructor implementation

        :param parameter_segments:   [in] uv domain quadratic curves parametrizing the other curves
        :param spatial_segments:     [in] spatial rational curves before projection. 
                                          degree 4, dimension 3
        :param planar_segments:      [in] planar rational curves after projection.
                                          degree 4, dimension 2
        :param chain_labels:         [in] list of maps of labels for each segment 
                                          (e.g., patch label)
        :param chains:               [in] list of lists of chained curve indices
        :param chain_labels:         [in] list of chain labels for each segment
        :param interior_cusps:       [in] list of lists of cusp points per segment
        :param has_cusp_at_base:     [in] list of bools per segment indicating if the segment base 
                                          node is a cusp
        :param intersections:        [in] list of lists of intersection points per segment
        :param intersection_indices: [in] list of lists of indices of curves corresponding to 
                                          intersection points per segment

        :param self.__segments: [out]
        :param self.__nodes: [out]

        :return to_array: array mapping segments to their endpoints
        :return out_array: array mapping nodes to their outgoing segment
        :return intersection_array: list of intersection nodes
        """
        num_segments: int = len(planar_segments)

        if len(parameter_segments) != num_segments:
            logger.error("Inconsistent number of segments")
        if len(spatial_segments) != num_segments:
            logger.error("Inconsistent number of segments")
        if len(planar_segments) != num_segments:
            logger.error("Inconsistent number of segments")
        if len(segment_labels) != num_segments:
            logger.error("Inconsistent number of segments")
        if len(chain_labels) != num_segments:
            logger.error("Inconsistent number of segments")
        if len(interior_cusps) != num_segments:
            logger.error("Inconsistent number of segments")
        if len(has_cusp_at_base) != num_segments:
            logger.error("Inconsistent number of segments")
        if len(intersections) != num_segments:
            logger.error("Inconsistent number of segments")
        if len(intersection_indices) != num_segments:
            logger.error("Inconsistent number of segments")
        logger.info("Building projected curve network for %s segments",
                    num_segments)

        # Connect segments into chains before splitting at intersections
        to_array: list[NodeIndex]
        out_array: list[SegmentIndex]

        (to_array,
         out_array,
         self.__segments,
         self.__nodes) = build_projected_curve_network_without_intersections(
            parameter_segments,
            spatial_segments,
            planar_segments,
            segment_labels,
            chains,
            has_cusp_at_base)
        assert self._is_valid_curve_data(to_array, out_array)
        self.__mark_open_chain_endpoints(to_array, out_array, chains, self.__nodes)

        # Remove intersections that are redundant
        remove_redundant_intersections(to_array,
                                       out_array,
                                       num_intersections,
                                       intersection_data_ref)
        assert self._is_valid_curve_data(to_array, out_array)

        # Split segments at intersections while maintaining a record of the original
        # segments
        original_segment_indices: list[SegmentIndex]
        split_segment_indices: list[list[SegmentIndex]]
        intersection_nodes: list[list[NodeIndex]]

        # TODO: make below more Pythonic. to_array, out_array, __segments, and __nodes
        # are modified by reference
        (original_segment_indices,
         split_segment_indices,
         intersection_nodes) = split_segments_at_intersections(
            intersection_data_ref,
            num_intersections,
            to_array,
            out_array,
            self.__segments,
            self.__nodes)

        assert self._is_valid_curve_data(to_array, out_array)

        for i, _ in enumerate(intersection_nodes):
            logger.info("Intersection %s: %s", i, intersection_nodes[i])
            if (len(intersection_nodes[i]) != 2) and (len(intersection_nodes[i]) != 0):
                logger.warning("Intersection %s does not have two nodes: %s",
                               i,
                               intersection_nodes[i])

        # Link intersection nodes
        # Initialize all intersection indices to -1
        num_nodes: int = len(out_array)
        intersection_array: list[NodeIndex] = [-1] * num_nodes

        # NOTE: below modifies intersection_array and self.__nodes by reference.
        connect_segment_intersections(
            # self.segments,
            #   intersection_data_ref,
            intersection_nodes,
            #   to_array,
            #   out_array,
            #   split_segment_indices,
            intersection_array,
            self.__nodes)

        assert self._is_valid_minimal_curve_network_data(to_array, out_array, intersection_array)

        # Further split segments at cusps
        split_segments_at_cusps(interior_cusps,
                                original_segment_indices,
                                split_segment_indices,
                                to_array,
                                out_array,
                                intersection_array,
                                self.__segments,
                                self.__nodes)

        return to_array, out_array, intersection_array

    def _is_valid_projected_curve_network(self) -> bool:
        """
        Checks if valid projected curve network.
        """
        # Check that all segments and nodes are hit by iteration from the start node
        is_covered_node: list[bool] = [False] * self.num_nodes
        is_covered_segment: list[bool] = [False] * self.num_segments

        for i, ni in enumerate(self.__chain_start_nodes):
            is_covered_node[ni] = True
            start_si: SegmentIndex = self.out(ni)
            if not self._is_valid_segment_index(start_si):
                logger.error("Start node is an end point")
                raise ValueError("Start node is an end point")
                return False

            # Check chain from start node
            is_covered_segment[start_si] = True
            iter_: SegmentChainIterator = self.get_segment_chain_iterator(start_si)

            while not iter_.at_end_of_chain:
                si: SegmentIndex = iter_.current_segment_index
                is_covered_segment[si] = True
                is_covered_node[self.to(si)] = True
                iter_.increment()

        num_missed_nodes = 0
        for ni in range(self.num_nodes):
            if not is_covered_node[ni]:
                num_missed_nodes += 1
                logger.error("%s node %s is not covered by chain iteration",
                             self.nodes[ni].formatted_node(),
                             ni)
                raise ValueError("%s node %s is not covered by chain iteration",
                                 self.nodes[ni].formatted_node(),
                                 ni)

        num_missed_segments = 0
        for si in range(self.num_segments):
            if not is_covered_segment[si]:
                logger.error("Segment %s is not covered by chain iteration", si)
                raise ValueError("Segment %s is not covered by chain iteration", si)

        if (num_missed_segments > 0) or (num_missed_nodes > 0):
            logger.error("Missed %s nodes and %s segments",
                         num_missed_nodes,
                         num_missed_segments)
            raise ValueError("Missed %s nodes and %s segments",
                             num_missed_nodes,
                             num_missed_segments)

        return True

    # ***************
    # Private Helpers
    # ***************
    def __init_chain_start_nodes(self) -> None:
        """
        Add all special nodes except the path end nodes to the list of chain start nodes
        WARNING: This method is a little dangerous; it modifies the segments as it
        iterates over them
        """
        num_nodes: NodeIndex = len(self.__nodes)

        # Get all nodes that are special (and not path end nodes)
        self.__chain_start_nodes = []

        for ni in range(num_nodes):
            if (not self.__nodes[ni].is_knot()) and (not self.__nodes[ni].is_path_end_node()):
                if self.out(ni) < 0:
                    continue  # Hack to skip intersection path end nodes
                self.__chain_start_nodes.append(ni)

        # Mark any missed chains
        all_nodes_covered = False
        while not all_nodes_covered:
            # Get list of all covered nodes
            is_covered_node: list[bool] = [False] * num_nodes
            for i, ni in enumerate(self.__chain_start_nodes):
                is_covered_node[ni] = True
                start_si: SegmentIndex = self.out(ni)
                if not self._is_valid_segment_index(start_si):
                    raise ValueError("Start node is an end point")

                # Check chain from start node
                # FIXME: potential mistranslation from C++ to Python (esp w/ pointers)
                iteration: SegmentChainIterator = self.get_segment_chain_iterator(start_si)
                while not iteration.at_end_of_chain:
                    si: SegmentIndex = iteration.current_segment_index
                    is_covered_node[self.to(si)] = True
                    iteration.increment()

            # Determine if all nodes covered
            all_nodes_covered = True
            for ni in range(num_nodes):
                if not is_covered_node[ni]:
                    logger.debug("Marking node %s on closed featureless contour", ni)
                    self.__nodes[ni].mark_as_marked_knot()
                    self.__chain_start_nodes.append(ni)
                    all_nodes_covered = False
                    break

    def __discretize_segment_chain(self,
                                   iter_: SegmentChainIterator) -> tuple[list[PlanarPoint1d],
                                                                         list[int]]:
        """
        """
        points: list[PlanarPoint1d] = []
        polyline: list[int] = []

        curve_disc_params = CurveDiscretizationParameters()
        if iter_.at_end_of_chain:
            return points, polyline

        # Start the chain polyline
        start_polyline: list[int] = []
        planar_curve_start: RationalFunction = (
            self.segments[iter_.current_segment_index].planar_curve)
        points, start_polyline = planar_curve_start.discretize(curve_disc_params)

        # Add other chain segment points, skipping shared endpoints
        # FIXME: check for duplicates with the above...
        while not iter_.at_end_of_chain:
            planar_curve_segment: RationalFunction = (
                self.segments[iter_.current_segment_index].planar_curve)
            segment_points: list[PlanarPoint1d]
            segment_polyline: list[int]
            segment_points, segment_polyline = planar_curve_segment.discretize(curve_disc_params)

            # FIXME: utilize len just to be safe.
            for i in range(1, len(segment_points)):
                # for _, segment_point in enumerate(segment_points, 1):
                points.append(segment_points[i])

            iter_.increment()

        # Build trivial polyline
        polyline = [PLACEHOLDER_VALUE] * len(points)
        for l, _ in enumerate(points):
            polyline[l] = l

        return points, polyline

    def __discretize_curve(self,
                           start_node_index: NodeIndex,
                           end_node_index: NodeIndex,
                           curve_disc_params: CurveDiscretizationParameters
                           ) -> tuple[list[PlanarPoint1d],
                                      list[int],
                                      list[bool]]:
        """
        """
        points: list[PlanarPoint1d] = []
        polyline: list[int] = []
        is_cusp: list[bool] = []

        if not self._is_valid_node_index(start_node_index):
            return points, polyline, is_cusp

        # Get the points of the curve
        node_index: SegmentIndex = start_node_index
        points.append(self.node_planar_point(start_node_index))
        is_cusp.append((self.is_interior_cusp_node(node_index) or
                       self.is_boundary_cusp_node(node_index)))

        while True:
            # Get point of the outgoing segment
            segment_index: SegmentIndex = self.out(node_index)
            if not self._is_valid_segment_index(segment_index):
                break
            planar_curve_segment_ref: RationalFunction = self.segments[segment_index].planar_curve
            segment_points: list[PlanarPoint1d]
            segment_polyline: list[int]
            segment_points, segment_polyline = planar_curve_segment_ref.discretize(
                curve_disc_params)

            # FIXME: using enumerate with start value is a bit weird
            for i in range(1, len(segment_points)):
                # for _, segment_point in enumerate(segment_points, 1):
                # points.append(segment_point)
                points.append(segment_points[i])
                is_cusp.append(False)

            # Get the next node in the curve, check if it's a cusp and break if it's
            # the end
            node_index = self.to(segment_index)
            is_cusp[-1] = (self.is_interior_cusp_node(node_index) or
                           self.is_boundary_cusp_node(node_index))
            if node_index == end_node_index:
                break

        # Build trivial polyline
        polyline = [PLACEHOLDER_VALUE] * len(points)
        for l, _ in enumerate(points):
            polyline[l] = l

        # If the segment is closed, make the loop closed
        if start_node_index == end_node_index:
            points.pop()
            is_cusp.pop()
            polyline[-1] = polyline[0]

        return points, polyline, is_cusp

    def __simplify_curves(self,
                          visible_curve_start_nodes: list[NodeIndex],
                          visible_curve_end_nodes: list[NodeIndex],
                          all_points: list[list[PlanarPoint1d]]
                          ) -> tuple[list[list[PlanarPoint1d]],
                                     list[list[int]]]:
        """
        :return simplified_points: 
        :return simplified_polylines: 
        """
        simplified_points: list[list[PlanarPoint1d]] = []
        simplified_polylines: list[list[int]] = []
        num_curves: int = len(visible_curve_start_nodes)

        # Do an O(n^2) search for joined curves
        prev: list[int] = [-1] * num_curves
        next_: list[int] = [-1] * num_curves
        for i in range(num_curves):
            end_node_i: NodeIndex = visible_curve_end_nodes[i]
            for j in range(num_curves):
                # Don't check for self overlap
                if i == j:
                    continue

                start_node_j: NodeIndex = visible_curve_start_nodes[j]

                # Check if the two curves are adjacent up to some threshold
                pi: PlanarPoint1d = self.node_planar_point(end_node_i)
                pj: PlanarPoint1d = self.node_planar_point(start_node_j)
                difference: PlanarPoint1d = pj - pi
                if difference.dot(difference) >= 1e-7:
                    continue

                # Get next visible node after the end node i
                node_index: int = end_node_i
                while True:
                    # Get outgoing segment
                    out_segment_index: SegmentIndex = self.out(node_index)

                    # Add node if it is a terminal endpoint
                    if ((not self._is_valid_segment_index(out_segment_index)) or
                            self.get_segment_quantitative_invisibility(out_segment_index) == 0):
                        break

                    # Get next node (always valid)
                    node_index = self.to(out_segment_index)

                    # Check if closed loop
                    if node_index == end_node_i:
                        break

                # Check if the two curves are adjacent in the connectivity
                if node_index != start_node_j:
                    continue

                # Mark connectivity
                next_[i] = j
                prev[j] = i

        # TODO: originally was a CHECK_VALIDITY pragma, may want to reimplement something like that
        if CHECK_VALIDITY:
            # Check if the connectivity has consistent next/prev pairs
            if not is_valid_next_prev_pair(next_, prev):
                logger.error("Invalid next/prev pair found")
                raise ValueError("Invalid next/prev pair found")
                return simplified_points, simplified_polylines

        # Combine the sorted lines
        covered: list[bool] = [False] * num_curves
        for i in range(num_curves):
            # Skip already covered curves
            if covered[i]:
                continue

            # Go back until the start of the chain is found
            start_index: int = i
            logger.info("Processing start index %s", start_index)
            while prev[start_index] != -1:
                start_index = prev[start_index]

                # Avoid infinite loop
                if start_index == i:
                    break

            # Iterate until end of chain is found
            points: list[PlanarPoint1d] = []
            polyline: list[int] = []
            current_index: int = start_index
            prev_index: int

            # Do the below at least once: translated do-while loop
            condition = True
            while condition:
                covered[current_index] = True

                # FIXME: using enumerate with start value is a bit weird.
                for j in range(1, len(all_points)):
                    # for j, _ in enumerate(all_points, start=1):
                    polyline.append(len(points))
                    points.append(all_points[current_index][j - 1])
                prev_index = current_index
                current_index = next_[current_index]
                logger.info("Processing index %s", current_index)

                # Break if current index is invalid or check for consistency
                if current_index == -1:
                    break
                end_point: PlanarPoint1d = all_points[prev_index][-1]
                start_point: PlanarPoint1d = all_points[current_index][0]
                difference: PlanarPoint1d = start_point - end_point
                if difference.dot(difference) > 2e-4:
                    logger.error("Points %s and %s are distant", end_point, start_point)
                    raise ValueError("Points %s and %s are distant", end_point, start_point)
                condition: bool = (current_index != start_index)

            # Close loop or add last point
            if current_index == start_index:
                polyline.append(0)
            else:
                polyline.append(len(points))
                points.append(all_points[prev_index][-1])

            # Add polyline to the global list
            simplified_points.append(points)
            simplified_polylines.append(polyline)

        return simplified_points, simplified_polylines

    def __write_uniform_segments(self, svg_elements: list[svg.Element]) -> None:
        """
        """
        curve_disc_params: CurveDiscretizationParameters = CurveDiscretizationParameters()
        for i in range(self.num_segments):
            planar_curve: RationalFunction = self.__segments[i].planar_curve
            write_planar_curve_segment(planar_curve, curve_disc_params, svg_elements, 800, 400)

    def __write_uniform_visible_segments(self,  svg_elements: list[svg.Element]) -> None:
        """
        """
        curve_disc_params: CurveDiscretizationParameters = CurveDiscretizationParameters()
        for i in range(self.num_segments):
            if self.__segments[i].quantitative_invisibility > 0:
                planar_curve: RationalFunction = self.__segments[i].planar_curve
                write_planar_curve_segment(planar_curve, curve_disc_params, svg_elements, 800, 400)

    def __write_contrast_invisible_segments(self,  svg_elements: list[svg.Element]) -> None:
        """
        """
        curve_disc_params: CurveDiscretizationParameters = CurveDiscretizationParameters()
        invisible_color: Color = (1, 0, 0, 1)
        for i in range(self.num_segments):
            planar_curve: RationalFunction = self.__segments[i].planar_curve
            if self.__segments[i].quantitative_invisibility > 0:
                write_planar_curve_segment(
                    planar_curve, curve_disc_params, svg_elements, 800, 400, invisible_color)
            else:
                write_planar_curve_segment(
                    planar_curve, curve_disc_params, svg_elements, 800, 400)

    def __write_random_chains(self, svg_elements: list[svg.Element]) -> None:
        """
        """
        for start_node_index in self.chain_start_nodes:
            # Get random chain color
            color_without_alpha: Vector3f = np.random.rand(3, )
            color: Color = (color_without_alpha[0],
                            color_without_alpha[1],
                            color_without_alpha[2],
                            1.0)

            # Write chain
            start_segment_index: SegmentIndex = self.out(start_node_index)
            iteration: SegmentChainIterator = self.get_segment_chain_iterator(start_segment_index)
            points: list[PlanarPoint1d]
            polyline: list[int]
            points, polyline = self.__discretize_segment_chain(iteration)
            add_curve_to_svg(points, polyline, svg_elements, 800, 400, color)

    def __write_uniform_chains(self, svg_elements: list[svg.Element]) -> None:
        """
        """
        color: Color = (0, 0, 0, 1)
        for start_node_index in self.chain_start_nodes:
            start_segment_index: SegmentIndex = self.out(start_node_index)
            iteration: SegmentChainIterator = self.get_segment_chain_iterator(start_segment_index)
            points: list[PlanarPoint1d]
            polyline: list[int]
            points, polyline = self.__discretize_segment_chain(iteration)
            add_curve_to_svg(points, polyline, svg_elements, 800, 400, color)

    def __write_uniform_visible_chains(self, svg_elements: list[svg.Element]) -> None:
        """
        """
        color: Color = (0, 0, 0, 1)

        for start_node_index in self.chain_start_nodes:
            start_segment_index: SegmentIndex = self.out(start_node_index)
            if self.__segments[start_segment_index].quantitative_invisibility > 0:
                continue
            iteration: SegmentChainIterator = self.get_segment_chain_iterator(start_segment_index)
            points: list[PlanarPoint1d]
            polyline: list[int]
            points, polyline = self.__discretize_segment_chain(iteration)
            add_curve_to_svg(points, polyline, svg_elements, 800, 400, color)

    def __write_uniform_visible_curves(self, svg_elements: list[svg.Element]) -> None:
        """
        """
        color: Color = (0, 0, 0, 1)
        curve_disc_params: CurveDiscretizationParameters = CurveDiscretizationParameters()

        # Get visible curve indices
        visible_curve_start_nodes: list[NodeIndex]
        visible_curve_end_nodes: list[NodeIndex]
        visible_curve_start_nodes, visible_curve_end_nodes = self.get_visible_curves()
        logger.info("%s start points and %s end points found",
                    len(visible_curve_start_nodes),
                    len(visible_curve_end_nodes))
        logger.info("Start nodes are %s", visible_curve_start_nodes)
        logger.info("End nodes are %s", visible_curve_end_nodes)

        # Write all curves
        for i, _ in enumerate(visible_curve_start_nodes):
            points: list[PlanarPoint1d]
            polyline: list[int]
            is_cusp: list[bool]
            (points,
             polyline,
             is_cusp) = self.__discretize_curve(visible_curve_start_nodes[i],
                                                visible_curve_end_nodes[i],
                                                curve_disc_params)
            add_curve_to_svg(points, polyline, svg_elements, 800, 400, color)

    def __write_uniform_closed_curves(self, svg_elements: list[svg.Element]) -> None:
        """
        """
        color: Color = (0, 0, 0, 1)
        curve_disc_params: CurveDiscretizationParameters = CurveDiscretizationParameters()

        # Get closed curve indices
        closed_curve_start_nodes: list[NodeIndex]
        closed_curve_end_nodes: list[NodeIndex]
        closed_curve_start_nodes, closed_curve_end_nodes = self.get_closed_curves()
        logger.info("%s start points and %s end points found",
                    len(closed_curve_start_nodes),
                    len(closed_curve_end_nodes))
        logger.info("Start nodes are %s", closed_curve_start_nodes)
        logger.info("End nodes are %s", closed_curve_end_nodes)

        # Write all curves
        for i, _ in enumerate(closed_curve_start_nodes):
            points: list[PlanarPoint1d]
            polyline: list[int]
            is_cusp: list[bool]
            (points,
             polyline,
             is_cusp) = self.__discretize_curve(closed_curve_start_nodes[i],
                                                closed_curve_end_nodes[i],
                                                curve_disc_params)
            add_curve_to_svg(points, polyline, svg_elements, 800, 400, color)

    def __write_uniform_simplified_visible_curves(self, svg_elements: list[svg.Element]) -> None:
        """
        """
        color: Color = (0, 0, 0, 1)
        curve_disc_params: CurveDiscretizationParameters = CurveDiscretizationParameters()

        # Get visible curve indices
        visible_curve_start_nodes: list[NodeIndex]
        visible_curve_end_nodes: list[NodeIndex]
        visible_curve_start_nodes, visible_curve_end_nodes = self.get_visible_curves()
        logger.info("%s start points and %s end points found",
                    len(visible_curve_start_nodes),
                    len(visible_curve_end_nodes))

        # Get all curves
        all_points: list[list[PlanarPoint1d]] = []
        for i, _ in enumerate(visible_curve_start_nodes):
            points: list[PlanarPoint1d]
            polyline: list[int]
            is_cusp: list[bool]
            (points,
             polyline,
             is_cusp) = self.__discretize_curve(visible_curve_start_nodes[i],
                                                visible_curve_end_nodes[i],
                                                curve_disc_params)
            all_points.append(points)

        # Simplify curves
        simplified_points: list[list[PlanarPoint1d]]
        simplified_polylines: list[list[int]]
        (simplified_points,
         simplified_polylines) = self.__simplify_curves(visible_curve_start_nodes,
                                                        visible_curve_end_nodes,
                                                        all_points)

        # Write curves
        for i, _ in enumerate(simplified_polylines):
            add_curve_to_svg(simplified_points[i],
                             simplified_polylines[i],
                             svg_elements,
                             800,
                             400,
                             color)

    def __add_planar_segments_to_viewer(self) -> None:
        """
        Add planar curves to the polyscope viewer
        # NOTE: method not used
        """
        unimplemented()

    def __add_spatial_segments_to_viewer(self) -> None:
        """
        Add spatial curves to the polyscope viewer
        """
        # Get spatial curve list
        spatial_curves_ref: list[RationalFunction] = self.enumerate_spatial_curves()
        curve_disc_params: CurveDiscretizationParameters = CurveDiscretizationParameters()

        # Discretize the spatial curves
        points: list[PlanarPoint1d]
        polylines: list[list[int]]
        points, polylines = discretize_curve_segments(4, 3, spatial_curves_ref, curve_disc_params)

        # FIXME: apparently there's a problem with the method below
        points_mat: MatrixXf = convert_nested_vector_to_matrix(points)
        edges: list[tuple[int, int]] = convert_polylines_to_edges(polylines)
        spatial_curve_network_segments: polyscope.CurveNetwork = polyscope.register_curve_network(
            "spatial_curve_network_segments", points_mat, edges)
        spatial_curve_network_segments.set_radius(0.001)
        spatial_curve_network_segments.set_color((0.600, 0.000, 0.067))

        # Add QI values
        quantitative_invisibility: list[int] = self.enumerate_quantitative_invisibility()
        num_segments: int = len(quantitative_invisibility)
        QI_labels: list[int] = [PLACEHOLDER_VALUE] * (curve_disc_params.num_samples * num_segments)
        for i in range(num_segments):
            for j in range(curve_disc_params.num_samples):
                QI_labels[curve_disc_params.num_samples * i + j] = quantitative_invisibility[i]
        colormap: MatrixXf = generate_random_category_colormap(QI_labels)
        spatial_curve_network_segments.add_color_quantity("quantitative_invisibility", colormap)

        # Get visible curves separately
        visible_spatial_curves: list[RationalFunction] = []
        for i, spatial_curve in enumerate(spatial_curves_ref):
            if quantitative_invisibility[i] == 0:
                visible_spatial_curves.append(spatial_curve)

        # Discretize the spatial curves
        visible_points: list[SpatialVector1d]
        visible_polylines: list[list[int]]
        (visible_points,
         visible_polylines) = discretize_curve_segments(4, 3, visible_spatial_curves,
                                                        curve_disc_params)
        visible_points_mat: MatrixXf = convert_nested_vector_to_matrix(visible_points)
        visible_edges: list[tuple[int, int]] = convert_polylines_to_edges(visible_polylines)
        visible_spatial_curves_network: polyscope.CurveNetwork = polyscope.register_curve_network(
            "visible_spatial_curves", visible_points_mat, visible_edges)
        visible_spatial_curves_network.set_radius(0.0025)
        visible_spatial_curves_network.set_color((0.0, 0.0, 0.0))

    def __add_spatial_nodes_to_viewer(self) -> None:
        """
        Add spatial curves to the polyscope viewer
        """
        spatial_knot_nodes: list[SpatialVector1d]
        spatial_marked_knot_nodes: list[SpatialVector1d]
        spatial_intersection_nodes: list[SpatialVector1d]
        spatial_interior_cusp_nodes: list[SpatialVector1d]
        spatial_boundary_cusp_nodes: list[SpatialVector1d]
        spatial_path_start_nodes: list[SpatialVector1d]
        spatial_path_end_nodes: list[SpatialVector1d]
        (spatial_knot_nodes,
         spatial_marked_knot_nodes,
         spatial_intersection_nodes,
         spatial_interior_cusp_nodes,
         spatial_boundary_cusp_nodes,
         spatial_path_start_nodes,
         spatial_path_end_nodes) = self.enumerate_spatial_nodes()

        # Get spatial tangents
        spatial_interior_cusp_in_tangents: list[SpatialVector1d]
        spatial_interior_cusp_out_tangents: list[SpatialVector1d]
        spatial_boundary_cusp_in_tangents: list[SpatialVector1d]
        spatial_boundary_cusp_out_tangents: list[SpatialVector1d]
        (spatial_interior_cusp_in_tangents,
         spatial_interior_cusp_out_tangents,
         spatial_boundary_cusp_in_tangents,
         spatial_boundary_cusp_out_tangents) = self.enumerate_cusp_spatial_tangents()

        # Register all spatial nodes
        spatial_knot_nodes_cloud: polyscope.PointCloud = polyscope.register_point_cloud(
            "spatial_knot_nodes", spatial_knot_nodes)
        spatial_knot_nodes_cloud.set_enabled(False)

        spatial_marked_knot_nodes_cloud: polyscope.PointCloud = polyscope.register_point_cloud(
            "spatial_marked_knot_nodes", spatial_marked_knot_nodes)
        spatial_marked_knot_nodes_cloud.set_enabled(False)

        spatial_intersection_nodes_cloud: polyscope.PointCloud = polyscope.register_point_cloud(
            "spatial_intersection_nodes", spatial_intersection_nodes)
        spatial_intersection_nodes_cloud.set_color((0.227, 0.420, 0.208))

        spatial_interior_cusp_nodes_cloud: polyscope.PointCloud = polyscope.register_point_cloud(
            "spatial_interior_cusp_nodes",                             spatial_interior_cusp_nodes)
        spatial_interior_cusp_nodes_cloud.set_color((0.537, 0.671, 0.890))
        spatial_interior_cusp_nodes_cloud.add_vector_quantity(
            "in_tangents", spatial_interior_cusp_in_tangents)
        spatial_interior_cusp_nodes_cloud .add_vector_quantity(
            "out_tangents", spatial_interior_cusp_out_tangents)

        spatial_boundary_cusp_nodes_cloud: polyscope.PointCloud = polyscope.register_point_cloud(
            "spatial_boundary_cusp_nodes", spatial_boundary_cusp_nodes)
        spatial_boundary_cusp_nodes_cloud.set_color((0.0, 0.0, 0.545))
        spatial_boundary_cusp_nodes_cloud.add_vector_quantity(
            "in_tangents", spatial_boundary_cusp_in_tangents)
        spatial_boundary_cusp_nodes_cloud.add_vector_quantity(
            "out_tangents", spatial_boundary_cusp_out_tangents)

        spatial_path_start_nodes_cloud: polyscope.PointCloud = polyscope.register_point_cloud(
            "spatial_path_start_nodes", spatial_path_start_nodes)
        spatial_path_start_nodes_cloud.set_enabled(False)

        spatial_path_end_nodes_cloud: polyscope.PointCloud = polyscope.register_point_cloud(
            "spatial_path_end_nodes", spatial_path_end_nodes)
        spatial_path_end_nodes_cloud.set_enabled(False)

    def __clear_geometry(self) -> None:
        """
        """
        self.__segments.clear()
        self.__nodes.clear()
        self.__chain_start_nodes.clear()

    @staticmethod
    def __mark_open_chain_endpoints(to_array: list[NodeIndex],
                                    out_array: list[SegmentIndex],
                                    chains: list[list[SegmentIndex]],
                                    nodes_ref: list[NodeGeometry]) -> None:
        """
        Record the start of open chains and also mark an arbitrary node on each
        closed contour
        """
        # Build from array from the network topology
        from_array: list[NodeIndex] = AbstractCurveNetwork.build_from_array(to_array, out_array)

        for i, _ in enumerate(chains):
            # Get the first and last segments in the chain
            first_segment: SegmentIndex = chains[i][0]
            last_segment: SegmentIndex = chains[i][-1]

            if to_array[last_segment] != from_array[first_segment]:
                start_node: NodeIndex = from_array[first_segment]
                end_node: NodeIndex = to_array[last_segment]
                nodes_ref[start_node].mark_as_path_start_node()
                nodes_ref[end_node].mark_as_path_end_node()

    @staticmethod
    def testing_mark_open_chain_endpoints(to_array: list[NodeIndex],
                                          out_array: list[SegmentIndex],
                                          chains: list[list[SegmentIndex]],
                                          nodes_ref: list[NodeGeometry]) -> None:
        """
        Exposed mark_open_chain_endpoints() for testing.
        """
        ProjectedCurveNetwork.__mark_open_chain_endpoints(to_array,
                                                          out_array,
                                                          chains,
                                                          nodes_ref)
