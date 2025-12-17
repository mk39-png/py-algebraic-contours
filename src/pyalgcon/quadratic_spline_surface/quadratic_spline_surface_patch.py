"""
Representation for quadratic surface patches with convex domains.
"""
import json
import logging
import os
import pathlib
from typing import TextIO

import numpy as np
import polyscope as ps

from pyalgcon.core.bivariate_quadratic_function import (
    evaluate_quadratic_mapping, generate_monomial_to_bezier_matrix,
    generate_quadratic_coordinate_domain_triangle_normalization_matrix)
from pyalgcon.core.common import (ROWS, Matrix2x2f, Matrix3x2r, Matrix6x3f,
                                  Matrix6x3r, Matrix6x6r, MatrixNx2f,
                                  PlanarPoint1d, SpatialVector,
                                  SpatialVector1d, Vector3f,
                                  compute_point_cloud_bounding_box, load_json,
                                  todo, unimplemented)
from pyalgcon.core.convex_polygon import ConvexPolygon
from pyalgcon.core.evaluate_surface_normal import \
    generate_quadratic_surface_normal_coeffs
from pyalgcon.core.line_segment import LineSegment
from pyalgcon.core.rational_function import RationalFunction

logger: logging.Logger = logging.getLogger(__name__)


# **************************************
# Quadratic Spline Surface Patch Helpers
# **************************************


def compute_normalized_surface_mapping(surface_mapping_coeffs: Matrix6x3r,
                                       domain: ConvexPolygon) -> Matrix6x3r:
    """
    Compute the surface mapping with normalized domain.

    :return: normalized_surface_mapping_coeffs of shape (6, 3)
    """
    assert surface_mapping_coeffs.shape == (6, 3)
    domain_vertices: Matrix3x2r = domain.vertices
    v0: PlanarPoint1d = domain_vertices[0, :]
    v1: PlanarPoint1d = domain_vertices[1, :]
    v2: PlanarPoint1d = domain_vertices[2, :]
    change_of_basis_matrix: Matrix6x6r = (
        generate_quadratic_coordinate_domain_triangle_normalization_matrix(v0, v1, v2))

    # shape (6, 3) = (6, 6) @ (6, 3)
    normalized_surface_mapping_coeffs = change_of_basis_matrix @ surface_mapping_coeffs
    return normalized_surface_mapping_coeffs


def compute_bezier_points(normalized_surface_mapping_coeffs: Matrix6x3r) -> Matrix6x3r:
    """
    Compute bezier points.
    """
    monomial_to_bezier_matrix: Matrix6x6r = generate_monomial_to_bezier_matrix()

    # shape (6, 3) = (6, 6) @ (6, 3)
    bezier_points: Matrix6x3r = monomial_to_bezier_matrix @ normalized_surface_mapping_coeffs
    return bezier_points


class QuadraticSplineSurfacePatch:
    """A quadratic surface patch with convex polygonal domain.

    Supports:
    - evaluation and sampling of points and normals on the surface
    - triangulation
    - conversion to Bezier form
    - bounding box computation
    - boundary curve parameterization
    - cone point annotation
    - (de)serialization
    """

    # **************
    # Public Methods
    # **************
    def __init__(self, surface_mapping_coeffs: Matrix6x3r, domain: ConvexPolygon,
                 normal_mapping_coeffs: Matrix6x3r | None = None,
                 normalized_surface_mapping_coeffs: Matrix6x3r | None = None,
                 bezier_points: Matrix6x3f | None = None,
                 min_point: SpatialVector1d | None = None,
                 max_point: SpatialVector1d | None = None,
                 cone_index: int | None = None) -> None:
        """
        Constructor for QuadraticSplineSurfacePatch
        """
        # -- Core independent data --
        self.m_surface_mapping_coeffs: Matrix6x3r = surface_mapping_coeffs
        self.m_domain: ConvexPolygon = domain

        # If any none, then do normal calculation
        # TODO: check if else statement is being entered
        if any(x is None for x in (
                normal_mapping_coeffs,
                normalized_surface_mapping_coeffs,
                bezier_points,
                min_point,
                max_point,
                cone_index)):

            # -- Inferred dependent data --
            self.m_normal_mapping_coeffs: Matrix6x3r = np.zeros(shape=(6, 3), dtype=np.float64)
            self.m_normalized_surface_mapping_coeffs: Matrix6x3r = np.zeros(
                shape=(6, 3), dtype=np.float64)
            self.m_bezier_points: Matrix6x3r = np.zeros(shape=(6, 3), dtype=np.float64)
            self.m_min_point: SpatialVector1d = np.zeros(shape=(3, ), dtype=np.float64)
            self.m_max_point: SpatialVector1d = np.zeros(shape=(3, ), dtype=np.float64)

            # Compute derived mapping information from the surface mapping and domain
            self.m_normal_mapping_coeffs: Matrix6x3f = (
                generate_quadratic_surface_normal_coeffs(surface_mapping_coeffs))
            self.m_normalized_surface_mapping_coeffs = (
                compute_normalized_surface_mapping(surface_mapping_coeffs, domain))
            self.m_bezier_points = compute_bezier_points(surface_mapping_coeffs)
            (self.m_min_point,
             self.m_max_point) = compute_point_cloud_bounding_box(self.m_bezier_points)

            # -- Additional cone marker to handle degenerate configurations --
            # NOTE: Do not mark a cone by default
            self.__cone_index: int = -1
        else:
            # Else, assuming deserializing from file, assign parameters
            assert normal_mapping_coeffs.shape == (6, 3)
            assert normalized_surface_mapping_coeffs.shape == (6, 3)
            assert bezier_points.shape == (6, 3)
            assert min_point.shape == (3, )
            assert max_point.shape == (3, )

            self.m_normal_mapping_coeffs = normal_mapping_coeffs
            self.m_normalized_surface_mapping_coeffs = normalized_surface_mapping_coeffs
            self.m_bezier_points = bezier_points
            self.m_min_point = min_point
            self.m_max_point = max_point
            self.__cone_index = cone_index

    @classmethod
    def init_from_json_file(cls, filename: pathlib.Path):
        """
        Initializes QuadraticSplineSurfacePatch object from JSON file.
        Used for testing retrieving files from algebraic_contours/test/

        :param filename: name of file to deserialize from
        """
        spline_surface_patch_json: dict = load_json(filename)

        # FIXME: check if ok
        surface_mapping_coeffs: Matrix6x3f = np.array(
            spline_surface_patch_json.get("surface_mapping_coeffs"))

        domain_json: dict = spline_surface_patch_json.get("domain")

        # TODO: might have to flip ordering of vertices matrix
        domain: ConvexPolygon = ConvexPolygon(
            [np.array(arr).squeeze() for arr in domain_json.get("boundary_segment_coeffs")],
            np.array(domain_json.get("vertices")).squeeze())
        normal_mapping_coeffs: Matrix6x3f = np.array(
            spline_surface_patch_json.get("normal_mapping_coeffs"))
        normalized_surface_mapping_coeffs: Matrix6x3f = np.array(
            spline_surface_patch_json.get("normalized_surface_mapping_coeffs"))
        bezier_points: Matrix6x3r = np.array(spline_surface_patch_json.get("bezier_points"))
        min_point: Vector3f = np.array(spline_surface_patch_json.get("min_point"))
        max_point: Vector3f = np.array(spline_surface_patch_json.get("max_point"))
        cone_index: int = spline_surface_patch_json.get("cone_index")

        return cls(surface_mapping_coeffs,
                   domain,
                   normal_mapping_coeffs,
                   normalized_surface_mapping_coeffs,
                   bezier_points,
                   min_point,
                   max_point,
                   cone_index)

    @property
    def dimension(self) -> int:
        """
        Get the dimension of the surface patch ambient space

        :return: dimension of the ambient space
        """
        return self.m_surface_mapping_coeffs.shape[1]

    def mark_cone(self, cone_index: int) -> None:
        """Mark one of the vertices as a cone

        :param cone_index: index of the cone in the triangle
        """
        self.__cone_index = cone_index

    def has_cone(self) -> bool:
        """ Determine if the patch has a cone

        :return: true iff the patch has a cone
        :rtype: bool
        """
        return ((self.__cone_index >= 0) and (self.__cone_index < 3))

    @property
    def cone_index(self) -> int:
        """
        Get the cone index, or -1 if none exists.

        :return: true iff the patch has a cone
        :rtype: int
        """
        return self.__cone_index

    @property
    def surface_mapping(self) -> Matrix6x3f:
        """
        Get the surface mapping coefficients.

        :return: reference to the surface mapping. shape == (6,3)
        :rtype: np.ndarray
        """
        return self.m_surface_mapping_coeffs

    @property
    def normal_mapping(self) -> Matrix6x3f:
        """
        Get the surface normal mapping coefficients.

        :return: reference to the surface normal mapping. shape==(6,3)
        :rtype: np.ndarray
        """
        return self.m_normal_mapping_coeffs

    def get_normalized_surface_mapping(self) -> Matrix6x3r:
        """
        Get the surface mapping coefficients with normalized domain.

        Compute them if they haven't been computed yet.

        :return: reference to the normalized surface mapping. shape==(6,3)
        :rtype: np.ndarray
        """
        return self.m_normalized_surface_mapping_coeffs

    def get_bezier_points(self) -> Matrix6x3r:
        """
        Get the surface mapping coefficients with normalized domain.

        Compute them if they haven't been computed yet.

        :return: reference to the bezier points. shape==(6,3)
        :rtype: np.ndarray
        """
        return self.m_bezier_points

    def get_bounding_box(self) -> tuple[SpatialVector, SpatialVector]:
        """
        Compute the bounding box for the surface patch.

        :return: (self.m_min_point, self.m_max_point) where min_point is the minimum coordinates 
            bounding box point and max_point is the maximum coordinates bounding box point.
        :rtype: tuple[SpatialVector, SpatialVector]
        """
        return self.m_min_point, self.m_max_point

    def get_bounding_box_min_point(self) -> SpatialVector:
        """
        Compute the minimum point of the bounding box for the surface patch.

        :return: min_point: minimum coordinates bounding box point
        :rtype: SpatialVector
        """
        assert self.m_min_point.shape == (3,)
        return self.m_min_point

    def get_bbox_x_min(self) -> float:
        """
        Get the minimum x-coordinate of the bounding box.
        Note that min_point is NumPy shape (3, ) as it is a point in 3D space as represented
        by x, y, z

        :return: self.m_min_point[0]: x-coordinate of the minimum point of the bounding box
        :rtype: float
        """
        return self.m_min_point[0]

    def get_bbox_y_min(self) -> float:
        """
        Get the minimum y-coordinate of the bounding box.
        Note that min_point is NumPy shape (3, )

        :return: self.m_min_point[1]: y-coordinate of the minimum point of the bounding box
        :rtype: float
        """
        return self.m_min_point[1]

    def get_bounding_box_max_point(self) -> SpatialVector:
        """
        Compute the maximum point of the bounding box for the surface patch.

        :return: max_point: maximum coordinates bounding box point
        :rtype: SpatialVector
        """
        assert self.m_max_point.shape == (3,)
        return self.m_max_point

    def get_bbox_x_max(self) -> float:
        """
        Get the maximum x-coordinate of the bounding box.
        Note that max_point is NumPy shape (1, 2)

        :return: self.m_max_point[0]: x-coordinate of the maximum point of the bounding box
        :rtype: float
        """
        return self.m_max_point[0]

    def get_bbox_y_max(self) -> float:
        """
        Get the maximum y-coordinate of the bounding box.

        :return: self.m_max_point[1]: y-coordinate of the maximum point of the bounding box
        :rtype: float
        """
        return self.m_max_point[1]

    @property
    def domain(self) -> ConvexPolygon:
        """
        Get the convex domain of the patch.

        :return: reference to the convex domain
        :rtype: ConvexPolygon
        """
        return self.m_domain

    def get_patch_boundaries(self) -> list[RationalFunction]:
        """
        Get the patch boundaries as spatial curves.

        :return: patch_boundaries: patch boundary spatial curves. degree=4, dimension=3
        :rtype: list[RationalFunction]
        """
        # Get parametrized domain boundaries.
        domain_boundaries: list[LineSegment] = self.domain.parametrize_patch_boundaries()
        # Checking len == 3 since ASOC code has domain_boundarys as array of 3 LineSegment elements
        assert len(domain_boundaries) == 3

        # Lift the domain boundaries to the surface
        __surface_mapping_coeffs_ref: Matrix6x3r = self.surface_mapping

        patch_boundaries: list[RationalFunction] = []

        # FIXME: Something might go wrong with the things below, especially since
        # I'm unsure about surface_mapping_coeffs
        for i, domain_boundary in enumerate(domain_boundaries):
            patch_boundaries.append(domain_boundary.pullback_quadratic_function(
                3, __surface_mapping_coeffs_ref))

        assert len(patch_boundaries) == 3
        assert patch_boundaries[0].degree == 4
        assert patch_boundaries[0].dimension == 3

        return patch_boundaries

    def normalize_patch_domain(self) -> "QuadraticSplineSurfacePatch":
        """
        Construct a spline surface patch with the same image but where the domain
        is normalized to the triangle u + v <= 1 in the positive quadrant.

        :return: normalized_spline_surface_patch: normalized patch
        :rtype: QuadraticSplineSurfacePatch
        """
        # Generate the standard u + v <= 1 triangle
        normalized_domain_vertices: Matrix3x2r = np.array([[0, 0],
                                                           [1, 0],
                                                           [0, 1]])
        assert normalized_domain_vertices.shape == (3, 2)
        normalized_domain: ConvexPolygon = ConvexPolygon.init_from_vertices(
            normalized_domain_vertices)

        # Build the normalized surface patch
        normalized_surface_mapping_coeffs: Matrix6x3r = self.get_normalized_surface_mapping()
        normalized_spline_surface_patch = QuadraticSplineSurfacePatch(
            normalized_surface_mapping_coeffs, normalized_domain)

        return normalized_spline_surface_patch

    def denormalize_domain_point(self, normalized_domain_point: PlanarPoint1d) -> PlanarPoint1d:
        """
        Given a normalized domain point in the triangle u + v <= 1, map it to the
        corresponding point in the patch domain.

        :param normalized_domain_point: normalized (barycentric) domain point
        :type normalized_domain_point: PlanarPoint

        :return: corresponding point in the domain triangle
        :rtype: PlanarPoint
        """
        assert normalized_domain_point.shape == (2, )

        # Get domain triangle vertices
        domain_ref: ConvexPolygon = self.domain
        domain_vertices: Matrix3x2r = domain_ref.vertices
        v0: PlanarPoint1d = domain_vertices[0, :]
        v1: PlanarPoint1d = domain_vertices[1, :]
        v2: PlanarPoint1d = domain_vertices[2, :]
        assert domain_vertices.shape == (3, 2)
        assert v0.shape == (2, )

        # Generate affine transformation mapping the standard triangle to the domain triangle
        # FIXME: double check implementation with C++ code...
        linear_transformation: Matrix2x2f = np.array([v1 - v0,
                                                     v2 - v0], dtype=np.float64)
        assert linear_transformation.shape == (2, 2)
        translation: PlanarPoint1d = v0

        # Denormalize the domain point
        # shapes: (2, ) @ (2, 2) + (2, )
        return normalized_domain_point @ linear_transformation + translation

    def evaluate(self, domain_point: PlanarPoint1d) -> SpatialVector1d:
        """
        Evaluate the surface at a given domain point.

        :param domain_point: domain evaluation point of shape (2, )
        :type domain_point: PlanarPoint

        :return: surface_point: image of the domain point on the surface of shape (3, )
        :rtype: SpatialVector
        """
        assert domain_point.shape == (2, )
        surface_point: SpatialVector1d = evaluate_quadratic_mapping(self.m_surface_mapping_coeffs,
                                                                    domain_point)
        assert surface_point.shape == (3, )
        return surface_point

    def evaluate_normal(self, domain_point: PlanarPoint1d) -> SpatialVector1d:
        """
        Evaluate the surface normal at a given domain point.

        :param domain_point: domain evaluation point of shape (2, )
        :type domain_point: PlanarPoint

        :return: surface_normal: surface normal at the image of the domain point
        :rtype: SpatialVector
        """
        assert domain_point.shape == (2, )
        surface_normal: SpatialVector1d = evaluate_quadratic_mapping(self.m_normal_mapping_coeffs,
                                                                     domain_point)
        assert surface_normal.shape == (3, )
        return surface_normal

    def sample(self, sampling_density: int) -> list[SpatialVector1d]:
        """
        Sample points on the surface.

        :param sampling_density: sampling density parameter
        :type sampling_density: int

        :return: spline_surface_patch_points: sampled points on the surface
        :rtype: list[SpatialVector]
        """
        # Sample the convex domain
        domain_points: list[PlanarPoint1d] = self.m_domain.sample(sampling_density)

        # Lift the domain points to the surface
        num_points: int = len(domain_points)

        spline_surface_patch_points: list[SpatialVector1d] = []

        # TODO: change for loop to utilize enumerate
        for i in range(num_points):
            spline_surface_patch_points.append(self.evaluate(domain_points[i]))

        return spline_surface_patch_points

    def triangulate(self,
                    num_refinements: int,
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Triangulate the surface patch.
        NOTE: Changed to return by value since this creates a new patch that has been
        triangulated anyways, no need to reference the original, if that makes sense.

        :param num_refinements: number of refinements of the domain to perform.
        :type num_refinements: int
        :return: (V_triangulate, F_triangulate, N_triangulate). 
            Triangulated patch vertex positions (V), triangulated patch faces (F), and triangulated 
            patch vertex normals (N). V and N shape (n, 3)
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        # Triangulate the domain
        # NOTE: domain refers to UV coords, which is (N, 2) shape
        V_domain: MatrixNx2f
        F_triangulate: np.ndarray[tuple[int, int], np.dtype[np.int64]]
        V_domain, F_triangulate = self.m_domain.triangulate(num_refinements)

        # Lift the domain vertices to the surface and also compute the normals
        V_triangulate: np.ndarray[tuple[int, int], np.dtype[np.float64]]
        N_triangulate: np.ndarray[tuple[int, int], np.dtype[np.float64]]
        V_triangulate = np.ndarray(shape=(V_domain.shape[ROWS], self.dimension), dtype=np.float64)
        N_triangulate = np.ndarray(shape=(V_domain.shape[ROWS], self.dimension), dtype=np.float64)

        for i in range(V_domain.shape[ROWS]):
            # V_domain of shape
            surface_point: SpatialVector1d = self.evaluate(V_domain[i, :])
            surface_normal: SpatialVector1d = self.evaluate_normal(V_domain[i, :])
            V_triangulate[i, :] = surface_point
            N_triangulate[i, :] = surface_normal

        return V_triangulate, F_triangulate, N_triangulate

    def add_patch_to_viewer(self, patch_name: str = "surface_patch") -> None:
        """
        Add triangulated patch to the polyscope viewer.

        :param patch_name: name to assign the patch in the viewer.
        :type patch_name: str
        """
        # Generate mesh discretization
        num_refinements: int = 2

        # TODO: does the logic below work? Triangulate modifies by reference and that may bring up
        # some issues with this Python code.
        V: np.ndarray
        F: np.ndarray
        N: np.ndarray
        V, F, N = self.triangulate(num_refinements)

        # Add patch mesh
        ps.init()
        ps.register_surface_mesh(patch_name, V, F)

    def serialize(self, output_file: TextIO) -> None:
        """
        Write the patch information to the output stream in the format
            c a_0 a_u a_v a_uv a_uu a_vv
            p1 p1_u p1_v
            p2 p2_u p2_v
            p3 p3_u p3_v

        :return: stream to write serialization to
        :rtype: str
        """
        precision: int = 17
        output_file.write("patch\n")

        # Serialize x coordinate coefficients
        output_file.write("cx")
        for i in range(self.m_surface_mapping_coeffs.shape[ROWS]):
            output_file.write(f" {self.m_surface_mapping_coeffs[i, 0]:.{precision}f}")
        output_file.write("\n")

        # Serialize y coordinate coefficients
        output_file.write("cy")
        for i in range(self.m_surface_mapping_coeffs.shape[ROWS]):
            output_file.write(f" {self.m_surface_mapping_coeffs[i, 1]:.{precision}f}")
        output_file.write("\n")

        # Serialize z coordinate coefficients
        output_file.write("cz")
        for i in range(self.m_surface_mapping_coeffs.shape[ROWS]):
            output_file.write(f" {self.m_surface_mapping_coeffs[i, 2]:.{precision}f}")
        output_file.write("\n")

        # Serialize domain boundary
        vertices: Matrix3x2r = self.m_domain.vertices
        if self.__cone_index == 0:
            output_file.write("cp1 ")
        else:
            output_file.write("p1 ")
        output_file.write(f"{vertices[0, 0]:.{precision}f} {vertices[0, 1]:.{precision}f}\n")

        if self.__cone_index == 1:
            output_file.write("cp2 ")
        else:
            output_file.write("p2 ")
        output_file.write(f"{vertices[1, 0]:.{precision}f} {vertices[1, 1]:.{precision}f}\n")

        if self.__cone_index == 2:
            output_file.write("cp3 ")
        else:
            output_file.write("p3 ")
        output_file.write(f"{vertices[2, 0]:.{precision}f} {vertices[2, 1]:.{precision}f}\n")

    def write_patch(self, filepath: str) -> None:
        """
        Write patch to file.

        :param filepath: file path to write serialized patch to.
        :type filepath: str
        """
        logger.info("Writing spline patch to %s", filepath)

        if os.path.isfile(filepath):
            logger.warning("Overwriting file at %s.", filepath)

        with open(filepath, 'w', encoding='utf-8') as output_file:
            self.serialize(output_file)
            output_file.close()

    # ***************
    # Private methods
    # ***************

    def __formatted_patch(self) -> None:
        """
        Method not used
        """
        unimplemented("Use serialize instead")
