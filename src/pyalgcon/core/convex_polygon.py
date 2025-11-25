"""
Convex polygons. Used in Quadratic Surface Patch files.
Convex polygon formed by intersecting half planes.
"""

import json
from venv import logger

import numpy as np

from pyalgcon.core.common import (ROWS, Index, Matrix2x2f,
                                  Matrix3x2f, MatrixNx3f,
                                  MatrixNx3i, PlanarPoint,
                                  PlanarPoint1d, Vector1D,
                                  Vector3f, float_equal,
                                  generate_linspace,
                                  unreachable)
from pyalgcon.core.interval import Interval
from pyalgcon.core.line_segment import LineSegment

# *******
# Helpers
# *******


def compute_line_between_points(point_0: PlanarPoint1d, point_1: PlanarPoint1d) -> Vector3f:
    """
    Compute the implicit form of a line between two points

    :param point_0: first point of shape (1, 2)
    :type point_0: PlanarPoint
    :param point_1: second point of shape (1, 2)
    :type point_1: PlanarPoint

    :return: line_coeff of shape (3, 1) made from the points
    :rtype: np.ndarray 
    """
    assert point_0.shape == (2, )
    assert point_1.shape == (2, )

    x0: np.float64 = point_0[0]
    y0: np.float64 = point_0[1]
    x1: np.float64 = point_1[0]
    y1: np.float64 = point_1[1]
    line_coeffs: Vector3f = np.array([(x0 * y1 - x1 * y0),
                                      (y0 - y1),
                                      (x1 - x0)],
                                     dtype=np.float64)

    assert line_coeffs.shape == (3, )
    return line_coeffs


def compute_parametric_line_between_points(point_0: PlanarPoint1d,
                                           point_1: PlanarPoint1d) -> LineSegment:
    """
    Compute the parametric form of a line between two points

    :param point_0: first point of shape (2, )
    :type point_0: PlanarPoint
    :param point_1: second point of shape (2, )
    :type point_1: PlanarPoint

    :return: line_segment
    :rtype: LineSegment
    """
    assert point_0.shape == (2, )
    assert point_1.shape == (2, )

    # Set numerator
    numerators: Matrix2x2f = np.array([
        [point_0[0], point_0[1]],
        [point_1[0] - point_0[0], point_1[1] - point_0[1]]], dtype=np.float64)

    assert numerators.shape == (2, 2)

    # Set domain interval [0, 1]
    # TODO: below may not be the most Python way of making the Interval object domain
    domain: Interval = Interval()
    domain.set_lower_bound(0, False)
    domain.set_upper_bound(1, False)

    line_segment = LineSegment(numerators, domain)
    return line_segment


def refine_triangles(V: MatrixNx3f, F: MatrixNx3i) -> tuple[MatrixNx3f, MatrixNx3i]:
    """
    Refine a mesh with midpoint subdivision.
    Logic of this method does not modify V and F by reference and instead creates 
    new np.ndarray V_refined and F_refined.

    :param V: vertices
    :type V: np.ndarray
    :param F: faces
    :type F: np.ndarray

    :return: Vertices and Faces refined
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    # Need proper datatype for F or else indexing from V will not work
    # (because cannot use floats for accesing indices)
    assert V.dtype == np.float64
    assert F.dtype == np.int64

    num_faces: Index = F.shape[ROWS]  # rows
    V_refined: MatrixNx3f = np.ndarray(shape=(num_faces * 6, 2), dtype=np.float64)
    F_refined: MatrixNx3i = np.ndarray(shape=(num_faces * 4, 3), dtype=np.int64)
    for i in range(num_faces):
        # We have vectors below, this time of shape (n, ) because we're using NumPy broadcasting
        v0: Vector1D = V[F[i, 0], :]
        v1: Vector1D = V[F[i, 1], :]
        v2: Vector1D = V[F[i, 2], :]
        assert v0.ndim == 1
        assert v1.ndim == 1
        assert v2.ndim == 1

        # Add vertices for refined face
        V_refined[6 * i + 0, :] = v0
        V_refined[6 * i + 1, :] = v1
        V_refined[6 * i + 2, :] = v2
        V_refined[6 * i + 3, :] = (v0 + v1) / 2.0
        V_refined[6 * i + 4, :] = (v1 + v2) / 2.0
        V_refined[6 * i + 5, :] = (v2 + v0) / 2.0

        # Add refined faces
        F_refined[4 * i + 0, :] = np.array([6 * i + 0, 6 * i + 3, 6 * i + 5], dtype=np.int64)
        F_refined[4 * i + 1, :] = np.array([6 * i + 1, 6 * i + 4, 6 * i + 3], dtype=np.int64)
        F_refined[4 * i + 2, :] = np.array([6 * i + 2, 6 * i + 5, 6 * i + 4], dtype=np.int64)
        F_refined[4 * i + 3, :] = np.array([6 * i + 3, 6 * i + 4, 6 * i + 5], dtype=np.int64)

    return V_refined, F_refined


class ConvexPolygon:
    """
    Representation of a convex polygon in R^2 that supports containment queries,
    sampling, boundary segments and vertices computation, triangulation, and
    boundary parametrization.
    """
    # TODO (from ASOC code): Implement constructor from collection of points

    def __init__(self,
                 boundary_segments_coeffs: list[Vector3f],
                 vertices: Matrix3x2f) -> None:
        """
        Constructor that is called by classmethod init_from_boundary_segments_coeffs 
        or init_from_vertices.
        NOTE: Do not call this constructor directly. As in, do not call 
        ConvexPolygon(boundary_segments_coeffs, vertices). Instead, use 
        ConvexPolygon.init_from_boundary_segments_coeffs or ConvexPolygon.init_from_vertices().

        :param boundary_segments_coeffs: boundary segment coefficients list of size 3 with 
        element np.ndarray of shape (3, 1) and type np.float64
        :type boundary_segments_coeffs: list[Vector3f]
        :param vertices: vertices of type np.ndarray of shape (3, 2)
        :type vertices: Matrix3x2r
        """
        # Assertions to match ASOC code C++ code
        assert len(boundary_segments_coeffs) == 3
        # Checks if empty in 2nd case
        assert (boundary_segments_coeffs[0].shape == (3, ) or
                boundary_segments_coeffs[0].shape == (3, 0))   # lazily checking elements.
        assert vertices.shape == (3, 2)

        # *******
        # Private
        # *******
        self.m_boundary_segments_coeffs: list[Vector3f] = boundary_segments_coeffs
        self.m_vertices: Matrix3x2f = vertices

    # TODO: make a method to serialize to JSON... or something like that.
    # def serialize_to_json_str(self) -> str:
    #     """
    #     Saves ConvexPolygon to a string in JSON format.
    #     """
    #     # TODO: save ConvexPolygon to a string in JSON format
    #     # But I feel that if I could get some serialziation and deserialization of the Spline Surface,
    #     # then that would REALLY help cut down on development time.

    @classmethod
    def init_from_json(cls, convex_polygon_json: dict[str, np.ndarray]):
        """
        Initialzes ConvexPolygon class from a dict made from a JSON-format string. 
        Primarily used for deserializing domain for Quadratic Spline Surface Patch.
        """
        # Attempt to get from JSON.
        # NOTE: use this version of the method since clear separation betwen data-reading functionality
        # and data processing.
        # TODO: error check for when key is mistyped or does not exist
        # TODO: convert a nested list of list into...
        boundary_segment_coeffs: list[Vector3f]
        vertices: Matrix3x2f
        boundary_segment_coeffs = [np.array(arr, dtype=np.float64)
                                   for arr in convex_polygon_json.get("boundary_segment_coeffs")]
        vertices = np.array(convex_polygon_json.get("vertices"), dtype=np.float64)

        assert vertices.shape == (3, 2)
        assert boundary_segment_coeffs[0].shape == (3, )  # lazy shape checking

        return cls(boundary_segment_coeffs, vertices)

    # @classmethod
    # def init_from_json_str(cls, convex_polygon_json_str: str):
    #     """
    #     Initialzes ConvexPolygon class from a JSON-format string.
    #     Primarily used for deserializing domain for Quadratic Spline Surface Patch.
    #     """
    #     convex_polygon_json: dict[str, np.ndarray] = json.loads(convex_polygon_json_str)

    #     # Attempt to get from JSON.
    #     # TODO: error check for when key is mistyped or does not exist
    #     # TODO: convert a nested list of list into...
    #     boundary_segment_coeffs: list[Vector3f]
    #     vertices: Matrix3x2f
    #     boundary_segment_coeffs = [np.array(arr, dtype=np.float64)
    #                                for arr in convex_polygon_json.get("boundary_segment_coeffs")]
    #     vertices = np.array(convex_polygon_json.get("vertices"), dtype=np.float64)

    #     assert vertices.shape == (3, 2)
    #     assert boundary_segment_coeffs[0].shape == (3, )  # lazy shape checking

    #     return cls(boundary_segment_coeffs, vertices)

    @classmethod
    def init_from_boundary_segments_coeffs(cls, boundary_segments_coeffs: list[Vector3f]):
        """
        Only boundary_segments_coeffs passed in. Construct m_vertices.
        """
        v0: PlanarPoint1d = cls.intersect_patch_boundaries(boundary_segments_coeffs[1],
                                                           boundary_segments_coeffs[2])
        v1: PlanarPoint1d = cls.intersect_patch_boundaries(boundary_segments_coeffs[2],
                                                           boundary_segments_coeffs[0])
        v2: PlanarPoint1d = cls.intersect_patch_boundaries(boundary_segments_coeffs[0],
                                                           boundary_segments_coeffs[1])

        vertices: Matrix3x2f = np.array([v0, v1, v2], dtype=np.float64)
        assert vertices.shape == (3, 2)

        # return vertices
        return cls(boundary_segments_coeffs, vertices)

    @classmethod
    def init_from_vertices(cls, vertices: Matrix3x2f):
        """
        Only vertices passed in. Construct m_boundary_segments_coeffs.
        """
        assert vertices.shape == (3, 2)
        num_vertices: int = vertices.shape[ROWS]
        boundary_segments_coeffs: list[Vector3f] = []

        # TODO: is the below the dynamic sizing of arrays that I needed to avoid?
        for i in range(num_vertices):
            line_coeffs: Vector3f
            line_coeffs = compute_line_between_points(vertices[i, :],
                                                      vertices[(i + 1) % num_vertices, :])
            boundary_segments_coeffs.append(line_coeffs)

        assert len(boundary_segments_coeffs) == num_vertices
        assert boundary_segments_coeffs[0].shape == (3, )  # lazy check

        # return boundary_segments_coeffs
        return cls(boundary_segments_coeffs, vertices)

    def contains(self, point: PlanarPoint1d) -> bool:
        """
        Return true iff point is in the convex polygon
        """
        # FIXME: if not done already, rework method to utilize 1D PlanarPoint type
        assert point.shape == (2, )

        for _, L_coeffs in enumerate(self.m_boundary_segments_coeffs):
            # NOTE: redundant check
            assert L_coeffs.shape == (3, )

            # NOTE: index accessing was wrong beforehand...
            if (L_coeffs[0] + L_coeffs[1] * point[0] + L_coeffs[2] * point[1]) < 0.0:
                return False

        return True

    @staticmethod
    def intersect_patch_boundaries(first_boundary_segment_coeffs: Vector3f,
                                   second_boundary_segment_coeffs: Vector3f) -> PlanarPoint1d:
        """
        Computes intersecting patch boundaries.

        Method has decorator @staticmethod to work with @classmethod 
        init_from_boundary_segments_coeffs.
        """
        assert first_boundary_segment_coeffs.shape == (3, )
        assert second_boundary_segment_coeffs.shape == (3, )

        a00: float = first_boundary_segment_coeffs[0]
        a10: float = first_boundary_segment_coeffs[1]
        a01: float = first_boundary_segment_coeffs[2]
        b00: float = second_boundary_segment_coeffs[0]
        b10: float = second_boundary_segment_coeffs[1]
        b01: float = second_boundary_segment_coeffs[2]

        x: float
        y: float

        # Solve for y in terms of x first
        if not float_equal(a01, 0.0):
            my: float = -a10 / a01
            by: float = -a00 / a01
            assert not float_equal(b10 + b01 * my, 0.0)
            x = -(b00 + b01 * by) / (b10 + b01 * my)
            y = my * x + by
        # Solve for x in terms of y first
        elif not float_equal(a10, 0.0):
            mx: float = -a01 / a10
            bx: float = -a00 / a10
            assert not float_equal(b01 + b10 * mx, 0.0)
            y = -(b00 + b10 * bx) / (b01 + b10 * mx)
            x = mx * y + bx
        else:
            logger.error("Degenerate line")
            # FIXME: maybe exception is a bit too extreme... originally there was a "return"
            # here in the ASOC code.
            # Since we don't want the program to fail if the user inputs an invalid mesh.
            raise ValueError("Degenerate line trying to form ConvexPolygon")

        # Build intersection
        # TODO: I really need a subclass of np.ndarray called PlanarPoint that automatically checks
        # the sizing of the array for me... Because manually checking shape==(1,2) is cumbersome
        intersection: PlanarPoint1d = np.array([x, y], dtype=np.float64)
        assert intersection.shape == (2, )
        return intersection

    @property
    def boundary_segments(self) -> list[Vector3f]:
        """
        Retrieves boundary segments coefficients of shape (3, )

        :rtype: list[Vector3f]
        """
        # HACK: flattening this accessor to return list[Vector3f] rather than
        # a list of (3, 1) matrices
        boundary_segments_coeffs_flattened: list[Vector3f] = []
        for boundary_segment_coeffs in self.m_boundary_segments_coeffs:
            boundary_segments_coeffs_flattened.append(boundary_segment_coeffs.flatten())
        assert (np.array(boundary_segments_coeffs_flattened).shape ==
                np.array(self.m_boundary_segments_coeffs).squeeze().shape)

        return boundary_segments_coeffs_flattened

    @property
    def vertices(self) -> Matrix3x2f:
        """Gets vertices from Convex Polygon"""
        return self.m_vertices

    def parametrize_patch_boundaries(self) -> list[LineSegment]:
        """
        Parametrizes patch boundaries

        :return patch_boundaries: list of length 3
        """
        patch_boundaries: list[LineSegment] = []

        # Get rows of m_vertices.
        # NOTE: num_verices should be 3
        num_vertices: int = self.m_vertices.shape[ROWS]
        for i in range(num_vertices):
            line_segment: LineSegment = compute_parametric_line_between_points(
                self.m_vertices[i, :],
                self.m_vertices[((i + 1) % num_vertices), :])
            patch_boundaries.append(line_segment)

        # Double checking that we indeed only have 3 elements inside patch_boundaries as per
        # original code
        assert len(patch_boundaries) == num_vertices
        return patch_boundaries

    def triangulate(self, num_refinements: int) -> tuple[MatrixNx3f, MatrixNx3i]:
        """
        Triangulate domain with.
        This takes in self.m_vertices and creates a new F matrix to return.
        TODO Can generalize to arbitrary domain if needed

        :param num_refinements: number of iterations to refine V and F
        :type num_refinements: int

        :return: tuple of V (shape (3, 2)) and F (shape (1, 3))
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        V: MatrixNx3f = self.m_vertices
        F: MatrixNx3i = np.array([[0, 1, 2]], dtype=np.int64)
        # NOTE: keep F as shape (1, 3) since F can have any number of rows and should not be 1D
        assert F.shape == (1, 3)

        for _ in range(num_refinements):
            V_refined: MatrixNx3f
            F_refined: MatrixNx3i
            V_refined, F_refined = refine_triangles(V, F)
            V = V_refined
            F = F_refined

        return V, F

    def sample(self, num_samples: int) -> list[PlanarPoint1d]:
        """
        Sample points from ConvexPolygon.

        :param num_samples: number of samples from ConvexPolygon
        :return domain_points: [out]
        """
        domain_points: list[PlanarPoint1d] = []
        lower_left_corner: PlanarPoint1d = np.array([-1, -1], dtype=np.float64)
        upper_right_corner: PlanarPoint1d = np.array([1, 1], dtype=np.float64)
        # Checking if shape (2, ) since that is the shape of PlanarPoint1d type
        assert lower_left_corner.shape == (2, )
        assert upper_right_corner.shape == (2, )

        x0: float = lower_left_corner[0]
        y0: float = lower_left_corner[1]
        x1: float = upper_right_corner[0]
        y1: float = upper_right_corner[1]

        # Compute points
        x_axis: Vector1D = generate_linspace(x0, x1, num_samples)
        y_axis: Vector1D = generate_linspace(y0, y1, num_samples)
        # Asserting ndim == 1 because ASOC code has x_axis and y_axis as VectorXr
        assert x_axis.ndim == 1
        assert y_axis.ndim == 1

        for i in range(num_samples):
            for j in range(num_samples):
                point: PlanarPoint1d = np.array([x_axis[i], y_axis[j]], dtype=np.float64)
                assert point.shape == (2, )
                if self.contains(point):
                    domain_points.append(point)

        return domain_points
