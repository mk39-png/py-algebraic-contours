import logging
import os
import pathlib
from dataclasses import dataclass
from typing import TextIO

import igl
import numpy as np
import polyscope

from pyalgcon.core.common import (COLS, DISCRETIZATION_LEVEL, HASH_TABLE_SIZE,
                                  ROWS, SKY_BLUE, Matrix3x2f, Matrix6x3f,
                                  MatrixNx3f, MatrixNx3i, MatrixXf, PatchIndex,
                                  PlanarPoint, PlanarPoint1d, SpatialVector,
                                  SpatialVector1d, Vector1D,
                                  convert_nested_vector_to_matrix,
                                  convert_polylines_to_edges)
from pyalgcon.core.convex_polygon import ConvexPolygon
from pyalgcon.core.line_segment import LineSegment
from pyalgcon.quadratic_spline_surface.quadratic_spline_surface_patch import \
    QuadraticSplineSurfacePatch

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class SurfaceDiscretizationParameters:
    """
    Parameters for the discretization of a quadratic spline
    """
    # Number of subdivisions per triangle of the domain
    num_subdivisions: int = 2
    # If true, compute unit length surface normal vectors
    normalize_surface_normals: bool = True


class QuadraticSplineSurface:
    """
    A piecewise quadratic surface.

    Supports:
    - evaluation
    - patch and subsurface extraction
    - triangulation
    - sampling
    - visualization
    - (basic) rendering
    - (de)serialization
    """

    def __init__(self, patches: list[QuadraticSplineSurfacePatch]) -> None:
        """
        Constructor from patches
        @param[in] patches: quadratic surface patches
        """

        # NOTE: self._patches protected since it's accessed and used by
        # subclass TwelveSplitSplineSurface
        self._patches: list[QuadraticSplineSurfacePatch] = patches

        # Hash table parameters
        # TODO: these have been commented because they are not used for anything.
        self.__patches_bbox_x_min: float = 0.0
        # self.__patches_bbox_x_max: float = 0.0
        self.__patches_bbox_y_min: float = 0.0
        # self.__patches_bbox_y_max: float = 0.0
        self.hash_x_interval: float = 0.0
        self.hash_y_interval: float = 0.0

        # Hash table data in a 2D list of list[int]
        # NOTE: hash_table is HASH_TABLE_SIZE x HASH_TABLE_SIZE 2D list with elements list[int]
        # TODO: utilize some sort of pythonic hash table type
        # FIXME: I think the hash table is where everything goes wrong and is the one function
        self._hash_table: list[list[list[int]]] = self.compute_patch_hash_tables(patches)

        # TODO: the below does not seem like it's used for anything yet
        # self.reverse_hash_table: list[list[tuple[int, int]]]

    @classmethod
    def from_file(cls, filepath: str | pathlib.Path) -> "QuadraticSplineSurface":
        """
        Read a surface serialization from file.\n
        NOTE: method used for testing with ASOC code and to make sure that
        implementation is correct.

        FIXME: this method is somewhat broken for QuadraticSplineSurfaces used in ContourNetwork...
        i.e. the test_compute_spline_surface_contours_and_boundaries_spot_mesh() case

        :param filepath: [in] file path for the serialized surface
        :return: patches to save to.
        :rtype: list[QuadraticSplineSurfacePatch]
        """
        input_file: TextIO
        patches: list[QuadraticSplineSurfacePatch] = []

        if not os.path.isfile(filepath):
            raise OSError("File does not exist at %s. Choose a file to read spline.", filepath)

        with open(filepath, 'r', encoding='utf-8') as input_file:
            patches = cls.deserialize(input_file)
        input_file.close()

        return cls(patches)

    @property
    def num_patches(self) -> PatchIndex:
        """
        Get the number of patches in the surface
        @return number of patches
        """
        return len(self._patches)

    @property
    def hash_table(self) -> dict[int, dict[int, list[int]]]:
        """
        Get the hash table of the surface
        """
        return self._hash_table

    def get_patch(self, patch_index: PatchIndex) -> QuadraticSplineSurfacePatch:
        """
        Get a reference to a spline patch at patch_index
        @return spline patch
        """
        return self._patches[patch_index]

    def evaluate_patch(self, patch_index: PatchIndex, domain_point: PlanarPoint) -> SpatialVector:
        """
        Evaluate the surface at a given patch and domain point
        @param[in] patch_index: index of the patch to evaluate
        @param[in] domain_point: point in the patch domain to evaluate
        @param[out] surface_point: output point on the surface
        """
        surface_point: SpatialVector = self.get_patch(
            patch_index).evaluate(domain_point)
        assert surface_point.shape == (1, 3)
        return surface_point

    def evaluate_patch_normal(
            self, patch_index: PatchIndex, domain_point: PlanarPoint) -> SpatialVector:
        """
        Evaluate the surface normal at a given patch and domain point.

        :param patch_index: index of the patch to evaluate
        :type patch_index: PatchIndex
        :param domain_point: point in the patch domain to evaluate
        :type domain_point: PlanarPoint

        :return: output point on the surface
        :rtype: SpatialVector
        """

        surface_normal: SpatialVector = self.get_patch(
            patch_index).evaluate_normal(domain_point)
        assert surface_normal.shape == (1, 3)
        return surface_normal

    def empty(self) -> bool:
        """
        Determine if the surface is empty

        :return: true iff the surface is empty
        """
        return len(self._patches) == 0

    def clear(self) -> None:
        """
        Clear the surface
        """
        self._patches.clear()

    def subsurface(self, patch_indices: list[PatchIndex]) -> "QuadraticSplineSurface":
        """
        Generate a subsurface with the given patch indices.

        :param patch_indices: indices of the patches to keep.
        :type patch_indices: list[PatchIndex]
        :return: subsurface with the given patches
        :rtype: QuadraticSplineSurface
        """
        sub_patches: list[QuadraticSplineSurfacePatch] = []

        for i, _ in enumerate(patch_indices):
            sub_patches.append(self._patches[patch_indices[i]])

        subsurface_spline = QuadraticSplineSurface(sub_patches)
        return subsurface_spline

    def triangulate_patch(self,
                          patch_index: PatchIndex,
                          num_refinements: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Triangulate a given patch.

        :param patch_index: patch to triangulate
        :type patch_index: PatchIndex
        :param num_refinements: number of refinements for the triangulation
        :type num_refinements: int

        :return: vertices (V), faces (F), and vertex normals (N) of the triangulation
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        V: MatrixNx3f
        F: MatrixNx3i
        N: MatrixNx3f
        V, F, N = self.get_patch(patch_index).triangulate(num_refinements)

        return V, F, N

    def discretize(self,
                   surface_disc_params: SurfaceDiscretizationParameters
                   ) -> tuple[MatrixNx3f, MatrixNx3i, MatrixNx3f]:
        """
        Triangulate the surface.

        :param surface_disc_params: discretization parameters
        :type surface_disc_params: SurfaceDiscretizationParameters
        :return: vertices of the triangulation (V_tri), faces of the triangulation(F_tri),
        and vertex normals (N_tri)
        :rtype: tuple[MatrixNx3f, MatrixNx3i, MatrixNx3f]
        """
        num_subdivisions: int = surface_disc_params.num_subdivisions

        if self.empty():
            V: MatrixNx3f = np.ndarray(shape=(0, 0), dtype=np.float64)
            F: MatrixNx3i = np.ndarray(shape=(0, 0), dtype=np.int64)
            N: MatrixNx3f = np.ndarray(shape=(0, 0), dtype=np.float64)
            return V, F, N

        # ** Build triangulated surface as a copy **
        patch_index: PatchIndex = 0

        # Build patch 0 to get information like num_patch_vertices and num_patch_faces.
        V_patch_0: np.ndarray[tuple[int], np.dtype[np.float64]]
        F_patch_0: np.ndarray[tuple[int], np.dtype[np.int64]]
        N_patch_0: np.ndarray[tuple[int], np.dtype[np.float64]]
        V_patch_0, F_patch_0, N_patch_0 = self.triangulate_patch(patch_index, num_subdivisions)

        num_patch_vertices: int = V_patch_0.shape[ROWS]
        num_patch_faces: int = F_patch_0.shape[ROWS]
        patch_index += 1

        # Set the patch 0 inside V, F, and N of the surface.
        V_tri: np.ndarray = np.zeros(shape=(num_patch_vertices * self.num_patches, 3),
                                     dtype=np.float64)
        F_tri: np.ndarray = np.zeros(shape=(num_patch_faces * self.num_patches, 3),
                                     dtype=np.int64)
        N_tri: np.ndarray = np.zeros(shape=(num_patch_vertices * self.num_patches, 3),
                                     dtype=np.float64)
        V_tri[:num_patch_vertices, :] = V_patch_0
        F_tri[:num_patch_faces, :] = F_patch_0
        N_tri[:num_patch_vertices, :] = N_patch_0

        # Building the rest of the patches. (e.g. the rest of the shape V, F, and N)
        # NOTE: expect V to be shape (235008, 3) for the Spot Control mesh
        while patch_index < self.num_patches:
            # NOTE: expect patches shape (24, 3) with num_subdivisions = 2 and the Spot Control mesh
            V_patch: np.ndarray[tuple[int, int], np.dtype[np.float64]]
            F_patch: np.ndarray[tuple[int, int], np.dtype[np.int64]]
            N_patch: np.ndarray[tuple[int, int], np.dtype[np.float64]]
            V_patch, F_patch, N_patch = self.triangulate_patch(patch_index, num_subdivisions)

            # FIXME: values stop being set after a certain point....
            V_tri[num_patch_vertices * patch_index: num_patch_vertices * (patch_index + 1),
                  : V_tri.shape[COLS]] = V_patch

            F_tri[num_patch_faces * patch_index: num_patch_faces * (patch_index + 1),
                  : F_tri.shape[COLS]] = (
                F_patch + np.full(shape=(num_patch_faces, F_tri.shape[COLS]),
                                  fill_value=num_patch_vertices * patch_index,
                                  dtype=np.int64))

            N_tri[num_patch_vertices * patch_index: num_patch_vertices * (patch_index + 1),
                  : N_tri.shape[COLS]] = N_patch

            # XXX: need to increment patch_index
            patch_index += 1

        logger.info("%s surface vertices", V_tri.shape[ROWS])
        logger.info("%s surface faces", F_tri.shape[ROWS])
        logger.info("%s surface normals", N_tri.shape[ROWS])

        return V_tri, F_tri, N_tri

    def discretize_patch_boundaries(self) -> tuple[list[SpatialVector], list[list[int]]]:
        """
        Discretize all patch boundaries as polylines.
        NOTE: This also appears in contour_network folder in discretize.py, 
        but is here for convenience and also for organization purposes.


        TODO: MOVE FUNCTION OUTSIDE OF CLASS AND ACCEPT QUADRATICSPLINESURFACE AS
        ARGUMENT. MAKES LIFE EEASIER FOR US AS WE DONT NEED TO REIMPLEMENT THIS FUNCTION
        INSIDE OF DISCRETIZE...
        ACTUALLY, JUST MOVE THIS FUNCTION INTO DISCRETIZE FOR OUR CONVENIENCE... YEAH
        OR JUST KEEP IT HERE.... ACTUALLY... JUST KEEP IT HERE...
        MAYBE?
        THIS HAS MUCH MORE TO DO WITH THE 12 SPLIT SPLINE PRINTING AND THE SURFACE THAN
        WITH THE POLYLINES ITSELF, I FEEL LIKE.



        :return points: list of polyline points.
        :rtype points: list[SpatialVector]
        :return polyline: list of lists of polyline edges
        :rtype polyline: list[list[int]]
        """
        points: list[SpatialVector] = []
        polylines: list[list[int]] = []

        for patch_index in range(self.num_patches):
            spline_surface_patch: QuadraticSplineSurfacePatch = self.get_patch(patch_index)
            # list of size 3
            patch_boundaries: list[LineSegment]
            patch_boundaries = spline_surface_patch.domain.parametrize_patch_boundaries()

            for k, _ in enumerate(patch_boundaries):
                # Get points on the boundary curve
                parameter_points_k: list[PlanarPoint1d] = patch_boundaries[k].sample_points(5)
                points_k: list[SpatialVector1d] = []

                for i, _ in enumerate(parameter_points_k):
                    points_k.append(spline_surface_patch.evaluate(parameter_points_k[i]))

                # Build polyline for the given curve
                polyline: list[int] = []
                for l, _ in enumerate(points_k):
                    polyline.append(len(points) + l)

                points.extend(points_k)
                polylines.append(polyline)

        return points, polylines

    def save_obj(self, filename: str) -> None:
        """
        Save the triangulated surface as an obj.

        NOTE: Used in contour_network.py

        :param filename: filepath to save the obj
        :type filename: str
        """
        # Generate mesh discretization
        V: np.ndarray
        # NOTE: TC and FTC intialization... is it equivalent to ASOC eigen code?
        TC: np.ndarray = np.ndarray(shape=(0, 0))
        F: np.ndarray
        FTC: np.ndarray = np.ndarray(shape=(0, 0))
        N: np.ndarray
        surface_disc_params: SurfaceDiscretizationParameters = SurfaceDiscretizationParameters()
        V, F, N = self.discretize(surface_disc_params)

        # Write mesh to file
        igl.writeOBJ(filename, V, F, N, F, TC, FTC)

    def add_surface_to_viewer(self,
                              color: tuple[float, float, float] = SKY_BLUE,
                              num_subdivisions: int = DISCRETIZATION_LEVEL) -> None:
        """
        Add the surface to the viewer.
        NOTE: Used in twelve_split_spline.py and contour_network.py

        :param color: color for the surface in the viewer
        :type color: np.ndarray

        :param num_subdivisions: number of subdivisions for the surface
        :type num_subdivisions: int
        """

        # TODO: adjust parameter naming of SurfaceDiscretizationParameters
        # Generate mesh discretization
        surface_disc_params = SurfaceDiscretizationParameters(num_subdivisions=num_subdivisions)
        V: MatrixNx3f
        F: MatrixNx3i
        _N: MatrixNx3f
        V, F, _N = self.discretize(surface_disc_params)  # NOTE: this takes approx 10 sec to do

        # Add surface mesh
        polyscope.init()
        surface: polyscope.SurfaceMesh = polyscope.register_surface_mesh("surface", V, F)
        surface.set_edge_width(0)
        surface.set_color(color)

        # Discretize patch boundaries
        boundary_points: list[SpatialVector]
        boundary_polylines: list[list[int]]
        boundary_points, boundary_polylines = self.discretize_patch_boundaries()

        # View contour curve network
        boundary_points_matrix: MatrixXf = convert_nested_vector_to_matrix(boundary_points)
        boundary_edges: list[tuple[int, int]] = convert_polylines_to_edges(boundary_polylines)

        # HACK: converting boundary edges to NumPy array so that polyscope works, but may
        # want to have convert_polylines_to_edges return a Nx2 matrix by default,
        # where each row is an edge.
        patch_boundaries: polyscope.CurveNetwork
        patch_boundaries = polyscope.register_curve_network("patch_boundaries",
                                                            boundary_points_matrix,
                                                            np.array(boundary_edges))
        patch_boundaries.set_color((0.670, 0.673, 0.292))
        patch_boundaries.set_radius(0.0005)
        patch_boundaries.set_radius(0.0005)
        patch_boundaries.set_enabled(False)

    def view(self,
             color: tuple[float, float, float] = SKY_BLUE,
             num_subdivisions: int = DISCRETIZATION_LEVEL) -> None:
        """
        View the surface.

        :param color: color for the surface in the viewer
        :type color: tuple[float, float, float]
        :param num_subdivisions: number of subdivisions for the surface
        :type num_subdivisions: int
        :return: None
        """
        self.add_surface_to_viewer(color, num_subdivisions)
        polyscope.show()

    def screenshot(self,
                   filename: str,
                   camera_position: SpatialVector = np.array([[0.0, 0.0, 2.0]], dtype=np.float64),
                   camera_target: SpatialVector = np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
                   use_orthographic: bool = False) -> None:
        """
        Save a screenshot of the surface in the viewer.

        :param filename: file to save the screenshot.
        :type filename: str
        :param camera_position: camera position for the screenshot. (np.ndarray shape (1, 3))
        :type camera_position: SpatialVector
        :param camera_target: camera target for the screenshot. (np.ndarray shape (1, 3))
        :type camera_target: SpatialVector
        :param use_orthographic: use orthographic perspective if true.
        :type use_orthographic: bool

        :return: None
        """
        self.add_surface_to_viewer()

        polyscope.look_at(camera_position, camera_target)
        if use_orthographic:
            polyscope.set_view_projection_mode("orthographic")
        else:
            polyscope.set_view_projection_mode("perspective")
        polyscope.screenshot(filename)
        logger.info("Screenshot saved to %s", filename)
        polyscope.remove_all_structures()

    def serialize(self, output_file: TextIO) -> None:
        """
        Serialize the surface
        NOTE: used by write_spline()

        patch information in the format
            patch
            cx a_0 a_u a_v a_uv a_uu a_vv
            cy a_1 a_u a_v a_uv a_uu a_vv
            cz a_2 a_u a_v a_uv a_uu a_vv
            p1 p1_u p1_v
            p2 p2_u p2_v
            p3 p3_u p3_v

        :param output_file: [in] output stream for the surface
        :type output_file: TextIO
        """
        for i, _ in enumerate(self._patches):
            self._patches[i].serialize(output_file)

    @staticmethod
    def deserialize(input_file: TextIO) -> list[QuadraticSplineSurfacePatch]:
        """
        Deserialize a surface

        NOTE: used for testing with original ASOC code
        TODO: future implementations could port over to JSON for univeral formatting
        and better parsing
        NOTE: staticmethod so that this method could work without relying on
        class "self".

        patch information in the format
            patch
            cx a_0 a_u a_v a_uv a_uu a_vv
            cy a_1 a_u a_v a_uv a_uu a_vv
            cz a_2 a_u a_v a_uv a_uu a_vv
            p1 p1_u p1_v
            p2 p2_u p2_v
            p3 p3_u p3_v

        :param input_file: [in] input stream for the surface
        :type input_file: TextIO
        """
        # self._patches.clear()
        patches: list[QuadraticSplineSurfacePatch] = []
        patch_info_lines: list[str] = input_file.readlines()
        ROWS_OF_PATCH_INFORMATION = 7
        NUM_OF_ROWS: int = len(patch_info_lines)
        assert NUM_OF_ROWS % ROWS_OF_PATCH_INFORMATION == 0

        # -- Read coordinate coefficients cx, cy, and cz along with point information --
        # NOTE: this relies on there being 7 rows for patch format
        for i in range(0, NUM_OF_ROWS, ROWS_OF_PATCH_INFORMATION):
            # TODO: add better checking and optional comments ettter

            # TODO: use regex to verify cx, cy, cz pattern
            # Read coordinates (skipping the label and reading the float data)
            cx: Vector1D = np.array(
                list(map(float, patch_info_lines[i + 1].split()[1:])),
                dtype=np.float64)
            cy: Vector1D = np.array(
                list(map(float, patch_info_lines[i + 2].split()[1:])),
                dtype=np.float64)
            cz: Vector1D = np.array(
                list(map(float, patch_info_lines[i + 3].split()[1:])),
                dtype=np.float64)
            surface_mapping_coeffs: Matrix6x3f = np.stack((cx, cy, cz), axis=1, dtype=np.float64)
            assert surface_mapping_coeffs.shape == (6, 3)

            # TODO: use regex to verify p1, p2, p3 pattern
            p1: Vector1D = np.array(
                list(map(float, patch_info_lines[i + 4].split()[1:])), dtype=np.float64)
            p2: Vector1D = np.array(
                list(map(float, patch_info_lines[i + 5].split()[1:])), dtype=np.float64)
            p3: Vector1D = np.array(
                list(map(float, patch_info_lines[i + 6].split()[1:])), dtype=np.float64)
            vertices: Matrix3x2f = np.array([p1, p2, p3], dtype=np.float64)
            assert vertices.shape == (3, 2)

            domain: ConvexPolygon = ConvexPolygon.init_from_vertices(vertices)

            # Add patch to the spline surface
            patches.append(QuadraticSplineSurfacePatch(surface_mapping_coeffs, domain))

        return patches

    # TODO: this should be accessible OUTSIDE the class.
    def write_spline(self, filepath: str) -> None:
        """
        Write the surface serialization to file.
        NOTE: used in contour_network.py

        patch information in the format
            patch
            cx a_0 a_u a_v a_uv a_uu a_vv
            cy a_1 a_u a_v a_uv a_uu a_vv
            cz a_2 a_u a_v a_uv a_uu a_vv
            p1 p1_u p1_v
            p2 p2_u p2_v
            p3 p3_u p3_v

        :param filepath: [in] file path for the serialized surface
        :type filepath: str
        """
        logger.info("Writing spline to %s", filepath)

        # filepath: str = os.path.abspath(f"src\\tests\\spot_control\\{filename}")
        if os.path.isfile(filepath):
            logger.warning("Overwriting file at %s.", filepath)

        with open(filepath, 'w', encoding='utf-8') as output_file:
            self.serialize(output_file)
        output_file.close()

    def read_spline(self, filepath: str) -> list[QuadraticSplineSurfacePatch]:
        """
        Read a surface serialization from file.\n
        NOTE: method used for testing with ASOC code and to make sure 
        that the implementation is correct.
        NOTE: this exists for any subclasses that want to read from a
        serialized spline file as well.

        :param filepath: [in] file path for the serialized surface
        :return: patches to save to.
        :rtype: list[QuadraticSplineSurfacePatch]
        """
        input_file: TextIO
        patches: list[QuadraticSplineSurfacePatch] = []

        if not os.path.isfile(filepath):
            raise OSError(f"File does not exist at {filepath}. Choose a file to read spline from.")

        with open(filepath, 'r', encoding='utf-8') as input_file:
            patches = self.deserialize(input_file)
        input_file.close()

        return patches

    def compute_patch_hash_tables(self,
                                  patches_ref: list[QuadraticSplineSurfacePatch]
                                  ) -> list[list[list[int]]]:
        """
        Compute hash tables for the surface.
        NOTE: Used in twelve_split_spline.py inside of init_twelve_split_patches().
        """
        num_patch: int = self.num_patches
        hash_size_x: int = HASH_TABLE_SIZE
        hash_size_y: int = HASH_TABLE_SIZE

        # Clear the hash table
        # NOTE: hash_table just going to be recreated in this method.
        # NOTE: hash_table is HASH_TABLE_SIZE x HASH_TABLE_SIZE 2D list with elements list[int]
        hash_table: list[list[list[int]]] = [
            [[] for _ in range(hash_size_x)]
            for _ in range(hash_size_x)
        ]

        # Compute bounding box for all the patches
        # Alias for readability
        x_min: float
        x_max: float
        y_min: float
        y_max: float
        x_min, x_max, y_min, y_max = self.__compute_patches_bbox(patches_ref)
        self.__patches_bbox_x_min = x_min
        # self.__patches_bbox_x_max = x_max
        self.__patches_bbox_y_min = y_min
        # self.__patches_bbox_y_max = y_max

        for i in range(1, num_patch):
            if x_min > patches_ref[i].get_bbox_x_min():
                x_min = patches_ref[i].get_bbox_x_min()
            if x_max < patches_ref[i].get_bbox_x_max():
                x_max = patches_ref[i].get_bbox_x_max()
            if y_min > patches_ref[i].get_bbox_y_min():
                y_min = patches_ref[i].get_bbox_y_min()
            if y_max < patches_ref[i].get_bbox_y_max():
                y_max = patches_ref[i].get_bbox_y_max()

        x_interval: float = (x_max - x_min) / hash_size_x
        y_interval: float = (y_max - y_min) / hash_size_y

        # TODO: below are not used anywhere...
        self.hash_x_interval = x_interval
        self.hash_y_interval = y_interval

        eps: float = 1e-10

        # Hash into each box
        for i in range(num_patch):
            left_x: int = int((patches_ref[i].get_bbox_x_min() - eps - x_min) / x_interval)
            right_x: int = int(
                hash_size_x - int((x_max - patches_ref[i].get_bbox_x_max() - eps) / x_interval) - 1)
            left_y: int = int((patches_ref[i].get_bbox_y_min() - eps - y_min) / y_interval)
            right_y: int = int(
                hash_size_y - int((y_max - patches_ref[i].get_bbox_y_max() - eps) / y_interval) - 1)

            for j in range(left_x, right_x + 1):
                for k in range(left_y, right_y + 1):
                    hash_table[j][k].append(i)

        return hash_table

    def compute_hash_indices(self, point: PlanarPoint) -> tuple[int, int]:
        """
        Compute the hash indices of a point in the plane.
        NOTE: Used in compute_ray_intersections.py

        :param point: PlanarPoint object of shape (1, 2) to convert to hash table x and y values
        :type point: PlanarPoint

        :return: tuple of hash_x and hash_y computed.
        """
        assert point.shape == (2, )

        # todo("Fix member variables")
        hash_x = int((point[0] - self.__patches_bbox_x_min) // self.hash_x_interval)
        hash_y = int((point[1] - self.__patches_bbox_y_min) // self.hash_y_interval)

        if (hash_x < 0) or (hash_x >= HASH_TABLE_SIZE):
            logger.error("x hash index out of bounds")
            hash_x: int = max(min(hash_x, HASH_TABLE_SIZE - 1), 0)

        if (hash_y < 0) or (hash_y >= HASH_TABLE_SIZE):
            logger.error("y hash index out of bounds")
            hash_y: int = max(min(hash_y, HASH_TABLE_SIZE - 1), 0)

        return (hash_x, hash_y)

    # ***************
    # Private Methods
    # ***************

    def __is_valid_patch_index(self, patch_index: PatchIndex) -> bool:
        """
        Determine if a patch index is valid
        """
        if patch_index >= self.num_patches:
            return False

        return True

    def __compute_patches_bbox(self,
                               patches_ref: list[QuadraticSplineSurfacePatch]
                               ) -> tuple[float, float, float, float]:
        """
        Compute bounding boxes for the patches.
        As in, calculates values for variables below:
        - patches_bbox_x_min
        - patches_bbox_x_max
        - patches_bbox_y_min
        - patches_bbox_y_max

        :return: tuple (x_min, x_max, y_min, y_max):
        :rtype: tuple[float, float, float, float]
        """
        x_min: float = patches_ref[0].get_bbox_x_min()
        x_max: float = patches_ref[0].get_bbox_x_max()
        y_min: float = patches_ref[0].get_bbox_y_min()
        y_max: float = patches_ref[0].get_bbox_y_max()

        for i in range(1, self.num_patches):
            if x_min > patches_ref[i].get_bbox_x_min():
                x_min = patches_ref[i].get_bbox_x_min()
            if x_max < patches_ref[i].get_bbox_x_max():
                x_max = patches_ref[i].get_bbox_x_max()
            if y_min > patches_ref[i].get_bbox_y_min():
                y_min = patches_ref[i].get_bbox_y_min()
            if y_max < patches_ref[i].get_bbox_y_max():
                y_max = patches_ref[i].get_bbox_y_max()

        # Alias for readability.
        patches_bbox_x_min: float = x_min
        patches_bbox_x_max: float = x_max
        patches_bbox_y_min: float = y_min
        patches_bbox_y_max: float = y_max

        return patches_bbox_x_min, patches_bbox_x_max, patches_bbox_y_min, patches_bbox_y_max
