from collections import defaultdict

from pyalgcon.core.affine_manifold import *
from pyalgcon.core.common import *
from pyalgcon.quadratic_spline_surface.optimize_spline_surface import (
    TriangleCornerData, TriangleMidpointData)

from pyalgcon.utils.generate_shapes import *

# ***************************
# Parametric surface functors
# ***************************


class QuadraticPositionFunction:
    """
    Used in twelve_split_spline testing.
    """

    def __init__(self,
                 input_uv_coeff: float,
                 input_uu_coeff: float,
                 input_vv_coeff: float) -> None:
        self.__uv_coeff: float = input_uv_coeff
        self.__uu_coeff: float = input_uu_coeff
        self.__vv_coeff: float = input_vv_coeff

    def __call__(self, u: float, v: float) -> SpatialVector:
        """
        Method that allows for translation of coeffs into UV point space... I think.
        """
        point: SpatialVector = np.array([
            u,
            v,
            self.uv_coeff * u * v + self.uu_coeff * u * u + self.vv_coeff * v * v
        ], dtype=float)

        assert point.shape == (3, )
        return point

    @property
    def uv_coeff(self) -> float:
        """Return uv coeff"""
        return self.__uv_coeff

    @property
    def uu_coeff(self) -> float:
        """Return uu coeff"""
        return self.__uu_coeff

    @property
    def vv_coeff(self) -> float:
        """Return vv coeff"""
        return self.__vv_coeff


class QuadraticGradientFunction:
    """
    Used in twelve_split_spline testing.
    """

    def __init__(self,
                 input_uv_coeff: float,
                 input_uu_coeff: float,
                 input_vv_coeff: float) -> None:
        self.__uv_coeff: float = input_uv_coeff
        self.__uu_coeff: float = input_uu_coeff
        self.__vv_coeff: float = input_vv_coeff

    def __call__(self, u: float, v: float) -> Matrix2x3r:
        """
        Method that allows for translation of coeffs into UV gradient space... I think.
        """
        gradient: Matrix2x3r = np.array([
            # First row, generate derivative with respect to u
            [1.0, 0.0, self.uv_coeff * v + 2 * self.uu_coeff * u],

            # Second row, generate derivative with respect to v
            [0.0, 1.0, self.uv_coeff * u + 2 * self.vv_coeff * v]

        ], dtype=float)

        assert gradient.shape == (2, 3)
        return gradient

    @property
    def uv_coeff(self) -> float:
        """Return uv coeff"""
        return self.__uv_coeff

    @property
    def uu_coeff(self) -> float:
        """Return uu coeff"""
        return self.__uu_coeff

    @property
    def vv_coeff(self) -> float:
        """Return vv coeff"""
        return self.__vv_coeff


# class TorusPositionFunction:
#     unimplemented()


# class TorusGradientFunction:
#     unimplemented()


def generate_parametric_affine_manifold_vertex_positions():
    unimplemented()


def generate_parametric_affine_manifold_vertex_gradients():
    unimplemented()


def generate_parametric_affine_manifold_edge_gradients():
    unimplemented()


def generate_parametric_affine_manifold_corner_data(position_func: QuadraticPositionFunction,
                                                    gradient_func: QuadraticGradientFunction,
                                                    parametric_affine_manifold: ParametricAffineManifold
                                                    ) -> dict[int, dict[int, TriangleCornerData]]:
    """
    Used in twelve_split_spline testing.
    FIXME: Generalize to any position function

    :return: corner_data of list of list of length 3
    """
    num_faces: int = parametric_affine_manifold.num_faces
    # corner_data: list[list[TriangleCornerData]] = [[] for _ in range(num_faces)]

    corner_data: dict[int, dict[int, TriangleCornerData]] = defaultdict(dict)

    for face_index in range(num_faces):
        # Get face vertex indices
        F: np.ndarray = parametric_affine_manifold.faces  # dtype == int...
        i: int = F[face_index, 0]
        j: int = F[face_index, 1]
        k: int = F[face_index, 2]

        # Get vertex uv positions
        uvi: PlanarPoint = parametric_affine_manifold.get_vertex_global_uv(i)
        uvj: PlanarPoint = parametric_affine_manifold.get_vertex_global_uv(j)
        uvk: PlanarPoint = parametric_affine_manifold.get_vertex_global_uv(k)

        # Get vertex positions
        vi: SpatialVector = position_func(uvi[0], uvi[1])
        vj: SpatialVector = position_func(uvj[0], uvj[1])
        vk: SpatialVector = position_func(uvk[0], uvk[1])

        # Get vertex gradients
        Gi: Matrix2x3r = gradient_func(uvi[0], uvi[1])
        Gj: Matrix2x3r = gradient_func(uvj[0], uvj[1])
        Gk: Matrix2x3r = gradient_func(uvk[0], uvk[1])

        # Get uv directions
        dij: PlanarPoint = uvj - uvi
        dik: PlanarPoint = uvk - uvi
        djk: PlanarPoint = uvk - uvj
        dji: PlanarPoint = uvi - uvj
        dki: PlanarPoint = uvi - uvk
        dkj: PlanarPoint = uvj - uvk

        # Building corner data
        first_corner_data = TriangleCornerData(input_function_value=vi,
                                               input_first_edge_derivative=dij @ Gi,
                                               input_second_edge_derivative=dik @ Gi)

        second_corner_data = TriangleCornerData(input_function_value=vj,
                                                input_first_edge_derivative=djk @ Gj,
                                                input_second_edge_derivative=dji @ Gj)

        third_corner_data = TriangleCornerData(input_function_value=vk,
                                               input_first_edge_derivative=dki @ Gk,
                                               input_second_edge_derivative=dkj @ Gk)
        corner_data[face_index][0] = first_corner_data
        corner_data[face_index][1] = second_corner_data
        corner_data[face_index][2] = third_corner_data

    return corner_data


def generate_parametric_affine_manifold_midpoint_data(gradient_func: QuadraticGradientFunction,
                                                      parametric_affine_manifold: ParametricAffineManifold,
                                                      ) -> dict[int, dict[int, TriangleMidpointData]]:
    """
    Used in twelve_split_spline testing.

    :return: list[list[TriangleMidpointData]] of list of list of length 3
    """
    num_faces: int = parametric_affine_manifold.num_faces

    # midpoint_data: list[list[TriangleMidpointData]] = [[] for _ in range(num_faces)]
    midpoint_data: dict[int, dict[int, TriangleMidpointData]] = defaultdict(dict)

    for face_index in range(num_faces):
        # Get face vertex indices
        F: np.ndarray = parametric_affine_manifold.faces
        i: int = F[face_index, 0]
        j: int = F[face_index, 1]
        k: int = F[face_index, 2]

        # Get vertex uv positions
        uvi: PlanarPoint = parametric_affine_manifold.get_vertex_global_uv(i)
        uvj: PlanarPoint = parametric_affine_manifold.get_vertex_global_uv(j)
        uvk: PlanarPoint = parametric_affine_manifold.get_vertex_global_uv(k)

        # Get midpoint uv positions
        uvij: PlanarPoint = 0.5 * (uvi + uvj)
        uvjk: PlanarPoint = 0.5 * (uvj + uvk)
        uvki: PlanarPoint = 0.5 * (uvk + uvi)

        # Get midpoint gradients
        # TODO: again with the confusing index accessing for PlanarPoint...
        Gij: Matrix2x3r = gradient_func(uvij[0], uvij[1])
        Gjk: Matrix2x3r = gradient_func(uvjk[0], uvjk[1])
        Gki: Matrix2x3r = gradient_func(uvki[0], uvki[1])

        # Get uv directions
        nij: PlanarPoint = uvk - uvij
        njk: PlanarPoint = uvi - uvjk
        nki: PlanarPoint = uvj - uvki

        # Build midpoint normals (indexed opposite the edge)
        midpoint_data[face_index][0] = TriangleMidpointData(njk @ Gjk)
        midpoint_data[face_index][1] = TriangleMidpointData(nki @ Gki)
        midpoint_data[face_index][2] = TriangleMidpointData(nij @ Gij)

    return midpoint_data
