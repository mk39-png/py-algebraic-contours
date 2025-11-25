"""
This is the user-facing interface to input a mesh and generate the algebraic contours.
"""
import os

import igl
import numpy as np
from mathutils import Matrix
from numpy.typing import ArrayLike

from pyalgcon.contour_network.compute_intersections import \
    IntersectionParameters
from pyalgcon.contour_network.contour_network import (
    ContourNetwork, InvisibilityParameters)
from pyalgcon.core.affine_manifold import AffineManifold
from pyalgcon.core.apply_transformation import \
    apply_transformation_to_vertices
from pyalgcon.core.common import MatrixNx3f
from pyalgcon.quadratic_spline_surface.optimize_spline_surface import \
    OptimizationParameters
from pyalgcon.quadratic_spline_surface.twelve_split_spline import (
    TwelveSplitSplineSurface, compute_twelve_split_spline_patch_boundary_edges)

from ...common import DIRECTORY_TEMP
from pyalgcon.utils.projected_curve_networks_utils import SVGOutputMode


# NOTE: this should also take some file...
# To preserve state or something?
def generate_algebraic_contours(projection_matrix: Matrix) -> None:
    """
    Testing contour network creation with control spot mesh.
    Reads from the temporary file.

    :param: projection matrix
    """
    # TODO: implement some ability for the user to change these parameters
    svg_output_mode = SVGOutputMode.UNIFORM_VISIBLE_CURVES
    optimization_params = OptimizationParameters()
    intersect_params = IntersectionParameters()
    invisibility_params = InvisibilityParameters()
    show_nodes: bool = False

    # Retrieve the uv unwrapped mesh
    V: ArrayLike
    uv: ArrayLike
    N: ArrayLike
    F: ArrayLike
    FT: ArrayLike
    FN: ArrayLike
    V, uv, N, F, FT, FN = igl.readOBJ(os.path.join(DIRECTORY_TEMP, "temp_out.obj"))
    print(projection_matrix)

    # Set up the camera
    # FIXME: probably using the WRONG PROJECTION MATRIX, hence we get missclassification.
    # Though, export .obj to original ASOC code to see how it runs
    # FIXME: also run with matrix to compare in ASOC code.
    # frame = np.array([[1, 0, 0],
    #                   [0, 1, 0],
    #                   [0, 0, 1]])
    # V_transformed: MatrixNx3f = apply_camera_frame_transformation_to_vertices(V, frame)
    # print(projection_matrix)

    # Preparing mesh data for use in contours calculation
    # TODO: will this work? The whole conversion of the MathUtils matrix to NumPy matrix?
    # TODO: maybe filter out non-numbers like 8.22e-16 and set to 0
    V_transformed: MatrixNx3f = apply_transformation_to_vertices(V, np.array(projection_matrix))
    # print(V_transformed)
    # print(V_transformed.shape)

    # Generate quadratic spline
    print("Computing spline surface...")
    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)
    # TODO: should cache this result somewhere since the camera can be
    spline_surface: TwelveSplitSplineSurface = TwelveSplitSplineSurface(V_transformed,
                                                                        affine_manifold,
                                                                        optimization_params)

    # Get the boundary edges
    patch_boundary_edges: list[tuple[int, int]] = (
        compute_twelve_split_spline_patch_boundary_edges(F, spline_surface.face_to_patch_indices))

    # Build the contours
    print("Computing contours...")
    contour_network: ContourNetwork = ContourNetwork(
        spline_surface,
        intersect_params,
        invisibility_params,
        patch_boundary_edges
    )

    # Save the contours to file
    # logger.info("Saving contours")
    contour_network_filename: str = "contours.svg"
    contour_network_filepath: str = os.path.join(DIRECTORY_TEMP, f"{contour_network_filename}")
    print(contour_network_filepath)
    contour_network.write(contour_network_filepath, svg_output_mode, show_nodes)
