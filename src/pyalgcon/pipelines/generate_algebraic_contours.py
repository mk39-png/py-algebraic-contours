"""
generate_algebraic_contours.py
This is the user-facing interface to input a mesh and generate the twelve split spline 
quadratic spline surface.
"""
import logging
import pathlib

import igl
from numpy.typing import ArrayLike

from pyalgcon.contour_network.compute_intersections import \
    IntersectionParameters
from pyalgcon.contour_network.contour_network import (ContourNetwork,
                                                      InvisibilityMethod,
                                                      InvisibilityParameters)
from pyalgcon.core.affine_manifold import AffineManifold
from pyalgcon.core.apply_transformation import (
    apply_camera_frame_transformation_to_vertices,
    apply_camera_matrix_transformation_to_vertices,
    apply_transformation_to_vertices)
from pyalgcon.core.common import Matrix4x4f, MatrixNx3f
from pyalgcon.core.generate_transformation import \
    origin_to_infinity_projective_matrix
from pyalgcon.quadratic_spline_surface.optimize_spline_surface import \
    OptimizationParameters
from pyalgcon.quadratic_spline_surface.twelve_split_spline import (
    TwelveSplitSplineSurface, compute_twelve_split_spline_patch_boundary_edges)
from pyalgcon.utils.projected_curve_networks_utils import SVGOutputMode

logger: logging.Logger = logging.getLogger(__name__)


def generate_algebraic_contours(camera_matrix: Matrix4x4f,
                                filepath: pathlib.Path) -> None:
    """
    Runs end-to-end pipeline of contour generation from mesh parsing to contour generation itself.

    :param camera_matrix: 4x4 matrix containing the rotation frame and translation vector
    :param filepath: output filepath 
    :return: None
    """
    # TODO: implement some ability for the user to change these parameters
    svg_output_mode: SVGOutputMode = SVGOutputMode.CONTRAST_INVISIBLE_SEGMENTS
    optimization_params = OptimizationParameters()
    intersect_params = IntersectionParameters()
    invisibility_params = InvisibilityParameters(invisibility_method=InvisibilityMethod.CHAINING)
    show_nodes: bool = False

    # Retrieve the uv unwrapped mesh
    V: ArrayLike
    uv: ArrayLike
    N: ArrayLike
    F: ArrayLike
    FT: ArrayLike
    FN: ArrayLike
    V, uv, N, F, FT, FN = igl.readOBJ(filepath)
    # print(projection_matrix)

    # Set up the camera
    # FIXME: probably using the WRONG PROJECTION MATRIX, hence we get missclassification.
    # Though, export .obj to original ASOC code to see how it runs
    # FIXME: also run with matrix to compare in ASOC code.
    # frame = np.array([[1, 0, 0],
    #                   [0, 1, 0],
    #                   [0, 0, 1]])
    # V_transformed: MatrixNx3f = apply_camera_frame_transformation_to_vertices(V, frame)
    # print(projection_matrix)

    # Utilize the origin to infinity projection matrix to see if that changes anything
    # camera_distance_to_plane: float = 1.0
    # camera_matrix: Matrix4x4f = projection_matrix  # alias for debugging
    # projection_matrix_ASOC: Matrix4x4f = origin_to_infinity_projective_matrix(
    #     camera_distance_to_plane)
    # print("prjection matrix asoc")
    # print(projection_matrix_ASOC)
    # projection_matrix_ASOC = projection_matrix_ASOC @ camera_matrix
    # V_transformed: MatrixNx3f = apply_transformation_to_vertices(V, projection_matrix_ASOC)

    V_transformed: MatrixNx3f = apply_camera_matrix_transformation_to_vertices(
        V, camera_matrix)

    # Preparing mesh data for use in contours calculation
    # TODO: will this work? The whole conversion of the MathUtils matrix to NumPy matrix?
    # TODO: maybe filter out non-numbers like 8.22e-16 and set to 0
    # V_transformed: MatrixNx3f = apply_transformation_to_vertices(V, projection_matrix)
    # print(V_transformed)
    # print(V_transformed.shape)

    # Generate quadratic spline
    logger.info("Computing spline surface...")
    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)
    # TODO: should cache this result somewhere since the camera can be
    spline_surface: TwelveSplitSplineSurface = TwelveSplitSplineSurface(V_transformed,
                                                                        affine_manifold,
                                                                        optimization_params)

    # Get the boundary edges
    patch_boundary_edges: list[tuple[int, int]] = (
        compute_twelve_split_spline_patch_boundary_edges(F, spline_surface.face_to_patch_indices))

    # Build the contours
    logger.info("Computing contours...")
    contour_network: ContourNetwork = ContourNetwork(
        spline_surface,
        intersect_params,
        invisibility_params,
        patch_boundary_edges
    )

    # Save the contours to file
    logger.info("Saving contours")
    # contour_network_filename: str = "contours.svg"
    # contour_network_filepath: pathlib.Path = directory_temp / f"{contour_network_filename}"
    try:
        # Write to SVG file
        # contour_network.write(directory_temp.parent / "contours.svg", svg_output_mode, show_nodes)
        contour_network.write(filepath, svg_output_mode, show_nodes)

        # Write to png
        # contour_network.write_rasterized_contours("contours.png")
    except (IOError):
        logger.error("FAILED TO WRITE CONTOURS TO DIRECTORY %s", filepath)
