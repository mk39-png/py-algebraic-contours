"""
Executable to generate a pipeline example from a mesh and camera specification.
Generates an image of the mesh, the quadratic surface (with front camera view),
the full contours with cusps and intersections, and the final occluding contours.

The following rotation angles were originally used for the below meshes, in
order x, y, z.

Spot 1 -30 1
Bigguy: 0 150 0
Blub: 0 150 0
Fertility: 0 150 0
Bob: 15 -30 15
Killaroo: 90 30 0
Monsterfrog: 0 150 0
Pawn: 90 0 0
Pipes: 0 170 0
Ogre 0 150 0
Toad 75 130 0
"""

import argparse
import logging
import pathlib
import sys

import igl
import numpy as np
from cholespy import CholeskySolverD
from contour_network.compute_intersections import IntersectionParameters
from contour_network.contour_network import (ContourNetwork,
                                             InvisibilityParameters)
from core.affine_manifold import AffineManifold
from core.apply_transformation import (
    apply_transformation_to_vertices,
    apply_transformation_to_vertices_in_place)
from core.common import (Matrix4x4f, MatrixNx3f, MatrixXi,
                         deserialize_eigen_matrix_csv_to_numpy)
from core.generate_transformation import origin_to_infinity_projective_matrix
from quadratic_spline_surface.optimize_spline_surface import \
    OptimizationParameters
from quadratic_spline_surface.twelve_split_spline import (
    TwelveSplitSplineSurface, compute_twelve_split_spline_patch_boundary_edges)
from scipy.sparse import csr_matrix
from utils.projected_curve_networks_utils import SVGOutputMode

logger: logging.Logger = logging.getLogger(__name__)


def main(args):
    """
    Main logic
    """
    # Get command line arguments
    # TODO: wrap in pathlib
    input_filename: pathlib.Path = pathlib.Path(args.input)
    output_dir: pathlib.Path = pathlib.Path(args.output)
    camera_filename: pathlib.Path = pathlib.Path(args.camera)

    # Set logger level
    logger.setLevel(logging.NOTSET)

    # Get input mesh
    initial_V: MatrixNx3f
    V: MatrixNx3f
    uv: np.ndarray
    N: np.ndarray
    F: MatrixXi
    FT: MatrixXi
    FN: MatrixXi
    initial_V, uv, N, F, FT, FN = igl.readOBJ(input_filename)

    # Set up the camera
    camera_to_plane_distance: float = 1.0
    camera_matrix: Matrix4x4f = deserialize_eigen_matrix_csv_to_numpy(camera_filename)
    logger.info("Using camera matrix:\n%s", camera_matrix)

    # Apply camera and perspective projection transformations
    projection_matrix: Matrix4x4f = origin_to_infinity_projective_matrix(camera_to_plane_distance)
    projection_matrix = projection_matrix @ camera_matrix
    V = apply_transformation_to_vertices(initial_V, projection_matrix)

    # Generate quadratic spline
    logger.info("Computing spline surface")
    optimization_params: OptimizationParameters = OptimizationParameters()
    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)
    spline_surface: TwelveSplitSplineSurface = TwelveSplitSplineSurface(
        V, affine_manifold, optimization_params)

    # Retrieving values from Twelve Split Spline surface
    face_to_patch_indices: list[list[int]] = spline_surface.face_to_patch_indices
    patch_to_face_indices: list[int] = spline_surface.patch_to_face_indices
    fit_matrix: csr_matrix = spline_surface.fit_matrix
    energy_hessian: csr_matrix = spline_surface.energy_hessian
    energy_hessian_inverse: CholeskySolverD = spline_surface.energy_hessian_inverse

    # Get the boundary edge
    patch_boundary_edges: list[tuple[int, int]] = compute_twelve_split_spline_patch_boundary_edges(
        F, face_to_patch_indices)

    # Build the contours
    logger.info("Computing contours")
    intersect_params = IntersectionParameters()
    invisibility_params = InvisibilityParameters()
    contour_network: ContourNetwork = ContourNetwork(spline_surface,
                                                     intersect_params, invisibility_params,
                                                     patch_boundary_edges)

    # View the contours
    affine_manifold.screenshot(
        output_dir / "mesh.png",
        V,
        np.array([0, 0, -1]),
        np.array([0, 0, 0]),
        True
    )
    contour_network.screenshot(
        output_dir / "spline_surface.png",
        spline_surface,
        (0, 0, -1),
        (0, 0, 0),
        True
    )
    contour_network.write(
        output_dir / "full_contours.svg",
        SVGOutputMode.CONTRAST_INVISIBLE_SEGMENTS,
        True
    )
    contour_network.write(
        output_dir / "contours.svg",
        SVGOutputMode.UNIFORM_VISIBLE_CHAINS,
        False
    )
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="animate_rotation",
        description="Generate example figure images for a given mesh and camera.")
    parser.add_argument("-i", "--input", type=str, help="Mesh filepath.", required=True)
    parser.add_argument("-c", "--camera", type=str, default="", help="Camera filepath")
    parser.add_argument("-o", "--output", type=str, default="./", help="Output directory")
    args: argparse.Namespace = parser.parse_args()

    sys.exit(main(args))
