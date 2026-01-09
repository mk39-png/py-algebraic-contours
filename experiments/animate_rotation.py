"""
Script that generates animated rotation of a particular mesh
"""

import argparse
import logging
import pathlib
import sys

import igl
import numpy as np
from cholespy import CholeskySolverD
from scipy.sparse import csr_matrix

from pyalgcon.contour_network.compute_intersections import \
    IntersectionParameters
from pyalgcon.contour_network.contour_network import (ContourNetwork,
                                                      InvisibilityParameters)
from pyalgcon.core.affine_manifold import AffineManifold
from pyalgcon.core.apply_transformation import \
    apply_camera_frame_transformation_to_vertices
from pyalgcon.core.common import (Matrix3x3f, MatrixNx3f, MatrixXf, MatrixXi,
                                  screenshot_mesh)
from pyalgcon.core.generate_transformation import \
    y_axis_rotation_projective_matrix
from pyalgcon.quadratic_spline_surface.optimize_spline_surface import \
    OptimizationParameters
from pyalgcon.quadratic_spline_surface.twelve_split_spline import (
    TwelveSplitSplineSurface, compute_twelve_split_spline_patch_boundary_edges)

logger: logging.Logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> int:
    """
    Runs the primary program
    """
    # Retrieve inputs
    # input_filename: str = ""
    # output_dir: str = "./"
    # view_surface: bool = False

    input_filename = pathlib.Path(args.input)
    output_dir = pathlib.Path(args.output)
    view_surface: bool = args.view_surface

    # Set logging and discretization level
    DISCRETIZATION_LEVEL: int = 2
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

    # Generate quadratic spline
    optimization_params: OptimizationParameters = OptimizationParameters()
    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)

    spline_surface: TwelveSplitSplineSurface = TwelveSplitSplineSurface(
        initial_V, affine_manifold, optimization_params)
    energy_hessian_inverse: CholeskySolverD = spline_surface.energy_hessian_inverse
    fit_matrix: csr_matrix = spline_surface.fit_matrix

    # Get the boundary edges
    face_to_patch_indices: list[list[int]] = spline_surface.face_to_patch_indices
    patch_boundary_edges: list[tuple[int, int]] = compute_twelve_split_spline_patch_boundary_edges(
        F, face_to_patch_indices)

    pert_frame: MatrixXf = np.array([
        [0.9218127938, -0.3785297887, 0.08352467839],
        [0.3815369887, 0.9240681946, -0.0229673378],
        [-0.06848867707, 0.0530393400, 0.9962410000]])
    pert: float = 1e-2
    theta: float = 0
    dx: float = 1
    output_path: pathlib.Path

    for i in range(360):
        # Rotate the mesh
        theta = dx * i + pert
        rotation_matrix: MatrixXf = y_axis_rotation_projective_matrix(theta)
        frame: Matrix3x3f = rotation_matrix[:3, :3] @ pert_frame
        V = apply_camera_frame_transformation_to_vertices(initial_V, frame)

        # Optionally screenshot the mesh
        if view_surface:
            output_path = output_dir / f"mesh_animation_frame_{i}.png"
            screenshot_mesh(V, F, output_path, np.array([0, 0, -1]), np.array([0, 0, 0]), True)

        # Update the spline surface vertex positions
        spline_surface.update_positions(V, fit_matrix, energy_hessian_inverse)

        # Optionally screenshot the surface
        if view_surface:
            output_path = output_dir / f"surface_animation_frame_{i}.png"
            spline_surface.screenshot(output_path, np.array([0, 0, -1]), np.array([0, 0, 0]), True)

        # Build the contours
        intersect_params: IntersectionParameters = IntersectionParameters()
        invisibility_params: InvisibilityParameters = InvisibilityParameters()
        contour_network: ContourNetwork = ContourNetwork(spline_surface,
                                                         intersect_params,
                                                         invisibility_params,
                                                         patch_boundary_edges)

        # Optionally write the raster contours if viewing the surface
        if view_surface:
            output_path = output_dir / f"animation_frame_{i}.png"
            contour_network.write_rasterized_contours(output_path)

        # Write the contours
        output_path = output_dir / f"animation_frame_{i}.svg"
        contour_network.write(output_path)

    return 0


# TODO: stack overflow post about checking if valid file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="animate_rotation", description="Generate rotation animation for a given mesh.")
    parser.add_argument("-i", "--input", type=str, help="Mesh filepath.", required=True)
    parser.add_argument("-o", "--output", type=str, default="./", help="Output directory")
    parser.add_argument("--view_surface",    action="store_true",
                        help="View mesh and surface rotation")
    args: argparse.Namespace = parser.parse_args()

    sys.exit(main(args))
