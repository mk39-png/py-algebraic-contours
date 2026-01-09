"""
"""
import argparse
import logging
import pathlib
import sys

import igl
import numpy as np
import polyscope
from cholespy import CholeskySolverD
from scipy.sparse import csr_matrix

from pyalgcon.core.affine_manifold import AffineManifold
from pyalgcon.core.apply_transformation import (
    apply_camera_frame_transformation_to_vertices,
    apply_transformation_to_vertices,
    apply_transformation_to_vertices_in_place)
from pyalgcon.core.common import (Matrix4x4f, MatrixNx3f, MatrixXi,
                                  deserialize_eigen_matrix_csv_to_numpy,
                                  screenshot_mesh)
from pyalgcon.core.generate_transformation import \
    axis_rotation_projective_matrix
from pyalgcon.quadratic_spline_surface.optimize_spline_surface import \
    OptimizationParameters
from pyalgcon.quadratic_spline_surface.twelve_split_spline import \
    TwelveSplitSplineSurface

logger: logging.Logger = logging.getLogger(__name__)


def main(args):
    # Get command line arguments
    # TODO: wrap in pathlib
    input_filename: pathlib.Path = pathlib.Path(args.input)
    output_dir: pathlib.Path = pathlib.Path(args.output)
    camera_filename: pathlib.Path = pathlib.Path(args.camera)
    w_f: float = args.weight

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

    # Set up Polyscope
    polyscope.init()
    polyscope.set_ground_plane_mode("shadow_only")

    # Set up the camera
    if camera_filename == "":
        frame: np.ndarray = np.identity(3)
        theta_x: float = 0
        theta_y: float = 160
        theta_z: float = 0
        rotation_matrix: Matrix4x4f = axis_rotation_projective_matrix(theta_x, theta_y, theta_z)
        logger.info("Projecting onto frame\n %s", frame)
        frame = rotation_matrix[:3, :3]
        orthographic: bool = False
        V = apply_camera_frame_transformation_to_vertices(V, frame, orthographic)
    else:
        camera_matrix: Matrix4x4f = deserialize_eigen_matrix_csv_to_numpy(camera_filename)
        logger.info("Using camera matrix:\n%s", camera_matrix)
        V = apply_transformation_to_vertices(initial_V, camera_matrix)

    # View the initial mesh
    screenshot_mesh(V,
                    F,
                    output_dir / "mesh.png",
                    np.array([0, 0, 0]),
                    np.array([0, 0, 1]))

    # Generate quadratic spline
    optimization_params: OptimizationParameters = OptimizationParameters(
        position_difference_factor=w_f)
    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)
    spline_surface: TwelveSplitSplineSurface = TwelveSplitSplineSurface(
        V, affine_manifold, optimization_params)

    # Retrieving values from Twelve Split Spline surface
    face_to_patch_indices: list[list[int]] = spline_surface.face_to_patch_indices
    patch_to_face_indices: list[int] = spline_surface.patch_to_face_indices
    fit_matrix: csr_matrix = spline_surface.fit_matrix
    energy_hessian: csr_matrix = spline_surface.energy_hessian
    energy_hessian_inverse: CholeskySolverD = spline_surface.energy_hessian_inverse

    # View the quadratic surface
    spline_surface.screenshot(
        output_dir / "spline_surface.png",
        np.array([0, 0, 0]),
        np.array([0, 0, 1]),
        False)

    return 0


if __name__ == '__main__':
    # Get command line arguments
    parser = argparse.ArgumentParser(
        prog="animate_rotation", description="Generate smoothness figure images for a given mesh and camera.")
    parser.add_argument("-i", "--input", type=str, help="Mesh filepath.", required=True)
    parser.add_argument("-c", "--camera", type=str, default="", help="Camera filepath")
    parser.add_argument("-o", "--output", type=str, default="./", help="Output directory")
    parser.add_argument("-w", "--weight", type=float, help="Fitting weight")
    args: argparse.Namespace = parser.parse_args()

    sys.exit(main(args))
