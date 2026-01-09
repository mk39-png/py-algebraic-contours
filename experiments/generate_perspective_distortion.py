"""
Generate perspective view of a mesh and its orthographic distortion equivalent.
"""
import argparse
import logging
import pathlib
import sys

import igl
import numpy as np
import polyscope
from cholespy import CholeskySolverD
from core.affine_manifold import AffineManifold
from quadratic_spline_surface.twelve_split_spline import \
    TwelveSplitSplineSurface
from scipy.sparse import csr_matrix

from pyalgcon.core.apply_transformation import (
    apply_camera_frame_transformation_to_vertices,
    apply_transformation_to_vertices)
from pyalgcon.core.common import (Matrix3x3f, Matrix4x4f, MatrixNx3f, MatrixXf,
                                  MatrixXi,
                                  deserialize_eigen_matrix_csv_to_numpy)
from pyalgcon.core.convert_conic import \
    compute_symmetric_matrix_eigen_decomposition
from pyalgcon.core.generate_transformation import (
    axis_rotation_projective_matrix, origin_to_infinity_projective_matrix)
from pyalgcon.quadratic_spline_surface.optimize_spline_surface import \
    OptimizationParameters

logger: logging.Logger = logging.getLogger(__name__)


def main(args):
    """
    Main logic utilizing user input
    """
    input_filename: pathlib.Path = pathlib.Path(args.input)
    output_dir: pathlib.Path = pathlib.Path(args.output)
    camera_filename: pathlib.Path = pathlib.Path(args.camera)
    translation: float = args.translation
    perspective_fov: float = args.perspective_fov
    orthographic_fov: float = args.orthographic_fov

    # Set logging and discretization level
    DISCRETIZATION_LEVEL = 2
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
    polyscope.set_ground_plane_mode("none")

    # Set up the camera
    if camera_filename == "":
        rotation_matrix: MatrixXf = axis_rotation_projective_matrix(100, 10, 0)
        rotation_frame: Matrix3x3f = rotation_matrix[:3, :3]
        logger.info("Projecting onto frame:\n%s", rotation_frame)
        orthographic: bool = False
        V = apply_camera_frame_transformation_to_vertices(initial_V, rotation_frame, orthographic)
    else:
        camera_matrix: Matrix4x4f = deserialize_eigen_matrix_csv_to_numpy(camera_filename)
        logger.info("Using camera matrix:\n%s", camera_matrix)
        V = apply_transformation_to_vertices(initial_V, camera_matrix)

    # Apply additional translation
    num_vertices: int = V.shape[0]
    for i in range(num_vertices):
        V[i, :] = V[i, :] + np.array([0, 0, translation])

    # Generate quadratic spline
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

    # View the perspective spline surface
    polyscope.set_vertical_fov_degrees(perspective_fov)
    spline_surface.screenshot(
        output_dir / "perspective_spline_surface.png",
        np.array([0, 0, 0], dtype=np.float64),
        np.array([0, 0, 1], dtype=np.float64),
        False
    )

    # Send the camera to infinity and update the vertex positions
    projection_matrix: Matrix4x4f = origin_to_infinity_projective_matrix(1.0)
    orthographic_V: MatrixNx3f = apply_transformation_to_vertices(V, projection_matrix)
    spline_surface.update_positions(orthographic_V, fit_matrix, energy_hessian_inverse)

    # View the orthographic spline surface
    polyscope.set_vertical_fov_degrees(orthographic_fov)
    spline_surface.screenshot(
        output_dir / "orthographic_spline_surface.png",
        np.array([0, 0, -5], dtype=np.float64),
        np.array([0, 0, 0], dtype=np.float64),
        True
    )

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="generate_perspective_distortion",
        description="Generate perspective distortion figure images for a given mesh and camera.")
    parser.add_argument("-i", "--input", type=str, help="Mesh filepath.", required=True)
    parser.add_argument("-i", "--camera", type=str, help="Camera filepath.", required=False)
    parser.add_argument("-o", "--output", type=str, default="./", help="Output directory")
    parser.add_argument("--translation", type=float, default=0.0, help="Translation amount")
    parser.add_argument("--perspective_fov", type=float, default=45.0,
                        help="Field of view for perspective rendering")
    parser.add_argument("--orthographic_fov", type=float, default=45.0,
                        help="Field of view for orthographic rendering")
    args: argparse.Namespace = parser.parse_args()
    sys.exit(main(args))
