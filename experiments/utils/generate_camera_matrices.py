#!/usr/bin/env python3

# Common script that generates camera matrices for testing both Python and original C++ versions

import argparse
import pathlib
import sys

import numpy as np

from pyalgcon.core.common import Matrix4x4f, MatrixXf, Vector3f
from pyalgcon.core.generate_transformation import (
    translation_projective_matrix, x_axis_rotation_projective_matrix,
    y_axis_rotation_projective_matrix, z_axis_rotation_projective_matrix)


def main(args):
    # Output filename (which can also be a directory specifying where the camera matrix should go)
    # output_dir: pathlib.Path = pathlib.Path(args.output_dir)
    output_dir: pathlib.Path = args.output
    num_matrices: int = args.num_matrices

    # Setup
    np.random.seed(0)
    Z_DISTANCE: float = 5.0

    # Generate an identity matrix as the default case.
    camera_matrix_identity: Matrix4x4f = np.identity(4, dtype=np.float64)
    camera_matrix_identity[2, 3] = Z_DISTANCE
    np.savetxt(output_dir / "camera_matrix_identity.csv",
               camera_matrix_identity,
               delimiter=",", fmt="%.6f")

    # NOTE: following pipeline similar to that of apply_transformation.py
    # NOTE: This used for testing each mesh with N number of cameras.
    #       So, for each M mesh, test each N camera, making for O(n^m) runtime.
    # NOTE: not having a for loop so that we do not have a hard dependency on filenames
    for i in range(num_matrices):
        # Generate random rotation matrix with rotations from 0 to 360 degrees
        angle_distribution: np.ndarray = np.random.uniform(low=0.0, high=360, size=3)
        # z_distance: float = np.random.uniform(low=3, high=5)
        degree_theta_x: float = angle_distribution[0]
        degree_theta_y: float = angle_distribution[1]
        degree_theta_z: float = angle_distribution[2]
        x_rotation_matrix: MatrixXf = x_axis_rotation_projective_matrix(degree_theta_x)
        y_rotation_matrix: MatrixXf = y_axis_rotation_projective_matrix(degree_theta_y)
        z_rotation_matrix: MatrixXf = z_axis_rotation_projective_matrix(degree_theta_z)
        frame_rotation_matrix: Matrix4x4f = x_rotation_matrix @ y_rotation_matrix @ z_rotation_matrix

        # Calculate translation matrix
        translation: Vector3f = np.array([0.0, 0.0, Z_DISTANCE], dtype=np.float64)
        translation_matrix: Matrix4x4f = translation_projective_matrix(translation)

        # Output rotation matrix to be read in by C++ and Python implementations.
        camera_matrix = translation_matrix @ frame_rotation_matrix

        filename: str = f"camera_matrix_{i}.csv"
        np.savetxt(output_dir / filename, camera_matrix, delimiter=",", fmt="%.6f")

    return 0


# Script entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="generate_rotation_matrices",
        description="Generates rotation matrices for experiments with ASOC.")
    parser.add_argument("-o", "--output", type=pathlib.Path, default=pathlib.Path("."),
                        help="Output directory")
    parser.add_argument("--num_matrices", type=int, default=1,
                        help="Number of matrices to generate. Nonnegative number")
    # TODO: add support for varying Z distances
    args: argparse.Namespace = parser.parse_args()
    sys.exit(main(args))
