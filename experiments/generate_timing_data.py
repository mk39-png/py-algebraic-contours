#!/usr/bin/env python3

import argparse
import logging
import pathlib
import sys
from datetime import datetime
import time

import igl
import numpy as np
from cholespy import CholeskySolverD
from scipy.sparse import csr_matrix

from pyalgcon.contour_network.compute_intersections import \
    IntersectionParameters
from pyalgcon.contour_network.contour_network import (ContourNetwork,
                                                      InvisibilityMethod,
                                                      InvisibilityParameters)
from pyalgcon.core.affine_manifold import AffineManifold
from pyalgcon.core.apply_transformation import \
    apply_camera_frame_transformation_to_vertices
from pyalgcon.core.common import (Matrix3x3f, Matrix4x4f, MatrixNx3f, MatrixXf,
                                  MatrixXi)
from pyalgcon.core.generate_transformation import (
    x_axis_rotation_projective_matrix, y_axis_rotation_projective_matrix,
    z_axis_rotation_projective_matrix)
from pyalgcon.quadratic_spline_surface.optimize_spline_surface import \
    OptimizationParameters
from pyalgcon.quadratic_spline_surface.twelve_split_spline import (
    TwelveSplitSplineSurface, compute_twelve_split_spline_patch_boundary_edges)


import logging
logging.disable(logging.CRITICAL)
logger: logging.Logger = logging.getLogger(__name__)


def _add_header_to_csv(output_dir: pathlib.Path) -> None:
    """
    Helper method to add a header to the timing .csv file, overwriting the pre-existing files
    with the same names.
    """
    # Write view independent timing data header
    with open(output_dir / "view_independent.csv", 'w', encoding='utf-8') as out_view_independent:
        out_view_independent.write(
            "mesh name, num triangles, time spline surface, time patch boundary edges\n")

    # Write view dependent timing data header
    with open(output_dir / "per_view.csv", 'w', encoding='utf-8') as out_per_view:
        out_per_view.write("mesh name, rotation matrix, total time per view, surface update, "
                           "compute contour, compute cusps, compute intersections, compute "
                           "visibility, graph building, num segments, num int cusps, num bound "
                           "cusps, num intersection call, num ray inter call, num patches\n")


def main(args: argparse.Namespace) -> None:
    """
    Main
    """
    input_filename = pathlib.Path(args.input)
    output_dir = pathlib.Path(args.output)
    num_tests: int = args.num_tests

    # Set logging level
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

    #
    # Start timer for quadratic spline generation
    #
    timer_start: float = time.perf_counter()

    # Generate quadratic spline
    optimization_params: OptimizationParameters = OptimizationParameters()
    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)
    spline_surface: TwelveSplitSplineSurface = TwelveSplitSplineSurface(
        initial_V, affine_manifold, optimization_params)

    # Retrieve values from twelve split spline
    face_to_patch_indices: list[list[int]] = spline_surface.face_to_patch_indices
    patch_to_face_indices: list[int] = spline_surface.patch_to_face_indices
    fit_matrix: csr_matrix = spline_surface.fit_matrix
    energy_hessian: csr_matrix = spline_surface.energy_hessian
    energy_hessian_inverse: CholeskySolverD = spline_surface.energy_hessian_inverse

    # Get spline construction_time
    timer_end: float = time.perf_counter()
    spline_surface_time: float = timer_end - timer_start

    #
    # Time and get the boundary edges
    #
    timer_start = time.perf_counter()
    patch_boundary_edges: list[tuple[int, int]] = compute_twelve_split_spline_patch_boundary_edges(
        F, face_to_patch_indices)
    timer_end: float = time.perf_counter()
    compute_patch_boundary_time: float = timer_end - timer_start

    #
    # Start the timer for the contour generation
    #
    timer_start = time.perf_counter()

    # Build the contours
    intersect_params: IntersectionParameters = IntersectionParameters()
    invisibility_params: InvisibilityParameters = InvisibilityParameters()
    invisibility_params.write_contour_soup = False
    invisibility_params.invisibility_method = InvisibilityMethod.PROPAGATION
    invisibility_params.check_propagation = False
    contour_network: ContourNetwork = ContourNetwork(
        spline_surface, intersect_params, invisibility_params, patch_boundary_edges)

    # Get contour network construction time
    timer_end: float = time.perf_counter()
    initial_contour_network_time: float = timer_end - timer_start

    # Write headers
    _add_header_to_csv(output_dir)

    # Write view independent timing data
    with open(output_dir / "view_independent.csv", 'a', encoding='utf-8', newline="") as out_view_independent:
        out_view_independent.write(
            f"{input_filename}, {F.shape[0]}, {spline_surface_time}, {compute_patch_boundary_time}\n")

    # Write view dependent timing data
    with open(output_dir / "per_view.csv", 'a', encoding='utf-8', newline="") as out_per_view:
        out_per_view.write(
            f"{input_filename}, \
                1 0 0 0 1 0 0 0 1, \
                {initial_contour_network_time},\
                0.0, \
                {contour_network.compute_contour_time},\
                {contour_network.compute_cusp_time},\
                {contour_network.compute_intersection_time},\
                {contour_network.compute_visibility_time},\
                {contour_network.compute_projected_time},\
                {contour_network.segment_number},\
                {contour_network.interior_cusp_number},\
                {contour_network.boundary_cusp_number},\
                {contour_network.intersection_call},\
                {contour_network.ray_intersection_call},\
                {spline_surface.num_patches}\n",)

    #
    #
    # Generating contours
    #
    #
    # Generate a random number generator for angles
    # TODO: translate properly
    np.random.seed(0)
    angle_distribution: np.ndarray = np.random.uniform(low=0.0, high=2.0 * np.pi, size=3)

    # Run tests
    for i in range(num_tests):
        # Generate random rotation matrix
        theta_x: float = angle_distribution[0]
        theta_y: float = angle_distribution[1]
        theta_z: float = angle_distribution[2]

        x_rotation_matrix: MatrixXf = x_axis_rotation_projective_matrix(theta_x)
        y_rotation_matrix: MatrixXf = y_axis_rotation_projective_matrix(theta_y)
        z_rotation_matrix: MatrixXf = z_axis_rotation_projective_matrix(theta_z)
        rotation_matrix: Matrix4x4f = x_rotation_matrix @ y_rotation_matrix @ z_rotation_matrix
        frame: Matrix3x3f = rotation_matrix[:3, :3]

        # Apply transformation
        V = apply_camera_frame_transformation_to_vertices(initial_V, frame)

        # Time update
        timer_start = datetime.now()
        spline_surface.update_positions(V, fit_matrix, energy_hessian)
        contour_network: ContourNetwork = ContourNetwork(
            spline_surface, intersect_params, invisibility_params, patch_boundary_edges)
        total_time: float = (timer_start - datetime.now()).seconds

        # Write timing data
        out_per_view.write(f"{input_filename}, \
                           {rotation_matrix[0, :]} \
                           {rotation_matrix[1, :]} \
                           {rotation_matrix[2, :]}, \
                           {total_time}, \
                           {contour_network.surface_update_position_time}, \
                           {contour_network.compute_contour_time}, \
                           {contour_network.compute_cusp_time}, \
                           {contour_network.compute_intersection_time}, \
                           {contour_network.compute_visibility_time}, \
                           {contour_network.compute_projected_time}, \
                           {contour_network.segment_number}, \
                           {contour_network.interior_cusp_number}, \
                           {contour_network.boundary_cusp_number}, \
                           {contour_network.intersection_call}, \
                           {contour_network.ray_intersection_call}, \
                           {spline_surface.num_patches}\n")

    # Close output file
    out_per_view.close()
    out_view_independent.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="animate_rotation", description="Generate rotation animation for a given mesh.")
    parser.add_argument("-i", "--input", type=str, help="Mesh filepath.", required=True)
    parser.add_argument("-o", "--output", type=str, default="./", help="Output directory")
    parser.add_argument("--num_tests", type=int, default=1,
                        help="Number of tests to run. Nonnegative number")

    args: argparse.Namespace = parser.parse_args()
    sys.exit(main(args))
