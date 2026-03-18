#!/usr/bin/env python3

# NOTE: this was originally generate_timing_data, but has been modified accept and input camera
#       matrix.

import argparse
import logging
import pathlib
import sys
import time

import igl
import numpy as np
import polyscope
from cholespy import CholeskySolverD
from scipy.sparse import csr_matrix

from pyalgcon.contour_network.compute_intersections import \
    IntersectionParameters
from pyalgcon.contour_network.contour_network import (ContourNetwork,
                                                      InvisibilityMethod,
                                                      InvisibilityParameters)
from pyalgcon.core.affine_manifold import AffineManifold
from pyalgcon.core.apply_transformation import (
    apply_camera_matrix_transformation_to_vertices,
    apply_normalization_to_vertices)
from pyalgcon.core.common import (Matrix3x3f, Matrix4x4f, MatrixNx3f, MatrixXi,
                                  deserialize_eigen_matrix_csv_to_numpy)
from pyalgcon.quadratic_spline_surface.optimize_spline_surface import \
    OptimizationParameters
from pyalgcon.quadratic_spline_surface.twelve_split_spline import (
    TwelveSplitSplineSurface, compute_twelve_split_spline_patch_boundary_edges)
from pyalgcon.utils.projected_curve_networks_utils import SVGOutputMode

logging.disable(logging.CRITICAL)
logger: logging.Logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """
    Main function
    NOTE: input_filename and camera_filename can also take in a path 
    """
    mesh_filepath = pathlib.Path(args.input)
    camera_filepath = pathlib.Path(args.camera)
    output_dir = pathlib.Path(args.output)

    # Set logging level
    logger.setLevel(logging.CRITICAL)

    # Get input mesh
    initial_V: MatrixNx3f
    transformed_V: MatrixNx3f
    uv: np.ndarray
    N: np.ndarray
    F: MatrixXi
    FT: MatrixXi
    FN: MatrixXi
    initial_V, uv, N, F, FT, FN = igl.readOBJ(mesh_filepath)

    # Start timer for vertex normalization
    # (this also helps with debugging it we got the correct contours in the end)
    timer_start: float = time.perf_counter()
    normalized_V: MatrixNx3f = apply_normalization_to_vertices(initial_V)
    timer_end: float = time.perf_counter()
    normalization_time: float = timer_end - timer_start

    # Start timer for vertex transformation
    timer_start = time.perf_counter()
    camera_matrix: Matrix4x4f = deserialize_eigen_matrix_csv_to_numpy(camera_filepath)
    assert camera_matrix.shape == (4, 4)
    transformed_V = apply_camera_matrix_transformation_to_vertices(normalized_V, camera_matrix)
    timer_end = time.perf_counter()
    transformation_time: float = timer_end - timer_start

    #
    # Start timer for quadratic spline generation
    #
    timer_start: float = time.perf_counter()

    # Generate quadratic spline
    optimization_params: OptimizationParameters = OptimizationParameters()
    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)
    spline_surface: TwelveSplitSplineSurface = TwelveSplitSplineSurface(
        transformed_V,
        affine_manifold,
        optimization_params
    )

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

    # Write view independent timing data (right now in case contour gen fails)
    with open(output_dir / "view_independent.csv", 'a', encoding="utf-8") as out_view_independent:
        out_view_independent.write(
            f"{mesh_filepath},"
            f"{camera_filepath},"
            f"{F.shape[0]},"
            f"{spline_surface_time},"
            f"{compute_patch_boundary_time},"
            f"{normalization_time}\n")

    # -------------------------------------
    # CONTOUR GEN BELOW
    # -------------------------------------

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

    # Start the timer for vector contour writing
    timer_start = time.perf_counter()
    contour_network.write(
        output_dir / f"timing_vector_contours_{mesh_filepath.name}.svg",
        SVGOutputMode.CONTRAST_INVISIBLE_SEGMENTS,
        True
    )
    timer_end = time.perf_counter()
    vector_contour_write_time: float = timer_end - timer_start

    # Start the timer for rasterized contour writing
    timer_start = time.perf_counter()
    contour_network.write_rasterized_contours(
        output_dir / f"timing_raster_contours_{mesh_filepath.name}.png",
        (0, 0, -1),
        (0, 0, 0)
    )
    timer_end = time.perf_counter()
    raster_contour_write_time: float = timer_end - timer_start

    # Write view dependent timing data
    # Save to write to file
    rotation_frame: Matrix3x3f = camera_matrix[:3, :3]
    z_distance: float = camera_matrix[2, 3]
    assert rotation_frame.shape == (3, 3)
    with open(output_dir / "per_view.csv", "a", encoding="utf-8") as out_per_view:
        out_per_view.write(f"{mesh_filepath.name},"
                           f"{camera_filepath.name},"
                           f"[{rotation_frame[0, :]}] "
                           f"[{rotation_frame[1, :]}] "
                           f"[{rotation_frame[2, :]}],"
                           f"{z_distance},"
                           f"{transformation_time},"
                           f"{initial_contour_network_time},"  # total time per view
                           f"{contour_network.surface_update_position_time},"
                           f"{contour_network.compute_contour_time},"  # compute contour
                           f"{contour_network.compute_cusp_time},"  # compute cusps
                           f"{contour_network.compute_intersection_time},"  # compute intersections
                           f"{contour_network.compute_visibility_time},"  # compute visibility
                           f"{contour_network.compute_projected_time},"  # graph building
                           f"{contour_network.segment_number},"
                           f"{contour_network.interior_cusp_number},"
                           f"{contour_network.boundary_cusp_number},"
                           f"{contour_network.intersection_call},"
                           f"{contour_network.ray_intersection_call},"
                           f"{spline_surface.num_patches},"
                           f"{vector_contour_write_time},"
                           f"{raster_contour_write_time}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate timing data", description="Append timing data CSVs for a given mesh and camera "
        " matrix.")
    parser.add_argument("-i", "--input", type=str, help="Mesh filepath.", required=True)
    parser.add_argument("-c", "--camera", type=str, help="Camera filepath.", required=True)
    parser.add_argument("-o", "--output", type=str, default="./", help="Output directory")
    args: argparse.Namespace = parser.parse_args()
    sys.exit(main(args))
