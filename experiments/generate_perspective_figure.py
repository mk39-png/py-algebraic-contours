
import argparse
import logging
import pathlib
import sys

import igl
import numpy as np
import polyscope
from core.apply_transformation import apply_transformation_to_vertices
from core.common import (Matrix3x3f, Matrix4x4f, MatrixNx3f, MatrixXf,
                         MatrixXi, SpatialVector1d,
                         compute_point_cloud_bounding_box,
                         convert_polylines_to_edges)
from core.generate_transformation import (axis_rotation_projective_matrix,
                                          origin_to_infinity_projective_matrix,
                                          rotate_frame_projective_matrix,
                                          translation_projective_matrix)

logger: logging.Logger = logging.getLogger(__name__)


def generate_frustum(distance: float, height: float, width: float) -> tuple[np.ndarray,
                                                                            list[tuple[int, int]]]:
    """
    Generate a camera frustum curve network
    """
    points: np.ndarray
    edges: list[tuple[int, int]]

    points = np.zeros(shape=(8, 3), dtype=np.float64)
    x0: float = -width / 2.0
    x1: float = width / 2.0
    y0: float = -height / 2.0
    y1: float = height / 2.0
    d: float = distance

    # Add points for bounding box
    points[0, :] = np.array([x0 / d, y0 / d, 1.0])
    points[1, :] = np.array([x1 / d, y0 / d, 1.0])
    points[2, :] = np.array([x1 / d, y1 / d, 1.0])
    points[3, :] = np.array([x0 / d, y1 / d, 1.0])
    points[4, :] = np.array([x0, y0, d])
    points[5, :] = np.array([x1, y0, d])
    points[6, :] = np.array([x1, y1, d])
    points[7, :] = np.array([x0, y1, d])

    # Add polylines for bounding box
    polyline_bottom: list[int] = [0, 1, 2, 3, 0]
    polyline_top: list[int] = [4, 5, 6, 7, 4]
    polyline_edge_0: list[int] = [0, 4]
    polyline_edge_1: list[int] = [1, 5]
    polyline_edge_2: list[int] = [2, 6]
    polyline_edge_3: list[int] = [3, 7]
    polylines: list[list[int]] = [
        polyline_bottom,
        polyline_top,
        polyline_edge_0,
        polyline_edge_1,
        polyline_edge_2,
        polyline_edge_3]
    edges = convert_polylines_to_edges(polylines)

    return points, edges


def initialize_vertices(input_V: MatrixNx3f, frame: Matrix3x3f) -> np.ndarray:
    """
    Initialize vertices for frustum view
    """
    # Get bounding box of mesh
    min_point: SpatialVector1d
    max_point: SpatialVector1d
    min_point, max_point = compute_point_cloud_bounding_box(input_V)
    mesh_midpoint: SpatialVector1d = 0.5 * (max_point + min_point)
    bounding_box_diagonal: SpatialVector1d = max_point - min_point
    logger.info("Initial mesh bounding box: %s, %s", min_point, max_point)
    logger.info("Initial mesh midpoint: %s", mesh_midpoint)

    # Normalize the vertices
    scale_factor: float = np.linalg.norm(bounding_box_diagonal)
    num_vertices: int = input_V.shape[0]
    output_V: np.ndarray = np.zeros(shape=(num_vertices, 3), dtype=np.float64)
    for i in range(num_vertices):
        output_V[i, :] = 4.0 * (input_V[i, :] - mesh_midpoint) / scale_factor
    min_point, max_point = compute_point_cloud_bounding_box(input_V)
    mesh_midpoint = 0.5 * (max_point + min_point)
    bounding_box_diagonal = max_point - min_point
    logger.info("Normalized mesh bounding box: %s, %s", min_point, max_point)
    logger.info("Normalized mesh midpoint: %s", mesh_midpoint)

    # Generate rotation matrix
    logger.info("Projecting onto frame:\n%s", frame)
    frame_rotation_matrix: Matrix4x4f = rotate_frame_projective_matrix(frame)

    # Generate translation matrix
    z_distance: float = 3.0
    translation: SpatialVector1d = np.array([0, 0, z_distance], dtype=np.float64)
    translation_matrix: Matrix4x4f = translation_projective_matrix(translation)

    # Apply the transformation
    projective_transformation: Matrix4x4f = translation_matrix @ frame_rotation_matrix
    return apply_transformation_to_vertices(output_V, projective_transformation)


def main(args):
    input_filename: pathlib.Path = pathlib.Path(args.input)

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

    # Initialize vertices
    rotation_matrix: Matrix4x4f = axis_rotation_projective_matrix(0, -90, 0)
    rotation_frame: Matrix3x3f = rotation_matrix[:3, :3]
    V = initialize_vertices(initial_V, rotation_frame)

    # Build camera frustum
    distance: float = 4.0
    height: float = 5.0
    width: float = 5.0
    perspective_points: MatrixNx3f
    edges: list[tuple[int, int]]
    perspective_points, edges = generate_frustum(distance, height, width)

    # Build perspective viewer
    polyscope.init()
    polyscope.set_ground_plane_mode("none")
    perspective_frustum: polyscope.CurveNetwork = polyscope.register_curve_network(
        "perspective_frustum", perspective_points, edges)
    perspective_frustum.set_color((0.0, 0.0, 0.0))
    perspective_frustum.set_radius(0.002)
    perspective_mesh: polyscope.SurfaceMesh = polyscope.register_surface_mesh(
        "perspective_mesh", V, F)
    perspective_mesh.set_color((0.670, 0.673, 0.292))

    # Look at viewer
    glm_camera_position: list[float] = [0.0, 0.0, 0.0]
    glm_camera_target: list[float] = [0.0, 0.0, 4.0]
    polyscope.look_at(glm_camera_position, glm_camera_target)
    polyscope.show()
    polyscope.remove_all_structures()

    # Project mesh and frustum to send the camera to infinity while fixing the
    # plane z = 1
    projection_matrix: Matrix4x4f = origin_to_infinity_projective_matrix(1.0)
    orthographic_points: MatrixNx3f
    orthographic_V: MatrixNx3f
    orthographic_points = apply_transformation_to_vertices(perspective_points, projection_matrix)
    orthographic_V = apply_transformation_to_vertices(V, projection_matrix)

    # Build orthographic viewer
    orthographic_frustum: polyscope.CurveNetwork = polyscope.register_curve_network(
        "orthographic_frustum", orthographic_points, edges)
    orthographic_mesh: polyscope.SurfaceMesh = polyscope.register_surface_mesh(
        "orthographic_mesh", orthographic_V, F)
    orthographic_frustum.set_color((0.0, 0.0, 0.0))
    orthographic_frustum.set_radius(0.002)
    orthographic_mesh.set_color((0.670, 0.673, 0.292))

    # Look at viewer
    glm_camera_position = [0.0, 0.0, -1.0]
    glm_camera_target = [0.0, 0.0, 1.0]
    polyscope.str_to_projection_mode("orthographic")
    polyscope.look_at(glm_camera_position, glm_camera_target)
    polyscope.show()
    polyscope.remove_all_structures()

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="generate_perspective_figure",
        description="Generate perspective figure example for a given mesh.")
    parser.add_argument("-i", "--input", type=str, help="Mesh filepath.", required=True)
    args: argparse.Namespace = parser.parse_args()

    sys.exit(main(args))
