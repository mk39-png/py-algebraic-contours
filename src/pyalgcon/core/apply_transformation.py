"""
Methods to apply projective transformation matrices to various point data
types.

TODO: try and utilize Blender's Python API for this to make life easier for us.
TODO: but this involves mathutils...
But anyways, could convert to MathUtils and back...
"""


import logging

import numpy as np

from pyalgcon.core.common import (ROWS, Matrix3x3f, Matrix4x4f, MatrixNx3,
                                  MatrixNx3f, SpatialVector1d, Vector4f,
                                  compute_point_cloud_bounding_box,
                                  float_equal, logger, todo, unimplemented)
from pyalgcon.core.generate_transformation import (
    origin_to_infinity_projective_matrix, rotate_frame_projective_matrix,
    translation_projective_matrix)

logger: logging.Logger = logging.getLogger(__name__)


def convert_point_to_homogeneous_coords(point: SpatialVector1d) -> Vector4f:
    """
    Convert point to homogeneous coordinates
    """
    # Simply append a homogeneous coordinate of value 1 to the point
    homogeneous_coords: Vector4f = np.ones(shape=(4, ), dtype=np.float64)
    homogeneous_coords[:3] = point

    return homogeneous_coords


def convert_homogeneous_coords_to_point(homogeneous_coords: Vector4f) -> SpatialVector1d:
    """
    Convert homogeneous coordinates to a point
    """
    point: SpatialVector1d = np.zeros(shape=(3, ), dtype=np.float64)

    # Extract homogeneous coordinates
    x: float = homogeneous_coords[0]
    y: float = homogeneous_coords[1]
    z: float = homogeneous_coords[2]
    w: float = homogeneous_coords[3]

    # If w is zero, return
    if float_equal(w, 0.0):
        return point

    # Divide by the homogeneous coordinate
    point[0] = x / w
    point[1] = y / w
    point[2] = z / w

    return point


def apply_transformation_to_point(point: SpatialVector1d,  projective_transformation: Matrix4x4f
                                  ) -> SpatialVector1d:
    """
    Apply projective transformation to a single point

    :param point: [in] point to transform
    :param projective_transformation: [in] transformation to apply to the point
    :return transformed_point: point after transformation
    """
    # Get homogeneous coordinates for the point
    homogeneous_coords: Vector4f = convert_point_to_homogeneous_coords(point)
    assert homogeneous_coords.shape == (4, )

    # Transform the homogeneous coordinates
    transformed_coords: Vector4f = projective_transformation @ homogeneous_coords

    # Convert transformed homogeneous coordinates to a point
    transformed_point: SpatialVector1d = convert_homogeneous_coords_to_point(transformed_coords)
    assert transformed_point.shape == (3, )

    return transformed_point


def apply_transformation_to_points():
    todo()


def apply_transformation_to_points_in_place():
    todo()


def apply_transformation_to_control_points():
    todo("Used internally")


def apply_transformation_to_control_points_in_place():
    unimplemented("Not used")


def apply_transformation_to_vertices(input_V: MatrixNx3f,
                                     projective_transformation: Matrix4x4f
                                     ) -> MatrixNx3f:
    """
    Used in generate_algebraic_contours.py

    Apply projective transformation to a matrix of vertices.

    :param[in] input_V: vertex matrix to transform
    :param[in] projective_transformation: transformation to apply to the vector

    :return output_V: transformed vertex matrix
    """
    # Apply transformation to each vertex point individually
    output_V: MatrixNx3f = np.ndarray(shape=input_V.shape, dtype=np.float64)
    for i in range(input_V.shape[ROWS]):
        v: SpatialVector1d = input_V[i, :]
        assert v.shape == (3, )
        Tv: SpatialVector1d = apply_transformation_to_point(v, projective_transformation)
        assert Tv.shape == (3, )
        output_V[i, :] = Tv

    return output_V


def apply_transformation_to_vertices_in_place(V_ref: MatrixNx3f,
                                              projective_transformation: Matrix4x4f) -> None:
    """
    Used in generate_algebraic_contours.py

    Apply projective transformation to a matrix of vertices in place.

    :param V_ref: [in, out]  vertex matrix to transform
    :param projective_transformation: [in]  transformation to apply to the vector
    """
    # Apply transformation to each vertex point individually
    for i in range(V_ref.shape[ROWS]):
        v: SpatialVector1d = V_ref[i, :]
        assert v.shape == (3, )
        Tv: SpatialVector1d = apply_transformation_to_point(v, projective_transformation)
        assert Tv.shape == (3, )
        V_ref[i, :] = Tv


def generate_projective_transformation():
    unimplemented("Not used ")


def initialize_control_points():
    unimplemented("Not used")


def initialize_vertices():
    """
    Used in generate_perspective_figure
    """
    todo()


def apply_camera_matrix_transformation_to_vertices(input_V: MatrixNx3f,
                                                   camera_matrix: Matrix4x4f,
                                                   projection_matrix: Matrix4x4f,
                                                   orthographic: bool = True,
                                                   recenter_mesh: bool = False
                                                   ) -> MatrixNx3f:
    """
    Used in generate_algebraic_contours.

    :param input_V:          [in] mesh vertices before the transformation
    :param camera_matrix:    [in] translation matrixs @ frame rotation matrix
    :param orthographic:     [in] project camera to infinity if true
    :param normalize_initial_positions: [in] normalize the initial vertices to a
                                             bounding box if true
    :return: vertices under projective transform
    """

    logger.info("Using camera matrix: %s\n", camera_matrix)
    camera_to_plane_distance: float = 1.0

    # DEBUG - overwrites parameter projection matrix
    projection_matrix: Matrix4x4f = origin_to_infinity_projective_matrix(camera_to_plane_distance)
    projective_transformation: Matrix4x4f = projection_matrix @ camera_matrix
    output_V: MatrixNx3f = apply_transformation_to_vertices(input_V,
                                                            projective_transformation)
    return output_V


def apply_camera_frame_transformation_to_vertices(input_V: MatrixNx3f,
                                                  frame: Matrix3x3f,
                                                  orthographic: bool = True,
                                                  recenter_mesh: bool = False) -> MatrixNx3f:
    """
    Used in generate_algebraic_contours.

    Apply transformations to a mesh to set the initial camera view direction
    with optional conversion to an orthographic perspective

    The vertices are optimally first normalized to be in a bounding box of
    diagonal 1 at the origin.

    :param input_V:          [in] mesh vertices before the transformation
    :param camera_direction: [in] view direction for the mesh
    :param orthographic:     [in] project camera to infinity if true
    :param normalize_initial_positions: [in] normalize the initial vertices to a
                                             bounding box if true

    :return output_V: mesh vertices after transformation
    """
    # Compute mesh midpoint and bounding box diagonal
    min_point: SpatialVector1d
    max_point: SpatialVector1d
    min_point, max_point = compute_point_cloud_bounding_box(input_V)
    mesh_midpoint: SpatialVector1d = 0.5 * (max_point + min_point)
    bounding_box_diagonal: SpatialVector1d = max_point - min_point
    logger.info("Initial mesh bounding box: %s, %s", min_point, max_point)
    logger.info("Initial mesh midpoint: %s", mesh_midpoint)

    # Normalize the vertices
    scale_factor: float = bounding_box_diagonal.max()
    num_vertices: int = input_V.shape[ROWS]
    output_V: MatrixNx3f = np.zeros((num_vertices, 3))
    for i in range(num_vertices):
        output_V[i, :] = 2.0 * (input_V[i, :] - mesh_midpoint) / scale_factor

    # min_point, max_point = compute_point_cloud_bounding_box(input_V)
    # mesh_midpoint = 0.5 * (max_point + min_point)
    # bounding_box_diagonal = max_point - min_point
    # logger.info("Normalized mesh bounding box: %s, %s", min_point, max_point)
    # logger.info("Normalized mesh midpoint: %s", mesh_midpoint)

    # Generate rotation matrix
    logger.info("Projecting onto frame:\n%s", frame)
    frame_rotation_matrix: Matrix4x4f = rotate_frame_projective_matrix(frame)
    logger.debug("Frame rotation matrix:\n%s", frame_rotation_matrix)

    # Generate translation matrix
    z_distance: float = 5.0
    translation: SpatialVector1d = np.array([0.0, 0.0, z_distance])
    translation_matrix: Matrix4x4f = translation_projective_matrix(translation)

    # Optionally generate matrix to send the origin to infinity
    projective_transformation: Matrix4x4f
    if orthographic:
        camera_to_plane_distance = 1.0
        projection_matrix: Matrix4x4f = (
            origin_to_infinity_projective_matrix(camera_to_plane_distance))
        projective_transformation = projection_matrix @ translation_matrix @ frame_rotation_matrix
    else:
        projective_transformation = translation_matrix @ frame_rotation_matrix

    # Apply the transformations
    logger.info("Apply transformation:\n%s", projective_transformation)
    apply_transformation_to_vertices_in_place(output_V, projective_transformation)

    if recenter_mesh:
        # Renormalize the projected vertices
        min_point, max_point = compute_point_cloud_bounding_box(output_V)
        mesh_midpoint = 0.5 * (max_point + min_point)
        logger.info("Projected mesh bounding box: %s, %s", min_point, max_point)
        logger.info("Projected mesh midpoint: %s", mesh_midpoint)
        bounding_box_diagonal = max_point - min_point
        scale_factor = np.linalg.norm(bounding_box_diagonal)
        for i in range(num_vertices):
            output_V[i, :] = (output_V[i, :] - mesh_midpoint) / scale_factor

    # Check final midpoint location
    min_point, max_point = compute_point_cloud_bounding_box(output_V)
    mesh_midpoint = 0.5 * (max_point + min_point)
    logger.info("Final mesh bounding box: %s, %s", min_point, max_point)
    logger.info("Final mesh midpoint: %s", mesh_midpoint)

    return output_V
