"""
generate_transformations.py
Methods to generate projective transformation matrices.
"""


import numpy as np

from pyalgcon.core.common import (Matrix3x3f, Matrix4x4f,
                                  SpatialVector1d, todo)


def get_frame():
    # Skip
    todo()


def origin_to_infinity_projective_matrix(plane_distance: float) -> Matrix4x4f:
    """
    Generate the projective matrix that sends the origin to infinity while
    fixing the plane z = plane_distance

    :param plane_distance: [in] distance from the origin to the plane
    :return: 4x4 projective matrix for the transformation
    """
    projection_matrix: Matrix4x4f = np.zeros(shape=(4, 4))

    # Scale by plane distance on x, y coordinates in the fixed plane
    projection_matrix[0, 0] = plane_distance
    projection_matrix[1, 1] = plane_distance

    # Ensure plane points remain in the plane (and invert z coordinate)
    projection_matrix[2, 3] = -plane_distance * plane_distance

    # The homogeneous coordinate is the original z coordinate
    projection_matrix[3, 2] = 1.0

    return projection_matrix


def infinity_to_origin_projective_matrix():
    """
    Generate the projective matrix that sends a point at infinity to the origin
    while fixing the plane z = plane_distance.

    This is the inverse of the map sending the origin to infinity.

    @param[in] plane_distance: distance from the origin to the plane
    @return 4x4 projective matrix for the transformation
    """
    pass


def rotate_frame_projective_matrix(frame: Matrix3x3f) -> Matrix4x4f:
    """
    Generate the rotation matrix that sends the given frame to the standard
    frame.

    :param frame: [in] 3x3 frame matrix to align with the standard frame

    :return: 4x4 projective matrix for the transformation
    """
    rotation_matrix: Matrix4x4f = np.zeros(shape=(4, 4), dtype=np.float64)

    # The desired rotation is the transpose of the frame
    rotation_matrix[0:3, 0:3] = frame.T

    # No homoegeneous scaling for the rotation
    rotation_matrix[3, 3] = 1

    return rotation_matrix


def translation_projective_matrix(translation: SpatialVector1d) -> Matrix4x4f:
    """
    Generate the projective matrix representing translation by the given
    translation vector.

    :param translation: [in] 1x3 translation vector

    :return: 4x4 projective matrix for the transformation    
    """
    assert translation.shape == (3, )

    # Initialize matrix to the identity
    translation_matrix: Matrix4x4f = np.identity(4, dtype=np.float64)
    assert translation_matrix.shape == (4, 4)

    # Add translation using homogeneous coordinates
    translation_matrix[:3, 3:4] = translation.reshape(3, 1)

    return translation_matrix


def scaling_projective_matrix():
    # Skip
    pass


def x_axis_rotation_projective_matrix():
    """
    Generate the projective matrix for rotation around the x axis.

    :param degree: [in] degree of rotation around the x axis
    :return: 4x4 projective matrix for the transformation
    """
    pass


def y_axis_rotation_projective_matrix():
    """
    Generate the projective matrix for rotation around the y axis.

    :param degree: [in] degree of rotation around the y axis
    :return: 4x4 projective matrix for the transformation
    """
    pass


def z_axis_rotation_projective_matrix():
    """
    Generate the projective matrix for rotation around the z axis.

    :param degree: [in] degree of rotation around the z axis
    :return: 4x4 projective matrix for the transformation
    """
    pass


def axis_rotation_projective_matrix():
    """
    Generate the projective matrix for chained rotation around the standard
    axes.

    The order of rotation is z axis -> y axis -> x axis

    :param x_degree: [in] degree of rotation around the x axis
    :param y_degree: [in] degree of rotation around the y axis
    :param z_degree: [in] degree of rotation around the z axis

    :return: 4x4 projective matrix for the transformation
    """
    pass
