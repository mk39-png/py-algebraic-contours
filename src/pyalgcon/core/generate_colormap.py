"""
TODO: only need implement one function, which is generate_random_category_colormap() for use in projected_curve_network in contour_network

# All other functions do not need to be implemented
"""

import numpy as np

from pyalgcon.core.common import MatrixXr, Vector3f


def _generate_random_color() -> Vector3f:
    """
    Generate random color
    """
    color: Vector3f = np.random.rand(3, )
    assert color.shape == (3, )

    return color


def generate_random_category_colormap(category_labels: list[int]) -> MatrixXr:
    """
    Given a vector of category labels, generate a colormap that assigns a random
    color to each label.

    :param category_labels: [in] labels for discrete categories
    :return colormap: RGB matrix of color values
    """
    colormap: MatrixXr = np.zeros(shape=(len(category_labels), 3))

    # Return if empty label set
    if len(category_labels) == 0:
        return colormap

    # Get array of random colors
    max_index: int = np.argmax(category_labels)
    category_colors: list[Vector3f] = []
    for _ in range(max_index + 1):
        category_colors.append(_generate_random_color())
    assert len(category_colors) == max_index + 2

    # Assign colors corresponding to indices
    for i, _ in enumerate(category_labels):
        colormap[i, :] = category_colors[category_labels[i] + 1]

    return colormap
