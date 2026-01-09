"""
compute_closed_contours.py
Methods to chain contour segments into closed contours.
"""

import logging

from pyalgcon.core.common import (PLACEHOLDER_VALUE, SpatialVector1d,
                                  float_equal_zero)
from pyalgcon.core.rational_function import RationalFunction

logger: logging.Logger = logging.getLogger(__name__)

# *******
# Helpers
# *******


def _point_distance_squared(point_1: SpatialVector1d, point_2: SpatialVector1d) -> float:
    """
    Distance helper function
    """
    assert point_1.shape == (3, )
    assert point_2.shape == (3, )
    displacement: SpatialVector1d = point_1 - point_2
    return displacement.dot(displacement)


def _are_overlapping_points(point_1: SpatialVector1d, point_2: SpatialVector1d) -> bool:
    """
    Return true iff the two points overlap
    """
    squared_distance: float = _point_distance_squared(point_1, point_2)
    return float_equal_zero(squared_distance)


def _is_closed_contour(contour: list[int],
                       contour_start_points: list[SpatialVector1d],
                       contour_end_points: list[SpatialVector1d]) -> bool:
    """
    Return true iff the contour chain is closed.
    """
    # lazy check
    assert contour_start_points[0].shape == (3, )
    assert contour_end_points[0].shape == (3, )

    # Treat empty contour as closed
    if len(contour) == 0:
        return True

    # Check if the start point of the first segment overlaps the end point of the last segment
    return _are_overlapping_points(contour_end_points[contour[-1]],  # back
                                   contour_start_points[contour[0]])  # front


def _is_valid_contours(contours: list[list[int]],
                       contour_start_points: list[SpatialVector1d],
                       contour_end_points: list[SpatialVector1d]) -> bool:
    """
    Return true iff contours defines valid contour chains
    """
    # Check contour segments are contiguous
    for i, _ in enumerate(contours):
        for j in range(1, len(contours[i])):
            if not _are_overlapping_points(contour_end_points[contours[i][j - 1]],
                                           contour_start_points[contours[i][j]]):
                logger.error("Segment %s in contour %s not adjacent to segment %s", j - 1, i, j)
                return False

    return True


def _get_starting_contour_segment(used_segments: list[bool]) -> int:
    """
    Get unused contour segment
    """
    # Find free segment
    for i, _ in enumerate(used_segments):
        if not used_segments[i]:
            return i

    # Return -1 if no free segment
    return -1


def _add_next_contour_segment(contour_segments: list[RationalFunction],
                              current_contour_ref: list[int],
                              used_segments_ref: list[bool],
                              contour_start_points: list[SpatialVector1d],
                              contour_end_points: list[SpatialVector1d]) -> bool:
    """
    Add next contour segment if possible and return false if not.

    :param contour_segments: [in]
    :param current_contour_ref: [out]
    :param used_segments_ref: [out]
    :param contour_start_points: [in]
    :param contour_end_points: [in]
    :return: bool is valid segment is found
    """
    min_distance_sq: float = 1e10  # FIXME
    next_candidate: int = -1

    # Find the closest next segment
    for i, _ in enumerate(contour_segments):
        # Skip used segments
        if used_segments_ref[i]:
            continue

        # Check if the current oriented contour is closer than the candidate
        cur_distance_sq: float = _point_distance_squared(
            contour_end_points[current_contour_ref[-1]],
            contour_start_points[i])
        if cur_distance_sq < min_distance_sq:
            min_distance_sq = cur_distance_sq
            next_candidate = i

    # Check if any candidate found
    if next_candidate == -1:
        return False

    # Determine if the candidate contour segment is adjacent to the growing chain
    if _are_overlapping_points(contour_end_points[current_contour_ref[-1]],
                               contour_start_points[next_candidate]):
        current_contour_ref.append(next_candidate)
        used_segments_ref[next_candidate] = True
        return True

    # Return false if no valid segment found
    return False


def _add_next_reverse_contour_segment(contour_segments: list[RationalFunction],
                                      current_contour_reverse_ref: list[int],
                                      used_segments_ref: list[bool],
                                      contour_start_points: list[SpatialVector1d],
                                      contour_end_points: list[SpatialVector1d]) -> bool:
    """
    Add previous contour segment to reverse list if possible and return false if not

    :param contour_segments: [in]
    :param current_contour_ref: [out]
    :param used_segments_ref: [out]
    :param contour_start_points: [in]
    :param contour_end_points: [in]
    :return: bool is valid segment is found
    """
    min_distance_sq: float = 1e10  # FIXME
    prev_candidate: int = -1

    # Find the closest prev segment
    for i, _ in enumerate(contour_segments):
        # Skip used segments
        if used_segments_ref[i]:
            continue

        # Check if the current oriented contour is closer than the candidate
        cur_distance_sq: float = _point_distance_squared(
            contour_start_points[current_contour_reverse_ref[-1]],
            contour_end_points[i])
        if cur_distance_sq < min_distance_sq:
            min_distance_sq = cur_distance_sq
            prev_candidate = i

    # Check if any candidate found
    if prev_candidate == -1:
        return False

    # Determine if the candidate contour segment is adjacent to the growing chain
    if _are_overlapping_points(contour_start_points[current_contour_reverse_ref[-1]],
                               contour_end_points[prev_candidate]):
        current_contour_reverse_ref.append(prev_candidate)
        used_segments_ref[prev_candidate] = True
        return True

    # Return false if no valid segment found
    return False


def _combine_forward_and_reverse_contour(contour: list[int],
                                         contour_reverse: list[int]) -> list[int]:
    """
    Combine a forward and reverse contour into one contour
    """
    assert len(contour) > 0
    assert len(contour_reverse) > 0
    assert contour[0] == contour_reverse[0]
    contour_full: list[int] = []

    # Add reverse contour segments in reverse (excluding the last one as it is redundant)
    # FIXME: check if right translation
    for i in range(len(contour_reverse) - 1, 0, -1):
        contour_full.append(contour_reverse[i])

    # Add contour segments
    for i, _ in enumerate(contour):
        contour_full.append(contour[i])

    num_segments: int = len(contour) + len(contour_reverse) - 1
    assert len(contour_full) == num_segments

    return contour_full


def _add_contour(contour_ref: list[int],
                 contours_ref: list[list[int]],
                 contour_labels_ref: list[int]) -> None:
    """
    Add contour to the list of all contours and assign it a new label

    :param contour_ref: [in]
    :param contours_ref: [in, out]
    :param contour_labels_ref: [in, out]
    """
    contours_ref.append(contour_ref)
    for i, _ in enumerate(contour_ref):
        contour_labels_ref[contour_ref[i]] = len(contours_ref)


def compute_closed_contours(contour_segments: list[RationalFunction]) -> tuple[list[list[int]],
                                                                               list[int]]:
    """
    Given contour segments on a surface, chain them together to generate the
    full closed contours.

    :param contour_segments: [in] surface contour segments (degree 4, dimension 3)
    :return contours: list of indices of segments for complete surface contours
    :return contour_labels: index of the contour corresponding to each segment
    """
    # lazy check
    assert (contour_segments[0].degree, contour_segments[0].dimension) == (4, 3)
    contours: list[list[int]] = []
    contour_labels: list[int] = [PLACEHOLDER_VALUE for _ in enumerate(contour_segments)]

    used_segments: list[bool] = [False] * len(contour_segments)
    contour_start_points: list[SpatialVector1d] = []
    contour_end_points: list[SpatialVector1d] = []

    for contour_segment in contour_segments:
        contour_start_points.append(contour_segment.start_point())
        contour_end_points  .append(contour_segment.end_point())
        assert contour_start_points[-1].shape == (3, )
        assert contour_end_points[-1].shape == (3, )

    while True:
        # Get next starting contour segment to process or return if none left
        starting_segment_index: int = _get_starting_contour_segment(used_segments)
        if starting_segment_index == -1:
            assert _is_valid_contours(contours, contour_start_points, contour_end_points)
            return contours, contour_labels

        # Initialize the next contour chain
        current_contour: list[int] = [starting_segment_index]
        used_segments[starting_segment_index] = True

        # Traverse forward until the contour is closed or no new contour is found
        closed_contour: bool = True  # Assume closed until proven otherwise

        while not _is_closed_contour(current_contour, contour_start_points, contour_end_points):
            adjacent_segment_found: bool = _add_next_contour_segment(contour_segments,
                                                                     current_contour,
                                                                     used_segments,
                                                                     contour_start_points,
                                                                     contour_end_points)

            # If no segment found, the contour is open
            if not adjacent_segment_found:
                closed_contour = False
                break

        if closed_contour:

            logger.debug("Closed contour of size %s found", len(current_contour))
            # TODO: check to see if function below modifies by reference properly.
            _add_contour(current_contour, contours, contour_labels)
        else:
            # Traverse an open contour in reverse to get the full chain
            current_contour_reverse: list[int] = [starting_segment_index]
            current_contour_full: list[int] = []
            while True:
                adjacent_segment_found = _add_next_reverse_contour_segment(contour_segments,
                                                                           current_contour_reverse,
                                                                           used_segments,
                                                                           contour_start_points,
                                                                           contour_end_points)

                # If no further segment found, combine the two chains and add it to the list
                if not adjacent_segment_found:
                    current_contour_full = _combine_forward_and_reverse_contour(
                        current_contour,
                        current_contour_reverse)
                    _add_contour(current_contour_full, contours, contour_labels)
                    break

        assert _is_valid_contours(contours, contour_start_points, contour_end_points)
