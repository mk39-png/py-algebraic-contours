"""
test_compute_ray_intersections_pencil_method.py

Testing various methods for calculating intersections.
"""


import numpy as np

from pyalgcon.contour_network.compute_ray_intersections_pencil_method import (
    compute_spline_surface_patch_ray_intersections_pencil_method,
    pencil_first_part, solve_quadratic_quadratic_equation_pencil_method)
from pyalgcon.core.common import (
    PlanarPoint1d, Vector6f, compare_eigen_numpy_matrix,
    compare_intersection_points, deserialize_eigen_matrix_csv_to_numpy)
from pyalgcon.quadratic_spline_surface.quadratic_spline_surface_patch import \
    QuadraticSplineSurfacePatch


def test_compute_spline_surface_patch_ray_intersections_pencil_method() -> None:
    """
    Testing values with default spot control mesh.
    """
    filepath: str = "spot_control\\contour_network\\compute_ray_intersections_pencil_method\\compute_spline_surface_patch_ray_intersections_pencil_method\\"
    for i in range(3162):
        (num_intersections,
            surface_intersections,
            ray_intersections,
            ray_int_call,
            ray_bbox_call) = compute_spline_surface_patch_ray_intersections_pencil_method(
                QuadraticSplineSurfacePatch.init_from_json_file(
                    filepath+f"spline_surface_patch\\{i}.json"),
                deserialize_eigen_matrix_csv_to_numpy(filepath+f"ray_mapping_coeffs\\{i}.csv"),
                deserialize_eigen_matrix_csv_to_numpy(
                    filepath+f"ray_intersections_call_in\\{i}.csv"),
                deserialize_eigen_matrix_csv_to_numpy(filepath+f"ray_bounding_box_call_in\\{i}.csv"))


def test_solve_quadratic_quadratic_equation_pencil_method_spot_mesh() -> None:
    """
    Testing with values from the default spot control mesh.
    Which is likely from the last iteration that solve_quadratic_quadratic_equation_pencil()
    is called, as used by compute_cusp_by_one_patch()

    NOTE: potentially failing because intersection_points is not set to array of 0s
    And if that is the case, then that would mean there are no intersections since the values
    of intersection_points have not been set off by any condition for intersecting points.
    """
    # Testing with 878 iterations
    filepath: str = "spot_control\\contour_network\\compute_ray_intersections_pencil_method\\solve_quadratic_equation_pencil_method\\"
    for i in range(879):
        a: Vector6f = deserialize_eigen_matrix_csv_to_numpy(filepath+f"a\\{i}.csv")
        b: Vector6f = deserialize_eigen_matrix_csv_to_numpy(filepath+f"b\\{i}.csv")
        num_intersections_control: int = int(deserialize_eigen_matrix_csv_to_numpy(
            filepath+f"num_intersections\\{i}.csv"))

        num_intersections_test: int
        intersection_points: list[PlanarPoint1d]
        num_intersections_test, intersection_points = (
            solve_quadratic_quadratic_equation_pencil_method(a, b))

        # TODO: test the intersection points nad whatnot...
        # If there rare no intersections, then we have no need to check for any intersections.
        assert num_intersections_test == num_intersections_control
        if num_intersections_control != 0:
            compare_intersection_points(filepath+f"intersection_points\\{i}.csv",
                                        np.array(intersection_points),
                                        num_intersections_control)


def test_pencil_first_part_qi_testing_spot_mesh() -> None:
    """
    Testing with values from the default spot control mesh.
    But this time for its usage in the QI calculation for the spot mesh
    """
    filepath: str = "spot_control\\contour_network\\compute_ray_intersections_pencil_method\\pencil_first_part_qi_testing\\"
    for i in range(3161):
        coeff_F: Vector6f = deserialize_eigen_matrix_csv_to_numpy(filepath+f"coeff_F\\{i}.csv")
        coeff_G: Vector6f = deserialize_eigen_matrix_csv_to_numpy(filepath+f"coeff_G\\{i}.csv")
        num_intersections_control: int = int(deserialize_eigen_matrix_csv_to_numpy(
            filepath+f"num_intersections\\{i}.csv"))

        num_intersections_test: int
        intersection_points: list[PlanarPoint1d]
        _, num_intersections_test, intersection_points = pencil_first_part(coeff_F, coeff_G)

        # TODO: test the intersection points nad whatnot...
        # If there rare no intersections, then we have no need to check for any intersections.
        assert num_intersections_test == num_intersections_control
        if num_intersections_control != 0:
            compare_intersection_points(filepath+f"intersection_points\\{i}.csv",
                                        np.array(intersection_points),
                                        num_intersections_control)


def test_pencil_first_part_spot_mesh() -> None:
    """
    Testing with values from the default spot control mesh.
    """
    filepath: str = "spot_control\\contour_network\\compute_ray_intersections_pencil_method\\pencil_first_part\\"
    for i in range(879):
        coeff_F: Vector6f = deserialize_eigen_matrix_csv_to_numpy(filepath+f"coeff_F\\{i}.csv")
        coeff_G: Vector6f = deserialize_eigen_matrix_csv_to_numpy(filepath+f"coeff_G\\{i}.csv")
        num_intersections_control: int = int(deserialize_eigen_matrix_csv_to_numpy(
            filepath+f"num_intersections\\{i}.csv"))

        num_intersections_test: int
        intersection_points: list[PlanarPoint1d]
        _, num_intersections_test, intersection_points = pencil_first_part(coeff_F, coeff_G)

        # TODO: test the intersection points nad whatnot...
        # If there rare no intersections, then we have no need to check for any intersections.
        assert num_intersections_test == num_intersections_control
        if num_intersections_control != 0:
            compare_intersection_points(filepath+f"intersection_points\\{i}.csv",
                                        np.array(intersection_points),
                                        num_intersections_control)
