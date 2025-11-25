"""
compute_cusps

Methods to compute cusps for a quadratic spline surface.
"""


import numpy as np

from pyalgcon.contour_network.compute_curve_frame import \
    compute_quadratic_surface_curve_frame
from pyalgcon.contour_network.compute_ray_intersections_pencil_method import \
    solve_quadratic_quadratic_equation_pencil_method
from pyalgcon.core.common import (Matrix3x2f, Matrix3x3f,
                                  Matrix6x3f, PatchIndex,
                                  PlanarPoint1d, Vector1D,
                                  Vector3f, Vector6f,
                                  deprecated, float_equal,
                                  float_equal_zero, todo)
from pyalgcon.core.conic import Conic
from pyalgcon.core.polynomial_function import \
    polynomial_real_roots
from pyalgcon.core.rational_function import RationalFunction
from pyalgcon.quadratic_spline_surface.quadratic_spline_surface import \
    QuadraticSplineSurface
from pyalgcon.quadratic_spline_surface.quadratic_spline_surface_patch import \
    QuadraticSplineSurfacePatch


def compute_quadratic_surface_cusp_function(surface_mapping_coeffs: Matrix6x3f,
                                            normal_mapping_coeffs: Matrix6x3f,
                                            frame: Matrix3x3f,
                                            contour_domain_curve_segment: Conic
                                            ) -> RationalFunction:
    """
    Compute an implicit function for a quadratic surface with cusps at the roots.

    Note that this function is high degree and may have spurious cusps.
    compute_spline_surface_cusps is preferred for finding cusps.

    :param surface_mapping_coeffs: [in] coefficients for the quadratic surface
    :param normal_mapping_coeffs:  [in] coefficients for the quadratic surface normal
    :param frame: [in] projection frame
    :param contour_domain_curve_segment: [in] local parametric domain contour segment
    :return cusp_functions: implicit cusp function
    """
    deprecated("Function does not appear to be used.")

    assert surface_mapping_coeffs.shape == (6, 3)
    assert normal_mapping_coeffs.shape == (6, 3)
    assert frame.shape == (3, 3)

    # Compute the contour tangent normal function
    contour_segment_tangent: RationalFunction  # degree 8, dimension 3
    contour_segment_normal: RationalFunction  # degree 4, dimension 3
    contour_segment_tangent_normal: RationalFunction  # degree 12, dimension 3
    (contour_segment_tangent,
     contour_segment_normal,
     contour_segment_tangent_normal) = compute_quadratic_surface_curve_frame(
        surface_mapping_coeffs,
        normal_mapping_coeffs,
        contour_domain_curve_segment)
    assert contour_segment_tangent.degree == 8
    assert contour_segment_tangent.dimension == 3
    assert contour_segment_normal.degree == 4
    assert contour_segment_normal.dimension == 3
    assert contour_segment_tangent_normal.degree == 12
    assert contour_segment_tangent_normal.dimension == 3

    # Compute the component of the tangent normal in the camera direction
    # TODO: double check C++ implementation
    tau: Vector3f = frame[:, 2]
    cusp_function: RationalFunction
    cusp_function = contour_segment_tangent_normal.apply_one_form(tau)
    assert cusp_function.degree == 12
    assert cusp_function.dimension == 1
    return cusp_function


def _compute_spline_surface_patch_cusp_function(spline_surface_patch: QuadraticSplineSurfacePatch,
                                                frame: Matrix3x3f,
                                                contour_domain_curve_segment: Conic
                                                ) -> RationalFunction:
    """
    Compute the cusp function for a single patch
    """
    assert frame.shape == (3, 3)

    # Generate surface and normal mappings
    surface_mapping_coeffs: Matrix6x3f = spline_surface_patch.surface_mapping
    normal_mapping_coeffs: Matrix6x3f = spline_surface_patch.normal_mapping
    assert surface_mapping_coeffs.shape == (6, 3)
    assert normal_mapping_coeffs.shape == (6, 3)

    # Compute cusp function for quadratic surface patch
    cusp_function: RationalFunction = compute_quadratic_surface_cusp_function(
        surface_mapping_coeffs,
        normal_mapping_coeffs,
        frame,
        contour_domain_curve_segment)
    assert (cusp_function.degree, cusp_function.dimension) == (12, 1)
    return cusp_function


def compute_spline_surface_cusp_functions(spline_surface: QuadraticSplineSurface,
                                          frame: Matrix3x3f,
                                          contour_domain_curve_segments: list[Conic],
                                          patch_indices: list[PatchIndex]
                                          ) -> list[RationalFunction]:
    """
    Compute implicit functions per patch for a spline surface with cusps at the roots.

    Note that these functions are high degree and may have spurious cusps.
    compute_spline_surface_cusps is preferred for finding cusps.

    :param spline_surface: [in] quadratic spline surface
    :param frame: [in] projection frame
    :param contour_domain_curve_segments: [in] local parametric domain contour segments
    :param patch_indices: [in] spline surface patch indices for the contour segments
    :return cusp_functions: implicit cusp functions per contour segment. degree 12, dimension 1.
    """
    deprecated("Function does not appear to be used.")

    assert frame.shape == (3, 3)
    cusp_functions: list[RationalFunction] = []

    for i, _ in enumerate(contour_domain_curve_segments):
        patch_index: PatchIndex = patch_indices[i]
        cusp_function: RationalFunction = _compute_spline_surface_patch_cusp_function(
            spline_surface.get_patch(patch_index),
            frame,
            contour_domain_curve_segments[i])
        assert (cusp_function.degree, cusp_function.dimension) == (12, 1)
        cusp_functions.append(cusp_function)

    return cusp_functions


def tangent_x(px: Vector6f, py: Vector6f) -> Vector6f:
    """
    Calculates the coefficients of the tangent of the x and y coefficients 
    of the normalized surface mappings.
    FIXME: autogenerated black box code
    """
    assert px.shape == (6, )
    assert py.shape == (6, )

    tx: Vector6f = np.zeros(shape=(6, ))
    t10: float
    t11: float
    t13: float
    t14: float
    t17: float
    t20: float
    t24: float
    t27: float
    t3: float
    t35: float
    t4: float
    t41: float
    t48: float
    t52: float
    t7: float
    t8: float
    t3 = px[1]
    t4 = py[5]
    t7 = px[2]
    t8 = py[3]
    t10 = px[3]
    t11 = py[2]
    t13 = px[5]
    t14 = py[1]
    t17 = -t11 * t10 + 0.2e1 * t14 * t13 - 0.2e1 * t3 * t4 + t7 * t8
    t20 = py[4]
    t24 = px[4]
    t27 = -t14 * t10 + 0.2e1 * t11 * t24 - 0.2e1 * t20 * t7 + t3 * t8
    t35 = 0.4e1 * t13 * t20 - 0.4e1 * t24 * t4
    t41 = -0.4e1 * t10 * t20 + 0.4e1 * t24 * t8
    t48 = -0.4e1 * t10 * t4 + 0.4e1 * t13 * t8
    t52 = -t35
    tx[0] = t17 * t3 + t27 * t7
    tx[1] = t10 * t27 + 0.2e1 * t17 * t24 + t3 * t35 + t41 * t7
    tx[2] = t10 * t17 + 0.2e1 * t13 * t27 + t3 * t48 + t52 * t7
    tx[3] = t10 * t35 + t10 * t52 + 0.2e1 * t13 * t41 + 0.2e1 * t24 * t48
    tx[4] = t10 * t41 + 0.2e1 * t24 * t35
    tx[5] = t10 * t48 + 0.2e1 * t13 * t52

    return tx


def _tangent_y(px: Vector6f, py: Vector6f) -> Vector6f:
    """
    Calculates coefficients of tangent vector from coefficients of x and y 3D
    FIXME: autogenerated black box code
    """
    assert px.shape == (6, )
    assert py.shape == (6, )

    ty: Vector6f = np.zeros(shape=(6,))
    t10: float
    t11: float
    t13: float
    t14: float
    t17: float
    t20: float
    t24: float
    t27: float
    t3: float
    t35: float
    t4: float
    t41: float
    t48: float
    t52: float
    t7: float
    t8: float
    t3 = px[1]
    t4 = py[5]
    t7 = px[2]
    t8 = py[3]
    t10 = px[3]
    t11 = py[2]
    t13 = px[5]
    t14 = py[1]
    t17 = -t11 * t10 + 0.2e1 * t14 * t13 - 0.2e1 * t3 * t4 + t7 * t8
    t20 = py[4]
    t24 = px[4]
    t27 = -t14 * t10 + 0.2e1 * t11 * t24 - 0.2e1 * t20 * t7 + t3 * t8
    t35 = 0.4e1 * t13 * t20 - 0.4e1 * t24 * t4
    t41 = -0.4e1 * t10 * t20 + 0.4e1 * t24 * t8
    t48 = -0.4e1 * t10 * t4 + 0.4e1 * t13 * t8
    t52 = -t35
    ty[0] = t11 * t27 + t14 * t17
    ty[1] = t11 * t41 + t14 * t35 + 0.2e1 * t17 * t20 + t27 * t8
    ty[2] = t11 * t52 + t14 * t48 + t17 * t8 + 0.2e1 * t27 * t4
    ty[3] = 0.2e1 * t20 * t48 + t35 * t8 + 0.2e1 * t4 * t41 + t52 * t8
    ty[4] = 0.2e1 * t20 * t35 + t41 * t8
    ty[5] = 0.2e1 * t4 * t52 + t48 * t8

    return ty


def _normal_x(py: Vector6f, pz: Vector6f) -> Vector6f:
    """
    Calculates coefficients of normal vector from coefficients of y and z 3D
    FIXME: autogenerated black box code
    """
    assert py.shape == (6, )
    assert pz.shape == (6, )

    nx: Vector6f = np.zeros(shape=(6,))
    t10: float
    t12: float
    t15: float
    t17: float
    t21: float
    t26: float
    t3: float
    t4: float
    t6: float
    t7: float
    t3 = py[1]
    t4 = pz[2]
    t6 = pz[1]
    t7 = py[2]
    t10 = pz[3]
    t12 = pz[4]
    t15 = py[3]
    t17 = py[4]
    t21 = pz[5]
    t26 = py[5]
    nx[0] = t3 * t4 - t6 * t7
    nx[1] = t10 * t3 - 0.2e1 * t12 * t7 - t15 * t6 + 0.2e1 * t17 * t4
    nx[2] = -t10 * t7 + t15 * t4 + 0.2e1 * t21 * t3 - 0.2e1 * t26 * t6
    nx[3] = -0.4e1 * t12 * t26 + 0.4e1 * t17 * t21
    nx[4] = 0.2e1 * t17 * t10 - 0.2e1 * t15 * t12
    nx[5] = -0.2e1 * t10 * t26 + 0.2e1 * t15 * t21

    return nx


def _normal_y(px: Vector6f, pz: Vector6f) -> Vector6f:
    """
    Calculates coefficients of normal vector from coefficients of x and z 3D
    FIXME: autogenerated black box code
    """
    assert px.shape == (6, )
    assert pz.shape == (6, )

    ny: Vector6f = np.zeros(shape=(6,))
    t10: float
    t12: float
    t15: float
    t17: float
    t21: float
    t26: float
    t3: float
    t4: float
    t6: float
    t7: float
    t3 = px[1]
    t4 = pz[2]
    t6 = pz[1]
    t7 = px[2]
    t10 = pz[3]
    t12 = pz[4]
    t15 = px[3]
    t17 = px[4]
    t21 = pz[5]
    t26 = px[5]
    ny[0] = -t3 * t4 + t6 * t7
    ny[1] = -t10 * t3 + 0.2e1 * t12 * t7 + t15 * t6 - 0.2e1 * t17 * t4
    ny[2] = t10 * t7 - t15 * t4 - 0.2e1 * t21 * t3 + 0.2e1 * t26 * t6
    ny[3] = 0.4e1 * t12 * t26 - 0.4e1 * t17 * t21
    ny[4] = -0.2e1 * t17 * t10 + 0.2e1 * t15 * t12
    ny[5] = 0.2e1 * t10 * t26 - 0.2e1 * t15 * t21

    return ny


def _evaluate(p: Vector6f, u: float, v: float) -> float:
    """
    FIXME: autogenerated black box code
    """
    return p[0] + p[1] * u + p[2] * v + p[3] * u * v + p[4] * u * u + p[5] * v * v


def _check_validity_by_equation(p: Vector6f, u: float, v: float) -> bool:
    if float_equal_zero(_evaluate(p, u, v)):
        return True
    return False


def _compute_cusp_by_one_patch(spline_surface_patch: QuadraticSplineSurfacePatch,
                               contour_domain_curve_segment: Conic) -> list[float]:
    """
    Builds list of Conic by computing cusp by one patch.
    FIXME: this method likely to go wrong.
    """
    conics: list[float] = []

    normalized_surface_mapping_coeffs: Matrix6x3f
    normalized_surface_mapping_coeffs = spline_surface_patch.get_normalized_surface_mapping()
    assert normalized_surface_mapping_coeffs.shape == (6, 3)

    # FIXME: below using autogenerated blackbox methods from above
    px: Vector6f = normalized_surface_mapping_coeffs[:, 0]
    py: Vector6f = normalized_surface_mapping_coeffs[:, 1]
    assert px.shape == (6, )
    assert py.shape == (6, )

    #
    # FIXME: tx and ty look correct for the first iteration that compute_cusp_by_one_patch() is called
    #
    tx: Vector6f = tangent_x(px, py)
    ty: Vector6f = _tangent_y(px, py)
    assert tx.shape == (6, )
    assert ty.shape == (6, )

    # Normalized solution
    num_solutions: int
    solutions: list[PlanarPoint1d]  # length 4  # FIXME: now it is correct below.
    num_solutions, solutions = solve_quadratic_quadratic_equation_pencil_method(tx, ty)

    solutions_in_domain: list[PlanarPoint1d] = []
    # Check whether the solution is inside the domain # FIXME: below loop looks good! works good for i == 285
    for i in range(num_solutions):
        u: float = solutions[i][0]
        v: float = solutions[i][1]
        if u >= 0 and v >= 0 and u + v <= 1:
            solutions_in_domain.append(
                spline_surface_patch.denormalize_domain_point(solutions[i]))

    if len(solutions_in_domain) == 0:
        return conics

    # Get t on the conic
    conic_numerators: Matrix3x2f = contour_domain_curve_segment.numerators
    conic_denominators: Vector3f = contour_domain_curve_segment.denominator
    assert conic_numerators.shape == (3, 2)
    assert conic_denominators.shape == (3, )
    pu: Vector3f = conic_numerators[:, 0]
    pv: Vector3f = conic_numerators[:, 1]
    q: Vector3f = conic_denominators
    assert pu.shape == (3, )

    linear_pu: bool = float_equal_zero(pu[2])
    linear_pv: bool = float_equal_zero(pv[2])

    for i, solution in enumerate(solutions_in_domain):
        u: float = solution[0]
        v: float = solution[1]

        if not linear_pu and not linear_pv:
            if abs(pu[2]) > abs(pv[2]):
                # Solve pu(t) = u * q(t)
                t_conic: Vector1D = polynomial_real_roots(pu - u * q)
                # Check which t is correct
                for _, t in enumerate(t_conic):
                    if float_equal(pv[0] + pv[1] * t + pv[2] * t * t,
                                   (q[0] + q[1] * t + q[2] * t * t) * v,
                                   1e-9):
                        if contour_domain_curve_segment.domain.contains(t):
                            conics.append(t)
                            break
            else:
                # Solve pv(t) = v * q(t)
                t_conic: Vector1D = polynomial_real_roots(pv - v * q)
                # Check which t is correct
                for _, t in enumerate(t_conic):
                    if float_equal(pu[0] + pu[1] * t + pu[2] * t * t,
                                   (q[0] + q[1] * t + q[2] * t * t) * u,
                                   1e-9):
                        if contour_domain_curve_segment.domain.contains(t):
                            conics.append(t)
                            break
        else:
            if not linear_pu:
                t_conic: Vector1D = polynomial_real_roots(pu - u * q)
                # Check which t is correct
                for _, t in enumerate(t_conic):
                    if float_equal(pv[0] + pv[1] * t + pv[2] * t * t,
                                   (q[0] + q[1] * t + q[2] * t * t) * v,
                                   1e-9):
                        if contour_domain_curve_segment.domain.contains(t):
                            conics.append(t)
                            break
            elif not linear_pv:
                # Solve pv(t) = v * q(t)
                t_conic = polynomial_real_roots(pv - v * q)
                # Check which t is correct
                for _, t in enumerate(t_conic):
                    if float_equal(pu[0] + pu[1] * t + pu[2] * t * t,
                                   (q[0] + q[1] * t + q[2] * t * t) * u,
                                   1e-9):
                        if contour_domain_curve_segment.domain.contains(t):
                            conics.append(t)
                            break
            else:
                if abs(pu[1]) > abs(pu[2]):
                    t_conic: Vector1D = polynomial_real_roots(pu - u * q)
                    # Check which t is correct
                    for _, t in enumerate(t_conic):
                        if float_equal(pv[0] + pv[1] * t + pv[2] * t * t,
                                       (q[0] + q[1] * t + q[2] * t * t) * v,
                                       1e-9):
                            if contour_domain_curve_segment.domain.contains(t):
                                conics.append(t)
                                break
                else:
                    t_conic: Vector1D = polynomial_real_roots(pv - v * q)
                    for _, t in enumerate(t_conic):
                        if float_equal(pu[0] + pu[1] * t + pu[2] * t * t,
                                       (q[0] + q[1] * t + q[2] * t * t) * u,
                                       1e-9):
                            if contour_domain_curve_segment.domain.contains(t):
                                conics.append(t)
                                break
    return conics


def _compute_cusp_start_end_points(spline_surface: QuadraticSplineSurface,
                                   contour_domain_curve_segments: list[Conic],
                                   patch_indices: list[PatchIndex]) -> tuple[list[float],
                                                                             list[float],
                                                                             list[float],
                                                                             list[float]]:
    """
    FIXME: utilizes autogenerated blackbox methods
    """
    function_start_points: list[float] = []
    function_end_points: list[float] = []
    function_start_points_param: list[float] = []
    function_end_points_param: list[float] = []

    for i, conic in enumerate(contour_domain_curve_segments):
        patch: QuadraticSplineSurfacePatch = spline_surface.get_patch(patch_indices[i])
        surface_mapping: Matrix6x3f = patch.surface_mapping
        t_start: float = conic.domain.lower_bound
        t_end: float = conic.domain.upper_bound

        px: Vector6f = surface_mapping[:, 0]
        py: Vector6f = surface_mapping[:, 1]
        pz: Vector6f = surface_mapping[:, 2]

        tx: Vector6f = tangent_x(px, py)
        ty: Vector6f = _tangent_y(px, py)
        nx: Vector6f = _normal_x(py, pz)
        ny: Vector6f = _normal_y(px, pz)

        uv_start: PlanarPoint1d = conic.evaluate(t_start)
        uv_end: PlanarPoint1d = conic.evaluate(t_end)

        nx_start: float = _evaluate(nx, uv_start[0], uv_start[1])
        ny_start: float = _evaluate(ny, uv_start[0], uv_start[1])
        tx_start: float = _evaluate(tx, uv_start[0], uv_start[1])
        ty_start: float = _evaluate(ty, uv_start[0], uv_start[1])
        nx_end: float = _evaluate(nx, uv_end[0], uv_end[1])
        ny_end: float = _evaluate(ny, uv_end[0], uv_end[1])
        tx_end: float = _evaluate(tx, uv_end[0], uv_end[1])
        ty_end: float = _evaluate(ty, uv_end[0], uv_end[1])

        z_start: float = nx_start * ty_start - ny_start * tx_start
        z_end: float = nx_end * ty_end - ny_end * tx_end

        function_start_points.append(z_start)
        function_end_points.append(z_end)
        function_start_points_param.append(t_start)
        function_end_points_param.append(t_end)

    return (function_start_points,
            function_end_points,
            function_start_points_param,
            function_end_points_param)


def _compute_boundary_cusps(cusp_function_start_points: list[float],
                            cusp_function_end_points: list[float],
                            cusp_function_start_points_param: list[float],
                            cusp_function_end_points_param: list[float],
                            closed_contours: list[list[int]]) -> tuple[list[list[float]],
                                                                       list[bool],
                                                                       list[bool]]:
    """

    """
    num_segments: PatchIndex = len(cusp_function_start_points)
    num_closed_contours: PatchIndex = len(closed_contours)
    boundary_cusps: list[list[float]] = [[] for _ in range(num_segments)]
    has_cusp_at_base: list[bool] = [False] * num_segments
    has_cusp_at_tip: list[bool] = [False] * num_segments

    for i in range(num_closed_contours):
        for j, _ in enumerate(closed_contours[i]):
            # Get cusp function values around the boundary....
            current_segment: PatchIndex = closed_contours[i][j]
            next_segment: int = closed_contours[i][(j + 1) % len(closed_contours[i])]
            cusp_function_limit_from_left: float = cusp_function_end_points[current_segment]
            cusp_function_limit_from_right: float = cusp_function_start_points[next_segment]

            # Add cusps if there is a sign change
            if (cusp_function_limit_from_left * cusp_function_limit_from_right) < 0:
                # Add cusp for current segments
                current_segment_cusp: float = cusp_function_end_points_param[current_segment]
                boundary_cusps[current_segment].append(current_segment_cusp)

                # Add cusp for next segment
                next_segment_cusp: float = cusp_function_start_points_param[next_segment]
                boundary_cusps[next_segment].append(next_segment_cusp)

                # Mark cusps
                has_cusp_at_base[next_segment] = True
                has_cusp_at_tip[current_segment] = True

    return boundary_cusps, has_cusp_at_base, has_cusp_at_tip


def compute_spline_surface_cusps(spline_surface: QuadraticSplineSurface,
                                 contour_domain_curve_segments: list[Conic],
                                 contour_segments: list[RationalFunction],
                                 patch_indices: list[PatchIndex],
                                 closed_contours: list[list[int]]) -> tuple[list[list[float]],
                                                                            list[list[float]],
                                                                            list[bool],
                                                                            list[bool]]:
    """
    Compute interior and boundary cusps per patch for a spline surface.

    :param spline_surface: [in] quadratic spline surface
    :param contour_domain_curve_segments: [in] local parametric domain contour segments
    :param contour_segments: [in] surface contour segments
    :param patch_indices:    [in] spline surface patch indices for the contour segments
    :param closed_contours:  [in] list of indices of segments for complete surface contours

    :return interior_cusps: paramater points of interior cusps per contour segment
    :return boundary_cusps: paramater points of boundary cusps per contour segment
    :return has_cusp_at_base: boolean per contour segment indicating if a cusp is at the base
    :return has_cusp_at_tip: boolean per contour segment indicating if a cusp is at the tip
    """
    # lazy check
    assert (contour_domain_curve_segments[0].degree,
            contour_domain_curve_segments[0].dimension) == (2, 2)
    assert (contour_segments[0].degree, contour_segments[0].dimension) == (4, 3)

    interior_cusps: list[list[float]] = []
    boundary_cusps: list[list[float]] = []
    has_cusp_at_base: list[bool] = []
    has_cusp_at_tip: list[bool] = []

    # Interior conics
    # FIXME: loop below looks good for spot_control
    for i, _ in enumerate(contour_domain_curve_segments):
        cusp: list[float] = _compute_cusp_by_one_patch(spline_surface.get_patch(patch_indices[i]),
                                                       contour_domain_curve_segments[i])
        interior_cusps.append(cusp)

    # FIXME: function below looks good.
    # Compute cusp function endpoints
    function_start_points: list[float]
    function_end_points: list[float]
    function_start_points_param: list[float]
    function_end_points_param: list[float]
    (function_start_points,
     function_end_points,
     function_start_points_param,
     function_end_points_param) = _compute_cusp_start_end_points(spline_surface,
                                                                 contour_domain_curve_segments,
                                                                 patch_indices)

    # Compute boundary cusps
    # FIXME: function below looks good.
    (boundary_cusps,
     has_cusp_at_base,
     has_cusp_at_tip) = _compute_boundary_cusps(function_start_points,
                                                function_end_points,
                                                function_start_points_param,
                                                function_end_points_param,
                                                closed_contours)

    return interior_cusps, boundary_cusps, has_cusp_at_base, has_cusp_at_tip


# ****************************
# Exposing methods for testing
# ****************************

def compute_cusp_by_one_patch_testing(spline_surface_patch: QuadraticSplineSurfacePatch,
                                      contour_domain_curve_segment: Conic) -> list[float]:
    """
    Exposed method for testing
    """
    cusp: list[float] = _compute_cusp_by_one_patch(spline_surface_patch,
                                                   contour_domain_curve_segment)
    return cusp


def compute_cusp_start_end_points_testing(spline_surface: QuadraticSplineSurface,
                                          contour_domain_curve_segments: list[Conic],
                                          patch_indices: list[PatchIndex]) -> tuple[list[float],
                                                                                    list[float],
                                                                                    list[float],
                                                                                    list[float]]:
    """
    Exposing method for testing
    """
    function_start_points: list[float]
    function_end_points: list[float]
    function_start_points_param: list[float]
    function_end_points_param: list[float]

    (function_start_points,
     function_end_points,
     function_start_points_param,
     function_end_points_param) = _compute_cusp_start_end_points(spline_surface,
                                                                 contour_domain_curve_segments,
                                                                 patch_indices)
    return (function_start_points,
            function_end_points,
            function_start_points_param,
            function_end_points_param)


def compute_boundary_cusps_testing(function_start_points: list[float],
                                   function_end_points: list[float],
                                   function_start_points_param: list[float],
                                   function_end_points_param: list[float],
                                   closed_contours: list[list[int]]) -> tuple[list[list[float]],
                                                                              list[bool],
                                                                              list[bool]]:
    """
    Exposing method for testing
    """
    boundary_cusps: list[list[float]] = []
    has_cusp_at_base: list[bool] = []
    has_cusp_at_tip: list[bool] = []

    (boundary_cusps,
     has_cusp_at_base,
     has_cusp_at_tip) = _compute_boundary_cusps(function_start_points,
                                                function_end_points,
                                                function_start_points_param,
                                                function_end_points_param,
                                                closed_contours)

    return boundary_cusps, has_cusp_at_base, has_cusp_at_tip
