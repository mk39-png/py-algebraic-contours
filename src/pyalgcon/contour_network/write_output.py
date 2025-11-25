"""
write_output.py

Methods for writing contour data to SVG file.
"""
import numpy as np
import svg

from pyalgcon.contour_network.discretize import \
    discretize_curve_segments
from pyalgcon.core.common import (Color, Matrix3x3f,
                                  PlanarPoint1d,
                                  SpatialVector1d, Vector2f)
from pyalgcon.core.rational_function import (
    CurveDiscretizationParameters, RationalFunction)


def _project_point(p: SpatialVector1d, frame: Matrix3x3f, scale: int, offset: int) -> Vector2f:
    """
    Helper method.
    """
    assert p.shape == (3, )
    assert frame.shape == (3, 3)

    p_3d: SpatialVector1d = p @ frame
    assert p_3d.shape == (3, )

    p_3d[0] = -scale * p_3d[0] + offset  # Inverted to account for SVG orientation
    p_3d[1] = scale * p_3d[1] + offset
    p_3d[2] = scale * p_3d[2] + offset

    p_2d: Vector2f = np.array([p_3d[0], p_3d[1]], dtype=np.float64)
    assert p_2d.shape == (2, )

    return p_2d


def _transform_point(p: PlanarPoint1d, scale: int, offset: int) -> Vector2f:
    """
    Helper method.
    """
    assert p.shape == (2, )

    p_transform: Vector2f = np.array([
        -scale * p[0] + offset,  # Inverted to account for SVG orientation
        scale * p[1] + offset], dtype=np.float64)
    assert p_transform.shape == (2, )

    return p_transform


def add_curve_to_svg(points: list[PlanarPoint1d],
                     polyline: list[int],
                     svg_elements_ref: list[svg.Element],
                     scale: int,
                     offset: int,
                     color: Color = (0, 0, 0, 1)) -> None:
    """
    Write curve to SVG without frame.

    :param points: [in] points to save to SVG
    :param polylines: [in] polylines to save to SVG
    :param svg_elements_ref: [out] list of SVG elements to save 
    :param scale: [in] scale of points
    :param offset: [in] offset of points
    :param color: [in] RGBA color of SVG element
    """
    polyline_points: list[float] = []

    for j, _ in enumerate(polyline):
        point: PlanarPoint1d = points[polyline[j]]
        transformed_point: Vector2f = _transform_point(point, scale, offset)
        polyline_points.append(transformed_point[0])
        polyline_points.append(transformed_point[1])

    svg_elements_ref.append(svg.Polyline(points=polyline_points,
                                         stroke=f"rgba{color}",
                                         fill="transparent",
                                         stroke_width=1.0))


def add_curve_network_to_svg(points: list[SpatialVector1d],
                             polylines: list[list[int]],
                             frame: Matrix3x3f,
                             svg_elements_ref: list[svg.Element],
                             scale: int = 800,
                             offset: int = 400) -> None:
    """
    Write curve network to SVG.

    :param points: [in] points to save to SVG
    :param polylines: [in] polylines to save to SVG
    :param frame: [in] 3x3 matrix defining the projection
    :param svg_elements_ref: [out] list of SVG elements to save 
    :param scale: [in] scale of points
    :param offset: [in] offset of points
    """
    assert frame.shape == (3, 3)

    # Write curve network
    for polyline in polylines:
        polyline_points: list[float] = []

        for j, _ in enumerate(polyline):
            point: SpatialVector1d = points[polyline[j]]
            assert point.shape == (3, )
            projected_point: Vector2f = _project_point(point, frame, scale, offset)
            polyline_points.append(projected_point[0])
            polyline_points.append(projected_point[1])

        svg_elements_ref.append(svg.Polyline(points=polyline_points,
                                             stroke="black",
                                             fill="transparent",
                                             stroke_width=1.0))


def add_vectors_to_svg(base_points: list[SpatialVector1d],
                       vectors: list[SpatialVector1d],
                       frame: Matrix3x3f,
                       svg_elements_ref: list[svg.Element],
                       normalize: bool = False,
                       scale: int = 800,
                       offset: int = 400) -> None:
    """
    Write vectors to SVG.

    :param base_points: [in] base points to save to SVG
    :param vectors: [in] vectors to save to SVG
    :param frame: [in] 3x3 matrix defining the projection
    :param svg_elements_ref: [out] list of SVG elements to save 
    :param normalize: [in] identifies whether to normalize vector tip
    :param scale: [in] scale of points
    :param offset: [in] offset of points
    """
    assert len(base_points) == len(vectors)
    assert frame.shape == (3, 3)

    # Write vectors
    for i, _ in enumerate(base_points):
        vector_line: list[float] = []

        base_point_3d: SpatialVector1d = base_points[i] @ frame
        vector_tip_3d: SpatialVector1d = vectors[i] @ frame
        assert base_point_3d.shape == (3, )
        assert vector_tip_3d.shape == (3, )

        base_point_2d: Vector2f = np.array([base_point_3d[0], base_point_3d[1]], dtype=np.float64)
        vector_tip_2d: Vector2f = np.array([vector_tip_3d[0], vector_tip_3d[1]], dtype=np.float64)

        base_point_2d = scale * base_point_2d + offset

        if normalize:
            vector_tip_2d /= np.linalg.norm(vector_tip_2d)
        vector_tip_2d = 0.5 * scale * vector_tip_2d + base_point_2d

        # Append x, y of base_point and vector_tip
        vector_line.append(base_point_2d[0])
        vector_line.append(base_point_2d[1])
        vector_line.append(vector_tip_2d[0])
        vector_line.append(vector_tip_2d[1])

        svg_elements_ref.append(svg.Polyline(points=vector_line,
                                             stroke="black",
                                             fill="transparent",
                                             stroke_width=1.0))


def add_point_to_svg(point: SpatialVector1d,
                     frame: Matrix3x3f,
                     svg_elements: list[svg.Element],
                     scale: int = 800,
                     offset: int = 400,
                     color: Color = (255, 0, 0, 1)) -> None:
    """
    Write spatial vector point as SVG.

    :param point: [in] point to write
    :param frame: [in] 3x3 matrix defining the projection
    :param svg_elements_ref: [out] list of SVG elements to save 
    :param scale: [in] scale of point
    :param offset: [in] offset of point
    :param color: [in] RGBA color of SVG element
    """
    assert point.shape == (3, )
    assert frame.shape == (3, 3)

    # Write point
    point_3d: SpatialVector1d = point @ frame
    assert point_3d.shape == (3, )
    point_2d: Vector2f = np.array([point_3d[0], point_3d[1]], dtype=np.float64)
    point_2d = scale * point_2d + offset
    svg_elements.append(svg.Circle(cx=point_2d[0], cy=point_2d[1], r=0.25, fill=f"rgba{color}"))


def write_planar_point(point: PlanarPoint1d,
                       svg_elements_ref: list[svg.Element],
                       scale: int = 800,
                       offset: int = 400,
                       color: tuple = (255, 0, 0, 1)) -> None:
    """
    Write planar point as SVG.

    :param point: [in] planar point to write
    :param svg_elements_ref: [out] list of SVG elements to save 
    :param scale: [in] scale of point
    :param offset: [in] offset of point
    :param color: [in] RGBA color of SVG element
    """
    assert point.shape == (2, )

    # Write planar point
    point_2d: Vector2f = _transform_point(point, scale, offset)
    svg_elements_ref.append(svg.Circle(cx=point_2d[0], cy=point_2d[1], r=2.0, fill=f"rgba{color}"))


def write_contours(frame: Matrix3x3f,
                   contour_segments: list[RationalFunction],
                   curve_disc_params: CurveDiscretizationParameters,
                   svg_elements_ref: list[svg.Element]) -> None:
    """
    Write contours to SVG.

    :param frame: [in] 3x3 matrix defining the projection
    :param contour_segments: [in] surface contour segments
    :param curve_disc_params: [in] parameters for the contour discretization
    :param svg_elements_ref: [out] list of SVG elements to save 
    """
    assert frame.shape == (3, 3)
    assert (contour_segments[0].degree, contour_segments[0].dimension) == (4, 3)  # lazy check

    # Write contours
    points: list[SpatialVector1d]
    polylines: list[list[int]]
    points, polylines = discretize_curve_segments(4, 3, contour_segments, curve_disc_params)
    add_curve_network_to_svg(points, polylines, frame, svg_elements_ref)


def write_planar_curve_segment(planar_curve_segment: RationalFunction,
                               curve_disc_params: CurveDiscretizationParameters,
                               svg_elements_ref: list[svg.Element],
                               scale: int = 800,
                               offset: int = 400,
                               color: Color = (0, 0, 0, 1)) -> None:
    """
    Write planar curve segment to SVG.

    :param planar_curve_segment: [in] planar curve segment to write
    :param curve_disc_params: [in] parameters for the contour discretization
    :param svg_elements_ref: [out] list of SVG elements to save 
    :param scale: [in] scale of planar curve 
    :param offset: [in] offset of planar curve
    :param color: [in] RGBA color of SVG element
    """
    assert (planar_curve_segment.degree, planar_curve_segment.dimension) == (4, 2)

    # Write curve
    points: list[PlanarPoint1d]
    polyline: list[int]
    points, polyline = planar_curve_segment.discretize(curve_disc_params)
    add_curve_to_svg(points, polyline, svg_elements_ref, scale, offset, color)


def _write_contour_point(frame: Matrix3x3f,
                         contour_segment: RationalFunction,  # degree 4, dimension 3
                         parameter_points: list[float],
                         svg_elements_ref: list[svg.Element],
                         scale: int = 800,
                         offset: int = 400,
                         color: Color = (1, 0, 0, 1)) -> None:
    """
    Helper to write_contour_points()
    """
    assert frame.shape == (3, 3)
    assert (contour_segment.degree, contour_segment.dimension) == (4, 3)

    # Write points
    for i, _ in enumerate(parameter_points):
        point: SpatialVector1d = contour_segment(parameter_points[i])
        assert point.shape == (3, )
        add_point_to_svg(point, frame, svg_elements_ref, scale, offset, color)


def write_contour_points(frame: Matrix3x3f,
                         contour_segments: list[RationalFunction],
                         parameter_points: list[list[float]],
                         svg_elements_ref: list[svg.Element],
                         scale: int = 800,
                         offset: int = 400,
                         color: Color = (255, 0, 0, 1)) -> None:
    """
    Write contour points to SVG.

    :param frame: [in] 3x3 matrix defining the projection
    :param contour_segments: [in] surface contour segments
    :param parameter_points: [in] points on curve
    :param svg_elements_ref: [out] list of SVG elements to save 
    :param scale: [in] scale of contour segments
    :param offset: [in] offset of contour segments
    :param color: [in] RGBA color of SVG element
    """
    assert frame.shape == (3, 3)
    assert (contour_segments[0].degree, contour_segments[0].dimension) == (4, 3)  # lazy check

    # Write points
    for i, _ in enumerate(parameter_points):
        _write_contour_point(frame,
                             contour_segments[i],
                             parameter_points[i],
                             svg_elements_ref,
                             scale,
                             offset,
                             color)


def write_contours_with_annotations(frame: Matrix3x3f,
                                    contour_segments: list[RationalFunction],
                                    interior_cusps: list[list[float]],
                                    boundary_cusps: list[list[float]],
                                    intersections: list[list[float]],
                                    curve_disc_params: CurveDiscretizationParameters,
                                    svg_elements_ref: list[svg.Element],
                                    scale: int = 800,
                                    offset: int = 400) -> None:
    """
    Write contours to SVG with annotations.

    :param frame: [in] 3x3 matrix defining the projection
    :param contour_segments: [in] surface contour segments
    :param interior_cusps: [in] paramater points of interior cusps per contour segment
    :param boundary_cusps: [in] paramater points of boundary cusps per contour segment
    :param intersections: [in] list of lists of intersection knots
    :param curve_disc_params: [in] parameters for the contour discretization
    :param svg_elements_ref: [out] list of SVG elements to save 
    :param scale: [in] scale of contour segments
    :param offset: [in] offset of contour segments
    """
    assert frame.shape == (3, 3)
    assert (contour_segments[0].degree, contour_segments[0].dimension) == (4, 3)

    # Write contours
    write_contours(frame, contour_segments, curve_disc_params, svg_elements_ref)

    # Write points
    write_contour_points(frame,
                         contour_segments,
                         interior_cusps,
                         svg_elements_ref,
                         scale,
                         offset,
                         (255, 0, 0, 1))  # red
    write_contour_points(frame,
                         contour_segments,
                         boundary_cusps,
                         svg_elements_ref,
                         scale,
                         offset,
                         (255, 191, 191, 1))  # Pink
    write_contour_points(frame,
                         contour_segments,
                         intersections,
                         svg_elements_ref,
                         scale,
                         offset,
                         (0, 255, 0, 1))  # Green
