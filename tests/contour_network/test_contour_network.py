import logging
import os

import numpy as np

from pyalgcon.contour_network.compute_intersections import \
    IntersectionParameters
from pyalgcon.contour_network.contour_network import (ContourNetwork,
                                                      InvisibilityMethod,
                                                      InvisibilityParameters,
                                                      _build_contour_labels)
from pyalgcon.core.affine_manifold import AffineManifold
from pyalgcon.core.apply_transformation import \
    apply_camera_frame_transformation_to_vertices
from pyalgcon.core.common import (Matrix3x3f, MatrixNx3f,
                                  deserialize_eigen_matrix_csv_to_numpy,
                                  initialize_spot_control_mesh)
from pyalgcon.quadratic_spline_surface.optimize_spline_surface import \
    OptimizationParameters
from pyalgcon.quadratic_spline_surface.twelve_split_spline import (
    TwelveSplitSplineSurface, compute_twelve_split_spline_patch_boundary_edges)
from pyalgcon.utils.projected_curve_networks_utils import (
    SVGOutputMode, compare_segment_labels)

logger: logging.Logger = logging.getLogger(__name__)


# TODO: the deserialization of rational functions and then printing of rational functions should be the same as the whole rational functions.txt file


def test_build_contour_labels_spot_control(root_folder) -> None:
    """

    """
    filepath: str = f"{root_folder}\\contour_network\\contour_network\\build_contour_labels\\"

    contour_patch_indices: list[int] = deserialize_eigen_matrix_csv_to_numpy(
        filepath+"contour_patch_indices.csv").tolist()
    contour_is_boundary: list[bool] = np.array(deserialize_eigen_matrix_csv_to_numpy(
        filepath+"contour_is_boundary.csv"), dtype=bool).tolist()

    contour_segment_labels_test: list[dict[str, int]] = _build_contour_labels(
        contour_patch_indices,
        contour_is_boundary)

    compare_segment_labels(filepath+"contour_segment_labels.json",
                           contour_segment_labels_test)


def test_contour_network() -> None:
    """
    Testing contour network creation with control spot mesh
    """
    svg_output_mode = SVGOutputMode.UNIFORM_VISIBLE_CURVES
    optimization_params = OptimizationParameters()
    intersect_params = IntersectionParameters()
    invisibility_params = InvisibilityParameters()
    weight: float = optimization_params.position_difference_factor
    trim: float = intersect_params.trim_amount
    pad: float = invisibility_params.pad_amount
    invisibility_method: InvisibilityMethod = invisibility_params.invisibility_method
    show_nodes: bool = False
    logger.setLevel(logging.INFO)

    # Set up the camera
    frame: Matrix3x3f = np.array([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]])
    logger.info("Projecting onto frame:\n%s", frame)
    V:  np.ndarray
    uv: np.ndarray
    F:  np.ndarray
    FT: np.ndarray
    V, uv, F, FT = initialize_spot_control_mesh()
    V_transformed: MatrixNx3f = apply_camera_frame_transformation_to_vertices(V, frame)
    # projection_matrix = np.array([
    #     [2.777777671813965, 0.0, 0.0, 0.0],
    #     [0.0, 4.938271522521973, 0.0, 0.0],
    #     [0.0, 0.0, -1.0020020008087158, -0.20020020008087158],
    #     [0.0, 0.0, -1.0, 0.0]], dtype=np.float64)

    # V_transformed: MatrixNx3f = apply_transformation_to_vertices(V, np.array(projection_matrix))

    # Generate quadratic spline
    logger.info("Computing spline surface")
    affine_manifold: AffineManifold = AffineManifold(F, uv, FT)
    spline_surface: TwelveSplitSplineSurface = TwelveSplitSplineSurface(V_transformed,
                                                                        affine_manifold,
                                                                        optimization_params)

    # Get the boundary edges
    patch_boundary_edges: list[tuple[int, int]]
    patch_boundary_edges: list[tuple[int, int]] = (
        compute_twelve_split_spline_patch_boundary_edges(F, spline_surface.face_to_patch_indices))

    # Build the contours
    logger.info("Computing contours")
    contour_network = ContourNetwork(
        spline_surface,
        intersect_params,
        invisibility_params,
        patch_boundary_edges
    )

    # Save the contours to file
    logger.info("Saving contours")
    contour_network_file: str = "contours.svg"
    contour_network_path: str = os.path.abspath(f"spot_control\\{contour_network_file}")
    contour_network.write(contour_network_path, svg_output_mode, show_nodes)


def test_contour_network_pytest(initialize_contour_network) -> None:
    """
    Utilizing Pytest utility methods.
    """
    contour_network: ContourNetwork = initialize_contour_network
    contour_network_file: str = "contours.svg"
    contour_network_path: str = os.path.abspath(f"\\spot_control\\{contour_network_file}")
    print(contour_network_path)
    # contour_network.write(contour_network_path, svg_output_mode, show_nodes)
