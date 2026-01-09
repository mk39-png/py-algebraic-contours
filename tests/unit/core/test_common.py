"""
Test common methods
"""


from pyalgcon.core.affine_manifold import AffineManifold
from pyalgcon.core.common import *
from pyalgcon.core.halfedge import Halfedge

# *******************
# Test Methods
# *******************


def test_compute_point_cloud_bounding_box(parsed_control_mesh) -> None:
    """
    Test compute_point_cloud_bounding_box
    """
    # Retrieve parameters
    V: np.ndarray
    V, _, _, _ = parsed_control_mesh

    # Execute method
    min_point_test: Vector3f
    max_point_test: Vector3f
    min_point_test, max_point_test = compute_point_cloud_bounding_box(V)

    # Compare results
    # Values from testing with spot_control_mesh-cleaned_conf_simplified_with_uv.obj
    # and printing to terminal to get control points
    min_point_control: Vector3f = np.array([-0.585967, -0.871576, -0.865963])
    max_point_control: Vector3f = np.array([0.509437, 0.866886, 0.810781])
    npt.assert_allclose(min_point_test, min_point_control, atol=1e-7)
    npt.assert_allclose(max_point_test, max_point_control, atol=1e-7)


def test_index_vector_complement(testing_fileinfo,
                                 parsed_control_mesh,
                                 initialize_affine_manifold) -> None:
    """
    Tests index_vector_complement() to initialize variable_vertices and variable_edges 
    for generate_optimized_twelve_split_position_data() for the spot_control mesh.
    NOTE: these values should be the same for the fit and full cases hessian cases.
    """
    # Retrieve parameters
    base_data_folderpath: pathlib.Path
    base_data_folderpath, _ = testing_fileinfo
    filepath: pathlib.Path = base_data_folderpath / "quadratic_spline_surface" / \
        "optimize_spline_surface" / "generate_optimized_twelve_split_position_data"
    V: np.ndarray
    V, _, _, _ = parsed_control_mesh
    affine_manifold: AffineManifold = initialize_affine_manifold
    halfedge: Halfedge = affine_manifold.halfedge
    num_vertices: int = V.shape[ROWS]
    num_edges: int = halfedge.num_edges

    # Execute method
    fixed_vertices: list[int] = []
    fixed_edges: list[int] = []
    variable_vertices: list[int] = index_vector_complement(fixed_vertices, num_vertices)
    variable_edges: list[int] = index_vector_complement(fixed_edges, num_edges)

    # Compare results
    compare_eigen_numpy_matrix(filepath / "variable_vertices.csv", np.array(variable_vertices))
    compare_eigen_numpy_matrix(filepath / "variable_edges.csv", np.array(variable_edges))


def test_cross_product() -> None:
    """
    Testing cross product functionality.
    """
    v: np.ndarray = np.array([1, 2, 3])
    w: np.ndarray = np.array([4, 5, 6])
    assert v.shape == (3, )
    assert w.shape == (3, )

    n_test = cross_product(v, w)
    n_numpy_control = np.cross(v, w, axis=0)

    assert np.array_equal(n_test, n_numpy_control)


def test_convert_nested_vector_to_matrix() -> None:
    """
    Testing if original C++ code is equivalent to NumPy operation
    """
    boundary_points: list[SpatialVector1d] = [np.array([0, 1, 2], dtype=np.float64),
                                              np.array([3, 4, 5], dtype=np.float64),
                                              np.array([6, 7, 8], dtype=np.float64)]
    matrix_test: Matrix3x3f = convert_nested_vector_to_matrix(boundary_points)
    matrix_control: Matrix3x3f = np.asarray(boundary_points)
    assert matrix_test.shape == (3, 3)
    assert matrix_control.shape == (3, 3)

    np.testing.assert_allclose(matrix_test, matrix_control)
