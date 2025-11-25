"""
powell_sabin_local_to_global_indexing.py
Methods to generate local to global indexing maps for six and twelve split
Powell-Sabin spline surfaces.

The variables we are optimizing need to be linearized and initialized with
some values. Similarly, the final optimized results need to be extracted.
Furthermore, for the optimization it is useful to combine the variable values
into a single vector so that, e.g., gradient descent or Newton's method can
be applied.

Used by optimize_spline_surface.py
"""
import logging
from typing import Literal

import numpy as np

from pyalgcon.core.common import (COLS, PLACEHOLDER_VALUE,
                                  ROWS, Index, Matrix2x3r,
                                  SpatialVector1d,
                                  TwelveSplitGradient,
                                  TwelveSplitHessian,
                                  Vector1D)
from pyalgcon.core.differentiable_variable import \
    generate_local_variable_matrix_index
from pyalgcon.core.halfedge import Halfedge

logger: logging.Logger = logging.getLogger(__name__)


# *******************
# Local block indices
# *******************


def generate_local_vertex_position_variables_start_index(vertex_index: int,
                                                         dimension: int = 3) -> int:
    """
    Get the local start index of the block of position variable indices for a given vertex
    """
    relative_vertex_index: int = 3 * dimension * vertex_index
    return relative_vertex_index


def generate_local_vertex_gradient_variables_start_index(vertex_index: int,
                                                         dimension: int = 3) -> int:
    """
    Get the local start index of the block of gradient variable indices for a given vertex
    """
    relative_vertex_index: int = 3 * dimension * vertex_index
    position_block_size: int = dimension
    return relative_vertex_index + position_block_size


def generate_local_edge_gradient_variables_start_index(edge_index: int,
                                                       dimension: int = 3) -> int:
    """ 
    Get the local start index of the block of gradient variable indices for a given edge
    """
    vertex_block_size: int = 9 * dimension
    relative_edge_index: int = dimension * edge_index

    return vertex_block_size + relative_edge_index


# **********************
# Local variable indices
# **********************

def generate_local_vertex_position_variable_index(face_vertex_index: int,
                                                  coord: int,
                                                  dimension: int = 3) -> int:
    """
    Used in optimize_spline_surface.py

    Compute the index of a vertex position variable in a local DOF vector. 
    i.e. Get the local index of the position variable indices for a given coordinate
    and vertex index

    :param face_vertex_index: [in] index of the vertex in the face
    :param coord:             [in] coordinate of the variable
    :param dimension:         [in] number of coordinate dimensions
    :return: index of the variable in the local DOF vector
    """
    start_index: int = generate_local_vertex_position_variables_start_index(face_vertex_index,
                                                                            dimension)

    return start_index + coord


def generate_local_vertex_gradient_variable_index(face_vertex_index: int,
                                                  row: int,
                                                  col: int,
                                                  dimension: int = 3) -> int:
    """
    Used in optimize_spline_surface.py

    Compute the index of a vertex gradient variable in a local DOF vector.
    i.e. Get the local index of the gradient variable indices for a given matrix index
    pair and vertex index

    :param face_vertex_index: [in] index of the vertex in the face
    :param row:               [in] row of the gradient matrix variable
    :param col:               [in] column of the gradient matrix variable
    :param dimension:         [in] number of coordinate dimensions
    :return: index of the variable in the local DOF vector
    """
    start_index: int = generate_local_vertex_gradient_variables_start_index(
        face_vertex_index, dimension)
    matrix_index: int = generate_local_variable_matrix_index(row, col, dimension)

    return start_index + matrix_index


def generate_local_edge_gradient_variable_index(face_edge_index: int,
                                                coord: int,
                                                dimension: int = 3) -> int:
    """
    Used in optimize_spline_surface.py

    Compute the index of a edge gradient variable in a local DOF vector.
    i.e. Get the local index of the gradient variable indices for a given coordinate
    and edge index pair

    :param face_vertex_index: [in] index of the edge in the face
    :param coord: [in] coordinate of the variable
    :param dimension: [in] number of coordinate dimensions
    :return: index of the variable in the local DOF vector
    """

    start_index: int = generate_local_edge_gradient_variables_start_index(
        face_edge_index, dimension)

    return start_index + coord


# ********************
# Global block indices
# ********************

def generate_global_vertex_position_variables_block_start_index() -> Literal[0]:
    """
    Get the start index of the block of vertex position variable indices
    """
    return 0


def generate_global_vertex_gradient_variables_block_start_index(num_variable_vertices: int,
                                                                dimension: int) -> int:
    """
    Get the start index of the block of vertex gradient variable indices
    """

    # There are dimension many position variables per variable vertex
    return dimension * num_variable_vertices


def generate_global_edge_gradient_variables_block_start_index(num_variable_vertices: int,
                                                              dimension: int) -> int:
    """ 
    Get the start index of the block of edge gradient variable indices
    """
    # There are dimension many position variables and 2 * dimension many vector
    # gradient variables per variable vertex
    return 3 * dimension * num_variable_vertices


def generate_global_vertex_position_variables_start_index(vertex_index: int,
                                                          dimension: int) -> int:
    """
    Get the start index of the block of position variable indices for a given vertex
    """
    start_index: int = generate_global_vertex_position_variables_block_start_index()
    relative_vertex_index: int = dimension * vertex_index

    return start_index + relative_vertex_index


def generate_global_vertex_gradient_variables_start_index(num_variable_vertices: int,
                                                          vertex_index: int, dimension: int) -> int:
    """
    Get the start index of the block of gradient variable indices for a given vertex
    """
    start_index: int = generate_global_vertex_gradient_variables_block_start_index(
        num_variable_vertices, dimension)
    relative_vertex_index: int = 2 * dimension * vertex_index

    return start_index + relative_vertex_index


def generate_global_edge_gradient_variables_start_index(num_variable_vertices: int,
                                                        edge_index: int,
                                                        dimension: int) -> int:
    """
    Get the start index of the block of gradient variable indices for a given edge
    """
    start_index: int = generate_global_edge_gradient_variables_block_start_index(
        num_variable_vertices, dimension)
    relative_edge_index: int = dimension * edge_index

    return start_index + relative_edge_index


# ***********************
# Global variable indices
# ***********************

def generate_global_vertex_position_variable_index(vertex_index: int,
                                                   coord: int,
                                                   dimension: int = 3) -> int:
    """
    Used locally.

    Compute the index of a vertex position variable in a global DOF vector.
    i.e. Get the global index of the position variable indices for a given coordinate
    and vertex index

    :param vertex_index: [in] index of the vertex in the mesh
    :param coord:        [in] coordinate of the variable
    :param dimension:    [in] number of coordinate dimensions
    :return: index of the variable in the global DOF vector
    """
    start_index: int = generate_global_vertex_position_variables_start_index(
        vertex_index, dimension)

    return start_index + coord


def generate_global_vertex_gradient_variable_index(num_variable_vertices: int,
                                                   vertex_index: int,
                                                   row: int,
                                                   col: int,
                                                   dimension: int = 3) -> int:
    """
    Used locally.

    Compute the index of a vertex gradient variable in a global DOF vector.
    i.e. Get the index of the gradient variable indices for a given matrix index pair
    and vertex index

    @param[in] num_variable_vertices: number of variable vertices for the optimization
    @param[in] vertex_index: index of the vertex in the mesh
    @param[in] row: row of the gradient matrix variable
    @param[in] col: column of the gradient matrix variable
    @param[in] dimension: number of coordinate dimensions
    @return index of the variable in the global DOF vector
    """
    start_index: int = generate_global_vertex_gradient_variables_start_index(
        num_variable_vertices, vertex_index, dimension)
    matrix_index: int = generate_local_variable_matrix_index(row, col, dimension)

    return start_index + matrix_index


def generate_global_edge_gradient_variable_index(num_variable_vertices: int,
                                                 edge_index: int,
                                                 coord: int,
                                                 dimension: int = 3) -> int:
    """
    Compute the index of an edge gradient variable in a global DOF vector.
    i.e. Get the index of the gradient variable indices for a given coordinate and
    edge index pair

    :param num_variable_vertices: [in] number of variable vertices for the optimization
    :param edge_index:            [in] index of the edge in the mesh
    :param coord:                 [in] coordinate of the variable
    :param dimension:             [in] number of coordinate dimensions
    :return index of the variable in the global DOF vector
    """
    start_index: int = generate_global_edge_gradient_variables_start_index(num_variable_vertices,
                                                                           edge_index,
                                                                           dimension)

    return start_index + coord


# *******************
# Variable flattening
# *******************

def generate_six_split_variable_value_vector(vertex_positions: list[SpatialVector1d],
                                             vertex_gradients: list[Matrix2x3r],
                                             variable_vertices: list[int]) -> Vector1D:
    """"
    Given vertex positions and gradients and a list of variable vertices, assemble
    the vector of global vertex degrees of freedom

    This is the complete list of degrees of freedom for the six split, and it is
    a subset of the degrees of freedom for the twelve split.

    i.e. Get flat vector of all current variable values for the six-split
    NOTE: Also used as a subroutine to generate the twelve split maps.
    NOTE: keeping variable_values as 1D array since generate_twelve_split_variable_value_vector() 
    expects 1D array back.

    :param vertex_positions:  (in) list of vertex position values. SpatialVector shape (1, 3)
    :param vertex_gradients:  (in) list of vertex gradient matrices 
    :param variable_vertices: (in)  list of variable vertex indices
    :return variable_values: vertex DOF vector of shape (n, )
    """
    num_variable_vertices: int = len(variable_vertices)
    if num_variable_vertices == 0:
        # FIXME: determine severity of warning (i.e. Exception or not)
        logger.warning("Building value vector for zero variable vertices")
        variable_values: Vector1D = np.ndarray(shape=(0, 0))
        raise Exception("building value vector for zero variable vertices")
        return variable_values
    dimension: int = 3
    variable_values: Vector1D = np.ndarray(shape=(3 * dimension * num_variable_vertices, ))
    assert variable_values.ndim == 1

    # Get postion values
    for vertex_index in range(num_variable_vertices):
        start_index: int = generate_global_vertex_position_variables_start_index(
            vertex_index,
            dimension)

        for i in range(dimension):
            # NOTE: SpatialVector shape is (3, )
            assert vertex_positions[variable_vertices[vertex_index]].shape == (3, )
            variable_values[start_index + i] = vertex_positions[variable_vertices[vertex_index]][i]

    # Get gradient values
    for vertex_index in range(num_variable_vertices):
        start_index: int = generate_global_vertex_gradient_variables_start_index(
            num_variable_vertices,
            vertex_index,
            dimension)
        variable_matrix: Matrix2x3r = vertex_gradients[variable_vertices[vertex_index]]
        # TODO: change below to use numpy indexing?
        for i in range(variable_matrix.shape[ROWS]):  # rows
            for j in range(variable_matrix.shape[COLS]):  # columns
                local_index: int = generate_local_variable_matrix_index(i, j, dimension)
                variable_values[start_index + local_index] = variable_matrix[i, j]

    assert variable_values.ndim == 1
    return variable_values


def generate_twelve_split_variable_value_vector(
        vertex_positions: list[SpatialVector1d],
        vertex_gradients: list[Matrix2x3r],
        edge_gradients: list[list[SpatialVector1d]],
        variable_vertices: list[int],
        variable_edges: list[int],
        halfedge: Halfedge, he_to_corner: list[tuple[Index, Index]]) -> Vector1D:
    """
    Used in optimize_spline_surface.py

    Given vertex positions and gradients, edge gradients, and lists
    of variable vertices and edges assemble the vector of global
    degrees of freedom for the twelve split
    i.e. Get flat vector of all current variable values for the twelve-split

    NOTE: keeping variable_values as a 1D array since optimize_twelve_split_spline_surface() 
        expects a 1D array back for initial_variable_values

    :param vertex_positions:  [in] list of vertex position values
    :param vertex_gradients:  [in] list of vertex gradient matrices 
    :param edge_gradients:    [in] list of edge gradient normal vectors 
    :param variable_vertices: [in] list of variable vertex indices
    :param variable_edges:    [in] list of variable edge indices
    :param halfedge:          [in] halfedge data structure
    :param he_to_corner:      [in] map from halfedges to opposite triangle corners
    :return variable_values: twelve-split DOF vector of shape (n, )
    """
    # Get the variable values shared with the six-split
    # TODO: wait, can the below be None or no?
    six_split_variable_values: Vector1D = generate_six_split_variable_value_vector(
        vertex_positions,
        vertex_gradients,
        variable_vertices)

    #  Add six split to the variable value vector
    #  Build a halfedge representation to get unique edge values
    num_variable_vertices: int = len(variable_vertices)
    num_variable_edges: int = len(variable_edges)
    dimension: int = 3

    variable_values: Vector1D = np.ndarray(shape=(3 * dimension * num_variable_vertices +
                                                  dimension * num_variable_edges, ))
    variable_values[:six_split_variable_values.size] = six_split_variable_values
    assert variable_values.ndim == 1

    # Get flat values for edge gradients
    for variable_edge_index in range(num_variable_edges):
        # Get one corner for the given edge
        edge_index: Index = variable_edges[variable_edge_index]
        halfedge_index: Index = halfedge.edge_to_first_halfedge(edge_index)
        face_index: Index = he_to_corner[halfedge_index][0]
        face_vertex_index: Index = he_to_corner[halfedge_index][1]

        # Extract each coordinate for the corner
        num_coordinates: int = 3
        for coord in range(num_coordinates):
            # Get edge variable index
            variable_index: int = generate_global_edge_gradient_variable_index(
                num_variable_vertices, variable_edge_index, coord)

            # Extract variable value of the edge to the corner
            # FIXME: problem with setting an array element with a sequence.
            # NOTE: edge_gradients[face_index][face_vertex_index] gets SpatialVector of shape (3, )
            assert edge_gradients[face_index][face_vertex_index].shape == (3, )
            variable_values[variable_index] = edge_gradients[face_index][face_vertex_index][coord]

    assert variable_values.ndim == 1
    return variable_values


def generate_six_split_local_to_global_map(global_vertex_indices: list[int],
                                           num_variable_vertices: int) -> list[int]:
    """
    Given the global vertex indices of a triangle, compute the map from the
    local DOF vector indices for this triangle to their indices in the global
    DOF vector for the six-split

    This is used as a subroutine for the twelve-split local to global map.

    i.e. Map local triangle vertex indices to their global variable indices
    NOTE: Also used as a subroutine to generate the twelve split maps

    :param global_vertex_indices: [in] global indices of the triangle vertices 
    :param num_variable_vertices: [in] number of variable vertices
    :return local_to_global_map: map from local to global DOF indices
    """
    assert len(global_vertex_indices) == 3
    dimension: int = 3
    # TODO: check if having -1 initialized into local_to_global_map is the way to go
    # TODO: seems like it since the below sets global_index to -1... somewhere.
    # TODO: what if instead of list[int]... we use numpy arrays???
    local_to_global_map: list[int] = [PLACEHOLDER_VALUE for _ in range(27)]

    for local_vertex_index in range(3):
        global_vertex_index: int = global_vertex_indices[local_vertex_index]

        # Add vertex position index values
        for coord in range(dimension):
            # TODO: check generate_local_vertex_position_variable_index
            local_index: int = generate_local_vertex_position_variable_index(
                local_vertex_index, coord, dimension)

            global_index: int
            if global_vertex_index < 0:
                global_index = -1
            else:
                global_index = generate_global_vertex_position_variable_index(
                    global_vertex_index, coord, dimension)

            local_to_global_map[local_index] = global_index

        # Add vertex gradient index values
        for row in range(2):
            for col in range(dimension):
                local_index: int = generate_local_vertex_gradient_variable_index(
                    local_vertex_index, row, col, dimension)
                global_index: int
                if global_vertex_index < 0:
                    global_index = -1
                else:
                    global_index = generate_global_vertex_gradient_variable_index(
                        num_variable_vertices, global_vertex_index, row, col, dimension)

                local_to_global_map[local_index] = global_index

    return local_to_global_map


def generate_twelve_split_local_to_global_map(global_vertex_indices: list[int],
                                              global_edge_indices: list[int],
                                              num_variable_vertices: int) -> list[int]:
    """
    Used in optimize_spline_surface.py

    Given the global vertex and edge indices of a triangle, compute the map
    from the local DOF vector indices for this triangle to their indices in
    the global DOF vector for the twelve-split.

    :param global_vertex_indices: [in] global indices of the triangle vertices
    :param global_vertex_indices: [in] global indices of the triangle edges 
    :param num_variable_vertices: [in] number of variable vertices
    :return local_to_global_map: map from local to global DOF indices
    """
    # Making sure that "arrays" of length 3 are passed in.
    assert len(global_vertex_indices) == 3
    assert len(global_edge_indices) == 3

    # Get index map for the Powell-Sabin shared variables
    dimension: int = 3
    local_to_global_map: list[int] = [PLACEHOLDER_VALUE for _ in range(36)]
    six_split_local_to_global_map: list[int] = generate_six_split_local_to_global_map(
        global_vertex_indices, num_variable_vertices)
    assert len(six_split_local_to_global_map) == 27
    local_to_global_map[0:len(six_split_local_to_global_map)] = six_split_local_to_global_map

    for local_edge_index in range(3):
        global_edge_index: int = global_edge_indices[local_edge_index]

        # Add edge gradient index values
        for coord in range(dimension):
            local_index: int = generate_local_edge_gradient_variable_index(
                local_edge_index, coord, dimension)
            global_index: int
            if global_edge_index < 0:
                global_index = -1
            else:
                global_index = generate_global_edge_gradient_variable_index(
                    num_variable_vertices, global_edge_index, coord, dimension)

            local_to_global_map[local_index] = global_index

    assert len(local_to_global_map) == 36
    return local_to_global_map


def update_independent_variable_vector(variable_values: Vector1D,
                                       variable_vector_ref: SpatialVector1d,
                                       start_index: int) -> None:
    """
    Update variables in a vector from the vector of all variable values from some
    start index.
    NOTE: modifies variable_vector_ref by reference!
    NOTE: variable_values 1D since this method is used by update_position_variables(), 
        which is then used  in optimize_twelve_split_spline_surface() with optimized_variable_values
        passed into this method. optimized_variable_values is generated from hessian_inverse.solve,  
        which "returns" a 1D array.

    :param variable_values: (in)
    :param variable_vector_ref: (out)
    :param start_index: (in)
    """
    assert variable_values.ndim == 1
    assert variable_vector_ref.shape == (3, )

    # TODO: could probably do this with NumPy indexing
    for i in range(variable_vector_ref.size):
        variable_index: int = start_index + i
        # NOTE: SpatialVector is of shape (3, )
        variable_vector_ref[i] = variable_values[variable_index]


def update_independent_variable_matrix(variable_values: Vector1D,
                                       variable_matrix_ref: Matrix2x3r,
                                       start_index: int) -> None:
    """
    Update variables in a matrix from the vector of all variable values from some
    start index The flattening of the matrix is assumed to be row major.

    NOTE: modifies variable_matrix_ref by reference.
    """
    assert variable_values.ndim == 1
    assert variable_matrix_ref.shape == (2, 3)

    dimension: int = variable_matrix_ref.shape[COLS]  # Columns
    for i in range(variable_matrix_ref.shape[ROWS]):  # rows
        for j in range(variable_matrix_ref.shape[COLS]):  # columns
            local_index: int = generate_local_variable_matrix_index(i, j, dimension)
            variable_index: int = start_index + local_index
            variable_matrix_ref[i, j] = variable_values[variable_index]


def update_position_variables(variable_values: Vector1D,
                              variable_vertices: list[int],
                              vertex_positions_ref: list[SpatialVector1d]) -> None:
    """
    Used in optimize_spline_surface.py

    Extract vertex positions from the global DOF vector.
    i.e. Update all position variables
    NOTE: modifies vertex_positions_ref by reference.

    :param variable_values:   [in] twelve-split DOF vector
    :param variable_vertices: [in] list of variable vertex indices
    :param vertex_positions_ref: [out] list of vertex position values
    """
    if len(vertex_positions_ref) == 0:
        return

    num_variable_vertices: int = len(variable_vertices)
    dimension: int = 3

    for vertex_index in range(num_variable_vertices):
        start_index: int = generate_global_vertex_position_variables_start_index(
            vertex_index, dimension)

        update_independent_variable_vector(
            variable_values,
            vertex_positions_ref[variable_vertices[vertex_index]],
            start_index)


def update_vertex_gradient_variables(variable_values: Vector1D,
                                     variable_vertices: list[int],
                                     vertex_gradients_ref: list[Matrix2x3r]) -> None:
    """
    Used in optimize_spline_surface.py

    Extract vertex gradients from the global DOF vector.
    i.e. Update all vertex gradient variables

    NOTE: modifies vertex_gradients by reference.

    :param variable_values:   [in] twelve-split DOF vector
    :param variable_vertices: [in] list of variable vertex indices
    :param vertex_gradients_ref: [out] list of vertex gradient values
    """
    num_variable_vertices: int = len(variable_vertices)
    dimension: int = 3
    for vertex_index in range(num_variable_vertices):
        start_index: int = generate_global_vertex_gradient_variables_start_index(
            num_variable_vertices, vertex_index, dimension)

        update_independent_variable_matrix(
            variable_values,
            vertex_gradients_ref[variable_vertices[vertex_index]],
            start_index)


def update_edge_gradient_variables(variable_values: Vector1D,
                                   variable_vertices: list[int],
                                   variable_edges: list[int],
                                   halfedge: Halfedge,
                                   he_to_corner: list[tuple[Index, Index]],
                                   edge_gradients_ref: list[list[SpatialVector1d]]) -> None:
    """
    Used in optimize_spline_surface.py

    Extract edge gradients from the global DOF vector.
    i.e. Update all edge gradient variables

    NOTE: modifies edge_gradients_ref by reference.

    @param[in] variable_values: twelve-split DOF vector
    @param[in] variable_vertices: list of variable vertex indices
    @param[in] variable_edges: list of variable edge indices
    @param[in] halfedge: halfedge data structure
    @param[in] he_to_corner: map from halfedges to opposite triangle corners
    @param[out] edge_gradients: list of edge gradient values (element arrays of length 3)
    """
    dimension: int = 3
    num_variable_vertices: int = len(variable_vertices)
    num_variable_edges: int = len(variable_edges)

    # Get flat values for edge gradients
    for variable_edge_index in range(num_variable_edges):
        # Get corner for the given edge
        edge_index: int = variable_edges[variable_edge_index]
        first_halfedge_index: int = halfedge.edge_to_first_halfedge(edge_index)
        first_face_index: int = he_to_corner[first_halfedge_index][0]
        first_face_vertex_index: int = he_to_corner[first_halfedge_index][1]

        # Get index in the flattened variable vector
        start_index: int = generate_global_edge_gradient_variables_start_index(
            num_variable_vertices, variable_edge_index, dimension)

        # Update the gradients for the first corner
        update_independent_variable_vector(
            variable_values,
            edge_gradients_ref[first_face_index][first_face_vertex_index],
            start_index)

        # Update the gradients for the second corner if it exists
        if not halfedge.is_boundary_edge(edge_index):
            second_halfedge_index: int = halfedge.edge_to_second_halfedge(edge_index)
            second_face_index: int = he_to_corner[second_halfedge_index][0]
            second_face_vertex_index: int = he_to_corner[second_halfedge_index][1]

            update_independent_variable_vector(
                variable_values,
                edge_gradients_ref[second_face_index][second_face_vertex_index],
                start_index)


def build_variable_vertex_indices_map(num_vertices: int, variable_vertices: list[int]) -> list[int]:
    """
    Used in optimize_spline_surface.py

    Generate a map from all vertices to a list of variable vertices or -1 for
    vertices that are not variable.

    @param[in] num_vertices: total number of vertices
    @param[in] variable_vertices: list of variable vertex indices
    @param[out] global_vertex_indices: map from vertex indices to variable vertices
    """
    # Get variable vertex indices
    global_vertex_indices: list[int] = [PLACEHOLDER_VALUE for _ in range(num_vertices)]
    for i, _ in enumerate(variable_vertices):
        global_vertex_indices[variable_vertices[i]] = i

    return global_vertex_indices


def build_variable_edge_indices_map(num_faces: int,
                                    variable_edges: list[int],
                                    halfedge: Halfedge,
                                    he_to_corner: list[tuple[Index, Index]]) -> list[list[int]]:
    """
    Used in optimize_spline_surface.py

    NOTE: returns global_edge_indices since this method is only used internally within 
        build_twelve_split_spline_energy_system()

    Generate a map from all edges to a list of variable edges or -1 for
    edges that are not variable.

    :param num_faces:      [in] total number of faces
    :param variable_edges: [in] list of variable edge indices
    :param halfedge:       [in] halfedge data structure
    :param he_to_corner:   [in] map from halfedges to opposite triangle corners
    :return global_edge_indices: map from edge indices to variable edges
    """
    PLACEHOLDER_INT = -1
    global_edge_indices: list[list[Index]] = [[PLACEHOLDER_INT, PLACEHOLDER_INT, PLACEHOLDER_INT]
                                              for _ in range(num_faces)]

    for i, _ in enumerate(variable_edges):
        edge_index: Index = variable_edges[i]
        h0: Index = halfedge.edge_to_first_halfedge(edge_index)
        f0: Index = he_to_corner[h0][0]
        f0_vertex_index: Index = he_to_corner[h0][1]
        global_edge_indices[f0][f0_vertex_index] = i

        if not halfedge.is_boundary_edge(edge_index):
            h1: Index = halfedge.edge_to_second_halfedge(edge_index)
            f1: Index = he_to_corner[h1][0]
            f1_vertex_index: Index = he_to_corner[h1][1]
            global_edge_indices[f1][f1_vertex_index] = i

    return global_edge_indices


def update_energy_quadratic(local_energy: float,
                            local_derivatives: TwelveSplitGradient,  # shape (36, 1)
                            local_hessian: TwelveSplitHessian,  # shape (36, 36)
                            local_to_global_map: list[int],
                            energy: float,
                            derivatives_ref: Vector1D,
                            hessian_entries_ref: list[tuple[int, int, float]]
                            ) -> float:
    """
    Update global energy, derivatives, and hessian with local per face values

    NOTE: since update_energy_quadratic() is called in a loop to update the energy variable, 
    we cannot simply just reassign energy to local_energy.
    TODO: is it right to pass in parameters that are then modified and then pass those same 
    parameters back as return values?

    :param local_energy:        [in] local energy value
    :param local_derivatives:   [in] local energy gradient
    :param local_hessian:       [in] local energy Hessian
    :param local_to_global_map: [in] map from local to global DOF indices
    :return energy:           global energy value
    :param  derivatives_ref: [out] global energy gradient
    :param  hessian_ref:     [out] global energy Hessian
    """
    logger.info("Adding local face energy %s", local_energy)
    logger.info("Local to global map: %s", local_to_global_map)
    assert derivatives_ref.ndim == 1  # shape (36, 1)
    assert local_derivatives.ndim == 1
    assert local_hessian.ndim == 2

    # Update energy
    energy += local_energy

    # Update derivatives NOTE: looks fine. things are being modified by refernece and changes are reflected back in caller method
    num_local_indices: int = len(local_to_global_map)
    for local_index in range(num_local_indices):
        global_index: int = local_to_global_map[local_index]
        if global_index < 0:
            continue  # Skip fixed variables with no global index
        derivatives_ref[global_index] += local_derivatives[local_index]

    # Update hessian entries
    for local_index_i in range(num_local_indices):
        # Get global row index, skipping fixed variables with no global index
        global_index_i: int = local_to_global_map[local_index_i]
        if global_index_i < 0:
            continue

        for local_index_j in range(num_local_indices):
            # Get global column index, skipping fixed variables with no global index
            global_index_j: int = local_to_global_map[local_index_j]
            if global_index_j < 0:
                continue

            # Get Hessian entry value
            hessian_value: float = local_hessian[local_index_i, local_index_j]

            # Assemble global Hessian entry
            hessian_entries_ref.append((global_index_i, global_index_j, hessian_value))

    return energy


def build_face_variable_vector(variables: list,
                               i: int,
                               j: int,
                               k: int) -> list:
    """
    Build a triplet of face vertex values from a global array of vertex variables

    :param variables: [in] global variables
    :param i: [in] first variable index
    :param j: [in] second variable index
    :param k: [in] third variable index
    :return face_variable_vector: variables for face Tijk
    """
    # TODO: test this with some NumPy implementation
    face_variable_vector: list = [variables[i], variables[j], variables[k]]
    return face_variable_vector
