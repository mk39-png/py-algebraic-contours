import numpy as np
cimport numpy as cnp
cnp.import_array()
DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t

cdef dot3(cnp.ndarray[DTYPE_t, ndim=1] v,cnp.ndarray[DTYPE_t, ndim=1] w):
    """
    Returns dot product.
    NOTE: for some reason, this implementation is preferred by the C++ code over the numpy.dot 
    method
    """
    return v[0] * w[0] + v[1] * w[1] + v[2] * w[2]

def point_distance_squared(cnp.ndarray[DTYPE_t, ndim=1] point_1, 
                           cnp.ndarray[DTYPE_t, ndim=1] point_2):
    """
    Distance helper function
    """
    # TODO: cythonize and call dot3_c internally
    assert point_1.size == 3
    assert point_2.size == 3
    cdef cnp.ndarray displacement = np.empty(3, dtype=DTYPE)
    displacement[0] = point_1[0] - point_2[0]
    displacement[1] = point_1[1] - point_2[1]
    displacement[2] = point_1[2] - point_2[2]

    return dot3(displacement, displacement)
