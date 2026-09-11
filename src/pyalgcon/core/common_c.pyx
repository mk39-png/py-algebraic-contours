import numpy as np
cimport numpy as cnp
cnp.import_array()
DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t

def cross_product(cnp.ndarray[DTYPE_t, ndim=1] v, cnp.ndarray[DTYPE_t, ndim=1] w):
    """
    Compute the cross product of two vectors of arbitrary scalars.
    Dedicated method to check for NumPy shapes before cross product calculation.

    :param v: [in] first vector to cross product
    :param w: [in] second vector to cross product
    :return: cross product v x w in shape (3, )
    """
    # TODO: make this 1D only.
    assert v.shape[0] == 3
    assert w.shape[0] == 3
    assert v.size == 3 and w.size == 3

    # NOTE: for some reason, this implementation is preferred by the C++ code over the numpy.cross
    # method. Since trying to use np.cross will cause some parts of PYAC to fail.
    cdef cnp.ndarray n = np.empty(3, dtype=DTYPE) 
    n[0] = (v[1] * w[2] - v[2] * w[1])
    n[1] = (-(v[0] * w[2] - v[2] * w[0]))
    n[2] = (v[0] * w[1] - v[1] * w[0])
    assert n.shape[0] == 3
    return n

