

import numpy as np

from pyalgcon.core.abstract_curve_network import \
    AbstractCurveNetwork
from pyalgcon.core.common import (
    NodeIndex, SegmentIndex, compare_eigen_numpy_matrix,
    deserialize_eigen_matrix_csv_to_numpy)


def test_init_abstract_curve_network() -> None:
    """
    """
    filepath_base: str = "spot_control\\core\\abstract_curve_network\\init_abstract_curve_network\\"
    to_array: list[NodeIndex] = np.array(deserialize_eigen_matrix_csv_to_numpy(
        filepath_base+"to_array.csv"),
        dtype=np.int64).tolist()
    out_array: list[SegmentIndex] = np.array(deserialize_eigen_matrix_csv_to_numpy(
        filepath_base+"out_array.csv"),
        dtype=np.int64).tolist()

    filepath_next: str = "spot_control\\core\\abstract_curve_network\\build_next_array\\"
    next_array: list[SegmentIndex] = AbstractCurveNetwork.build_next_array(to_array, out_array)
    compare_eigen_numpy_matrix(filepath_next+"next_array.csv",
                               np.array(next_array, dtype=np.int64))

    filepath_prev: str = "spot_control\\core\\abstract_curve_network\\build_prev_array\\"
    prev_array: list[SegmentIndex] = AbstractCurveNetwork.build_prev_array(to_array, out_array)
    compare_eigen_numpy_matrix(filepath_prev+"prev_array.csv",
                               np.array(prev_array, dtype=np.int64))

    filepath_from: str = "spot_control\\core\\abstract_curve_network\\build_from_array\\"
    from_array: list[NodeIndex] = AbstractCurveNetwork.build_from_array(to_array, out_array)
    compare_eigen_numpy_matrix(filepath_from+"from_array.csv",
                               np.array(from_array, dtype=np.int64))

    filepath_in: str = "spot_control\\core\\abstract_curve_network\\build_in_array\\"
    in_array: list[SegmentIndex] = AbstractCurveNetwork.build_in_array(to_array, out_array)
    compare_eigen_numpy_matrix(filepath_in+"in_array.csv",
                               np.array(in_array, dtype=np.int64))
