"""
Original ASOC code utilizes the AutoDiff library by Wenzel Jakob. Based on code by Jon Kaldor
and Eitan Grinspun.
"""


def generate_local_variable_matrix_index(row: int, col: int, dimension=3) -> int:
    """
    Used in powell_sabin_local_to_global.py.
    """
    return dimension * row + col
