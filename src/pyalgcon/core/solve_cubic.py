# CUBIC ROOT SOLVER

# Date Created   :    24.05.2017
# Created by     :    Shril Kumar [(shril.iitdhn@gmail.com),(github.com/shril)] &
#                     Devojoyti Halder [(devjyoti.itachi@gmail.com),(github.com/devojoyti)]

# Project        :    Classified
# Use Case       :    Instead of using standard numpy.roots() method for finding roots,
#                     we have implemented our own algorithm which is ~10x faster than
#                     in-built method.

# Modifications  :    Removed check for quadratic equations, only solving cubic equations.
#                     Replaced direct floating-point comparison with threshold-based comparison.
#                     Renamed function from solve() to solve_cubic()

# Algorithm Link :    www.1728.org/cubic2.htm

# This script (Cubic Equation Solver) is an independent program for computation of roots of Cubic
# Polynomials. This script, however, has no relation with original project code or calculations.
# It is to be also made clear that no knowledge of it's original project is included or used to
# device this script. This script is complete freeware developed by above signed users, and may
# further be used or modified for commercial or non-commercial purpose.

import math

import numpy as np

from pyalgcon.core.common import float_equal_zero


def _solve_cubic(coeffs: list[float] | np.ndarray | tuple[float, float, float, float],
                 ) -> np.ndarray:
    """
    Solve cubic equation for roots using some version of Cardano's Method 
    given a cubic equation with unknown "t" and non-zero leading "a" coefficient.

    Format: ax^3 + bx^2 + cx + d = 0
    """
    # TODO: add threshold parameter for float_equal_zero comparisons
    a, b, c, d = coeffs

    f = ((3.0 * c / a) - ((b ** 2.0) / (a ** 2.0))) / 3.0
    g = (((2.0 * (b ** 3.0)) / (a ** 3.0)) - ((9.0 * b * c) / (a ** 2.0)) + (27.0 * d / a)) / 27.0
    h = ((g ** 2.0) / 4.0 + (f ** 3.0) / 27.0)

    # All 3 Roots are Real and Equal
    if float_equal_zero(f) and float_equal_zero(g) and float_equal_zero(h):
        if (d / a) >= 0:
            x = (d / (1.0 * a)) ** (1 / 3.0) * -1
        else:
            x = (-d / (1.0 * a)) ** (1 / 3.0)
        return np.array([x, x, x])

    elif h <= 0:  # All 3 roots are Real
        i = math.sqrt(((g ** 2.0) / 4.0) - h)
        j = i ** (1 / 3.0)
        k = math.acos(-(g / (2 * i)))
        L = j * -1
        M = math.cos(k / 3.0)
        N = math.sqrt(3) * math.sin(k / 3.0)
        P = (b / (3.0 * a)) * -1

        x1 = 2 * j * math.cos(k / 3.0) - (b / (3.0 * a))
        x2 = L * (M + N) + P
        x3 = L * (M - N) + P

        return np.array([x1, x2, x3])

    elif h > 0:  # One Real Root and two Complex Roots
        R = -(g / 2.0) + math.sqrt(h)
        if R >= 0:
            S = R ** (1 / 3.0)
        else:
            S = (-R) ** (1 / 3.0) * -1
        T = -(g / 2.0) - math.sqrt(h)
        if T >= 0:
            U = (T ** (1 / 3.0))
        else:
            U = ((-T) ** (1 / 3.0)) * -1

        x1 = (S + U) - (b / (3.0 * a))
        x2 = -(S + U) / 2 - (b / (3.0 * a)) + (S - U) * math.sqrt(3) * 0.5j
        x3 = -(S + U) / 2 - (b / (3.0 * a)) - (S - U) * math.sqrt(3) * 0.5j

        return np.array([x1, x2, x3])

    raise ValueError(f"Invalid coeffs when solving cubic: {coeffs}")
