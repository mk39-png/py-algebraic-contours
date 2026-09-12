import numpy as np
import numpy.testing as npt

from pyalgcon.core.solve_cubic import _solve_cubic


def test_solve_cubic() -> None:
    """
    Comparing with NumPy implementation.
    """
    trials = 10000
    degree = 3

    for _ in range(trials):
        coeffs = np.random.uniform(-100, 100, degree + 1)
        coeffs_np = coeffs[::-1]

        # Compute the complex roots
        solver = np.polynomial.Polynomial(coeffs_np)
        # Reverse to match order of descending degree
        control_roots: np.ndarray = solver.roots()[::-1]
        test_roots: np.ndarray = _solve_cubic(coeffs)

        # NOTE: root may have different ordering, which for our case is fine for now.
        npt.assert_allclose(actual=np.sort_complex(test_roots),
                            desired=np.sort_complex(control_roots),
                            atol=1e-7)
