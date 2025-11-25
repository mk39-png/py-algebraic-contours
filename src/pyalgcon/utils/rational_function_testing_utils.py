"""
rational_function_utils.py

Methods used to serialize, deserialize, and compare rational functions for testing.
"""

from typing import Any

import numpy as np
import numpy.testing as npt

from pyalgcon.core.common import float_equal, load_json
from pyalgcon.core.interval import Interval
from pyalgcon.core.rational_function import RationalFunction

# These utils are more for testing than anything else.
# TODO: move/rename file to better reflect that this file is only for testing and not to be used for
# rational_function.py itself.


def compare_rational_functions_from_file(filename: str,
                                         rational_functions_test: list[RationalFunction]) -> None:
    """
    Reads from file to compare rational functions
    """
    rational_functions_control: list[RationalFunction] = deserialize_rational_functions(filename)
    compare_rational_functions(rational_functions_control, rational_functions_test)


def compare_rational_functions(rational_functions_control: list[RationalFunction],
                               rational_functions_test: list[RationalFunction]) -> None:
    """
    Takes in two lists of rational functions and compares their member variables for closeness.
    """
    assert len(rational_functions_control) == len(rational_functions_test)
    num_rational_functions: int = len(rational_functions_control)

    # TODO: override Python's "==" comparison to make this process below modular/more flexible.
    for i in range(num_rational_functions):
        rational_function_control_ref: RationalFunction = rational_functions_control[i]
        rational_function_test_ref: RationalFunction = rational_functions_test[i]

        assert rational_function_control_ref.degree == rational_function_test_ref.degree
        assert rational_function_control_ref.dimension == rational_function_test_ref.dimension
        npt.assert_allclose(
            rational_function_test_ref.numerator_coeffs,
            rational_function_control_ref.numerator_coeffs)
        npt.assert_allclose(
            rational_function_test_ref.denominator_coeffs,
            rational_function_control_ref.denominator_coeffs)
        float_equal(rational_function_test_ref.domain.lower_bound,
                    rational_function_control_ref.domain.lower_bound)
        float_equal(rational_function_test_ref.domain.upper_bound,
                    rational_function_control_ref.domain.upper_bound)
        assert (rational_function_test_ref.domain.is_bounded_below() ==
                rational_function_control_ref.domain.is_bounded_below())
        assert (rational_function_test_ref.domain.is_bounded_above() ==
                rational_function_control_ref.domain.is_bounded_above())
        assert (rational_function_test_ref.domain.is_open_below() ==
                rational_function_control_ref.domain.is_open_below())
        assert (rational_function_test_ref.domain.is_open_above() ==
                rational_function_control_ref.domain.is_open_above())


def deserialize_rational_function(rational_function_intermediate: dict[str, Any]
                                  ) -> RationalFunction:
    """
    Takes in JSON dict representation of RationalFunction and converts to RationalFunction object.
    """
    # Intermediate processing.
    # TODO: add typehinting for dict

    # Extract the following:
    degree: int = rational_function_intermediate.get("degree")
    dimension: int = rational_function_intermediate.get("dimension")

    # NOTE: must transpose to be (degree + 1, dimension) shape.
    numerator_coeffs: list[list[float]] = np.array(
        rational_function_intermediate.get("numerator_coeffs"),
        dtype=np.float64).T

    denominator_coeffs: list[list[float]] = np.array(
        rational_function_intermediate.get("denominator_coeffs"),
        dtype=np.float64).squeeze()

    # Getting the interval.
    t0: float = rational_function_intermediate.get("domain").get("t0")
    t1: float = rational_function_intermediate.get("domain").get("t1")

    # NOTE: the below probably is not needed, but it's good to confirm nonetheless
    bounded_below: bool = rational_function_intermediate.get("domain").get("bounded_below")
    bounded_above: bool = rational_function_intermediate.get("domain").get("bounded_above")
    open_below: bool = rational_function_intermediate.get("domain").get("open_below")
    open_above: bool = rational_function_intermediate.get("domain").get("open_above")

    domain: Interval = Interval(t0, t1)
    domain.set_lower_bound(t0)
    domain.set_upper_bound(t1)

    rational_function_final: RationalFunction = RationalFunction(
        degree,
        dimension,
        numerator_coeffs,
        denominator_coeffs,
        domain)

    return rational_function_final


def deserialize_rational_functions(filepath: str) -> list[RationalFunction]:
    """
    Takes in a JSON file and deserializes it to list of RationalFunction objects.
    """
    # Intermediate processing.
    # TODO: add proper typehinting for dict
    rational_functions_intermediate: list[dict[str, int | float | bool | list]] = load_json(
        filepath)
    rational_functions_final: list[RationalFunction] = []

    for rational_function_intermediate in rational_functions_intermediate:
        rational_function_final: RationalFunction = deserialize_rational_function(
            rational_function_intermediate)

        # # Extract the following:
        # degree: int = rational_function.get("degree")
        # dimension: int = rational_function.get("dimension")

        # # NOTE: must transpose to be (degree + 1, dimension) shape.
        # numerator_coeffs: list[list[float]] = np.array(
        #     rational_function.get("numerator_coeffs"),
        #     dtype=np.float64).T

        # denominator_coeffs: list[list[float]] = np.array(
        #     rational_function.get("denominator_coeffs"),
        #     dtype=np.float64).squeeze()

        # # Getting the interval.
        # t0: float = rational_function.get("domain").get("t0")
        # t1: float = rational_function.get("domain").get("t1")

        # # NOTE: the below probably is not needed, but it's good to confirm nonetheless
        # bounded_below: bool = rational_function.get("domain").get("bounded_below")
        # bounded_above: bool = rational_function.get("domain").get("bounded_above")
        # open_below: bool = rational_function.get("domain").get("open_below")
        # open_above: bool = rational_function.get("domain").get("open_above")

        # domain: Interval = Interval(t0, t1)
        # domain.set_lower_bound(t0)
        # domain.set_upper_bound(t1)

        # rational_function_final: RationalFunction = RationalFunction(
        #     degree,
        #     dimension,
        #     numerator_coeffs,
        #     denominator_coeffs,
        #     domain
        # )

        rational_functions_final.append(rational_function_final)

    return rational_functions_final
