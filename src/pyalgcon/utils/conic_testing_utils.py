"""
conic_testing_utils.py

Methods used to serialize, deserialize, and compare conics for testing.
"""
import pathlib
from typing import Any

import numpy as np
import numpy.testing as npt

from pyalgcon.core.common import float_equal, load_json
from pyalgcon.core.conic import Conic, ConicType
from pyalgcon.core.interval import Interval


def _string_to_conictype(type_string: str) -> ConicType:
    """
    Turns string representation to enum ConicType
    """
    if type_string == "ELLIPSE":
        return ConicType.ELLIPSE
    if type_string == "HYPERBOLA":
        return ConicType.HYPERBOLA
    if type_string == "PARABOLA":
        return ConicType.PARABOLA
    if type_string == "PARALLEL_LINES":
        return ConicType.PARALLEL_LINES
    if type_string == "INTERSECTING_LINES":
        return ConicType.INTERSECTING_LINES
    if type_string == "LINE":
        return ConicType.LINE
    if type_string == "POINT":
        return ConicType.POINT
    if type_string == "EMPTY":
        return ConicType.EMPTY
    if type_string == "PLANE":
        return ConicType.PLANE
    if type_string == "ERROR":
        return ConicType.ERROR
    if type_string == "UNKNOWN":
        return ConicType.UNKNOWN
    raise ValueError(f"Given {type_string} does not match any of the ConicTypes")


def compare_conics_from_file(filepath: pathlib.Path,
                             conics_test: list[Conic]) -> None:
    """
    Reads Conics from file to compare to.
    """
    conics_control: list[Conic] = deserialize_conics_from_file(filepath)
    compare_conics(conics_control, conics_test)


def compare_conics(conics_control: list[Conic],
                   conics_test: list[Conic]) -> None:
    """
    Takes in two lists of conics and compares their member variables for closeness.
    """
    assert len(conics_control) == len(conics_test)
    num_conics: int = len(conics_control)

    # TODO: override Python's "==" comparison to make this process below modular/more flexible.
    for i in range(num_conics):
        conic_control_ref: Conic = conics_control[i]
        conic_test_ref: Conic = conics_test[i]

        assert conic_control_ref.degree == conic_test_ref.degree
        assert conic_control_ref.dimension == conic_test_ref.dimension
        assert conic_control_ref.type == conic_test_ref.type
        npt.assert_allclose(
            conic_test_ref.numerator_coeffs,
            conic_control_ref.numerator_coeffs)
        npt.assert_allclose(
            conic_test_ref.denominator_coeffs,
            conic_control_ref.denominator_coeffs)
        float_equal(conic_test_ref.domain.lower_bound,
                    conic_control_ref.domain.lower_bound)
        float_equal(conic_test_ref.domain.upper_bound,
                    conic_control_ref.domain.upper_bound)
        assert (conic_test_ref.domain.is_bounded_below() ==
                conic_control_ref.domain.is_bounded_below())
        assert (conic_test_ref.domain.is_bounded_above() ==
                conic_control_ref.domain.is_bounded_above())
        assert (conic_test_ref.domain.is_open_below() ==
                conic_control_ref.domain.is_open_below())
        assert (conic_test_ref.domain.is_open_above() ==
                conic_control_ref.domain.is_open_above())


def deserialize_conic(conic_intermediate: dict[str, Any]) -> Conic:
    """
    Takes in JSON dict representation of Conic and converts to Conic object.
    """
    # Extract the following:
    degree: int = conic_intermediate.get("degree")
    dimension: int = conic_intermediate.get("dimension")

    type: str = conic_intermediate.get("type")

    # NOTE: must transpose to be (degree + 1, dimension) shape.
    numerator_coeffs: list[list[float]] = np.array(
        conic_intermediate.get("numerator_coeffs"),
        dtype=np.float64).T

    denominator_coeffs: list[list[float]] = np.array(
        conic_intermediate.get("denominator_coeffs"),
        dtype=np.float64).squeeze()

    # Getting the interval.
    t0: float = conic_intermediate.get("domain").get("t0")
    t1: float = conic_intermediate.get("domain").get("t1")

    # NOTE: the below probably is not needed, but it's good to confirm nonetheless
    bounded_below: bool = conic_intermediate.get("domain").get("bounded_below")
    bounded_above: bool = conic_intermediate.get("domain").get("bounded_above")
    open_below: bool = conic_intermediate.get("domain").get("open_below")
    open_above: bool = conic_intermediate.get("domain").get("open_above")

    domain: Interval = Interval(t0, t1)
    domain.set_lower_bound(t0)
    domain.set_upper_bound(t1)

    conic_final: Conic = Conic(
        _string_to_conictype(type),
        numerator_coeffs,
        denominator_coeffs,
        domain
    )

    return conic_final


def deserialize_conics_from_file(filepath: pathlib.Path) -> list[Conic]:
    """
    Takes in a JSON file and deserializes it to list of Conic objects.
    """
    # Intermediate processing.
    # TODO: add typehinting for dict
    conics_intermediate: list[dict] = load_json(filepath)
    conics_final: list[Conic] = []

    for conic_intermediate in conics_intermediate:
        conic_final: Conic = deserialize_conic(conic_intermediate)
        # # Extract the following:
        # degree: int = conic.get("degree")
        # dimension: int = conic.get("dimension")

        # type: str = conic.get("type")

        # # NOTE: must transpose to be (degree + 1, dimension) shape.
        # numerator_coeffs: list[list[float]] = np.array(
        #     conic.get("numerator_coeffs"),
        #     dtype=np.float64).T

        # denominator_coeffs: list[list[float]] = np.array(
        #     conic.get("denominator_coeffs"),
        #     dtype=np.float64).squeeze()

        # # Getting the interval.
        # t0: float = conic.get("domain").get("t0")
        # t1: float = conic.get("domain").get("t1")

        # # NOTE: the below probably is not needed, but it's good to confirm nonetheless
        # bounded_below: bool = conic.get("domain").get("bounded_below")
        # bounded_above: bool = conic.get("domain").get("bounded_above")
        # open_below: bool = conic.get("domain").get("open_below")
        # open_above: bool = conic.get("domain").get("open_above")

        # domain: Interval = Interval(t0, t1)
        # domain.set_lower_bound(t0)
        # domain.set_upper_bound(t1)

        # conic_final: Conic = Conic(
        #     _string_to_conictype(type),
        #     numerator_coeffs,
        #     denominator_coeffs,
        #     domain
        # )

        conics_final.append(conic_final)

    return conics_final
