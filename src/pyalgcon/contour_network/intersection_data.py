"""
File for intersection data.
"""

from pyalgcon.core.common import float_equal
from pyalgcon.core.interval import Interval


class IntersectionData():
    """
    Data for intersections.
    Originally a struct in C++ code.
    """

    def __init__(self,
                 knot: float,
                 intersection_index: int,
                 intersection_knot: float,
                 id_: int,
                 is_base: bool = False,
                 is_tip: bool = False,
                 is_redundant: bool = False) -> None:
        """Constructor for IntersectionData"""
        # Parameter value for the the intersection in the given curve
        self.knot: float = knot
        # ID of the intersecting curve
        self.intersection_index: int = intersection_index
        # Parameter of the intersection in the intersecting curve's domain
        self.intersection_knot: float = intersection_knot
        # Unique identifier for the intersection
        self.id: int = id_
        # True iff the intersection is at the base of the curve
        self.is_base: bool = is_base
        # True iff the intersection is at the tip of the curve
        self.is_tip: bool = is_tip
        # Flag for redundant intersections
        self.is_redundant: bool = is_redundant

    def check_if_tip(self, domain_ref: Interval, eps: float) -> None:
        """
        Check if the knot is the tip of an oriented curve
        @param[in] domain: domain for the curve
        @param[in] eps: epsilon tolerance for the check
        """
        self.is_tip = float_equal(domain_ref.upper_bound, self.knot, eps)

    def check_if_base(self, domain_ref: Interval, eps: float) -> None:
        """
        Check if the knot is the base of an oriented curve

        :param domain: [in] domain for the curve
        :param eps: [in] epsilon tolerance for the check
        """
        self.is_base = float_equal(domain_ref.lower_bound, self.knot, eps)
