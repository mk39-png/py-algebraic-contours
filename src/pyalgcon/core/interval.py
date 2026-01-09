"""
File for Interval class.
"""


class Interval:
    """
    Representation of an interval in R that can be bounded or unbounded
    and closed or open on both ends. Also supports sampling and predicates such
    as compactness and containment.
    """

    def __init__(self,
                 lower_bound: float = -float("inf"),
                 upper_bound: float = float("inf")) -> None:
        """
        Constructor for Interval class.
        Which also has support for constructor where user passes in lower_bound and upper_bound.
        """

        # *****************
        # Private Members
        # *****************
        self.__t0: float = lower_bound
        self.__t1: float = upper_bound
        self.m_bounded_below: bool = False
        self.m_bounded_above: bool = False
        self.m_open_below: bool = True
        self.m_open_above: bool = True

        # Case where values are passed into the constructor, calls special functions
        if lower_bound != -float("inf"):
            self.set_lower_bound(lower_bound)
        if upper_bound != float("inf"):
            self.set_upper_bound(upper_bound)

        # TODO: do I really need reset_bounds after everything above?
        # self.reset_bounds()

    def set_lower_bound(self, lower_bound: float,  is_open: bool = False) -> None:
        """
        Sets t0 to lower_bound.
        Sets bounded_below to True.
        Sets open_below to is_open.
        """
        self.__t0 = lower_bound
        self.m_bounded_below = True
        self.m_open_below = is_open

    def set_upper_bound(self, upper_bound: float, is_open: bool = False) -> None:
        """
        Sets t1 to upper_bound.
        Sets bounded_above to True.
        Sets open_above to is_open.
        """
        self.__t1 = upper_bound
        self.m_bounded_above = True
        self.m_open_above = is_open

    def trim_lower_bound(self, trim_amount: float) -> None:
        """Adds self.t0 by trim_amount"""
        # Don't trim if the interval is not bounded or has length of order trim amount
        if ((not self.m_bounded_below) or (2.0 * abs(trim_amount) > self.get_length())):
            return
        self.__t0 += trim_amount

    def trim_upper_bound(self, trim_amount: float) -> None:
        """Subtracts self.t1 by trim_amount"""
        # Don't trim if the interval is not bounded or has length of order trim amount
        if ((not self.m_bounded_above) or (2.0 * abs(trim_amount) > self.get_length())):
            return
        self.__t1 -= trim_amount

    def pad_lower_bound(self, pad_amount: float) -> None:
        """Subtracts self.t0 by pad_amount"""
        #  Don't trim if the interval is not bounded or has length of order trim amount
        if ((not self.m_bounded_below) or (pad_amount < 0)):
            return
        self.__t0 -= pad_amount

    def pad_upper_bound(self, pad_amount: float) -> None:
        """Adds self.t1 by pad_amount"""
        # Don't trim if the interval is not bounded or has length of order trim amount
        if ((not self.m_bounded_above) or (pad_amount < 0)):
            return
        self.__t1 += pad_amount

    # TODO: float("inf") may be wrong...
    def reset_bounds(self) -> None:
        self.__t0 = -float("inf")
        self.__t1 = float("inf")
        self.m_bounded_below = False
        self.m_bounded_above = False
        self.m_open_below = True
        self.m_open_above = True

    def is_bounded_above(self) -> bool:
        return self.m_bounded_above

    def is_bounded_below(self) -> bool:
        return self.m_bounded_below

    def is_open_above(self) -> bool:
        return self.m_open_above

    def is_open_below(self) -> bool:
        return self.m_open_below

    def is_finite(self) -> bool:
        return (self.is_bounded_above() and self.is_bounded_below())

    def is_compact(self) -> bool:
        return (self.is_finite() and (not self.is_open_above()) and (not self.is_open_below()))

    @property
    def lower_bound(self) -> float:
        """Gets lower bound of interval."""
        return self.__t0

    @property
    def upper_bound(self) -> float:
        """Gets upper bound of interval."""
        return self.__t1

    def get_center(self) -> float:
        """Gets center of interval."""
        return 0.5 * (self.__t0 + self.__t1)

    def get_length(self) -> float:
        """Gets length of interval."""
        if self.is_finite():
            return self.upper_bound - self.lower_bound
        else:
            return float("inf")

    def contains(self, t: float) -> bool:
        """Checks if Interval contains t."""

        if ((not self.m_bounded_below) and (not self.m_bounded_above)):
            return True

        if ((not self.m_bounded_below) and (t <= self.__t1)):
            return True

        if ((not self.m_bounded_above) and (t >= self.__t0)):
            return True

        if ((t < self.__t0) and (not self.m_open_below)):
            return False

        if ((t <= self.__t0) and (self.m_open_below)):
            return False

        if ((t > self.__t1) and (not self.m_open_above)):
            return False

        if ((t >= self.__t1) and (self.m_open_above)):
            return False

        return True

    def is_in_interior(self, t: float) -> bool:
        """Checks if t is in interior of Interval."""
        if t <= self.__t0:
            return False
        if t >= self.__t1:
            return False
        return True

    def sample_points(self, num_points: int) -> list[float]:
        """Samples points from Interval."""
        points: list[float] = []

        if num_points <= 1:
            return points

        # Unbounded
        if not self.is_bounded_above() and not self.is_bounded_below():
            for i in range(num_points):
                points.append(-num_points / 20.0 + 0.1 * i)
            return points

        # Unbounded below, closed above
        elif not self.is_bounded_below() and not self.is_open_above():
            upper: float = self.upper_bound
            for i in range(num_points):
                points.append(upper - 0.1 * i)
            return points

        # Unbounded below, open above
        elif not self.is_bounded_below() and self.is_open_above():
            upper: float = self.upper_bound
            for i in range(1, num_points + 1):
                points.append(upper - 0.1 * i)
            return points

        # Unbounded above, closed below
        elif not self.is_bounded_above() and not self.is_open_below():
            lower: float = self.lower_bound
            for i in range(num_points):
                points.append(lower + 0.1 * i)
            return points

        # Unbounded above, open below
        elif not self.is_bounded_above() and self.is_open_below():
            lower: float = self.lower_bound
            for i in range(1, num_points + 1):
                points.append(lower + 0.1 * i)
            return points

        # Bounded cases
        t0: float = self.lower_bound
        t1: float = self.upper_bound

        # Closed
        if not self.is_open_below() and not self.is_open_above():
            for i in range(num_points):
                s: float = i / (num_points - 1)
                t: float = (1.0 - s) * t0 + s * t1
                points.append(t)

        # Open below, closed above
        elif self.is_open_below() and not self.is_open_above():
            for i in range(1, num_points + 1):
                s: float = i / num_points
                t: float = (1.0 - s) * t0 + s * t1
                assert t != t0
                points.append(t)

        # Closed below, open above
        elif not self.is_open_below() and self.is_open_above():
            for i in range(num_points):
                s: float = i / num_points
                t: float = (1.0 - s) * t0 + s * t1
                assert t != t1
                points.append(t)

        # Open
        elif self.is_open_below() and self.is_open_above():
            for i in range(1, num_points + 1):
                s: float = i / (num_points + 1)
                t: float = (1.0 - s) * t0 + s * t1
                points.append(t)

        return points

    def formatted_interval(self) -> str:
        """Formatted Interval for printing."""

        interval_string: str = ""
        # Unbounded cases
        if (not self.is_bounded_above()) and (not self.is_bounded_below()):
            interval_string += "(-infty, infty)"
        elif (not self.is_bounded_below()) and (not self.is_open_above()):
            interval_string += f"(-infty, {self.__t1:.17g}]"
        elif (not self.is_bounded_below()) and self.is_open_above():
            return f"(-infty, {self.__t1:.17g})"
        elif (not self.is_bounded_above()) and (not self.is_open_below()):
            return f"[{self.__t0:.17g}, infty)"
        elif (not self.is_bounded_above()) and self.is_open_below():
            return f"({self.__t0:.17g}, infty)"

        # Bounded cases
        t0: float = self.__t0
        t1: float = self.__t1

        if (not self.is_open_below()) and (not self.is_open_above()):
            interval_string += f"[{t0:.17g}, {t1:.17g}]"
        elif self.is_open_below() and (not self.is_open_above()):
            interval_string += f"({t0:.17g}, {t1:.17g}]"
        elif (not self.is_open_below()) and self.is_open_above():
            interval_string += f"[{t0:.17g}, {t1:.17g})"
        elif self.is_open_below() and self.is_open_above():
            interval_string += f"({t0:.17g}, {t1:.17g})"

        return interval_string

    def __repr__(self) -> str:
        return self.formatted_interval()
