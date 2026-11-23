"""Exception hierarchy for pydisks. Every error subclasses RevolutionError."""

__all__ = [
    "RevolutionError",
    "AxisError",
    "VariableMismatchError",
    "BoundsError",
    "SelfIntersectingSolidError",
    "CurveCrossingError",
    "DomainError",
    "UnevaluatedIntegralError",
    "ProfileError",
]


class RevolutionError(Exception):
    """Base class for all pydisks errors."""


class AxisError(RevolutionError):
    """The axis specification is malformed."""


class VariableMismatchError(RevolutionError):
    """A curve or bound is expressed in the wrong variable for this setup."""


class BoundsError(RevolutionError):
    """The integration bounds are invalid (non-constant, a >= b, or outside
    the sampled range of a Profile)."""


class SelfIntersectingSolidError(RevolutionError):
    """The axis of rotation passes through the interior of the region."""


class CurveCrossingError(RevolutionError):
    """The two boundary curves cross or are given in the wrong order (washer
    outer/inner swap, or shell f/bottom cross or invert)."""


class DomainError(RevolutionError):
    """A curve is undefined or discontinuous somewhere on the interval."""


class UnevaluatedIntegralError(RevolutionError):
    """SymPy could not find a closed form for the integral."""


class ProfileError(RevolutionError):
    """Sample data for a Profile is malformed (unsorted, mismatched lengths,
    non-finite values, too few points, or an unknown interpolation kind)."""
