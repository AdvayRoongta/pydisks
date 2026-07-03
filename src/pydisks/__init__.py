"""pydisks — volumes and surface areas of solids of revolution via SymPy.

A small, explicit library for the disc, washer, and shell methods plus
surface area of revolution. The axis of rotation is always given
explicitly as a line, e.g. "x" for the x-axis or "x=2" for the vertical
line x = 2. The correct variable of integration is derived from the axis
and the method, and mismatches are reported as errors rather than
silently "fixed" (the library never tries to invert your function for
you).

Conventions
-----------
The axis of rotation is written as a string:

    "x"    -> the x-axis          (the line y = 0)
    "y"    -> the y-axis          (the line x = 0)
    "x=2"  -> the vertical line   x = 2
    "y=-1" -> the horizontal line y = -1

A bare letter names an axis; an equation is a literal line. Note the
asymmetry this creates: "x" is the x-axis (y=0), but "x=0" is the line
x=0 (the y-axis).

The variable you must express your curve(s) in depends on the method:

    disc / washer : horizontal axis -> f(x);  vertical axis -> f(y)
    shell         : horizontal axis -> f(y);  vertical axis -> f(x)
"""

import sympy as sp

__version__ = "0.1.0"

x = sp.Symbol("x")
y = sp.Symbol("y")

__all__ = [
    "x",
    "y",
    "disc",
    "washer",
    "shell",
    "surface_area",
    "revolve",
    "RevolutionError",
    "AxisError",
    "VariableMismatchError",
    "BoundsError",
    "SelfIntersectingSolidError",
    "CurveCrossingError",
    "DomainError",
    "UnevaluatedIntegralError",
]


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------

class RevolutionError(Exception):
    """Base class for all pydisks errors."""


class AxisError(RevolutionError):
    """The axis specification is malformed."""


class VariableMismatchError(RevolutionError):
    """A curve or bound is expressed in the wrong variable for this setup."""


class BoundsError(RevolutionError):
    """The integration bounds are invalid (non-constant or a >= b)."""


class SelfIntersectingSolidError(RevolutionError):
    """The axis of rotation passes through the interior of the region."""


class CurveCrossingError(RevolutionError):
    """The two boundary curves cross or are given in the wrong order (washer
    outer/inner swap, or shell f/bottom cross or invert)."""


class DomainError(RevolutionError):
    """A curve is undefined or discontinuous somewhere on the interval."""


class UnevaluatedIntegralError(RevolutionError):
    """SymPy could not find a closed form for the integral."""


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------

def _parse_axis(axis):
    """Normalize an axis spec to (coord, offset), where ``coord`` names the
    coordinate held constant along the line of rotation.

    Accepted forms:
      "x" / "y"       -> the x-axis (line y=0) / the y-axis (line x=0)
      "x=2" / "y=-1"  -> the literal line, written as its equation
      (coord, value)  -> equivalent pair form, e.g. ("x", 2) is x=2

    Note the deliberate asymmetry: the bare string "x" is *the x-axis*
    (y=0), whereas the equation "x=0" is *the line x=0* (the y-axis).
    A bare letter names an axis; an equation is a literal line.
    """
    if isinstance(axis, str):
        s = axis.replace(" ", "")
        if s == "x":               # the x-axis is the line y = 0
            return "y", sp.Integer(0)
        if s == "y":               # the y-axis is the line x = 0
            return "x", sp.Integer(0)
        if "=" in s:
            coord, _, value = s.partition("=")
            if coord not in ("x", "y"):
                raise AxisError(
                    f"axis equation must start with 'x=' or 'y='; got {axis!r}"
                )
            if value == "":
                raise AxisError(f"axis equation {axis!r} is missing a value.")
            try:
                offset = sp.sympify(value)
            except (sp.SympifyError, SyntaxError, TypeError):
                raise AxisError(f"could not read the axis value in {axis!r}")
            if offset.free_symbols:
                raise AxisError(f"axis value must be a constant; got {axis!r}")
            return coord, offset
        raise AxisError(
            f"axis string must be 'x', 'y', or an equation like 'x=2' or "
            f"'y=-1'; got {axis!r}"
        )
    if isinstance(axis, (tuple, list)) and len(axis) == 2:
        coord, value = axis
        if coord not in ("x", "y"):
            raise AxisError(
                f"axis coordinate must be 'x' or 'y' (the constant coordinate "
                f"of the line); got {coord!r}"
            )
        offset = sp.sympify(value)
        if offset.free_symbols:
            raise AxisError(f"axis offset must be a constant; got {value!r}")
        return coord, offset
    raise AxisError(
        f"axis must be 'x', 'y', an equation like 'x=2', or a (coord, value) "
        f"pair; got {axis!r}"
    )


def _var_for(method, axis_coord):
    """Derive the integration variable from the method and axis orientation."""
    if method in ("disc", "washer", "surface_area"):
        # slices are perpendicular to the axis: integrate along the axis line
        return x if axis_coord == "y" else y
    # shell: shells run parallel to the axis
    return y if axis_coord == "y" else x


def _check_variable(expr, var, name):
    """Ensure ``expr`` uses only ``var`` (constants are fine)."""
    expr = sp.sympify(expr)
    other = y if var is x else x
    if other in expr.free_symbols:
        raise VariableMismatchError(
            f"{name} is written in terms of {other}, but this rotation "
            f"integrates with respect to {var}. Express it as a function of "
            f"{var} (e.g. solve for {var} yourself); pydisks will not invert "
            f"it for you."
        )
    stray = expr.free_symbols - {var}
    if stray:
        raise VariableMismatchError(
            f"{name} contains unexpected symbol(s) {stray}; only {var} is allowed."
        )
    return expr


def _check_bounds(a, b, var):
    """Validate integration bounds and return them as SymPy objects."""
    a, b = sp.sympify(a), sp.sympify(b)
    for label, bound in (("a", a), ("b", b)):
        if var in bound.free_symbols:
            raise BoundsError(f"bound {label}={bound} must not depend on {var}.")
    diff = sp.simplify(b - a)
    if diff.is_number:
        if diff == 0:
            raise BoundsError(f"bounds are equal (a = b = {a}); the region is empty.")
        if diff < 0:
            raise BoundsError(
                f"upper bound b={b} is not greater than lower bound a={a}; "
                f"swap them."
            )
    return a, b


def _real_roots_in(expr, var, a, b):
    """Real roots of ``expr`` strictly inside the open interval (a, b)."""
    try:
        lo, hi = float(a), float(b)
    except (TypeError, ValueError):
        return []  # symbolic bounds: skip the geometric check
    try:
        solutions = sp.solve(sp.Eq(sp.sympify(expr), 0), var)
    except Exception:
        return []
    roots = []
    for s in solutions:
        if s.free_symbols or s.is_real is False:
            continue
        try:
            val = float(s)
        except (TypeError, ValueError):
            continue
        if lo < val < hi:
            roots.append(s)
    return roots


def _sign_at_mid(expr, var, a, b):
    """Sign (-1, 0, +1) of ``expr`` at the midpoint of [a, b].

    Returns 0 when the bounds or value are not numeric. Only meaningful for
    an ``expr`` already known to keep a constant sign on the interval.
    """
    try:
        mid = (float(a) + float(b)) / 2
        value = float(sp.sympify(expr).subs(var, mid))
    except (TypeError, ValueError):
        return 0
    return (value > 0) - (value < 0)


def _check_domain(expr, var, a, b):
    """Raise DomainError if ``expr`` is discontinuous on the closed interval."""
    try:
        lo, hi = float(a), float(b)
    except (TypeError, ValueError):
        return
    try:
        from sympy.calculus.util import continuous_domain
        domain = continuous_domain(sp.sympify(expr), var, sp.Interval(a, b))
    except Exception:
        return
    if not sp.Interval(a, b).is_subset(domain):
        raise DomainError(
            f"{expr} is not continuous on the whole interval [{lo}, {hi}] "
            f"(continuous domain: {domain}). Split the problem at the "
            f"discontinuity."
        )


def _integrate(integrand, var, a, b):
    """Integrate, raising if SymPy returns an unevaluated result."""
    result = sp.integrate(integrand, (var, a, b))
    if result.has(sp.Integral):
        raise UnevaluatedIntegralError(
            f"SymPy could not find a closed form for the volume of "
            f"{sp.integrate(integrand, var)} over [{a}, {b}]."
        )
    return sp.simplify(result)


# --------------------------------------------------------------------------
# Public methods
# --------------------------------------------------------------------------

def disc(f, a, b, axis="x"):
    """Volume of the solid formed by revolving the region between ``f`` and
    the axis, using the disc method.

    ``axis`` defaults to "x", i.e. the x-axis.
    """
    coord, offset = _parse_axis(axis)
    var = _var_for("disc", coord)
    f = _check_variable(f, var, "f")
    a, b = _check_bounds(a, b, var)
    _check_domain(f, var, a, b)

    radius = f - offset
    # The curve may cross the axis: disc slices are stacked along the axis and
    # never overlap, so (f-offset)**2 still gives the correct volume (e.g.
    # y=x on [-1, 1] is a valid double cone, not a self-intersecting solid).
    return _integrate(sp.pi * radius ** 2, var, a, b)


def washer(outer, inner, a, b, axis="x"):
    """Volume by the washer method: region between two curves, revolved.

    ``outer`` must be the curve farther from the axis on all of [a, b] and
    ``inner`` the nearer one. If they cross, a CurveCrossingError is raised.
    """
    coord, offset = _parse_axis(axis)
    var = _var_for("washer", coord)
    outer = _check_variable(outer, var, "outer")
    inner = _check_variable(inner, var, "inner")
    a, b = _check_bounds(a, b, var)
    _check_domain(outer, var, a, b)
    _check_domain(inner, var, a, b)

    R = outer - offset
    r = inner - offset
    # Each radius must keep a fixed sign (region stays on one side of axis)...
    for radius, name in ((R, "outer"), (r, "inner")):
        if _real_roots_in(radius, var, a, b):
            raise SelfIntersectingSolidError(
                f"the {name} curve crosses the axis {coord}={offset} inside "
                f"({a}, {b}); the solid overlaps itself. Split the interval."
            )
    # ...and both curves must lie on the same side of the axis. If they
    # straddle it, the region contains the axis and the washer formula is
    # invalid (e.g. outer=1, inner=-1 about y=0 gives 0, not the true pi).
    if _sign_at_mid(R, var, a, b) * _sign_at_mid(r, var, a, b) < 0:
        raise SelfIntersectingSolidError(
            f"the outer and inner curves lie on opposite sides of the axis "
            f"{coord}={offset}, so the region contains the axis and the solid "
            f"overlaps itself. Split the region at the axis and revolve each "
            f"piece, or choose an axis outside the region."
        )
    # ...and the outer radius must dominate the inner one everywhere. Each
    # radius has constant sign (checked above), so R**2 - r**2 flips exactly
    # where |outer| and |inner| swap -- and it avoids unreliable Abs solving.
    crossing = _real_roots_in(R ** 2 - r ** 2, var, a, b)
    if crossing:
        raise CurveCrossingError(
            f"outer and inner curves are equidistant from the axis at {var}="
            f"{crossing} inside ({a}, {b}), so they swap order. Split the "
            f"interval at those points (and check which curve is 'outer' on "
            f"each piece)."
        )
    return _integrate(sp.pi * (R ** 2 - r ** 2), var, a, b)


def shell(f, a, b, axis="y", bottom=0):
    """Volume by the cylindrical shell method.

    ``axis`` defaults to "y", i.e. the y-axis. For a region between two
    curves, pass the lower curve as ``bottom``; the shell height is
    ``f - bottom``.
    """
    coord, offset = _parse_axis(axis)
    var = _var_for("shell", coord)
    f = _check_variable(f, var, "f")
    bottom = _check_variable(bottom, var, "bottom")
    a, b = _check_bounds(a, b, var)
    _check_domain(f, var, a, b)
    _check_domain(bottom, var, a, b)

    try:
        if float(a) < float(offset) < float(b):
            raise SelfIntersectingSolidError(
                f"the axis {coord}={offset} passes through the interior of "
                f"[{a}, {b}]; shells on opposite sides overlap. Split the "
                f"interval at {var}={offset}."
            )
    except (TypeError, ValueError):
        pass  # symbolic bounds/offset: skip the geometric check

    height = f - bottom
    if _real_roots_in(height, var, a, b):
        crossings = _real_roots_in(height, var, a, b)
        raise CurveCrossingError(
            f"f and bottom cross at {var}={crossings} inside ({a}, {b}); the "
            f"shell height changes sign there. Split the interval at those "
            f"points (the upper curve differs on each piece)."
        )
    if _sign_at_mid(height, var, a, b) < 0:
        raise CurveCrossingError(
            f"f lies below bottom on ({a}, {b}); f must be the upper curve. "
            f"Swap f and bottom."
        )
    radius = sp.Abs(var - offset)
    return _integrate(2 * sp.pi * radius * height, var, a, b)


def surface_area(f, a, b, axis="x"):
    """Surface area of the surface of revolution of ``f`` about ``axis``."""
    coord, offset = _parse_axis(axis)
    var = _var_for("surface_area", coord)
    f = _check_variable(f, var, "f")
    a, b = _check_bounds(a, b, var)
    _check_domain(f, var, a, b)

    distance = sp.Abs(f - offset)
    f_prime = sp.diff(f, var)
    return _integrate(2 * sp.pi * distance * sp.sqrt(1 + f_prime ** 2), var, a, b)


def revolve(f, a, b, axis="x", inner=None):
    """Convenience dispatcher: disc if ``inner`` is None, else washer.

    Returns a dict with the chosen ``method``, the exact ``volume``, and a
    floating-point ``approx`` for a quick sanity check.
    """
    if inner is None:
        method, volume = "disc", disc(f, a, b, axis=axis)
    else:
        method, volume = "washer", washer(f, inner, a, b, axis=axis)
    return {"method": method, "volume": volume, "approx": sp.N(volume)}
