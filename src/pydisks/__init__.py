"""pydisks — volumes and surface areas of solids of revolution.

A small, explicit library for the disc, washer, and shell methods plus
surface area of revolution. The axis of rotation is always given
explicitly as a line, e.g. "x" for the x-axis or "x=2" for the vertical
line x = 2. The correct variable of integration is derived from the axis
and the method, and mismatches are reported as errors rather than
silently "fixed" (the library never tries to invert your function for
you).

Curves can be SymPy expressions (exact answers via symbolic integration)
or ``Profile`` objects built from sample points (measured shapes; numeric
answers as floats). Pass ``numeric=True`` to get a float for a formula
SymPy cannot integrate in closed form.

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

import numpy as np
import sympy as sp

from .errors import (
    AxisError,
    BoundsError,
    CurveCrossingError,
    DomainError,
    ProfileError,
    RevolutionError,
    SelfIntersectingSolidError,
    UnevaluatedIntegralError,
    VariableMismatchError,
)
from .profile import (
    Profile,
    as_derivative,
    as_function,
    breakpoints,
    check_grid,
    quadrature,
)

__version__ = "0.1.0"

x = sp.Symbol("x")
y = sp.Symbol("y")

__all__ = [
    "x",
    "y",
    "Profile",
    "disc",
    "washer",
    "shell",
    "surface_area",
    "revolve",
    "plot_disc",
    "plot_washer",
    "animate_disc",
    "animate_washer",
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
    """Ensure ``expr`` uses only ``var`` (constants are fine).

    A Profile must have been sampled along ``var``; it is returned as is.
    """
    if isinstance(expr, Profile):
        if expr.var != var:
            raise VariableMismatchError(
                f"{name} is sampled along {expr.var}, but this rotation "
                f"integrates with respect to {var}. Provide samples taken "
                f"along {var} instead; pydisks will not invert or resample "
                f"the data for you."
            )
        return expr
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


def _numeric_bounds(a, b, offset, why):
    """Convert bounds and axis offset to floats or raise BoundsError."""
    try:
        return float(a), float(b), float(offset)
    except (TypeError, ValueError):
        raise BoundsError(
            f"{why} needs numeric bounds and axis offset; got a={a}, b={b}, "
            f"offset={offset}. Substitute values for any symbols first."
        )


def _integrate(integrand, var, a, b, numeric=False):
    """Integrate, raising if SymPy returns an unevaluated result.

    With ``numeric=True`` the integral is evaluated by adaptive quadrature
    (mpmath, via SymPy's evalf) and returned as a float.
    """
    if numeric:
        _numeric_bounds(a, b, 0, "numeric=True")
        value = sp.Integral(integrand, (var, a, b)).evalf()
        try:
            re, im = (float(part) for part in value.as_real_imag())
        except (TypeError, ValueError):
            raise UnevaluatedIntegralError(
                f"could not evaluate the integral of {integrand} over "
                f"[{a}, {b}] numerically."
            )
        if not np.isfinite(re) or abs(im) > 1e-9 * max(1.0, abs(re)):
            raise DomainError(
                f"the integral of {integrand} over [{a}, {b}] is not a finite "
                f"real number ({value}); check the curve is defined on the "
                f"whole interval."
            )
        return re
    result = sp.integrate(integrand, (var, a, b))
    if result.has(sp.Integral):
        raise UnevaluatedIntegralError(
            f"SymPy could not find a closed form for the integral of "
            f"{sp.integrate(integrand, var)} over [{a}, {b}]. Pass "
            f"numeric=True for a floating-point answer."
        )
    return sp.simplify(result)


# --------------------------------------------------------------------------
# Numeric path (any curve is a Profile)
# --------------------------------------------------------------------------

def _has_profile(*curves):
    return any(isinstance(c, Profile) for c in curves)


def _numeric_setup(named_curves, a, b, var, offset):
    """Validate a numeric problem and return what the integrators need.

    Returns ``(lo, hi, c, fns, breaks, grid)``: float bounds and axis offset,
    one vectorized function per curve, quadrature breakpoints (Profile knots
    included), and a dense grid for the geometry checks.
    """
    a, b = _check_bounds(a, b, var)
    lo, hi, c = _numeric_bounds(a, b, offset, "integrating a Profile")
    curves = [curve for _, curve in named_curves]
    for name, curve in named_curves:
        if isinstance(curve, Profile):
            start, end = curve.span
            slack = 1e-12 * max(1.0, abs(start), abs(end))
            if lo < start - slack or hi > end + slack:
                raise BoundsError(
                    f"[{lo:g}, {hi:g}] extends past the sampled range of "
                    f"{name}, [{start:g}, {end:g}]. pydisks does not "
                    f"extrapolate measured data: narrow the bounds or add "
                    f"samples."
                )
        else:
            _check_domain(curve, var, a, b)
    # Clamp to the data so a bound equal to an endpoint (up to float noise)
    # never evaluates a Profile just outside its span.
    for curve in curves:
        if isinstance(curve, Profile):
            lo, hi = max(lo, curve.span[0]), min(hi, curve.span[1])
    fns = [as_function(curve, var) for curve in curves]
    breaks = breakpoints(curves, lo, hi)
    grid = check_grid(breaks)
    for (name, _), fn in zip(named_curves, fns):
        bad = ~np.isfinite(fn(grid))
        if np.any(bad):
            raise DomainError(
                f"{name} is undefined or infinite near {var}="
                f"{grid[np.argmax(bad)]:.6g} inside [{lo:g}, {hi:g}]."
            )
    return lo, hi, c, fns, breaks, grid


def _tolerance(*arrays):
    scale = max(1.0, *(float(np.max(np.abs(arr))) for arr in arrays))
    return 1e-9 * scale


def _first_sign_change(grid, values, tol):
    """Approximate location where ``values`` changes sign, or None."""
    signs = np.where(values > tol, 1, np.where(values < -tol, -1, 0))
    nonzero = np.flatnonzero(signs)
    if nonzero.size == 0:
        return None
    flips = np.flatnonzero(signs[nonzero] != signs[nonzero[0]])
    if flips.size == 0:
        return None
    return float(grid[nonzero[flips[0]]])


def _dominant_sign(values, tol):
    nonzero = values[np.abs(values) > tol]
    return int(np.sign(nonzero[0])) if nonzero.size else 0


# --------------------------------------------------------------------------
# Public methods
# --------------------------------------------------------------------------

def disc(f, a, b, axis="x", *, numeric=False):
    """Volume of the solid formed by revolving the region between ``f`` and
    the axis, using the disc method.

    ``axis`` defaults to "x", i.e. the x-axis. ``f`` may be a SymPy
    expression or a Profile. The result is exact for an expression, or a
    float for a Profile or when ``numeric=True``.
    """
    coord, offset = _parse_axis(axis)
    var = _var_for("disc", coord)
    f = _check_variable(f, var, "f")

    if _has_profile(f):
        lo, hi, c, (F,), breaks, _ = _numeric_setup([("f", f)], a, b, var, offset)
        return quadrature(lambda t: np.pi * (F(t) - c) ** 2, breaks)

    a, b = _check_bounds(a, b, var)
    _check_domain(f, var, a, b)

    radius = f - offset
    # The curve may cross the axis: disc slices are stacked along the axis and
    # never overlap, so (f-offset)**2 still gives the correct volume (e.g.
    # y=x on [-1, 1] is a valid double cone, not a self-intersecting solid).
    return _integrate(sp.pi * radius ** 2, var, a, b, numeric)


def washer(outer, inner, a, b, axis="x", *, numeric=False):
    """Volume by the washer method: region between two curves, revolved.

    ``outer`` must be the curve farther from the axis on all of [a, b] and
    ``inner`` the nearer one. If they cross, a CurveCrossingError is raised.
    Either curve may be a Profile (the result is then a float).
    """
    coord, offset = _parse_axis(axis)
    var = _var_for("washer", coord)
    outer = _check_variable(outer, var, "outer")
    inner = _check_variable(inner, var, "inner")

    if _has_profile(outer, inner):
        lo, hi, c, (F_out, F_in), breaks, grid = _numeric_setup(
            [("outer", outer), ("inner", inner)], a, b, var, offset
        )
        R, r = F_out(grid) - c, F_in(grid) - c
        tol = _tolerance(R, r)
        for values, name in ((R, "outer"), (r, "inner")):
            where = _first_sign_change(grid, values, tol)
            if where is not None:
                raise SelfIntersectingSolidError(
                    f"the {name} curve crosses the axis {coord}={offset} near "
                    f"{var}={where:.6g}; the solid overlaps itself. Split the "
                    f"interval there."
                )
        if _dominant_sign(R, tol) * _dominant_sign(r, tol) < 0:
            raise SelfIntersectingSolidError(
                f"the outer and inner curves lie on opposite sides of the axis "
                f"{coord}={offset}, so the region contains the axis. Split the "
                f"region at the axis, or choose an axis outside the region."
            )
        gap = R ** 2 - r ** 2
        gap_tol = _tolerance(R ** 2, r ** 2)
        bad = gap < -gap_tol
        if np.any(bad):
            where = float(grid[np.argmax(bad)])
            if np.all(gap <= gap_tol):
                hint = "inner is farther from the axis everywhere: swap them."
            else:
                hint = ("they swap order; split the interval where they are "
                        "equidistant from the axis.")
            raise CurveCrossingError(
                f"inner is farther from the axis than outer near {var}="
                f"{where:.6g}; {hint}"
            )
        return quadrature(
            lambda t: np.pi * ((F_out(t) - c) ** 2 - (F_in(t) - c) ** 2), breaks
        )

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
    return _integrate(sp.pi * (R ** 2 - r ** 2), var, a, b, numeric)


def _check_shell_axis(coord, offset, var, a, b):
    try:
        inside = float(a) < float(offset) < float(b)
    except (TypeError, ValueError):
        return  # symbolic bounds/offset: skip the geometric check
    if inside:
        raise SelfIntersectingSolidError(
            f"the axis {coord}={offset} passes through the interior of "
            f"[{a}, {b}]; shells on opposite sides overlap. Split the "
            f"interval at {var}={offset}."
        )


def shell(f, a, b, axis="y", bottom=0, *, numeric=False):
    """Volume by the cylindrical shell method.

    ``axis`` defaults to "y", i.e. the y-axis. For a region between two
    curves, pass the lower curve as ``bottom``; the shell height is
    ``f - bottom``. Either curve may be a Profile (the result is then a
    float).
    """
    coord, offset = _parse_axis(axis)
    var = _var_for("shell", coord)
    f = _check_variable(f, var, "f")
    bottom = _check_variable(bottom, var, "bottom")

    if _has_profile(f, bottom):
        lo, hi, c, (F, B), breaks, grid = _numeric_setup(
            [("f", f), ("bottom", bottom)], a, b, var, offset
        )
        _check_shell_axis(coord, offset, var, lo, hi)
        height = F(grid) - B(grid)
        tol = _tolerance(height)
        if np.any(height < -tol):
            where = _first_sign_change(grid, height, tol)
            if where is None:
                raise CurveCrossingError(
                    f"f lies below bottom on ({lo:g}, {hi:g}); f must be the "
                    f"upper curve. Swap f and bottom."
                )
            raise CurveCrossingError(
                f"f and bottom cross near {var}={where:.6g}; the shell height "
                f"changes sign there. Split the interval at that point."
            )
        return quadrature(lambda t: 2 * np.pi * np.abs(t - c) * (F(t) - B(t)), breaks)

    a, b = _check_bounds(a, b, var)
    _check_domain(f, var, a, b)
    _check_domain(bottom, var, a, b)
    _check_shell_axis(coord, offset, var, a, b)

    height = f - bottom
    crossings = _real_roots_in(height, var, a, b)
    if crossings:
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
    return _integrate(2 * sp.pi * radius * height, var, a, b, numeric)


def surface_area(f, a, b, axis="x", *, numeric=False):
    """Surface area of the surface of revolution of ``f`` about ``axis``.

    ``f`` may be a Profile (the result is then a float). For a linear
    Profile the surface is a chain of cone frustums.
    """
    coord, offset = _parse_axis(axis)
    var = _var_for("surface_area", coord)
    f = _check_variable(f, var, "f")

    if _has_profile(f):
        lo, hi, c, (F,), breaks, _ = _numeric_setup([("f", f)], a, b, var, offset)
        dF = as_derivative(f, var)
        return quadrature(
            lambda t: 2 * np.pi * np.abs(F(t) - c) * np.sqrt(1 + dF(t) ** 2), breaks
        )

    a, b = _check_bounds(a, b, var)
    _check_domain(f, var, a, b)

    distance = sp.Abs(f - offset)
    f_prime = sp.diff(f, var)
    return _integrate(
        2 * sp.pi * distance * sp.sqrt(1 + f_prime ** 2), var, a, b, numeric
    )


def revolve(f, a, b, axis="x", inner=None, *, numeric=False):
    """Convenience dispatcher: disc if ``inner`` is None, else washer.

    Returns a dict with the chosen ``method``, the ``volume`` (exact, or a
    float for Profiles / ``numeric=True``), and a float ``approx`` for a
    quick sanity check.
    """
    if inner is None:
        method, volume = "disc", disc(f, a, b, axis=axis, numeric=numeric)
    else:
        method, volume = "washer", washer(f, inner, a, b, axis=axis, numeric=numeric)
    approx = volume if isinstance(volume, float) else sp.N(volume)
    return {"method": method, "volume": volume, "approx": approx}


# Imported last: these modules reuse the helpers and methods defined above.
from .plot import plot_disc, plot_washer  # noqa: E402
from .animate import animate_disc, animate_washer  # noqa: E402
