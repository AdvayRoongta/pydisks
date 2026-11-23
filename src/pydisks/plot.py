"""Visualization for the disc and washer methods.

Each function returns a matplotlib Figure with two panels:

* left  -- the 2D region, the axis of rotation, and a representative slice
           with its radius (disc: R; washer: R and r) marked;
* right -- the 3D solid of revolution with the same slice highlighted.

Both panels share one orientation: world-x is always the horizontal plot
axis and world-y is always "up" (the 3D z-axis), so the 3D solid passes
through the 2D curve at angle 0.

Inputs are validated by calling the real ``disc`` / ``washer`` first, so a
plot raises exactly the same errors as the volume computation would.
"""

import numpy as np
import sympy as sp

from . import (
    BoundsError,
    _check_bounds,
    _check_variable,
    _parse_axis,
    _var_for,
    disc,
    washer,
)
from .profile import Profile, as_function

__all__ = ["plot_disc", "plot_washer"]

_THETA_SAMPLES = 60

_REGION = "#4C72B0"
_INNER = "#DD8452"
_SLICE = "#C44E52"
_AXIS = "#333333"


# --------------------------------------------------------------------------
# Numeric helpers (pure, no rendering -- tested directly)
# --------------------------------------------------------------------------

def _numeric(expr, var):
    """Return a NumPy function of ``var`` for an expression or Profile;
    constants broadcast to the input."""
    return as_function(expr, var)


def _revolve_points(t, rho, theta, coord, offset):
    """Map (position along axis, signed radius, angle) to plot coordinates.

    Horizontal axis y=c: X = t,             Y = rho*sin, Z = c + rho*cos
    Vertical axis   x=c: X = c + rho*cos,   Y = rho*sin, Z = t
    """
    c = float(offset)
    along = np.broadcast_to(t, np.broadcast(t, rho, theta).shape)
    across = c + rho * np.cos(theta)
    depth = rho * np.sin(theta)
    if coord == "y":
        return along, depth, across
    return across, depth, along


def _surface_mesh(fn, a, b, coord, offset, n=200, m=_THETA_SAMPLES,
                  sweep=2 * np.pi):
    """(X, Y, Z) mesh of the surface swept by ``fn`` over [a, b] about the
    axis, through angles 0..``sweep`` (a full turn by default)."""
    t = np.linspace(float(a), float(b), n)
    rho = fn(t) - float(offset)
    theta = np.linspace(0.0, sweep, m)
    T, THETA = np.meshgrid(t, theta, indexing="ij")
    RHO = np.broadcast_to(rho[:, None], T.shape)
    return _revolve_points(T, RHO, THETA, coord, offset)


def _annulus_mesh(t0, rho_in, rho_out, coord, offset, m=_THETA_SAMPLES,
                  sweep=2 * np.pi):
    """(X, Y, Z) mesh of a flat annulus (a disc when rho_in == 0) at t0,
    through angles 0..``sweep``."""
    s = np.linspace(0.0, 1.0, 8)
    theta = np.linspace(0.0, sweep, m)
    S, THETA = np.meshgrid(s, theta, indexing="ij")
    RHO = rho_in + S * (rho_out - rho_in)
    return _revolve_points(np.full_like(S, float(t0)), RHO, THETA, coord, offset)


def _sheet_mesh(t, rho_in, rho_out, angle, coord, offset, k=6):
    """(X, Y, Z) mesh of the flat 2D region (between radii ``rho_in`` and
    ``rho_out`` at each position ``t``) rotated to ``angle``."""
    s = np.linspace(0.0, 1.0, k)
    T, S = np.meshgrid(t, s, indexing="ij")
    RHO = rho_in[:, None] + S * (rho_out - rho_in)[:, None]
    return _revolve_points(T, RHO, np.full_like(T, angle), coord, offset)


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------

def _axis_label(coord, offset):
    if offset == 0:
        return "the x-axis" if coord == "y" else "the y-axis"
    return f"{coord} = {offset}"


def _math(expr):
    """``$latex$`` if matplotlib's mathtext can render it, else plain text."""
    from matplotlib.mathtext import MathTextParser

    if isinstance(expr, Profile):
        return f"{expr.kind} profile ({len(expr)} points)"
    text = f"${sp.latex(expr)}$"
    try:
        MathTextParser("path").parse(text)
        return text
    except Exception:
        return sp.sstr(expr)


def _title(method, curves, coord, offset, volume):
    curve_text = ", ".join(f"{name} = {_math(expr)}" for name, expr in curves)
    if isinstance(volume, float):  # numeric (Profile) result: no exact form
        volume_text = f"$V$ ≈ {volume:.6g}"
    else:
        volume_text = f"$V$ = {_math(volume)} ≈ {float(sp.N(volume)):.6g}"
    return (
        f"{method} method: {curve_text}, revolved about "
        f"{_axis_label(coord, offset)}\n{volume_text}"
    )


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

def _prepare(curves, a, b, axis, method, slice_at):
    """Shared setup: axis, variable, numeric bounds, numeric curve functions."""
    coord, offset = _parse_axis(axis)
    var = _var_for(method, coord)
    exprs = [_check_variable(expr, var, name) for name, expr in curves]
    a, b = _check_bounds(a, b, var)
    try:
        lo, hi, c = float(a), float(b), float(offset)
    except (TypeError, ValueError):
        raise BoundsError(
            f"plotting requires numeric bounds and axis offset; got a={a}, "
            f"b={b}, offset={offset}. Substitute values for any symbols first."
        )
    if slice_at is None:
        s = (lo + hi) / 2
    else:
        try:
            s = float(sp.sympify(slice_at))
        except (TypeError, ValueError, sp.SympifyError):
            raise BoundsError(f"slice_at must be a number; got {slice_at!r}")
        if not lo <= s <= hi:
            raise BoundsError(
                f"slice_at={slice_at} lies outside the interval [{a}, {b}]."
            )
    fns = [_numeric(expr, var) for expr in exprs]
    return coord, offset, c, var, lo, hi, s, exprs, fns


def _draw_2d(ax, coord, c, var, t, curves, fns, s, labels):
    """Region, its mirror image, the axis, and the slice radii."""
    horizontal = coord == "y"
    values = [fn(t) for fn in fns]
    lower = values[1] if len(values) > 1 else np.full_like(t, c)
    upper = values[0]
    fill = ax.fill_between if horizontal else ax.fill_betweenx

    fill(t, upper, lower, color=_REGION, alpha=0.35, linewidth=0, label="region")
    fill(t, 2 * c - upper, 2 * c - lower, color=_REGION, alpha=0.10, linewidth=0,
         label="mirror image")

    colors = [_REGION, _INNER]
    for (name, expr), v, color in zip(curves, values, colors):
        xy = (t, v) if horizontal else (v, t)
        ax.plot(*xy, color=color, linewidth=2, label=f"{name} = {_math(expr)}")
        if isinstance(expr, Profile):  # show the measured samples themselves
            keep = (expr.positions >= t[0]) & (expr.positions <= t[-1])
            pts = (expr.positions[keep], expr.values[keep])
            ax.plot(*(pts if horizontal else pts[::-1]), "o", color=color,
                    markersize=3.5, alpha=0.8)

    if horizontal:
        ax.axhline(c, color=_AXIS, linestyle="--", linewidth=1.2, label="axis")
    else:
        ax.axvline(c, color=_AXIS, linestyle="--", linewidth=1.2, label="axis")

    # Slice radii: from the axis out to each curve at var = s. The first
    # (outer) radius is drawn thick and translucent so the inner one shows.
    styles = [dict(linewidth=6, alpha=0.45), dict(linewidth=2.5, alpha=1.0)]
    # Outer label goes right/above the line, inner label left/below.
    nudges = [(8, 0), (-8, 0)] if horizontal else [(0, 8), (0, -8)]
    aligns = [("left", "center"), ("right", "center")] if horizontal else \
        [("center", "bottom"), ("center", "top")]
    for fn, label, style, nudge, (ha, va) in zip(fns, labels, styles, nudges, aligns):
        end = float(fn(np.array([s]))[0])
        seg = ([s, s], [c, end]) if horizontal else ([c, end], [s, s])
        ax.plot(*seg, color=_SLICE, solid_capstyle="butt", **style)
        mid = (c + end) / 2
        pos = (s, mid) if horizontal else (mid, s)
        ax.annotate(f"${label}$", pos, xytext=nudge, textcoords="offset points",
                    ha=ha, va=va, color=_SLICE, fontsize=13)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"region and slice at {var} = {s:.4g}", fontsize=10)


def _draw_axis_3d(ax, coord, c, lo, hi):
    pad = 0.1 * (hi - lo)
    ends = np.array([lo - pad, hi + pad])
    if coord == "y":
        ax.plot(ends, [0, 0], [c, c], color=_AXIS, linestyle="--", linewidth=1.2)
    else:
        ax.plot([c, c], [0, 0], ends, color=_AXIS, linestyle="--", linewidth=1.2)


def _finish_3d(ax, coord, var, s):
    if coord == "x":
        # Vertical axis: a low camera shows the slice at its height instead
        # of reading as a lid on top of a squat solid.
        ax.view_init(elev=12, azim=-60)
    else:
        # Horizontal axis: more side-on than the default so short, wide
        # solids show their profile instead of just their end cap.
        ax.view_init(elev=22, azim=-70)
    ax.locator_params(axis="x", nbins=5)
    ax.locator_params(axis="z", nbins=5)
    ax.set_xlabel("x")
    ax.set_ylabel("")
    ax.set_yticks([])  # depth direction: purely the sweep, no coordinate
    ax.set_zlabel("y")
    ax.set_aspect("equal")
    ax.set_title(f"solid of revolution (slice at {var} = {s:.4g})", fontsize=10)


def _surface(ax, mesh, **kw):
    """plot_surface without numpy warnings from degenerate cells.

    Meshes collapse to a point on the axis (sphere poles, disc centres);
    matplotlib's shading then normalizes zero-length normals, which numpy 2
    reports as divide/overflow warnings. The rendering is unaffected.
    """
    kw.setdefault("shade", True)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return ax.plot_surface(*mesh, linewidth=0, antialiased=True, **kw)


def _slice_surface(ax, mesh):
    _surface(ax, mesh, color=_SLICE, alpha=0.95, shade=False, zorder=10)


def _figure():
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 5.5))
    ax2d = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")
    # matplotlib's per-collection depth sorting would bury the highlighted
    # slice inside the solid; draw in call order instead, slice last.
    ax3d.computed_zorder = False
    return fig, ax2d, ax3d


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def plot_disc(f, a, b, axis="x", *, slice_at=None, n=200):
    """Figure of the disc-method solid for ``disc(f, a, b, axis)``.

    Left: the region between ``f`` and the axis with a representative slice
    at ``slice_at`` (default: midpoint) and its radius R. Right: the 3D solid
    with that disc highlighted. Raises the same errors as ``disc``, plus
    ``BoundsError`` for symbolic bounds or a ``slice_at`` outside [a, b].

    Returns the matplotlib Figure; call ``plt.show()`` or ``fig.savefig()``.
    """
    volume = disc(f, a, b, axis=axis)
    coord, offset, c, var, lo, hi, s, exprs, fns = _prepare(
        [("f", f)], a, b, axis, "disc", slice_at
    )
    (fn,) = fns
    t = np.linspace(lo, hi, n)

    fig, ax2d, ax3d = _figure()
    _draw_2d(ax2d, coord, c, var, t, [("f", exprs[0])], fns, s, ["R"])

    _surface(ax3d, _surface_mesh(fn, lo, hi, coord, offset, n), color=_REGION,
             alpha=0.35)
    for end in (lo, hi):
        rho = float(fn(np.array([end]))[0]) - c
        if np.isfinite(rho) and rho != 0:
            _surface(ax3d, _annulus_mesh(end, 0.0, rho, coord, offset),
                     color=_REGION, alpha=0.35)
    rho_s = float(fn(np.array([s]))[0]) - c
    _slice_surface(ax3d, _annulus_mesh(s, 0.0, rho_s, coord, offset))
    _draw_axis_3d(ax3d, coord, c, lo, hi)
    _finish_3d(ax3d, coord, var, s)

    fig.suptitle(_title("Disc", [("f", exprs[0])], coord, offset, volume))
    fig.tight_layout()
    return fig


def plot_washer(outer, inner, a, b, axis="x", *, slice_at=None, n=200):
    """Figure of the washer-method solid for ``washer(outer, inner, a, b, axis)``.

    Left: the region between the curves with a representative slice at
    ``slice_at`` (default: midpoint) and its radii R (outer) and r (inner).
    Right: the 3D solid -- translucent outer surface, opaque inner surface so
    the hole shows -- with that washer highlighted. Raises the same errors as
    ``washer``, plus ``BoundsError`` for symbolic bounds or a ``slice_at``
    outside [a, b].

    Returns the matplotlib Figure; call ``plt.show()`` or ``fig.savefig()``.
    """
    volume = washer(outer, inner, a, b, axis=axis)
    curves = [("outer", outer), ("inner", inner)]
    coord, offset, c, var, lo, hi, s, exprs, fns = _prepare(
        curves, a, b, axis, "washer", slice_at
    )
    named = list(zip(("outer", "inner"), exprs))
    f_out, f_in = fns
    t = np.linspace(lo, hi, n)

    fig, ax2d, ax3d = _figure()
    _draw_2d(ax2d, coord, c, var, t, named, fns, s, ["R", "r"])

    _surface(ax3d, _surface_mesh(f_in, lo, hi, coord, offset, n), color=_INNER,
             alpha=0.6)
    _surface(ax3d, _surface_mesh(f_out, lo, hi, coord, offset, n), color=_REGION,
             alpha=0.3)

    def radii(at):
        p = np.array([at])
        return float(f_in(p)[0]) - c, float(f_out(p)[0]) - c

    for end in (lo, hi):
        r_in, r_out = radii(end)
        if np.isfinite(r_in) and np.isfinite(r_out) and r_in != r_out:
            _surface(ax3d, _annulus_mesh(end, r_in, r_out, coord, offset),
                     color=_REGION, alpha=0.4)
    r_in, r_out = radii(s)
    _slice_surface(ax3d, _annulus_mesh(s, r_in, r_out, coord, offset))
    _draw_axis_3d(ax3d, coord, c, lo, hi)
    _finish_3d(ax3d, coord, var, s)

    fig.suptitle(_title("Washer", named, coord, offset, volume))
    fig.tight_layout()
    return fig
