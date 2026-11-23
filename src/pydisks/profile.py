"""Curves given by sample points, plus the numeric machinery they need.

A ``Profile`` stands in for a SymPy expression anywhere a curve is accepted
(``disc``, ``washer``, ``shell``, ``surface_area`` and the plot functions).
Any method that receives a Profile integrates numerically and returns a
``float``.

The explicit-variable rule still applies: a Profile records which variable
its sample positions are measured along, and pydisks raises
``VariableMismatchError`` rather than resampling or inverting it. Profiles
are never extrapolated past their first or last sample.
"""

import numpy as np
import sympy as sp

from .errors import ProfileError

__all__ = ["Profile"]

_KINDS = ("linear", "cubic")

# Gauss-Legendre rule used on every sub-interval. 8 nodes integrate
# polynomials up to degree 15 exactly, which covers pi*f**2 for linear and
# cubic profiles (degree <= 6) and the shell integrand (degree <= 4) without
# any quadrature error -- only the interpolation itself is approximate.
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(8)


class Profile:
    """A curve defined by sample points instead of a formula.

    Parameters
    ----------
    var : x or y (the pydisks symbols) or the strings "x" / "y"
        The variable the sample *positions* are measured along. For
        ``disc(Profile(x, ...), a, b, axis="x")`` the samples are heights
        ``y`` at positions ``x``, exactly like a formula ``f(x)``.
    positions, values : sequences of numbers
        Same length, at least 2 points, all finite, positions strictly
        increasing.
    kind : "linear" (default) or "cubic"
        "linear" joins the points with straight segments (a turned part
        drawn as a polyline is exact). "cubic" fits a natural cubic spline
        (smooth, zero curvature at both ends) for measured smooth shapes.

    Evaluating outside ``span`` returns NaN; pydisks never extrapolates.
    """

    def __init__(self, var, positions, values, kind="linear"):
        name = str(var)
        if name not in ("x", "y"):
            raise ProfileError(f"var must be x or y; got {var!r}")
        if kind not in _KINDS:
            raise ProfileError(f"kind must be one of {_KINDS}; got {kind!r}")
        try:
            t = np.array(positions, dtype=float)
            v = np.array(values, dtype=float)
        except (TypeError, ValueError):
            raise ProfileError("positions and values must be sequences of numbers.")
        if t.ndim != 1 or v.ndim != 1:
            raise ProfileError("positions and values must be one-dimensional.")
        if t.shape != v.shape:
            raise ProfileError(
                f"positions and values differ in length ({t.size} vs {v.size})."
            )
        if t.size < 2:
            raise ProfileError(f"a Profile needs at least 2 points; got {t.size}.")
        if not (np.all(np.isfinite(t)) and np.all(np.isfinite(v))):
            raise ProfileError("positions and values must all be finite (no NaN/inf).")
        steps = np.diff(t)
        if np.any(steps <= 0):
            i = int(np.argmax(steps <= 0))
            raise ProfileError(
                f"positions must be strictly increasing; position[{i + 1}]="
                f"{t[i + 1]} does not exceed position[{i}]={t[i]}. Sort the "
                f"samples and merge duplicate positions."
            )

        t.flags.writeable = False
        v.flags.writeable = False
        self.var = sp.Symbol(name)
        self.kind = kind
        self._t = t
        self._v = v
        self._h = steps
        self._m = _natural_spline_moments(t, v) if kind == "cubic" else None

    # -- read-only views ---------------------------------------------------

    @property
    def positions(self):
        return self._t

    @property
    def values(self):
        return self._v

    @property
    def span(self):
        """(first position, last position): the range the data covers."""
        return float(self._t[0]), float(self._t[-1])

    def __len__(self):
        return self._t.size

    def __repr__(self):
        lo, hi = self.span
        return (
            f"Profile({self.var}, {len(self)} points, kind={self.kind!r}, "
            f"span=[{lo:g}, {hi:g}])"
        )

    # -- evaluation ----------------------------------------------------------

    def _locate(self, q):
        q = np.asarray(q, dtype=float)
        i = np.clip(np.searchsorted(self._t, q, side="right") - 1, 0, self._t.size - 2)
        outside = (q < self._t[0]) | (q > self._t[-1])
        return q, i, outside

    def __call__(self, q):
        """Interpolated value(s) at position(s) ``q`` (NaN outside ``span``)."""
        q, i, outside = self._locate(q)
        t0, h = self._t[i], self._h[i]
        v0, v1 = self._v[i], self._v[i + 1]
        if self._m is None:
            out = v0 + (v1 - v0) * (q - t0) / h
        else:
            m0, m1 = self._m[i], self._m[i + 1]
            u, w = q - t0, self._t[i + 1] - q
            out = (
                m0 * w ** 3 / (6 * h) + m1 * u ** 3 / (6 * h)
                + (v0 / h - m0 * h / 6) * w + (v1 / h - m1 * h / 6) * u
            )
        return np.where(outside, np.nan, out)

    def derivative(self, q):
        """d(value)/d(position) at ``q`` (NaN outside ``span``)."""
        q, i, outside = self._locate(q)
        h = self._h[i]
        v0, v1 = self._v[i], self._v[i + 1]
        if self._m is None:
            out = np.broadcast_to((v1 - v0) / h, q.shape).astype(float)
        else:
            m0, m1 = self._m[i], self._m[i + 1]
            u, w = q - self._t[i], self._t[i + 1] - q
            out = (
                -m0 * w ** 2 / (2 * h) + m1 * u ** 2 / (2 * h)
                + (v1 - v0) / h - (m1 - m0) * h / 6
            )
        return np.where(outside, np.nan, out)


def _natural_spline_moments(t, v):
    """Second derivatives at the knots of the natural cubic spline.

    Solves the standard tridiagonal system with M[0] = M[-1] = 0 by the
    Thomas algorithm (O(n), so large measured datasets stay cheap).
    """
    n = t.size
    m = np.zeros(n)
    if n < 3:
        return m
    h = np.diff(t)
    slope = np.diff(v) / h
    sub = h[:-1].copy()                  # coefficient of M[i-1]
    diag = 2 * (h[:-1] + h[1:])          # coefficient of M[i]
    sup = h[1:].copy()                   # coefficient of M[i+1]
    rhs = 6 * np.diff(slope)
    k = n - 2
    for j in range(1, k):
        w = sub[j] / diag[j - 1]
        diag[j] -= w * sup[j - 1]
        rhs[j] -= w * rhs[j - 1]
    inner = np.empty(k)
    inner[-1] = rhs[-1] / diag[-1]
    for j in range(k - 2, -1, -1):
        inner[j] = (rhs[j] - sup[j] * inner[j + 1]) / diag[j]
    m[1:-1] = inner
    return m


# --------------------------------------------------------------------------
# Numeric helpers shared by the volume methods and the plots
# --------------------------------------------------------------------------

def as_function(curve, var):
    """Vectorized NumPy function of ``var`` for a Profile or SymPy expression.

    Constants broadcast to the input's shape; invalid points become NaN.
    """
    if isinstance(curve, Profile):
        return curve
    fn = sp.lambdify(var, sp.sympify(curve), "numpy")

    def evaluate(t):
        t = np.asarray(t, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            values = np.asarray(fn(t))
        if np.iscomplexobj(values):
            values = np.where(np.abs(values.imag) > 1e-12, np.nan, values.real)
        return np.broadcast_to(np.asarray(values, dtype=float), t.shape).copy()

    return evaluate


def as_derivative(curve, var):
    """Vectorized derivative of a Profile or SymPy expression."""
    if isinstance(curve, Profile):
        return curve.derivative
    return as_function(sp.diff(sp.sympify(curve), var), var)


def breakpoints(curves, lo, hi):
    """Sorted [lo, ..., hi] including every Profile knot strictly inside."""
    points = {lo, hi}
    for curve in curves:
        if isinstance(curve, Profile):
            points.update(float(k) for k in curve.positions if lo < k < hi)
    return np.array(sorted(points))


def check_grid(breaks, samples=2001):
    """Dense evaluation grid for the numeric geometry checks: every knot,
    every knot midpoint, and a uniform grid over the whole interval."""
    lo, hi = breaks[0], breaks[-1]
    mids = (breaks[:-1] + breaks[1:]) / 2
    return np.unique(np.concatenate([breaks, mids, np.linspace(lo, hi, samples)]))


def quadrature(integrand, breaks, min_pieces=512):
    """Composite Gauss-Legendre integral of ``integrand`` over [breaks[0],
    breaks[-1]], never straddling a breakpoint (where a Profile has a kink)."""
    total = breaks[-1] - breaks[0]
    edges = []
    for left, right in zip(breaks[:-1], breaks[1:]):
        pieces = max(1, int(np.ceil(min_pieces * (right - left) / total)))
        edges.append(np.linspace(left, right, pieces + 1)[:-1])
    edges.append([breaks[-1]])
    edges = np.concatenate(edges)
    mid = (edges[:-1] + edges[1:]) / 2
    half = (edges[1:] - edges[:-1]) / 2
    nodes = mid[:, None] + half[:, None] * _GL_NODES[None, :]
    values = np.asarray(integrand(nodes.ravel()), dtype=float).reshape(nodes.shape)
    return float(np.sum(values * _GL_WEIGHTS[None, :] * half[:, None]))
