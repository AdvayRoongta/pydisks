"""Tests for pydisks — checks textbook answers and the error surface."""

import matplotlib
matplotlib.use("Agg")  # headless: tests must not open windows

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import pytest

from matplotlib.animation import FuncAnimation

from pydisks import (
    x, y, disc, washer, shell, surface_area, revolve, plot_disc, plot_washer,
    animate_disc, animate_washer, Profile,
    AxisError, VariableMismatchError, BoundsError,
    SelfIntersectingSolidError, CurveCrossingError, DomainError,
    ProfileError, UnevaluatedIntegralError,
)
from pydisks.plot import _numeric, _sheet_mesh, _surface_mesh

pi = sp.pi


# --------------------------------------------------------------------------
# Known volumes / areas
# --------------------------------------------------------------------------

def test_disc_sphere():
    # y = sqrt(4 - x^2) about the x-axis -> sphere of radius 2 -> 32*pi/3
    assert disc(sp.sqrt(4 - x ** 2), -2, 2) == 32 * pi / 3


def test_disc_cone():
    # y = x on [0, 3] about the x-axis -> cone r=3 h=3 -> 9*pi
    assert disc(x, 0, 3) == 9 * pi


def test_disc_vertical_axis():
    # x = 2 on y in [0, 3] about the y-axis -> cylinder r=2 h=3 -> 12*pi
    assert disc(sp.Integer(2), 0, 3, axis="y") == 12 * pi


def test_disc_shifted_axis():
    # f = 1 on [0, 2] about y = -1 -> solid cylinder radius 2, length 2 -> 8*pi
    assert disc(sp.Integer(1), 0, 2, axis="y=-1") == 8 * pi


def test_axis_equation_matches_pair():
    # "x=2" and ("x", 2) must describe the same line.
    assert disc(y ** 2, 0, 1, axis="x=2") == disc(y ** 2, 0, 1, axis=("x", 2))


def test_bare_letter_names_the_axis():
    # "x" is the x-axis (y=0); the equation "x=0" is the *y*-axis instead.
    assert disc(x ** 2, 0, 1, axis="x") == disc(x ** 2, 0, 1, axis=("y", 0))
    assert disc(y ** 2, 0, 1, axis="y") == disc(y ** 2, 0, 1, axis=("x", 0))


def test_washer_between_curves():
    # region between y = x and y = x^2 on [0, 1] about the x-axis -> 2*pi/15
    assert washer(x, x ** 2, 0, 1) == 2 * pi / 15


def test_shell_paraboloid():
    # y = x^2 on [0, 2] about the y-axis -> 8*pi
    assert shell(x ** 2, 0, 2) == 8 * pi


def test_shell_with_bottom_curve():
    # region between y = x and y = x^2 on [0, 1] about the y-axis.
    # shell: 2*pi * integral_0^1 x*(x - x^2) dx = 2*pi*(1/3 - 1/4) = pi/6
    assert shell(x, 0, 1, bottom=x ** 2) == pi / 6


def test_surface_area_cone():
    # y = x on [0, 1] about the x-axis -> 2*pi*integral x*sqrt(2) dx = sqrt(2)*pi
    assert surface_area(x, 0, 1) == sp.sqrt(2) * pi


def test_revolve_dispatch():
    d = revolve(x, 0, 3)
    assert d["method"] == "disc" and d["volume"] == 9 * pi
    w = revolve(x, 0, 1, inner=x ** 2)
    assert w["method"] == "washer" and w["volume"] == 2 * pi / 15


def test_disc_double_cone_across_axis():
    # y = x on [-1, 1] about the x-axis -> two cones tip-to-tip, vol 2*pi/3.
    # The curve crosses the axis but the disc method is still valid.
    assert disc(x, -1, 1) == 2 * pi / 3


def test_symbolic_bounds():
    # disc of y = x from 0 to h about the x-axis -> pi*h**3/3 (a cone).
    h = sp.Symbol("h", positive=True)
    assert sp.simplify(disc(x, 0, h) - sp.pi * h ** 3 / 3) == 0


def test_washer_inner_zero_is_disc():
    # A washer with inner radius 0 is just a disc.
    assert washer(x ** 2, 0, 0, 1) == disc(x ** 2, 0, 1)


def test_shell_and_disc_agree():
    # Volume under y = x^2, x in [0, 2], revolved about the y-axis.
    # Shell in x must equal disc in y (region between x = sqrt(y) and x = 2).
    v_shell = shell(x ** 2, 0, 2)
    v_disc = washer(sp.Integer(2), sp.sqrt(y), 0, 4, axis="y")
    assert sp.simplify(v_shell - v_disc) == 0


# --------------------------------------------------------------------------
# Error surface
# --------------------------------------------------------------------------

def test_wrong_variable_raises():
    with pytest.raises(VariableMismatchError):
        disc(y ** 2, 0, 1)  # var is x here, f uses y


def test_swapped_bounds_raises():
    with pytest.raises(BoundsError):
        disc(x, 2, 0)


def test_equal_bounds_raises():
    with pytest.raises(BoundsError):
        disc(x, 1, 1)


def test_axis_through_region_raises():
    with pytest.raises(SelfIntersectingSolidError):
        shell(x ** 2, -1, 2)  # axis x=0 lies inside (-1, 2)


def test_curve_crossing_raises():
    with pytest.raises(CurveCrossingError):
        washer(x, x ** 2, 0, 2)  # x and x^2 swap order at x=1


def test_washer_straddling_axis_raises():
    # outer=1, inner=-1 about y=0: the region contains the axis.
    with pytest.raises(SelfIntersectingSolidError):
        washer(1, -1, 0, 1)


def test_shell_curves_cross_raises():
    # f and bottom cross at x=1 inside (0, 2): height changes sign.
    with pytest.raises(CurveCrossingError):
        shell(x, 0, 2, bottom=x ** 2)


def test_shell_inverted_curves_raises():
    # f below bottom on the whole interval (curves given in wrong order).
    with pytest.raises(CurveCrossingError):
        shell(x ** 2, 0, 1, bottom=x)


def test_bad_axis_spec_raises():
    with pytest.raises(AxisError):
        disc(x, 0, 1, axis="z=3")       # 'z' is not a valid coordinate
    with pytest.raises(AxisError):
        disc(x, 0, 1, axis="x=")        # missing value
    with pytest.raises(AxisError):
        disc(x, 0, 1, axis="diagonal")  # not a letter or an equation


def test_discontinuity_raises():
    with pytest.raises(DomainError):
        disc(1 / x, -1, 1)  # 1/x blows up at x=0


# --------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _assert_two_panel(fig):
    axes = fig.get_axes()
    assert len(axes) == 2
    assert axes[0].name == "rectilinear" and axes[1].name == "3d"
    fig.canvas.draw()  # forces mathtext/LaTeX rendering errors to surface


def test_plot_disc_sphere():
    _assert_two_panel(plot_disc(sp.sqrt(4 - x ** 2), -2, 2))


def test_plot_washer_horizontal_and_vertical():
    _assert_two_panel(plot_washer(x, x ** 2, 0, 1))
    _assert_two_panel(plot_washer(2, sp.sqrt(y), 0, 4, axis="y"))


def test_plot_constant_curve_shifted_axis():
    _assert_two_panel(plot_disc(1, 0, 2, axis="y=-1", slice_at=0.5))


def test_plot_errors_match_volume_methods():
    with pytest.raises(CurveCrossingError):
        plot_washer(x, x ** 2, 0, 2)
    with pytest.raises(VariableMismatchError):
        plot_disc(y ** 2, 0, 1)
    with pytest.raises(BoundsError):
        plot_disc(x, 0, sp.Symbol("h", positive=True))  # can't plot symbols
    with pytest.raises(BoundsError):
        plot_disc(x, 0, 1, slice_at=2)  # slice outside [0, 1]


@pytest.mark.parametrize("f, var, a, b, axis, coord, c", [
    (sp.sqrt(4 - x ** 2), x, -2, 2, "x", "y", 0.0),
    (y ** 2, y, 0, 1, "x=2", "x", 2.0),
])
def test_plot_mesh_matches_disc(f, var, a, b, axis, coord, c):
    # Independent cross-check: the drawn 3D surface must sit at distance
    # |f - c| from the axis, and a numeric disc sum over the mesh radii must
    # match the symbolic volume.
    fn = _numeric(f, var)
    X, Y, Z = _surface_mesh(fn, a, b, coord, c, n=2001)
    if coord == "y":   # horizontal axis y=c: along = X, radial plane = (Y, Z)
        t, dist = X[:, 0], np.hypot(Y, Z - c)
    else:              # vertical axis x=c: along = Z, radial plane = (X, Y)
        t, dist = Z[:, 0], np.hypot(X - c, Y)
    assert np.allclose(dist, np.abs(fn(t) - c)[:, None])
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    approx = trapezoid(np.pi * dist[:, 0] ** 2, t)
    assert approx == pytest.approx(float(disc(f, a, b, axis=axis)), rel=1e-4)


def test_plot_accepts_profiles():
    xs = np.linspace(0, 2, 30)
    _assert_two_panel(plot_disc(Profile(x, xs, xs ** 2, kind="cubic"), 0, 2))
    _assert_two_panel(plot_washer(Profile(x, xs, xs + 1), 0.5, 0, 2))


# --------------------------------------------------------------------------
# Numeric integration: formulas with numeric=True
# --------------------------------------------------------------------------

@pytest.mark.parametrize("call, exact", [
    (lambda n: disc(sp.sqrt(4 - x ** 2), -2, 2, numeric=n), 32 * pi / 3),
    (lambda n: washer(x, x ** 2, 0, 1, numeric=n), 2 * pi / 15),
    (lambda n: shell(x ** 2, 0, 2, numeric=n), 8 * pi),
    (lambda n: surface_area(x ** 3, 0, 1, numeric=n),
     pi * (10 * sp.sqrt(10) - 1) / 27),
])
def test_numeric_flag_matches_exact(call, exact):
    # Independent cross-check: mpmath quadrature vs. SymPy's closed form.
    assert call(False) == exact
    approx = call(True)
    assert isinstance(approx, float)
    assert approx == pytest.approx(float(exact), rel=1e-10)


def test_numeric_flag_rescues_unevaluated_integral():
    with pytest.raises(UnevaluatedIntegralError, match="numeric=True"):
        disc(x ** x, 1, 2)
    assert disc(x ** x, 1, 2, numeric=True) == pytest.approx(15.4257331686, rel=1e-9)
    assert revolve(x ** x, 1, 2, numeric=True)["approx"] == pytest.approx(15.4257331686)
    # SymPy has no closed form for the sphere's surface integrand either.
    with pytest.raises(UnevaluatedIntegralError):
        surface_area(sp.sqrt(4 - x ** 2), -2, 2)
    assert surface_area(sp.sqrt(4 - x ** 2), -2, 2, numeric=True) == \
        pytest.approx(float(16 * pi), rel=1e-10)


def test_numeric_flag_keeps_validation_and_needs_numeric_bounds():
    with pytest.raises(CurveCrossingError):
        washer(x, x ** 2, 0, 2, numeric=True)
    with pytest.raises(BoundsError):
        disc(x, 0, sp.Symbol("h", positive=True), numeric=True)


# --------------------------------------------------------------------------
# Profiles (curves from sample points)
# --------------------------------------------------------------------------

def test_linear_profile_is_exact_for_polyline_solids():
    # A cone and a frustum-plus-cylinder are exactly piecewise linear, so a
    # linear Profile must reproduce the symbolic volume to rounding error.
    xs = np.array([0.0, 3.0])
    assert disc(Profile(x, xs, xs), 0, 3) == pytest.approx(float(9 * pi), rel=1e-12)
    part = Profile(x, [0, 1, 3], [1, 2, 2])  # frustum r 1->2, then cylinder r=2
    exact = float(disc(1 + x, 0, 1) + disc(2, 1, 3))
    assert disc(part, 0, 3) == pytest.approx(exact, rel=1e-12)


def test_cubic_profile_converges_to_exact_volume():
    errors = []
    for n in (11, 41, 161):
        xs = np.linspace(0, 2, n)
        v = disc(Profile(x, xs, xs ** 2, kind="cubic"), 0, 2)
        errors.append(abs(v - float(32 * pi / 5)))
    assert errors[-1] < 1e-5 * float(32 * pi / 5)
    assert errors[0] > errors[1] > errors[2]  # refining the data helps


def test_profile_sub_interval_and_shifted_axis():
    xs = np.linspace(0, 4, 81)
    p = Profile(x, xs, np.sin(xs) + 2, kind="cubic")
    exact = disc(sp.sin(x) + 2, 1, 3, axis="y=-1")
    assert disc(p, 1, 3, axis="y=-1") == pytest.approx(float(exact), rel=1e-6)


def test_profile_shell_and_washer_agree():
    # Independent cross-check, mirroring test_shell_and_disc_agree: the
    # region under y = x^2 on [0, 1] revolved about the y-axis, from two
    # *different* sampled datasets and two methods. Exact answer pi/2.
    xs = np.linspace(0, 1, 101)
    ys = np.linspace(0, 1, 401)
    by_shell = shell(Profile(x, xs, xs ** 2, kind="cubic"), 0, 1)
    by_washer = washer(1, Profile(y, ys, np.sqrt(ys), kind="cubic"), 0, 1, axis="y")
    assert by_shell == pytest.approx(by_washer, rel=1e-4)
    assert by_shell == pytest.approx(float(pi / 2), rel=1e-6)


def test_profile_mixed_with_formula_washer():
    # Measured outer wall, machined constant bore.
    xs = np.linspace(0, 2, 3)
    outer = Profile(x, xs, [2, 3, 2])
    exact = float(washer(2 + x, 1, 0, 1) + washer(4 - x, 1, 1, 2))
    assert washer(outer, 1, 0, 2) == pytest.approx(exact, rel=1e-12)


def test_profile_surface_area_frustum_and_sphere():
    # Linear profile: lateral area of a frustum, pi*(r1+r2)*slant.
    frustum = surface_area(Profile(x, [0, 4], [1, 4]), 0, 4)
    assert frustum == pytest.approx(np.pi * 5 * 5, rel=1e-12)
    t = np.linspace(-2, 2, 801)
    sphere = surface_area(Profile(x, t, np.sqrt(4 - t ** 2), kind="cubic"), -2, 2)
    assert sphere == pytest.approx(float(16 * pi), rel=1e-4)


def test_profile_evaluation_and_derivative():
    p = Profile(x, [0, 1, 2], [0, 2, 2])
    assert p(np.array([0.5, 1.5])) == pytest.approx([1.0, 2.0])
    assert p.derivative(np.array([0.5, 1.5])) == pytest.approx([2.0, 0.0])
    assert np.isnan(p(3.0))  # never extrapolates
    xs = np.linspace(0, 1, 6)
    c = Profile(y, xs, xs ** 3, kind="cubic")
    assert c(xs) == pytest.approx(xs ** 3)  # interpolates the knots exactly
    assert c.span == (0.0, 1.0) and len(c) == 6 and c.var == y


@pytest.mark.parametrize("args, kwargs, match", [
    ((x, [0, 1], [1]), {}, "length"),
    ((x, [0], [1]), {}, "at least 2"),
    ((x, [0, 2, 1], [1, 1, 1]), {}, "strictly increasing"),
    ((x, [0, 1, 1], [1, 1, 1]), {}, "strictly increasing"),
    ((x, [0, 1], [1, np.nan]), {}, "finite"),
    ((x, [0, 1], [1, 2]), {"kind": "quadratic"}, "kind"),
    (("z", [0, 1], [1, 2]), {}, "var"),
])
def test_profile_rejects_bad_data(args, kwargs, match):
    with pytest.raises(ProfileError, match=match):
        Profile(*args, **kwargs)


def test_profile_errors_match_formula_errors():
    xs = np.linspace(0, 2, 21)
    with pytest.raises(VariableMismatchError, match="sampled along x"):
        disc(Profile(x, xs, xs), 0, 2, axis="y")  # vertical axis needs f(y)
    with pytest.raises(BoundsError, match="extrapolate"):
        disc(Profile(x, xs, xs), 0, 3)
    with pytest.raises(BoundsError):
        disc(Profile(x, xs, xs), 2, 0)
    # washer: y = x and y = x^2 swap order at x = 1 (like the formula case)
    with pytest.raises(CurveCrossingError, match="near x=1"):
        washer(Profile(x, xs, xs), Profile(x, xs, xs ** 2), 0, 2)
    with pytest.raises(CurveCrossingError, match="swap them"):
        washer(Profile(x, xs, xs + 1), Profile(x, xs, xs + 2), 0, 2)
    # outer crosses the axis y = 1 inside the interval
    with pytest.raises(SelfIntersectingSolidError):
        washer(Profile(x, xs, xs), 0.5, 0, 2, axis="y=1")
    # shell: axis x = 1 inside [0, 2]; f below bottom
    with pytest.raises(SelfIntersectingSolidError):
        shell(Profile(x, xs, xs + 1), 0, 2, axis="x=1")
    with pytest.raises(CurveCrossingError, match="Swap"):
        shell(Profile(x, xs, xs), 0, 2, bottom=5)
    # mixed with a formula that is undefined on part of the interval
    with pytest.raises(DomainError):
        washer(Profile(x, xs, xs + 5), sp.sqrt(1 - x), 0, 2)


# --------------------------------------------------------------------------
# Animation
# --------------------------------------------------------------------------

def test_sheet_mesh_lies_between_the_radii():
    # Independent geometric check: the rotating region drawn during the
    # sweep must sit between the inner and outer radii at the frame's angle.
    t = np.linspace(0, 1, 50)
    r_in, r_out = t ** 2, t
    X, Y, Z = _sheet_mesh(t, r_in, r_out, 1.0, "y", 0)
    dist = np.hypot(Y, Z)
    assert np.all(dist >= r_in[:, None] - 1e-12)
    assert np.all(dist <= r_out[:, None] + 1e-12)
    assert np.allclose(np.arctan2(Y[dist > 0], Z[dist > 0]), 1.0)


def test_partial_sweep_ends_at_requested_angle():
    fn = _numeric(x + 1, x)
    X, Y, Z = _surface_mesh(fn, 0, 1, "y", 0, n=5, m=7, sweep=np.pi / 2)
    assert np.allclose(np.arctan2(Y[:, -1], Z[:, -1]), np.pi / 2)
    assert np.allclose(np.arctan2(Y[:, 0], Z[:, 0]), 0.0)


@pytest.mark.parametrize("make, frames", [
    (lambda: animate_disc(sp.sqrt(4 - x ** 2), -2, 2, sweep_frames=3,
                          spin_frames=2, n=20, m=12), 5),
    (lambda: animate_washer(2, sp.sqrt(y), 0, 4, axis="y", sweep_frames=3,
                            spin_frames=2, n=20, m=12), 5),
    (lambda: animate_disc(Profile(x, [0, 1, 2], [1, 2, 1]), 0, 2, axis="y=-1",
                          sweep_frames=3, spin_frames=0, n=20, m=12), 3),
])
def test_animation_renders_every_frame(make, frames, tmp_path):
    from PIL import Image

    anim = make()
    assert isinstance(anim, FuncAnimation)
    path = tmp_path / "solid.gif"
    anim.save(path, writer="pillow", fps=5, dpi=40)
    with Image.open(path) as gif:
        assert gif.n_frames == frames  # one image per sweep/spin frame


def test_render_frame_is_order_independent():
    # Frames are drawn from scratch, so rendering out of order (as the
    # parallel GIF script does) must give the same pixels as in order.
    anim = animate_disc(x + 1, 0, 1, sweep_frames=3, spin_frames=2, n=10, m=8)
    assert anim.n_frames == 5
    in_order = [anim.render_frame(i, dpi=30) for i in range(5)]
    assert in_order[0].dtype == np.uint8 and in_order[0].shape[2] == 3
    for i in (4, 1, 3, 0, 2):
        assert np.array_equal(anim.render_frame(i, dpi=30), in_order[i])
    with pytest.raises(IndexError):
        anim.render_frame(5)


def test_animation_errors_match_volume_methods():
    with pytest.raises(CurveCrossingError):
        animate_washer(x, x ** 2, 0, 2)
    with pytest.raises(BoundsError):
        animate_disc(x, 0, 1, slice_at=5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
