# claude
"""Tests for pydisks — checks textbook answers and the error surface."""

import sympy as sp
import pytest

from main import (
    x, y, disc, washer, shell, surface_area, revolve,
    AxisError, VariableMismatchError, BoundsError,
    SelfIntersectingSolidError, CurveCrossingError, DomainError,
)

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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
