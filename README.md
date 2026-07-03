# pydisks

Compute **volumes and surface areas of solids of revolution** — the disc,
washer, and shell methods plus surface area of revolution — symbolically. Built for checking calculus homework:
you get an exact answer, and mistakes (wrong variable, self-intersecting
solids, curves in the wrong order) raise clear errors instead of silently
returning a wrong number.

## Install

```bash
pip install pydisks
```

## Quick start

```python
from pydisks import disc, washer, shell, surface_area, revolve, x, y
import sympy as sp

# Volume of a sphere of radius 2: revolve y = sqrt(4 - x^2) about the x-axis
disc(sp.sqrt(4 - x**2), -2, 2)          # -> 32*pi/3

# Washer: region between y = x and y = x^2 on [0, 1], about the x-axis
washer(x, x**2, 0, 1)                    # -> 2*pi/15

# Shell: revolve y = x^2 on [0, 2] about the y-axis
shell(x**2, 0, 2)                        # -> 8*pi

# Dispatcher: disc if no inner curve, else washer; also gives a float check
revolve(x, 0, 3)                         # -> {'method': 'disc', 'volume': 9*pi, 'approx': 28.27...}
```

## The axis of rotation

The axis is written as a string — either a named axis or a line equation:

| You write | Means | Line |
|-----------|-------|------|
| `"x"`     | the x-axis | y = 0 |
| `"y"`     | the y-axis | x = 0 |
| `"x=2"`   | vertical line | x = 2 |
| `"y=-1"`  | horizontal line | y = -1 |

A bare letter names an axis; an equation is a literal line. Note the
asymmetry this creates: `"x"` is the x-axis (y = 0), but `"x=0"` is the
line x = 0 (the y-axis).

The **variable you write your curve in** is derived from the axis and method:

- disc / washer / surface_area: horizontal axis → `f(x)`; vertical axis → `f(y)`
- shell: horizontal axis → `f(y)`; vertical axis → `f(x)`

If you pass a curve in the wrong variable, pydisks raises `VariableMismatchError`
rather than trying to invert it for you — solving for the right variable is part
of the problem.

## Functions

| Function | Method |
|----------|--------|
| `disc(f, a, b, axis="x")` | region between `f` and the axis |
| `washer(outer, inner, a, b, axis="x")` | region between two curves; `outer` stays farther from the axis |
| `shell(f, a, b, axis="y", bottom=0)` | cylindrical shells; pass `bottom` for a two-curve region |
| `surface_area(f, a, b, axis="x")` | surface area of revolution |
| `revolve(f, a, b, axis="x", inner=None)` | disc/washer dispatcher; returns `{method, volume, approx}` |

## Errors

Every error subclasses `RevolutionError`, and messages name the likely fix.
Validation runs before integration:

| Exception | When |
|-----------|------|
| `AxisError` | malformed axis string/pair |
| `VariableMismatchError` | curve or bound in the wrong variable |
| `BoundsError` | non-constant bounds, or `a >= b` |
| `SelfIntersectingSolidError` | the axis passes through the region |
| `CurveCrossingError` | two boundary curves cross or are given in the wrong order |
| `DomainError` | curve discontinuous on `[a, b]` |
| `UnevaluatedIntegralError` | SymPy found no closed form |

**Caveat:** the geometric and domain checks use `sp.solve` / `continuous_domain`,
which quietly no-op when the bounds are symbolic or SymPy can't solve. They catch
common concrete-number mistakes, not every symbolic edge case — treat a returned
volume as validated only for numeric bounds.

## Tests

```bash
python3 -m pytest test_main.py -v                  # all tests
python3 -m pytest test_main.py::test_disc_sphere   # a single test
```

## License

MIT — see [LICENSE](LICENSE).
