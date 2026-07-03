# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

pydisks is a single-file SymPy library for computing volumes and surface areas of solids of revolution (disc, washer, shell, and surface-area-of-revolution methods) via symbolic integration.

## Commands

- Run the demo: `python3 main.py`
- Run tests: `python3 -m pytest test_main.py -v`
- Run one test: `python3 -m pytest test_main.py::test_disc_sphere`
- Dependencies: `sympy` and `pytest` (no requirements.txt/pyproject.toml yet — `pip install sympy pytest`)

## Core design decision: explicit axis, no auto-inversion

The axis of rotation is always passed **explicitly** as a line `(coord, c)`, naming the coordinate held constant — `("y", 0)` is the x-axis, `("x", 2)` is the vertical line x=2. The integration variable is *derived* from the axis and the method (see `_var_for`), so a caller cannot silently pair them wrong:

- disc/washer/surface_area: horizontal axis → integrate in `x`; vertical axis → integrate in `y`
- shell: horizontal axis → integrate in `y`; vertical axis → integrate in `x`

The library **never inverts a function** for the caller. If a curve is written in the wrong variable for the chosen axis/method, it raises `VariableMismatchError` and tells the user to solve for the correct variable themselves. This is deliberate — auto-inversion via `sp.solve` can return multiple branches and silently produce a wrong-but-plausible volume.

## Error surface

All errors subclass `RevolutionError`, and messages name the likely fix. Validation runs before integration:

- `AxisError` — malformed axis spec or non-constant offset
- `VariableMismatchError` — curve/bound in the wrong variable
- `BoundsError` — non-constant bounds, or `a >= b`
- `SelfIntersectingSolidError` — axis passes through the region's interior (disc/washer: curve crosses the axis inside `(a,b)`; shell: offset lies inside `(a,b)`)
- `CurveCrossingError` — washer outer/inner curves swap order inside `(a,b)`
- `DomainError` — curve discontinuous on `[a,b]`
- `UnevaluatedIntegralError` — SymPy returned an unevaluated `Integral`

**Important caveat:** the geometric/domain checks rely on `sp.solve` and `continuous_domain`, which silently no-op when bounds are symbolic or SymPy can't solve. They are best-effort guards, not guarantees — do not assume a returned volume was fully validated.

## Public API (`main.py`)

- `disc(f, a, b, axis=("y", 0))` — region between `f` and the axis
- `washer(outer, inner, a, b, axis=("y", 0))` — region between two curves; `outer` must stay farther from the axis than `inner` on all of `[a,b]`
- `shell(f, a, b, axis=("x", 0), bottom=0)` — pass `bottom` for a two-curve region (height = `f - bottom`)
- `surface_area(f, a, b, axis=("y", 0))`
- `revolve(f, a, b, axis=("y", 0), inner=None)` — dispatches disc/washer; returns `{"method", "volume", "approx"}` (`approx` is a float sanity-check)

Module-level `x` and `y` symbols are exported for building curve expressions.

## Adding methods

`test_main.py` includes a cross-check test (`test_shell_and_disc_agree`) that computes one solid two independent ways and asserts they match. When adding a new method, add a similar independent cross-check rather than only asserting against a hand-computed constant.
