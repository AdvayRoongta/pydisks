"""Animations of the disc and washer methods.

Each function returns a matplotlib ``FuncAnimation`` in two phases:

1. **sweep** -- the flat 2D region (highlighted) rotates about the axis,
   and the surface it traces builds up behind it until the solid closes
   after a full turn;
2. **spin** -- the camera orbits the finished solid, with the
   representative disc/washer slice highlighted as in ``plot_disc`` /
   ``plot_washer``.

Save it with ``anim.save("solid.mp4", fps=30)`` (needs ffmpeg) or
``anim.save("solid.gif", writer="pillow", fps=20)``, or call ``plt.show()``
to watch it live. Inputs are validated by the real ``disc`` / ``washer``
first, so an animation raises exactly the same errors as the volume
computation would.
"""

import numpy as np
from matplotlib.animation import FuncAnimation

from . import disc, washer
from .plot import (
    _AXIS,
    _INNER,
    _REGION,
    _SLICE,
    _annulus_mesh,
    _draw_axis_3d,
    _prepare,
    _revolve_points,
    _sheet_mesh,
    _surface,
    _surface_mesh,
    _title,
)

__all__ = ["animate_disc", "animate_washer", "SolidAnimation"]

_FULL = 2 * np.pi


def _sweep_steps(m, angle):
    """Number of angular samples for a partial surface through ``angle``."""
    return max(2, int(np.ceil(m * angle / _FULL)) + 1)


def _animate(method, named, f_out, f_in, coord, offset, var, lo, hi, s, volume,
             sweep_frames, spin_frames, interval, n, m):
    import matplotlib.pyplot as plt

    if sweep_frames < 1 or spin_frames < 0:
        raise ValueError("sweep_frames must be >= 1 and spin_frames >= 0.")

    c = float(offset)
    t = np.linspace(lo, hi, n)
    rho_out = f_out(t) - c
    rho_in = f_in(t) - c if f_in is not None else np.zeros_like(t)

    def radii(at):
        p = np.array([float(at)])
        inner = float(f_in(p)[0]) - c if f_in is not None else 0.0
        return inner, float(f_out(p)[0]) - c

    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(projection="3d")
    # Draw in call order (slice/sheet last) instead of matplotlib's
    # per-collection depth sort, which would bury the highlight.
    ax.computed_zorder = False

    # Fixed limits sized to the finished solid, so the frame never jumps.
    reach = float(np.nanmax(np.abs(np.concatenate([rho_out, rho_in])))) or 1.0
    pad = 0.05 * (hi - lo)
    along = (lo - pad, hi + pad)
    across = (c - 1.05 * reach, c + 1.05 * reach)
    depth = (-1.05 * reach, 1.05 * reach)
    if coord == "y":   # horizontal axis: along = X, up = Z
        limits = (along, depth, across)
        elev, azim0 = 22, -70
    else:              # vertical axis: along = Z, across = X
        limits = (across, depth, along)
        elev, azim0 = 12, -60
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_box_aspect([hi_ - lo_ for lo_, hi_ in limits])
    ax.set_xlabel("x")
    ax.set_zlabel("y")
    ax.set_ylabel("")
    ax.set_yticks([])  # depth direction: purely the sweep, no coordinate

    # Static: the axis of rotation and the outline of the original region.
    _draw_axis_3d(ax, coord, c, lo, hi)
    outlines = [(rho_out, _REGION)] + ([(rho_in, _INNER)] if f_in is not None else [])
    for rho, color in outlines:
        ax.plot(*_revolve_points(t, rho, 0.0, coord, offset), color=color,
                linewidth=1.5, alpha=0.8)

    fig.suptitle(_title(method, named, coord, offset, volume), fontsize=9.5)
    status = ax.text2D(0.02, 0.97, "", transform=ax.transAxes, fontsize=10,
                       color=_AXIS)

    dynamic = []

    def surface(mesh, **kw):
        dynamic.append(_surface(ax, mesh, **kw))

    def update(frame):
        while dynamic:
            dynamic.pop().remove()
        sweeping = frame < sweep_frames
        if sweeping:
            angle = _FULL * (frame + 1) / sweep_frames
            azim = azim0
            status.set_text(f"revolving the region: {np.degrees(angle):.0f}°")
        else:
            angle = _FULL
            azim = azim0 + 360.0 * (frame - sweep_frames + 1) / max(spin_frames, 1)
            status.set_text(f"{method.lower()} slice at {var} = {s:.4g}")
        ax.view_init(elev=elev, azim=azim)

        steps = _sweep_steps(m, angle)
        if f_in is not None:
            surface(_surface_mesh(f_in, lo, hi, coord, offset, n, steps, angle),
                    color=_INNER, alpha=0.6)
        surface(_surface_mesh(f_out, lo, hi, coord, offset, n, steps, angle),
                color=_REGION, alpha=0.3 if f_in is not None else 0.35)
        for end in (lo, hi):
            r_in, r_out = radii(end)
            if np.isfinite(r_in) and np.isfinite(r_out) and r_in != r_out:
                surface(_annulus_mesh(end, r_in, r_out, coord, offset, steps, angle),
                        color=_REGION, alpha=0.4)

        if sweeping:  # the region itself, at its current angle
            surface(_sheet_mesh(t, rho_in, rho_out, angle, coord, offset),
                    color=_SLICE, alpha=0.85, shade=False)
        else:         # the representative slice of the finished solid
            r_in, r_out = radii(s)
            surface(_annulus_mesh(s, r_in, r_out, coord, offset),
                    color=_SLICE, alpha=0.95, shade=False)
        return dynamic

    return SolidAnimation(fig, update, frames=sweep_frames + spin_frames,
                          interval=interval, blit=False, repeat=True)


class SolidAnimation(FuncAnimation):
    """A FuncAnimation that can also render any single frame on demand.

    ``render_frame(i)`` draws frame ``i`` and returns it as an RGB ``uint8``
    array of shape (height, width, 3). Each frame is drawn from scratch, so
    frames can be rendered out of order or split across processes (see
    ``docs/make_readme_gifs.py``). ``n_frames`` is the total frame count.
    """

    def __init__(self, fig, func, frames, **kwargs):
        self._draw_one = func
        self.n_frames = frames
        super().__init__(fig, func, frames=frames, **kwargs)

    def render_frame(self, i, dpi=None):
        if not 0 <= i < self.n_frames:
            raise IndexError(f"frame {i} is outside 0..{self.n_frames - 1}")
        fig = self._fig
        if dpi is not None:
            fig.set_dpi(dpi)
        self._draw_one(i)
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def animate_disc(f, a, b, axis="x", *, slice_at=None, sweep_frames=90,
                 spin_frames=90, interval=33, n=80, m=48):
    """Animation of the disc-method solid for ``disc(f, a, b, axis)``.

    The region between ``f`` and the axis sweeps one full turn to build the
    solid (``sweep_frames``), then the camera orbits it (``spin_frames``)
    with the disc at ``slice_at`` (default: midpoint) highlighted.
    ``interval`` is milliseconds per frame for live playback; ``n`` and
    ``m`` are the mesh resolutions along and around the axis.

    Raises the same errors as ``disc``, plus ``BoundsError`` for symbolic
    bounds or a ``slice_at`` outside [a, b]. Returns a FuncAnimation: keep
    a reference to it, then ``anim.save(...)`` or ``plt.show()``.
    """
    volume = disc(f, a, b, axis=axis)
    coord, offset, c, var, lo, hi, s, exprs, fns = _prepare(
        [("f", f)], a, b, axis, "disc", slice_at
    )
    return _animate("Disc", [("f", exprs[0])], fns[0], None, coord, offset, var,
                    lo, hi, s, volume, sweep_frames, spin_frames, interval, n, m)


def animate_washer(outer, inner, a, b, axis="x", *, slice_at=None,
                   sweep_frames=90, spin_frames=90, interval=33, n=80, m=48):
    """Animation of the washer-method solid for
    ``washer(outer, inner, a, b, axis)``.

    The region between the curves sweeps one full turn to build the solid
    (translucent outer surface, opaque inner surface so the hole shows),
    then the camera orbits it with the washer at ``slice_at`` highlighted.
    Other parameters as in ``animate_disc``.

    Raises the same errors as ``washer``, plus ``BoundsError`` for symbolic
    bounds or a ``slice_at`` outside [a, b]. Returns a FuncAnimation.
    """
    volume = washer(outer, inner, a, b, axis=axis)
    coord, offset, c, var, lo, hi, s, exprs, fns = _prepare(
        [("outer", outer), ("inner", inner)], a, b, axis, "washer", slice_at
    )
    named = list(zip(("outer", "inner"), exprs))
    return _animate("Washer", named, fns[0], fns[1], coord, offset, var,
                    lo, hi, s, volume, sweep_frames, spin_frames, interval, n, m)
