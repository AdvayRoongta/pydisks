"""Demo: render animate_disc / animate_washer videos.

    python3 demo_animations.py                  # every example -> ./videos/*.mp4
    python3 demo_animations.py sphere washer_y  # just these (names below)
    python3 demo_animations.py --gif            # GIFs instead (no ffmpeg needed)
    python3 demo_animations.py --show sphere    # play one live in a window
    python3 demo_animations.py --out DIR        # write somewhere else

Each video sweeps the region once around the axis to build the solid, then
orbits the camera around the finished solid with a slice highlighted.
MP4 output needs ffmpeg on your PATH (e.g. `brew install ffmpeg`).
"""

import sys
from pathlib import Path

import sympy as sp

from demo_plots import vase_inner, vase_outer
from pydisks import animate_disc, animate_washer, x, y

EXAMPLES = {
    "sphere": lambda: animate_disc(sp.sqrt(4 - x**2), -2, 2),
    "parabola": lambda: animate_disc(x**2, 0, 2),
    "vertical": lambda: animate_disc(y**2, 0, 1, axis="x=2"),
    "washer": lambda: animate_washer(x, x**2, 0, 1, slice_at=0.4),
    "washer_y": lambda: animate_washer(2, sp.sqrt(y), 0, 4, axis="y"),
    "vase_hollow": lambda: animate_washer(vase_outer, vase_inner, 0, 14, axis="y"),
}


def _take(argv, flag, default=None):
    """Remove ``flag VALUE`` from argv; return (value, argv)."""
    if flag not in argv:
        return default, argv
    i = argv.index(flag)
    value = argv[i + 1] if i + 1 < len(argv) else default
    return value, argv[:i] + argv[i + 2:]


def main(argv):
    out, argv = _take(argv, "--out", "videos")
    gif = "--gif" in argv
    show = "--show" in argv
    argv = [a for a in argv if a not in ("--gif", "--show")]

    names = argv or list(EXAMPLES)
    unknown = [n for n in names if n not in EXAMPLES]
    if unknown:
        sys.exit(f"unknown example(s) {unknown}; choose from {list(EXAMPLES)}")

    if show:
        import matplotlib.pyplot as plt

        anims = [EXAMPLES[name]() for name in names]  # keep references alive
        plt.show()
        return anims

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        anim = EXAMPLES[name]()
        path = out_dir / f"{name}.{'gif' if gif else 'mp4'}"
        print(f"rendering {path} ...", flush=True)
        if gif:
            anim.save(path, writer="pillow", fps=20, dpi=80)
        else:
            anim.save(path, fps=30, dpi=100)
        plt.close("all")
    print("done")


if __name__ == "__main__":
    main(sys.argv[1:])
