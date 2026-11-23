"""Demo: see plot_disc / plot_washer in action.

    python3 demo_plots.py            # open every example in its own window
    python3 demo_plots.py sphere     # open just one (names listed below)
    python3 demo_plots.py --save out # write PNGs to ./out instead of showing

The 3D panels can be rotated by dragging with the mouse.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import sympy as sp

from pydisks import Profile, plot_disc, plot_washer, x, y

# A vase measured with calipers: radius (cm) at heights (cm) up its axis.
VASE_HEIGHTS = [0, 2, 4, 6, 8, 10, 12, 14]
VASE_RADII = [3.0, 4.2, 4.8, 4.5, 3.4, 2.4, 2.2, 2.8]
vase_outer = Profile(y, VASE_HEIGHTS, VASE_RADII, kind="cubic")
vase_inner = Profile(y, VASE_HEIGHTS, [r - 0.5 for r in VASE_RADII], kind="cubic")

EXAMPLES = {
    # Measured data: a solid vase shape, revolved about its vertical axis
    "vase": lambda: plot_disc(vase_outer, 0, 14, axis="y"),
    # Measured data: the same vase hollowed out with 0.5 cm walls
    "vase_hollow": lambda: plot_washer(vase_outer, vase_inner, 0, 14, axis="y"),
    # Disc: y = sqrt(4 - x^2) about the x-axis -> sphere of radius 2
    "sphere": lambda: plot_disc(sp.sqrt(4 - x**2), -2, 2),
    # Disc: y = x on [0, 3] about the x-axis -> cone
    "cone": lambda: plot_disc(x, 0, 3),
    # Disc: y = x^2 on [0, 2] about the x-axis -> horn shape, V = 32*pi/5
    "parabola": lambda: plot_disc(x**2, 0, 2),
    # Disc about a shifted horizontal axis: y = 1 about y = -1 -> cylinder
    "shifted": lambda: plot_disc(1, 0, 2, axis="y=-1"),
    # Disc about a vertical line: x = y^2 on y in [0, 1] about x = 2
    "vertical": lambda: plot_disc(y**2, 0, 1, axis="x=2"),
    # Washer: between y = x and y = x^2 on [0, 1] about the x-axis
    "washer": lambda: plot_washer(x, x**2, 0, 1, slice_at=0.4),
    # Washer about the y-axis: between x = 2 and x = sqrt(y), y in [0, 4]
    "washer_y": lambda: plot_washer(2, sp.sqrt(y), 0, 4, axis="y"),
    # Washer about a shifted axis: y = 3 and y = 1 + sin(x) about y = -1
    "washer_sin": lambda: plot_washer(3, 1 + sp.sin(x), 0, 2 * sp.pi, axis="y=-1"),
}


def main(argv):
    save_dir = None
    if "--save" in argv:
        i = argv.index("--save")
        save_dir = Path(argv[i + 1] if i + 1 < len(argv) else "plots")
        argv = argv[:i] + argv[i + 2:]

    names = argv or list(EXAMPLES)
    unknown = [n for n in names if n not in EXAMPLES]
    if unknown:
        sys.exit(f"unknown example(s) {unknown}; choose from {list(EXAMPLES)}")

    for name in names:
        fig = EXAMPLES[name]()
        fig.canvas.manager.set_window_title(f"pydisks: {name}")
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            path = save_dir / f"{name}.png"
            fig.savefig(path, dpi=120)
            print(f"saved {path}")
            plt.close(fig)

    if save_dir is None:
        plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
