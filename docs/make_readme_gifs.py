"""Regenerate the animated GIFs embedded in README.md.

    python3 docs/make_readme_gifs.py               # all GIFs
    python3 docs/make_readme_gifs.py disc_sphere   # just one

Pipeline (needs ffmpeg on your PATH):

1. Render: frames are drawn in parallel, one chunk of frames per worker
   process (``SolidAnimation.render_frame``), straight to raw RGB arrays.
   No intermediate video, so there is no compression loss before the GIF.
   Each worker is limited to one BLAS thread (see below).
2. Crop: the union of every frame's non-background pixels is computed and
   all frames are cropped to it, so the GIF stores no empty margin.
3. Encode: raw frames are piped into ffmpeg, which builds one 128-colour
   palette weighted toward the pixels that change (``stats_mode=diff``) and
   stores each frame as only the rectangle that changed since the previous
   one (``diff_mode=rectangle``). While the camera is still (the sweep),
   that rectangle is a small part of the frame.
"""

import os

# One BLAS thread per process. Workers already use every core; letting each
# also start OpenBLAS threads (for mplot3d's projection matmuls)
# oversubscribes the CPU, and the spinning threads more than double the
# wall time. Must be set before numpy is imported here and in the workers.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from demo_plots import vase_inner, vase_outer  # noqa: E402
from pydisks import animate_disc, animate_washer, x, y  # noqa: E402

OUT = Path(__file__).resolve().parent / "img"
FRAMES = dict(sweep_frames=60, spin_frames=60)
FPS = 20
# Size is driven by how many pixels change per frame (about 30% of the
# frame while the region sweeps, 65% while the camera orbits), so
# resolution matters far more than dithering. 64 colours looks the same as
# 128 for these shaded surfaces and saves about 20%. Override with the
# GIF_DPI / GIF_COLORS environment variables.
DPI = int(os.environ.get("GIF_DPI", 75))        # figure is 8.5 x 6.5 in
COLORS = int(os.environ.get("GIF_COLORS", 64))
CHUNK = 8       # frames per worker task
PAD = 6         # pixels of margin kept around the cropped content

GIFS = {
    "disc_sphere": lambda: animate_disc(sp.sqrt(4 - x**2), -2, 2, **FRAMES),
    "washer_y_axis": lambda: animate_washer(2, sp.sqrt(y), 0, 4, axis="y", **FRAMES),
    "profile_vase": lambda: animate_washer(vase_outer, vase_inner, 0, 14,
                                           axis="y", **FRAMES),
}

# --------------------------------------------------------------------------
# 1. Render (runs inside worker processes)
# --------------------------------------------------------------------------

_built = {}  # per-worker cache: each worker builds each animation only once


def _render_chunk(job):
    name, start, stop = job
    if name not in _built:
        _built[name] = GIFS[name]()
    anim = _built[name]
    return name, start, [anim.render_frame(i, dpi=DPI) for i in range(start, stop)]


# --------------------------------------------------------------------------
# 2. Crop and 3. encode (main process)
# --------------------------------------------------------------------------

def _crop_box(frames):
    """(top, bottom, left, right) bounding every non-background pixel."""
    background = frames[0][0, 0].astype(int)
    ink = np.zeros(frames[0].shape[:2], dtype=bool)
    for frame in frames:
        ink |= np.any(np.abs(frame.astype(int) - background) > 12, axis=2)
    rows, cols = np.flatnonzero(ink.any(axis=1)), np.flatnonzero(ink.any(axis=0))
    h, w = ink.shape
    return (max(rows[0] - PAD, 0), min(rows[-1] + PAD + 1, h),
            max(cols[0] - PAD, 0), min(cols[-1] + PAD + 1, w))


def _encode(frames, path):
    top, bottom, left, right = _crop_box(frames)
    clip = np.ascontiguousarray(np.stack(frames)[:, top:bottom, left:right])
    height, width = clip.shape[1:3]
    graph = (
        f"split[a][b];[a]palettegen=max_colors={COLORS}:stats_mode=diff[p];"
        f"[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
    )
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", str(FPS), "-i", "-",
         "-vf", graph, "-loop", "0", str(path)],
        input=clip.tobytes(), check=True,
    )
    return width, height


def main(names):
    names = names or list(GIFS)
    unknown = [n for n in names if n not in GIFS]
    if unknown:
        sys.exit(f"unknown GIF(s) {unknown}; choose from {list(GIFS)}")
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    counts = {name: FRAMES["sweep_frames"] + FRAMES["spin_frames"] for name in names}
    jobs = [(name, i, min(i + CHUNK, n))
            for name, n in counts.items() for i in range(0, n, CHUNK)]

    frames = {name: [None] * n for name, n in counts.items()}
    remaining = dict(counts)
    # "spawn" gives each worker a clean matplotlib state (fork is unsafe on
    # macOS once a GUI framework may have been touched).
    with get_context("spawn").Pool(os.cpu_count()) as pool:
        for name, start, chunk in pool.imap_unordered(_render_chunk, jobs):
            frames[name][start:start + len(chunk)] = chunk
            remaining[name] -= len(chunk)
            if remaining[name] == 0:  # encode as soon as one GIF is complete
                gif = OUT / f"{name}.gif"
                w, h = _encode(frames.pop(name), gif)
                print(f"{gif.relative_to(OUT.parent.parent)}  {w}x{h}  "
                      f"{gif.stat().st_size / 1e6:.2f} MB")
    print(f"done in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
