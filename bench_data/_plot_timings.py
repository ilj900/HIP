#!/usr/bin/env nix-shell
#!nix-shell -i python3 -p "python3.withPackages(ps: [ps.numpy ps.matplotlib])"
"""
Plot one or more timing files written by saveTimings() (see save_timings.hpp).

Each file gets its own plot window, plus a .png saved next to it so you
still get output even if no window shows up.

Usage from a terminal:
    ./plot_timings.py run1.tim run2.tim ...

Intended usage: drag one or more .tim files onto a launcher/icon pointing
at this script (see README.md for how to set that up on NixOS).
"""

import struct
import sys

import numpy as np
import matplotlib.pyplot as plt

MAGIC = b"TIM1"


def load_timings(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"not a recognized timings file (bad magic: {magic!r})")
        (count,) = struct.unpack("<I", f.read(4))
        data = np.fromfile(f, dtype="<f4", count=count)
        if data.size != count:
            raise ValueError(f"truncated file, expected {count} floats, got {data.size}")
    return data


def plot_file(path):
    try:
        data = load_timings(path)
    except (ValueError, OSError) as e:
        print(f"Skipping '{path}': {e}", file=sys.stderr)
        return

    fig, ax = plt.subplots()
    ax.plot(data, marker="o", markersize=3, linewidth=1)
    ax.set_title(path)
    ax.set_xlabel("sample index")
    ax.set_ylabel("timing")

    if data.size:
        stats = (
            f"n={data.size}  mean={data.mean():.6g}  "
            f"min={data.min():.6g}  max={data.max():.6g}  std={data.std():.6g}"
        )
        ax.text(
            0.01, 0.99, stats, transform=ax.transAxes,
            va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", alpha=0.7),
        )

    fig.tight_layout()

    png_path = path + ".png"
    try:
        fig.savefig(png_path, dpi=150)
        print(f"Saved {png_path}")
    except OSError as e:
        print(f"Could not save '{png_path}': {e}", file=sys.stderr)


def main():
    paths = sys.argv[1:]
    if not paths:
        print(
            "No files given.\n"
            "Drag one or more .tim files onto this script's launcher, "
            "or run it from a terminal:\n"
            "  ./plot_timings.py file1.tim [file2.tim ...]",
            file=sys.stderr,
        )
        return

    for path in paths:
        plot_file(path)

    plt.show()


if __name__ == "__main__":
    main()
