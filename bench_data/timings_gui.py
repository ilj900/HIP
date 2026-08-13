import os
import struct
import sys
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
)

MAGIC = b"TIM1"
WIN_W = 1920          # fixed window width  (px)
WIN_H = 1080           # fixed window height (px)
MAX_PLOTS = 5         # cap on simultaneously drawn subplots


# --------------------------------------------------------------------------- #
# Data loading (same format as plot_timings.py)
# --------------------------------------------------------------------------- #
def load_timings(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"bad magic: {magic!r}")
        (count,) = struct.unpack("<I", f.read(4))
        data = np.fromfile(f, dtype="<f4", count=count)
        if data.size != count:
            raise ValueError(f"truncated: expected {count} floats, got {data.size}")
    return data

# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
class TimingGUI:
    def __init__(self, root, folder):
        self.folder = folder
        self.files = []                       # list of filenames (basename)

        root.title(f"Timing viewer — {folder}")
        root.geometry(f"{WIN_W}x{WIN_H}")
        root.resizable(False, False)          # fixed-size window

        paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # ---- left: file list ------------------------------------------------
        left = ttk.Frame(paned, padding=4)
        paned.add(left, weight=0)

        ttk.Label(left, text="bench_data (*.dat)").pack(anchor="w")

        list_frame = ttk.Frame(left)
        list_frame.pack(fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.listbox = tk.Listbox(
            list_frame, selectmode=tk.EXTENDED, exportselection=False,
            width=36, activestyle="none", yscrollcommand=sb.set,
        )
        sb.config(command=self.listbox.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox.bind("<<ListboxSelect>>", lambda e: self.redraw())

        self.combine = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            left, text="Combine into one graph",
            variable=self.combine, command=self.redraw,
        ).pack(anchor="w", pady=(4, 0))

        btns = ttk.Frame(left)
        btns.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(btns, text="Refresh", command=self.refresh_files).pack(
            side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(btns, text="Select all", command=self.select_all).pack(
            side=tk.LEFT, expand=True, fill=tk.X)

        # ---- right: single embedded figure that fills the fixed pane --------
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        self.fig = Figure()
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.mpl_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH,
                                             expand=True)

        self.refresh_files()

    # -- file list --------------------------------------------------------- #
    def refresh_files(self):
        prev = set(self.selected_names())
        self.files = sorted(
            f for f in os.listdir(self.folder) if f.endswith(".dat"))
        self.listbox.delete(0, tk.END)
        for f in self.files:
            self.listbox.insert(tk.END, f)
        for i, f in enumerate(self.files):        # keep prior selection
            if f in prev:
                self.listbox.selection_set(i)
        self.redraw()

    def select_all(self):
        self.listbox.selection_set(0, tk.END)
        self.redraw()

    def selected_names(self):
        return [self.files[i] for i in self.listbox.curselection()]

    # -- plotting ---------------------------------------------------------- #
    def redraw(self):
        names = self.selected_names()
        self.fig.clear()

        if not names:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Select one or more files on the left",
                    ha="center", va="center", transform=ax.transAxes,
                    color="gray")
            ax.set_axis_off()
        elif self.combine.get():
            ax = self.fig.add_subplot(111)
            for name in names:
                self._plot_series(ax, name, label=name)
            ax.set_xlabel("sample index")
            ax.set_ylabel("timing")
            ax.set_title(f"{len(names)} file(s) combined")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        else:
            drawn = names[:MAX_PLOTS]          # cap simultaneous subplots
            hidden = len(names) - len(drawn)
            n = len(drawn)
            for idx, name in enumerate(drawn):
                ax = self.fig.add_subplot(n, 1, idx + 1)
                self._plot_series(ax, name)
                ax.set_title(name, fontsize=9)
                ax.set_ylabel("timing")
                ax.grid(True, alpha=0.3)
                if idx == n - 1:
                    ax.set_xlabel("sample index")
            if hidden:
                self.fig.suptitle(
                    f"showing first {MAX_PLOTS} of {len(names)} selected "
                    f"({hidden} hidden — use Combine, or deselect some)",
                    fontsize=8, color="darkred")

        try:
            self.fig.tight_layout()
        except Exception:
            pass
        self.mpl_canvas.draw()

    def _plot_series(self, ax, name, label=None):
        path = os.path.join(self.folder, name)
        try:
            data = load_timings(path)
        except (ValueError, OSError) as e:
            ax.text(0.5, 0.5, f"{name}\n{e}", ha="center", va="center",
                    transform=ax.transAxes, color="red", fontsize=8)
            return
        ax.plot(data, marker="o", markersize=3, linewidth=1, label=label)
        if data.size and label is None:
            stats = (f"n={data.size}  mean={data.mean():.4g}  "
                     f"min={data.min():.4g}  max={data.max():.4g}")
            ax.text(0.01, 0.98, stats, transform=ax.transAxes,
                    va="top", ha="left", fontsize=7,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.7))

def main():
    args = sys.argv[1:]
    folder = args[0] if args else os.path.dirname(os.path.abspath(__file__))
    if not os.path.isdir(folder):
        print(f"Not a directory: {folder}", file=sys.stderr)
        sys.exit(1)

    root = tk.Tk()
    TimingGUI(root, folder)
    root.mainloop()


if __name__ == "__main__":
    main()