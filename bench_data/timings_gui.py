import os
import struct
import sys
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)

MAGIC = b"TIM1"
BASE_DPI = 100
BASE_PER_PLOT_PX = 240     # height budget for each stacked subplot (unscaled)


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

def detect_scale():
    """UI scale factor. XWayland hides the real display scale, so we take it
    from an env var (Hyprland users can export GDK_SCALE) or a --scale flag."""
    for var in ("TIMING_GUI_SCALE", "GDK_SCALE", "QT_SCALE_FACTOR"):
        val = os.environ.get(var)
        if val:
            try:
                return max(0.5, float(val))
            except ValueError:
                pass
    return 1.0


def apply_widget_scaling(root, scale):
    """Scale Tk widget fonts. Negative size = absolute pixels, so the result
    is deterministic regardless of what DPI XWayland reports."""
    base_px = 13
    for name in ("TkDefaultFont", "TkTextFont", "TkFixedFont",
                 "TkMenuFont", "TkHeadingFont", "TkTooltipFont", "TkIconFont"):
        try:
            tkfont.nametofont(name).configure(size=-int(round(base_px * scale)))
        except tk.TclError:
            pass

# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
class TimingGUI:
    def __init__(self, root, folder, scale=1.0):
        self.root = root
        self.folder = folder
        self.scale = scale
        self.dpi = BASE_DPI * scale
        self.per_plot_px = int(BASE_PER_PLOT_PX * scale)
        self.files = []                       # list of filenames (basename)
        self._resize_job = None

        root.title(f"Timing viewer — {folder}")
        root.geometry(f"{int(1100 * scale)}x{int(700 * scale)}")

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
            width=40, activestyle="none", yscrollcommand=sb.set,
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

        # ---- right: scrollable embedded figure ------------------------------
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        self.fig = Figure(dpi=self.dpi)
        self.scroll_canvas = tk.Canvas(right, highlightthickness=0)
        vbar = ttk.Scrollbar(right, orient=tk.VERTICAL,
                             command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=vbar.set)

        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=self.scroll_canvas)
        self.mpl_widget = self.mpl_canvas.get_tk_widget()
        self.win_id = self.scroll_canvas.create_window(
            (0, 0), window=self.mpl_widget, anchor="nw")

        toolbar_frame = ttk.Frame(right)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        NavigationToolbar2Tk(self.mpl_canvas, toolbar_frame)

        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scroll_canvas.bind("<Configure>", self._on_container_resize)
        # mouse-wheel scrolling (Linux uses Button-4/5)
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.scroll_canvas.bind_all(seq, self._on_wheel)

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

        # -- scrolling / resize ------------------------------------------------ #
    def _on_wheel(self, event):
        if event.num == 4 or event.delta > 0:
            self.scroll_canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:
            self.scroll_canvas.yview_scroll(1, "units")

    def _on_container_resize(self, event):
        # debounce: redraw shortly after the last resize event
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(120, self.redraw)

    # -- plotting ---------------------------------------------------------- #
    def redraw(self):
        self._resize_job = None
        names = self.selected_names()
        self.fig.clear()

        avail_w = max(self.scroll_canvas.winfo_width(), 200)
        avail_h = max(self.scroll_canvas.winfo_height(), 200)
        w_in = avail_w / self.dpi

        if not names:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Select one or more files on the left",
                    ha="center", va="center", transform=ax.transAxes,
                    color="gray")
            ax.set_axis_off()
            total_h = avail_h
        elif self.combine.get():
            ax = self.fig.add_subplot(111)
            for name in names:
                self._plot_series(ax, name, label=name)
            ax.set_xlabel("sample index")
            ax.set_ylabel("timing")
            ax.set_title(f"{len(names)} file(s) combined")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            total_h = avail_h
        else:
            n = len(names)
            total_h = max(n * self.per_plot_px, avail_h)
            for idx, name in enumerate(names):
                ax = self.fig.add_subplot(n, 1, idx + 1)
                self._plot_series(ax, name)
                ax.set_title(name, fontsize=9)
                ax.set_ylabel("timing")
                ax.grid(True, alpha=0.3)
                if idx == n - 1:
                    ax.set_xlabel("sample index")

        self.fig.set_size_inches(w_in, total_h / self.dpi, forward=True)
        try:
            self.fig.tight_layout()
        except Exception:
            pass

        self.mpl_canvas.draw()
        self.scroll_canvas.itemconfigure(self.win_id, width=avail_w)
        self.scroll_canvas.configure(scrollregion=(0, 0, avail_w, total_h))

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
    scale = detect_scale()
    folder = None
    i = 0
    while i < len(args):
        if args[i] in ("--scale", "-s") and i + 1 < len(args):
            scale = float(args[i + 1])
            i += 2
        else:
            folder = args[i]
            i += 1

    if folder is None:
        folder = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isdir(folder):
        print(f"Not a directory: {folder}", file=sys.stderr)
        sys.exit(1)

    root = tk.Tk()
    apply_widget_scaling(root, scale)
    TimingGUI(root, folder, scale)
    root.mainloop()


if __name__ == "__main__":
    main()