"""Shared matplotlib style for HIL plots.

Every summary.png uses the same figure size, colors and title format:
current run in one accent color, baseline in grey, bounds as shaded bands.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGSIZE = (10, 6)
DPI = 110
ACCENT = "#1f6fb4"
BASELINE = "#8c8c8c"
BAND = "#cfe3f3"
FAIL = "#c0392b"
STATUS_COLORS = {"pass": "#2e8b57", "fail": FAIL, "inconclusive": "#d4a017", "info": BASELINE}

STYLE = {
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "legend.frameon": False,
}


def summary_figure(test_id, figure_ref, run_id, status, nrows=1, ncols=1, **kwargs):
    """Create a figure with the standard size and title. Returns (fig, axes)."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE, dpi=DPI, squeeze=False, **kwargs)
    title = f"{test_id.upper()} — {figure_ref}   |   {run_id}   |   "
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.text(0.99, 0.985, status.upper(), ha="right", va="top", fontsize=11, weight="bold",
             color=STATUS_COLORS.get(status, "black"))
    for ax in axes.flat:
        for key, val in STYLE.items():
            if key.startswith("axes.spines."):
                ax.spines[key.rsplit(".", 1)[1]].set_visible(val)
        ax.grid(True, alpha=STYLE["grid.alpha"])
    return fig, axes


def plot_current(ax, x, y, label="this run", **kwargs):
    return ax.plot(x, y, color=ACCENT, lw=1.2, label=label, **kwargs)


def plot_baseline(ax, x, y, label="baseline", **kwargs):
    return ax.plot(x, y, color=BASELINE, lw=1.0, label=label, **kwargs)


def shade_band(ax, x, lo, hi, label="bound", **kwargs):
    return ax.fill_between(x, lo, hi, color=BAND, alpha=0.6, lw=0, label=label, **kwargs)


def hband(ax, lo, hi, label="bound"):
    return ax.axhspan(lo, hi, color=BAND, alpha=0.6, lw=0, label=label)


def save(fig, path):
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path
