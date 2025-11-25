#***************************************************************************
# *  Visualize and plot an event through the RMM matrix *
#***************************************************************************

#!/usr/bin/env python3
# plot_rmm_event.py
# Read Map2RMM CSV, select one event, and plot the (optionally cropped) per-event RMM as a PNG.
# Default: event 10, MET + 10 jets, 10 bjets, 10 muons, 10 electrons, 10 photons
# Run from command line: 
#Run as : python plot_rmm.py --csv X2hh.csv

# Custom setting: MET, 10 jets, 8 bjets, 7 muons, 5 electrons, 6 photons
#Run as : python plot_rmm_event.py --csv X2hh.csv --event 42 --nj 10 --nb 8 --nm 7 --ne 5 --ng 6

# Choose by row index instead of event number, and adjust log range
#Run as : python plot_rmm_event.py --csv X2hh.csv --index 7 --nj 6 --nb 6 --nm 2 --ne 2 --ng 3 --vmin 1e-8 --vmax 1e-2


import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
import math

# ---- Default view configuration (you can change these) ----
DEFAULT_NJ = 7 # 10  # jets to show
DEFAULT_NB = 7 # 10  # bjets to show
DEFAULT_NM = 7 # 10  # muons to show
DEFAULT_NE = 7 # 10  # electrons to show
DEFAULT_NG = 7 # 10  # photons to show
DEFAULT_EVENT = 1

def build_full_labels(m_size: int):
    """Return the full per-slot labels in the canonical Map2RMM order."""
    if (m_size - 1) % 5 != 0 or m_size < 6:
        raise ValueError(f"Unexpected matrix size {m_size}; expected 1 + 5*maxNumber.")
    max_number = (m_size - 1) // 5
    labels = ["MET"]
    labels += [f"j{i}" for i in range(1, max_number + 1)]
    labels += [f"b{i}" for i in range(1, max_number + 1)]
    labels += [f"μ{i}" for i in range(1, max_number + 1)]
    labels += [f"e{i}" for i in range(1, max_number + 1)]
    labels += [f"γ{i}" for i in range(1, max_number + 1)]
    return labels, max_number

def selection_indices(nj:int, nb:int, nm:int, ne:int, ng:int, max_number:int):
    """Given desired counts and the file's max_number, build the row/col indices to keep."""
    nj = max(0, min(nj, max_number))
    nb = max(0, min(nb, max_number))
    nm = max(0, min(nm, max_number))
    ne = max(0, min(ne, max_number))
    ng = max(0, min(ng, max_number))

    off_j  = 1
    off_b  = off_j + max_number
    off_mu = off_b + max_number
    off_e  = off_mu + max_number
    off_g  = off_e + max_number

    idx = [0]  # MET
    idx += list(range(off_j,  off_j  + nj))
    idx += list(range(off_b,  off_b  + nb))
    idx += list(range(off_mu, off_mu + nm))
    idx += list(range(off_e,  off_e  + ne))
    idx += list(range(off_g,  off_g  + ng))
    return idx, (nj, nb, nm, ne, ng)

def cropped_labels(nj, nb, nm, ne, ng):
    labels = ["MET"]
    labels += [f"j{i}" for i in range(1, nj+1)]
    labels += [f"b{i}" for i in range(1, nb+1)]
    labels += [f"μ{i}" for i in range(1, nm+1)]
    labels += [f"e{i}" for i in range(1, ne+1)]
    labels += [f"γ{i}" for i in range(1, ng+1)]
    return labels

def load_event_matrix(csv_path: str, event_id: int | None, row_index: int | None):
    """Return (matrix, m_size, meta_dict) for the chosen event."""
    df = pd.read_csv(csv_path)
    if event_id is not None:
        sel = df[df["event"] == event_id]
        if sel.empty:
            raise ValueError(f"Event {event_id} not found in CSV.")
        row = sel.iloc[0]
    elif row_index is not None:
        if row_index < 0 or row_index >= len(df):
            raise IndexError(f"Row index {row_index} out of range [0, {len(df)-1}].")
        row = df.iloc[row_index]
    else:
        sel = df[df["event"] == DEFAULT_EVENT]
        row = sel.iloc[0] if not sel.empty else df.iloc[0]

    mat_vals = row.filter(regex=r"^R\d{2}C\d{2}$").to_numpy(dtype=float)
    m_size = int(np.sqrt(mat_vals.size))
    if m_size * m_size != mat_vals.size:
        raise RuntimeError("Matrix columns are not a perfect square; header mismatch?")
    M = mat_vals.reshape(m_size, m_size)

    meta = {"run": int(row["run"]), "event": int(row["event"])}  # weight omitted by request
    return M, m_size, meta

def sci_as_pow10(x, pos=None):
    """Formatter for colorbar ticks: 10^{n} with superscript."""
    if x <= 0 or not np.isfinite(x):
        return ""
    n = int(round(math.log10(x)))
    # Only format exact powers of 10 nicely; otherwise fallback to scientific
    if abs(x - 10**n) / x < 1e-9:
        return rf"$10^{{{n}}}$"
    return rf"$10^{{{math.log10(x):.1f}}}$"

def plot_rmm(M: np.ndarray, labels: list[str], meta: dict, out_png: str,
             vmin: float | None = None, vmax: float | None = None, dpi: int = 220):
    """Render a log-heatmap of the RMM and save to PNG.
       Y-axis is flipped (origin='upper'): row 0 (MET) is at the top."""
    m_size = M.shape[0]

    # Log range
    finite = M[np.isfinite(M) & (M > 0)]
    auto_vmin = np.percentile(finite, 5) if finite.size else 1e-8
    auto_vmax = np.percentile(finite, 99.5) if finite.size else 1.0
    vmin = vmin if vmin is not None else max(1e-8, float(auto_vmin))
    vmax = vmax if vmax is not None else max(vmin * 10, float(auto_vmax))

    fig = plt.figure(figsize=(11.8, 10.2))
    ax = plt.gca()

    im = ax.imshow(
        M,
        origin="upper",                   # top row is row 0 (MET)
        norm=LogNorm(vmin=vmin, vmax=vmax),
        interpolation="nearest",
        aspect="equal",
        cmap="viridis",
    )

    # Ticks and labels on BOTH bottom and top
    ax.set_xticks(range(m_size))
    ax.set_yticks(range(m_size))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=9)
    ax.tick_params(axis="x", which="both", top=True, labeltop=True, bottom=True, labelbottom=True)

    # Minor grid
    ax.set_xticks(np.arange(-0.5, m_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, m_size, 1), minor=True)
    ax.grid(which="minor", color="k", linestyle=":", linewidth=0.25)
    ax.tick_params(which="both", length=0)

    # Diagonal (white dashed)
    ax.plot([-0.5, m_size - 0.5], [-0.5, m_size - 0.5], color="w", ls="--", lw=1)

    # Title (only Event no.)
    ax.set_title(f"RMM for Event# {meta['event']}", fontsize=13, pad=10)

    # Colorbar with 10^n formatting
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Choose decade ticks within [vmin, vmax]
    decades = np.arange(math.floor(math.log10(vmin)), math.ceil(math.log10(vmax)) + 1)
    ticks = 10.0 ** decades
    ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]
    if ticks.size:
        cbar.set_ticks(ticks)
    cbar.formatter = FuncFormatter(sci_as_pow10)
    cbar.update_ticks()

    plt.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)

def main():
    p = argparse.ArgumentParser(
        description="Plot a single-event RMM from a Map2RMM CSV (MET at top), with configurable slots per object type."
    )
    p.add_argument("--csv", required=True, help="Input CSV path")
    p.add_argument("--out", default=None, help="Output PNG path (default: rmm_event_<event>_nj..png)")
    # event selection: either by 'event' value or by 0-based row index
    p.add_argument("--event", type=int, default=DEFAULT_EVENT, help=f"Event number to plot (default: {DEFAULT_EVENT})")
    p.add_argument("--index", type=int, default=None, help="Row index to plot (overrides --event)")
    # slot counts to visualize
    p.add_argument("--nj", type=int, default=DEFAULT_NJ, help=f"jets to show (default: {DEFAULT_NJ})")
    p.add_argument("--nb", type=int, default=DEFAULT_NB, help=f"bjets to show (default: {DEFAULT_NB})")
    p.add_argument("--nm", type=int, default=DEFAULT_NM, help=f"muons to show (default: {DEFAULT_NM})")
    p.add_argument("--ne", type=int, default=DEFAULT_NE, help=f"electrons to show (default: {DEFAULT_NE})")
    p.add_argument("--ng", type=int, default=DEFAULT_NG, help=f"photons to show (default: {DEFAULT_NG})")
    # optional display tuning
    p.add_argument("--vmin", type=float, default=None, help="Log color min (default: auto)")
    p.add_argument("--vmax", type=float, default=None, help="Log color max (default: auto)")
    args = p.parse_args()

    # Load full matrix
    M_full, m_size, meta = load_event_matrix(args.csv, args.event if args.index is None else None, args.index)
    full_labels, max_number = build_full_labels(m_size)

    # Build indices for the requested view, clipped to what's available
    idx, (nj, nb, nm, ne, ng) = selection_indices(args.nj, args.nb, args.nm, args.ne, args.ng, max_number)

    # Crop the matrix and labels
    M = M_full[np.ix_(idx, idx)]
    labels = cropped_labels(nj, nb, nm, ne, ng)

    # Output name
    suffix = f"nj{nj}_nb{nb}_nm{nm}_ne{ne}_ng{ng}"
    #out_png = args.out or f"rmm_event_{meta['event']}_{suffix}.png"
    out_png = args.out or f"rmm_event_{meta['event']}.png"

    plot_rmm(M, labels, meta, out_png, vmin=args.vmin, vmax=args.vmax)
    print(f"Saved {out_png}  (cropped matrix {M.shape[0]}×{M.shape[1]} from full {m_size}×{m_size})")

if __name__ == "__main__":
    main()
