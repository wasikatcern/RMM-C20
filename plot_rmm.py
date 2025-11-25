#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot a single RMM event as a heatmap, with full per-object labels.

Usage examples
--------------
# Select by Event ID (from the 'Event' or 'event' column)
python plot_rmm.py --csv out/pythia8_X500GeV_HH2bbll_data100percent.csv.gz --event 1

# Or select by 1-based row index
python plot_rmm.py --csv out/pythia8_X500GeV_HH2bbll_data100percent.csv.gz --index 1
"""

import argparse
import os
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Helpers to read RMM from V_* columns
# ----------------------------------------------------------------------

def pick_v_columns(df, prefix="V_"):
    """
    Return V_* columns sorted by numeric suffix (V_1, V_2, ...).
    """
    vcols = []
    for c in df.columns:
        if isinstance(c, str) and c.startswith(prefix):
            try:
                idx = int(c[len(prefix):])
                vcols.append((idx, c))
            except ValueError:
                continue
    vcols.sort(key=lambda x: x[0])
    return [c for _, c in vcols]


def infer_m_from_vcols(vcols):
    """
    Infer matrix size m from number of V_* columns.
    """
    n = len(vcols)
    m = int(round(math.sqrt(n)))
    if m * m != n:
        raise ValueError(f"Expected a perfect square number of V_* columns, got {n}.")
    return m


def build_matrix_from_row(row, vcols, m):
    """
    Rebuild m×m matrix A from flattened V_* columns in a row (row-major).
    """
    vals = pd.to_numeric(row[vcols], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return vals.reshape(m, m)


def load_event_matrix(csv_path, event_id=None, index=None, prefix="V_"):
    """
    Load a single event's RMM matrix from CSV.

    Selection:
      - If index is not None: treat it as 1-based row index.
      - Else if event_id is not None: match the 'Event'/'event' column.
      - Else: use the first row.

    Returns:
      M_full : (m, m) numpy array
      m      : matrix size
      meta   : dict with some metadata if present (Run, Event, Weight, Label)
    """
    df = pd.read_csv(csv_path, compression="infer")
    n_rows = len(df)
    if n_rows == 0:
        raise RuntimeError("Input CSV has no rows.")

    # Pick V_* columns and infer matrix size
    vcols = pick_v_columns(df, prefix=prefix)
    if not vcols:
        raise RuntimeError(f"No V_* columns found with prefix '{prefix}'.")
    m = infer_m_from_vcols(vcols)

    # Choose the row
    if index is not None:
        if index < 1 or index > n_rows:
            raise ValueError(f"--index must be in [1..{n_rows}] (got {index})")
        row = df.iloc[index - 1]
    elif event_id is not None:
        # Try to find a column named 'Event' or 'event'
        event_col = None
        for c in df.columns:
            if c.lower() == "event":
                event_col = c
                break
        if event_col is None:
            raise KeyError("No 'Event' or 'event' column found in the CSV.")
        sel = df[df[event_col] == event_id]
        if sel.empty:
            raise ValueError(f"No row found with {event_col} == {event_id}")
        row = sel.iloc[0]
    else:
        # default: first row
        row = df.iloc[0]

    # Build matrix
    M_full = build_matrix_from_row(row, vcols, m)

    # Some metadata (if available)
    meta = {}
    for key in ["Run", "run", "Event", "event", "Weight", "weight", "Label", "label"]:
        if key in df.columns:
            meta[key] = row[key]

    return M_full, m, meta


# ----------------------------------------------------------------------
# Helpers for labels, block structure and plotting
# ----------------------------------------------------------------------

def build_full_labels(m):
    """
    Build per-index labels:
      0          : MET
      1..10      : j1..j10
      11..20     : b1..b10
      21..30     : μ1..μ10
      31..40     : e1..e10
      41..50     : γ1..γ10
    Assumes m = 51 (or at least 1 + 5*10).
    """
    labels = []
    labels.append("MET")

    maxN = (m - 1) // 5  # usually 10

    # Jets j1..jN
    for i in range(maxN):
        labels.append(f"J{i+1}")

    # b-jets b1..bN
    for i in range(maxN):
        labels.append(f"b{i+1}")

    # muons μ1..μN
    for i in range(maxN):
        labels.append(f"μ{i+1}")

    # electrons e1..eN
    for i in range(maxN):
        labels.append(f"e{i+1}")

    # photons γ1..γN
    for i in range(maxN):
        labels.append(f"γ{i+1}")

    # If m is slightly different, truncate/pad
    if len(labels) > m:
        labels = labels[:m]
    elif len(labels) < m:
        labels.extend([f"x{i}" for i in range(len(labels), m)])

    return labels


def compute_block_boundaries(m):
    """
    Compute block boundaries around MET and each object-type block.
    """
    maxN = (m - 1) // 5
    boundaries = [0, 1]  # MET at index 0, then start of Jets
    for i in range(5):
        boundaries.append(1 + (i + 1) * maxN)
    # keep only those within [0, m]
    boundaries = sorted(set(b for b in boundaries if 0 <= b <= m))
    return boundaries


def plot_rmm_matrix(M, m, meta, out_path):
    """
    Plot full RMM matrix as a heatmap with:
      - per-object labels on both axes (j1.., b1.., μ1.., e1.., γ1..)
      - labels on bottom and top for x-axis
      - empty cells (value == 0) shown in white
      - block boundaries drawn in white lines
    """
    # Mask zeros and show them as white
    data = np.array(M)
    mask = (data == 0.0)
    M_plot = np.ma.masked_where(mask, data)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")  # masked (zero) cells -> white

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(M_plot, origin="upper", interpolation="nearest",
                   aspect="equal", cmap=cmap)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("RMM entry value")

    # Block boundaries
    boundaries = compute_block_boundaries(m)
    for b in boundaries:
        # horizontal and vertical lines at b
        ax.axhline(b - 0.5, color="white", linewidth=0.8)
        ax.axvline(b - 0.5, color="white", linewidth=0.8)

    # Full per-index labels
    labels = build_full_labels(m)
    positions = np.arange(m)

    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)

    ax.set_xticks(np.arange(-0.5, m, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, m, 1), minor=True)
    ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=0.3)

    # Show x labels on top AND bottom
    ax.tick_params(axis="x", labeltop=True, labelbottom=True, top=True, bottom=True)

    # Title
    title = "RMM matrix"

    # Normalize event ID (convert float → int)
    event_id = None
    if "Event" in meta:
        event_id = int(meta["Event"])
    elif "event" in meta:
        event_id = int(meta["event"])

    if event_id is not None:
        title += f" (for Event# {event_id})"
    ax.set_title(title)

    # === Add diagonal dashed line ===
    ax.plot([0, m-1], [0, m-1],
        linestyle="--",
        color="black",
        linewidth=1.0,
        alpha=0.7)


    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved RMM heatmap to: {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot a single RMM event as a heatmap.")
    parser.add_argument("--csv", required=True, help="Path to CSV or CSV.GZ with RMM data.")
    parser.add_argument("--event", type=int, default=None,
                        help="Event ID to select (from 'Event'/'event' column).")
    parser.add_argument("--index", type=int, default=None,
                        help="1-based row index to select (if given, overrides --event).")
    parser.add_argument("--prefix", default="V_",
                        help="Prefix for matrix columns (default: V_).")
    parser.add_argument("--out", default=None,
                        help="Output PDF/PNG path (default: derived from CSV name).")
    args = parser.parse_args()

    if args.index is not None and args.event is not None:
        print("[info] Both --index and --event given; using --index and ignoring --event.")

    M_full, m, meta = load_event_matrix(
        args.csv,
        event_id=None if args.index is not None else args.event,
        index=args.index,
        prefix=args.prefix
    )

    # Output path
    if args.out is not None:
        out_path = args.out
    else:
        base = os.path.splitext(os.path.basename(args.csv))[0]
        suffix = ""
        if args.index is not None:
            suffix = f"_idx{args.index}"
        elif args.event is not None:
            suffix = f"_evt{args.event}"
        out_path = f"{base}_rmm_matrix{suffix}.pdf"

    plot_rmm_matrix(M_full, m, meta, out_path)


if __name__ == "__main__":
    main()
