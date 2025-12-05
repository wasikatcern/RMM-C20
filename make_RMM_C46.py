#!/usr/bin/env python3
"""
RMM-Compact-46 extractor for V_* schema

Builds a 46-dimensional RMM-C46 representation:

  1  MET term:
       MET

  5  ET terms (diagonal of same-type blocks):
       ET_jets, ET_bjets, ET_muons, ET_electrons, ET_photons

  5  Transverse-mass (T) terms (from mT zones in the first row):
       T_jets, T_bjets, T_muons, T_electrons, T_photons

  5  Lorentz-vector (L) terms (from hL zones in the first column):
       L_jets, L_bjets, L_muons, L_electrons, L_photons

  15 rapidity (h) terms (lower blocks / triangles):
       h_j_j, h_bj_j, h_bj_bj,
       h_mu_j, h_mu_bj, h_mu_mu,
       h_e_j, h_e_bj, h_e_mu, h_e_e,
       h_y_j, h_y_bj, h_y_mu, h_y_e, h_y_y

  15 mass (m) terms (upper blocks / triangles):
       m_j_j, m_bj_j, m_bj_bj,
       m_mu_j, m_mu_bj, m_mu_mu,
       m_e_j, m_e_bj, m_e_mu, m_e_e,
       m_y_j, m_y_bj, m_y_mu, m_y_e, m_y_y

Two aggregation styles are supported:

  --style add   : simple sum of the values in each zone
  --style frob  : Frobenius / L2 norm over the values in each zone

NOTE:  T and L zones are built purely from the relevant cells in the first
row / first column.  We never mix them with MET[0,0] or combine row+column.
MET is used only as the single scalar "MET" feature.
"""

import argparse
import math
import os
import re

import numpy as np
import pandas as pd


TYPES = ["Jets", "bJets", "Muons", "Electrons", "Photons"]
# shorthand for labels
SHORT = {"Jets": "j", "bJets": "bj", "Muons": "mu", "Electrons": "e", "Photons": "y"}


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def pick_v_columns(df, prefix="V_"):
    """Return V_* columns sorted by numeric suffix (V_1, V_2, ...)."""
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    pairs = []
    for c in df.columns:
        if isinstance(c, str):
            m = pat.match(c)
            if m:
                pairs.append((int(m.group(1)), c))
    pairs.sort(key=lambda x: x[0])
    return [c for _, c in pairs]


def infer_m_from_vcols(vcols):
    """Infer matrix size m from number of V_* columns."""
    n = len(vcols)
    m = int(round(math.sqrt(n)))
    if m * m != n:
        raise ValueError(f"Expected a perfect square number of V_* columns, got {n}.")
    return m


def build_matrix_from_V(row, vcols, m):
    """Rebuild m×m matrix A from flattened V_* columns in a row (row-major)."""
    vals = pd.to_numeric(row[vcols], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return vals.reshape(m, m)


def choose_max_types(m, requested):
    """
    Ensure (m-1) divisible by number of types; pick best <= requested if needed.
    """
    if (m - 1) % requested == 0:
        return requested
    for k in [5, 4, 3, 2, 1]:
        if (m - 1) % k == 0 and k <= requested:
            return k
    for k in [5, 4, 3, 2, 1]:
        if (m - 1) % k == 0:
            return k
    raise ValueError(f"(m-1)={m-1} not divisible by any k in {{1,2,3,4,5}}.")


def type_slices(m, max_types):
    """
    Compute row/col index slices for each object type (Jets, bJets, ...).

    Layout is:

        index 0     : MET
        1..maxN     : type 0 (Jets)
        maxN+1..2*maxN : type 1 (bJets)
        ...

    Returns:
      names  : list of type names (subset of TYPES)
      slices : dict(name -> slice of indices)
      maxN   : number of slots per type
    """
    maxN = (m - 1) // max_types
    names = TYPES[:max_types]
    slices = {}
    for t, name in enumerate(names):
        start = 1 + t * maxN
        slices[name] = slice(start, start + maxN)
    return names, slices, maxN


def aggregate(block, style):
    """Aggregate a block according to style: 'add' or 'frob'."""
    arr = np.asarray(block, dtype=float)
    if style == "add":
        return float(np.nansum(arr))
    elif style == "frob":
        return float(np.linalg.norm(arr))
    else:
        raise ValueError(f"Unknown style: {style}")


# ----------------------------------------------------------------------
# C46 construction
# ----------------------------------------------------------------------

def compute_c46_for_matrix(A, names, slices, style="add"):
    """
    Compute the 46 RMM-C46 features for a single event matrix A.

    Order of outputs:

      1) MET
      2-6)  ET_<type>
      7-11) T_<type>
      12-16)L_<type>
      17-31)h_*_* (15 order, see below)
      32-46)m_*_* (same 15 order)

    The 15 (type,type) order follows:
       (j,j),
       (bj,j), (bj,bj),
       (mu,j), (mu,bj), (mu,mu),
       (e,j), (e,bj), (e,mu), (e,e),
       (y,j), (y,bj), (y,mu), (y,e), (y,y)
    """

    feats = []
    labels = []

    # ---------- 1: MET ----------
    MET_val = float(A[0, 0])
    feats.append(MET_val)
    labels.append("MET")

    # ---------- 5: ET terms (diagonals of same-type blocks) ----------
    for t in names:
        s = slices[t]
        B = A[s, s]
        diag = np.diag(B)
        feats.append(aggregate(diag, style))
        labels.append(f"ET_{SHORT[t]}")

    # ---------- 5: T terms (mT). Use only first ROW segments, no mixing with MET[0,0] ----------
    #   For each type, we take A[0, slice(type)] as the mT zone and aggregate.
    for t in names:
        s = slices[t]
        row_segment = A[0, s]   # first row, type columns
        feats.append(aggregate(row_segment, style))
        labels.append(f"T_{SHORT[t]}")

    # ---------- 5: L terms (hL). Use only first COLUMN segments ----------
    #   For each type, we take A[ slice(type), 0 ] as the hL zone.
    for t in names:
        s = slices[t]
        col_segment = A[s, 0]   # type rows, first column
        feats.append(aggregate(col_segment, style))
        labels.append(f"L_{SHORT[t]}")

    # ---------- 15 h-terms and 15 m-terms ----------
    # Following the specific order requested:
    #  (j,j),
    #  (bj,j), (bj,bj),
    #  (mu,j), (mu,bj), (mu,mu),
    #  (e,j), (e,bj), (e,mu), (e,e),
    #  (y,j), (y,bj), (y,mu), (y,e), (y,y)

    order_pairs = [
        ("Jets", "Jets"),
        ("bJets", "Jets"), ("bJets", "bJets"),
        ("Muons", "Jets"), ("Muons", "bJets"), ("Muons", "Muons"),
        ("Electrons", "Jets"), ("Electrons", "bJets"),
        ("Electrons", "Muons"), ("Electrons", "Electrons"),
        ("Photons", "Jets"), ("Photons", "bJets"),
        ("Photons", "Muons"), ("Photons", "Electrons"),
        ("Photons", "Photons"),
    ]

    # --- First the 15 rapidity (h) terms ---
    for ti, tj in order_pairs:
        si, sj = slices[ti], slices[tj]
        if ti == tj:
            # same-type: strictly lower triangle of block (i>j)
            B = A[si, sj]
            lower = np.tril(B, k=-1)
            feats.append(aggregate(lower, style))
            labels.append(f"h_{SHORT[ti]}_{SHORT[tj]}")
        else:
            # cross-type: lower-left block (tj rows, ti cols)
            B_h = A[slices[tj], slices[ti]]
            feats.append(aggregate(B_h, style))
            labels.append(f"h_{SHORT[ti]}_{SHORT[tj]}")

    # --- Then the 15 mass (m) terms ---
    for ti, tj in order_pairs:
        si, sj = slices[ti], slices[tj]
        if ti == tj:
            # same-type: strictly upper triangle of block (i<j)
            B = A[si, sj]
            upper = np.triu(B, k=1)
            feats.append(aggregate(upper, style))
            labels.append(f"m_{SHORT[ti]}_{SHORT[tj]}")
        else:
            # cross-type: upper-right block (ti rows, tj cols)
            B_m = A[si, sj]
            feats.append(aggregate(B_m, style))
            labels.append(f"m_{SHORT[ti]}_{SHORT[tj]}")

    assert len(feats) == 46
    assert len(labels) == 46
    return labels, feats


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RMM-Compact-46 extractor (C46-add / C46-frob) for V_* schema"
    )
    parser.add_argument("--csv", required=True, help="Input CSV/CSV.GZ with V_* columns")
    parser.add_argument("--prefix", default="V_", help="Prefix for matrix columns (default: V_)")
    parser.add_argument("--max_types", type=int, default=5, help="Number of object types (<=5)")
    parser.add_argument(
        "--style",
        choices=["add", "frob"],
        default="frob",
        help="Aggregation style: 'add' for sums, 'frob' for Frobenius/L2 norms",
    )
    parser.add_argument(
        "--event",
        type=int,
        default=None,
        help="1-based event index (if set, only that event is processed)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: rmm_C46_<style>_all.csv or ..._eventN.csv)",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.csv, compression="infer")
    vcols = pick_v_columns(df, prefix=args.prefix)
    if not vcols:
        raise RuntimeError(f"No V_* columns with prefix '{args.prefix}' found.")

    m = infer_m_from_vcols(vcols)
    max_types = choose_max_types(m, args.max_types)
    if max_types != args.max_types:
        print(
            f"[info] Requested max_types={args.max_types}, but (m-1)={m-1} "
            f"is not divisible by it. Using max_types={max_types}."
        )

    names, slices, maxN = type_slices(m, max_types)
    print(f"Matrix size m={m}, maxN={maxN}, types={names}")

    # Build header via a dummy zero matrix
    label_list, _ = compute_c46_for_matrix(np.zeros((m, m)), names, slices, style=args.style)
    header = ["event"] + label_list

    # Row selection
    if args.event is not None:
        assert 1 <= args.event <= len(df), f"--event must be in [1..{len(df)}]"
        rows = [args.event - 1]
        out_path = args.out or f"rmm_C46_{args.style}_event{args.event}.csv"
    else:
        rows = list(range(len(df)))
        out_path = args.out or f"rmm_C46_{args.style}_all.csv"

    # Compute features
    records = []
    for ridx in rows:
        A = build_matrix_from_V(df.iloc[ridx], vcols, m)
        _, vals = compute_c46_for_matrix(A, names, slices, style=args.style)
        records.append([ridx + 1] + vals)

    out_df = pd.DataFrame(records, columns=header)
    out_df.to_csv(out_path, index=False)

    print(f"Wrote {len(records)} events to: {out_path}")
    print("Preview:\n", out_df.head(min(5, len(out_df))).to_string(index=False))


if __name__ == "__main__":
    main()

