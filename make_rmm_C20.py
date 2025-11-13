#!/usr/bin/env python3
"""
RMM-Compact-20 extractor (supports V_1..V_m*m schema + bar plot)

- Assumes your CSV(.gz) has metadata columns like: Run, Event, Weight, Label
  and matrix columns named V_1, V_2, ..., V_{m*m} (row-major).

- If --event N is provided, also saves a 20-D bar chart:
    compact20_eventN_bar.png
"""
# Extract invariant masses from csv files :
#Mjj (jets–jets, i<j): R{1+i}C{1+j} e.g., leading–subleading jets → R01C02.
#Mbb (bjets–bjets, i<j): R{11+i}C{11+j} e.g., b1–b2 → R11C12.
#Mμμ (muons–muons, i<j): R{21+i}C{21+j} e.g., μ1–μ2 → R21C22.
#Mee (electrons–electrons, i<j):R{31+i}C{31+j} e.g., e1–e2 → R31C32.
#Mγγ (Myy) (photons–photons, i<j):R{41+i}C{41+j} e.g., γ1–γ2 → R41C42.
#Mjb (jet i with b-jet j):row = jets, col = b-jets → R{1+i}C{11+j}; e.g., j1–b1 → R01C11. (The transpose R11C01 is the angle, not mass.)
#Mjμ (jet i with muon j):R{1+i}C{21+j}; e.g., j1–μ1 → R01C21.
#Mbμ (b-jet i with muon j):R{11+i}C{21+j}; e.g., b1–μ1 → R11C21.
#Mjγ (Mjy) (jet i with photon j):R{1+i}C{41+j}; e.g., j1–γ1 → R01C41.

import argparse, numpy as np, pandas as pd, math, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TYPES_FULL = ["Jets", "bJets", "Muons", "Electrons", "Photons"]
MET_IDX = 0

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
    n = len(vcols)
    m = int(round(math.sqrt(n)))
    if m*m != n:
        raise ValueError(f"Expected a perfect square number of V_* columns, got {n}.")
    return m

def build_matrix_from_V(row, vcols, m):
    vals = pd.to_numeric(row[vcols], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return vals.reshape(m, m)   # row-major

def choose_max_types(m, requested):
    # Ensure (m-1) divisible by number of types; pick best <= requested if needed
    if (m-1) % requested == 0:
        return requested
    for k in [5,4,3,2,1]:
        if (m-1) % k == 0 and k <= requested:
            return k
    for k in [5,4,3,2,1]:
        if (m-1) % k == 0:
            return k
    raise ValueError(f"(m-1)={m-1} not divisible by any k in {{1,2,3,4,5}}.")

def type_slices(m, max_types):
    maxN = (m - 1) // max_types
    names = TYPES_FULL[:max_types]
    slices = {}
    for t, name in enumerate(names):
        start = 1 + t*maxN
        slices[name] = slice(start, start + maxN)
    return names, slices, maxN

def frob(block: np.ndarray) -> float:
    return float(np.linalg.norm(block))

def compact20_for_matrix(A: np.ndarray, names, slices):
    # 15 TYPE↔TYPE (unordered with replacement)
    pair_labels, pair_values = [], []
    for i, ti in enumerate(names):
        for j, tj in enumerate(names[i:], start=i):
            si, sj = slices[ti], slices[tj]
            if i == j:
                val = frob(A[si, sj])
            else:
                a = frob(A[si, sj])
                b = frob(A[sj, si])
                val = float(math.sqrt(a*a + b*b))
            pair_labels.append(f"{ti}↔{tj}")
            pair_values.append(val)
    # 5 MET↔TYPE
    met_labels, met_values = [], []
    for ti in names:
        si = slices[ti]
        a = np.linalg.norm(A[0, si])  # MET row segment (MT)
        b = np.linalg.norm(A[si, 0])  # MET col segment (HL)
        met_labels.append(f"MET↔{ti}")
        met_values.append(float(math.sqrt(a*a + b*b)))
    return pair_labels + met_labels, pair_values + met_values

def plot_compact20(labels, values, event_idx, out_png):
    fig = plt.figure(figsize=(12, 5))
    ax = plt.gca()
    x = np.arange(len(values))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Value")
    ax.set_title(f"RMM-Compact-20 — Event {event_idx}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="RMM-Compact-20 extractor for V_* schema")
    ap.add_argument("--csv", required=True, help="Path to CSV or CSV.GZ")
    ap.add_argument("--event", type=int, default=None, help="1-based event index (omit to process ALL events)")
    ap.add_argument("--max_types", type=int, default=5, help="Requested number of types (<=5)")
    ap.add_argument("--prefix", default="V_", help="Prefix for matrix columns (default: V_)")
    ap.add_argument("--id_cols", default="Run,Event,Weight,Label",
                    help="Comma-separated metadata columns to ignore when scanning (optional)")
    ap.add_argument("--out", default=None, help="Output CSV path (default: compact20_all.csv or compact20_event{N}.csv)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, compression="infer")

    # Pick V_* columns (matrix) and confirm m*m
    vcols = pick_v_columns(df, prefix=args.prefix)
    if not vcols:
        raise RuntimeError(f"No columns found with prefix '{args.prefix}'.")
    m = infer_m_from_vcols(vcols)

    # Adjust types if necessary
    max_types = choose_max_types(m, args.max_types)
    if max_types != args.max_types:
        print(f"[info] Requested --max_types={args.max_types}, but (m-1)={m-1} is not divisible by it. Using max_types={max_types}.")

    names, slices, maxN = type_slices(m, max_types)

    # Build header
    labels, _ = compact20_for_matrix(np.zeros((m,m)), names, slices)
    header = ["event"] + labels

    # Choose rows
    if args.event is not None:
        assert 1 <= args.event <= len(df), f"--event must be in [1..{len(df)}]"
        rows = [args.event - 1]
        out_path = args.out or f"compact20_event{args.event}.csv"
    else:
        rows = list(range(len(df)))
        out_path = args.out or "compact20_all.csv"

    # Compute features
    out_records = []
    last_vals = None
    for ridx in rows:
        A = build_matrix_from_V(df.iloc[ridx], vcols, m)
        _, vals = compact20_for_matrix(A, names, slices)
        out_records.append([ridx + 1] + vals)
        last_vals = vals

    out_df = pd.DataFrame(out_records, columns=header)
    out_df.to_csv(out_path, index=False)

    print(f"Schema: V_* flat | m={m} | max_types={max_types} | maxN={maxN} | types={names}")
    print(f"Processed {len(rows)} event(s). Wrote: {out_path}")
    print("\nPreview:\n", out_df.head(min(5, len(out_df))).to_string(index=False))

    # If single event, also plot the bar chart
    if args.event is not None:
        base_dir = os.path.dirname(out_path) or "."
        out_png = os.path.join(base_dir, f"compact20_event{args.event}_bar.png")
        plot_compact20(labels, last_vals, args.event, out_png)
        print(f"Saved bar chart: {out_png}")

if __name__ == "__main__":
    main()

