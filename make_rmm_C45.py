#!/usr/bin/env python3
"""
RMM-Compact extractor (20-D, 45-D or 60-D) for V_* schema

- Assumes your CSV(.gz) has metadata columns like: Run, Event, Weight, Label
  and matrix columns named V_1, V_2, ..., V_{m*m} (row-major).

- If --event N is provided, also saves a bar chart:
    compact{MODE}_eventN_bar.png

MODE:
  20  -> original RMM-C20 (20 features)
  45  -> [mass,y,Et] for same-type; [mass,y] for cross-type + MET↔TYPE (45 features)
  60  -> extended [mass, y, Et] per block (60 features)
  
# full 60-D
python make_rmm_C60.py --csv your_rmm.csv.gz --mode 60

# compressed 45-D (no useless zero ETs for cross-types & MET↔TYPE)
python make_rmm_C60.py --csv your_rmm.csv.gz --mode 45

# original 20-D
python make_rmm_C60.py --csv your_rmm.csv.gz --mode 20

"""

import argparse, numpy as np, pandas as pd, math, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TYPES_FULL = ["Jets", "bJets", "Muons", "Electrons", "Photons"]
MET_IDX = 0


# ----------------- utilities -----------------

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

      row/col 0 : MET
      1..maxN  : type 0 (Jets)
      maxN+1..2*maxN : type 1 (bJets)
      ...

    Returns:
      names  : list of type names (subset of TYPES_FULL)
      slices : dict name -> slice of indices
      maxN   : number of slots per type
    """
    maxN = (m - 1) // max_types
    names = TYPES_FULL[:max_types]
    slices = {}
    for t, name in enumerate(names):
        start = 1 + t * maxN
        slices[name] = slice(start, start + maxN)
    return names, slices, maxN


def frob(block: np.ndarray) -> float:
    """Frobenius norm convenience wrapper."""
    return float(np.linalg.norm(block))


# ----------------- original 20-D compact -----------------

def compact20_for_matrix(A: np.ndarray, names, slices):
    """
    Original RMM-C20:

      - 15 TYPE↔TYPE (unordered with replacement):
            ||block||_F   (same-type)
            sqrt(||A_ij||^2 + ||A_ji||^2)  (different-type)

      - 5 MET↔TYPE:
            sqrt( ||MET row segment||^2 + ||MET column segment||^2 )
    """
    # 15 TYPE↔TYPE
    pair_labels, pair_values = [], []
    for i, ti in enumerate(names):
        for j, tj in enumerate(names[i:], start=i):
            si, sj = slices[ti], slices[tj]
            if i == j:
                val = frob(A[si, sj])
            else:
                a = frob(A[si, sj])
                b = frob(A[sj, si])
                val = float(math.sqrt(a * a + b * b))
            pair_labels.append(f"{ti}↔{tj}")
            pair_values.append(val)

    # 5 MET↔TYPE
    met_labels, met_values = [], []
    for ti in names:
        si = slices[ti]
        a = np.linalg.norm(A[0, si])  # MET row segment
        b = np.linalg.norm(A[si, 0])  # MET col segment
        met_labels.append(f"MET↔{ti}")
        met_values.append(float(math.sqrt(a * a + b * b)))

    return pair_labels + met_labels, pair_values + met_values


# ----------------- extended 60-D compact -----------------

def compact60_for_matrix(A: np.ndarray, names, slices):
    """
    Extended compact representation (60-D) aligned with the RMM physics:

      For each TYPE↔TYPE pair we build 3 features:
        [mass, y, Et]
      and 3 for each MET↔TYPE, giving 60 = 3*(15 + 5).

    Conventions (following the RMM definition):

      SAME-TYPE (Jets↔Jets, bJets↔bJets, ...):
        Let B = A[ti, ti] (square block).
        - Upper triangle (i<j) holds m(i_n, i_k)/√s       -> "mass"
        - Lower triangle (i>j) holds h(i_n, i_k)          -> "y"
        - Diagonal holds eT and δeT                       -> "Et"

        mass = ||upper triangle of B (k=+1)||_F
        y    = ||lower triangle of B (k=-1)||_F
        Et   = ||diag(B)||_2

      DIFFERENT-TYPE (Jets↔bJets, Jets↔Muons, ...):
        - A[ti, tj] (rows ti, cols tj): m(ti_n, tj_k)/√s  -> "mass"
        - A[tj, ti] (rows tj, cols ti): h(ti_n, tj_k)     -> "y"
        - There is no pair-specific Et term in the RMM,
          so we set Et = 0.0 but keep a slot for symmetry.

      MET↔TYPE:
        - First row segment A[0, si]: mT(t_n)/√s         -> "mass"
        - First column segment A[si, 0]: hL(t_n)         -> "y"
        - No Et entry here; set Et = 0.0.
    """
    labels = []
    values = []

    # ----- TYPE↔TYPE pairs -----
    for i, ti in enumerate(names):
        for j, tj in enumerate(names[i:], start=i):
            si, sj = slices[ti], slices[tj]

            if i == j:
                # same-type square block
                B = A[si, sj]  # shape (N, N)

                # Upper triangle (excluding diagonal): masses m(i_n, i_k)
                upper = np.triu(B, k=1)
                mass = float(np.linalg.norm(upper))

                # Lower triangle (excluding diagonal): rapidity vars h(i_n, i_k)
                lower = np.tril(B, k=-1)
                y = float(np.linalg.norm(lower))

                # Diagonal: Et and δEt
                diag = np.diag(B)
                Et = float(np.linalg.norm(diag))

            else:
                # different-type: top-right vs bottom-left blocks
                B_mass = A[si, sj]   # m(ti, tj)
                B_y    = A[sj, si]   # h(ti, tj)

                mass = float(np.linalg.norm(B_mass))
                y    = float(np.linalg.norm(B_y))
                Et   = 0.0  # no pair-specific Et term in RMM for cross-types

            labels.extend([
                f"{ti}↔{tj} [mass]",
                f"{ti}↔{tj} [y]",
                f"{ti}↔{tj} [Et]",
            ])
            values.extend([mass, y, Et])

    # ----- MET↔TYPE -----
    for ti in names:
        si = slices[ti]
        row = A[0, si]  # MET → type: mT
        col = A[si, 0]  # type → MET: hL

        mass = float(np.linalg.norm(row))  # mT channel
        y    = float(np.linalg.norm(col))  # longitudinal / rapidity channel
        Et   = 0.0                         # no dedicated Et here

        labels.extend([
            f"MET↔{ti} [mass]",
            f"MET↔{ti} [y]",
            f"MET↔{ti} [Et]",
        ])
        values.extend([mass, y, Et])

    return labels, values


# ----------------- 45-D compact (drop ET for cross-type & MET↔TYPE) -----------------

def compact45_for_matrix(A: np.ndarray, names, slices):
    """
    45-D compact representation:

      - SAME-TYPE (5 blocks): keep [mass, y, Et]  -> 5 * 3 = 15
      - CROSS-TYPE TYPE↔TYPE (10 blocks): keep [mass, y] only (drop Et) -> 10 * 2 = 20
      - MET↔TYPE (5 blocks): keep [mass, y] only (drop Et) -> 5 * 2 = 10

      Total: 15 + 20 + 10 = 45 features.
    """
    labels = []
    values = []

    # ----- TYPE↔TYPE pairs -----
    for i, ti in enumerate(names):
        for j, tj in enumerate(names[i:], start=i):
            si, sj = slices[ti], slices[tj]

            if i == j:
                # same-type square block
                B = A[si, sj]

                upper = np.triu(B, k=1)      # masses
                lower = np.tril(B, k=-1)     # rapidity vars
                diag  = np.diag(B)           # Et / δEt

                mass = float(np.linalg.norm(upper))
                y    = float(np.linalg.norm(lower))
                Et   = float(np.linalg.norm(diag))

                labels.extend([
                    f"{ti}↔{tj} [mass]",
                    f"{ti}↔{tj} [y]",
                    f"{ti}↔{tj} [Et]",
                ])
                values.extend([mass, y, Et])
            else:
                # cross-type TYPE↔TYPE: keep only [mass,y]
                B_mass = A[si, sj]   # m(ti,tj)
                B_y    = A[sj, si]   # h(ti,tj)

                mass = float(np.linalg.norm(B_mass))
                y    = float(np.linalg.norm(B_y))

                labels.extend([
                    f"{ti}↔{tj} [mass]",
                    f"{ti}↔{tj} [y]",
                ])
                values.extend([mass, y])

    # ----- MET↔TYPE (keep only [mass,y]) -----
    for ti in names:
        si = slices[ti]
        row = A[0, si]  # MET → type: mT
        col = A[si, 0]  # type → MET: hL

        mass = float(np.linalg.norm(row))
        y    = float(np.linalg.norm(col))

        labels.extend([
            f"MET↔{ti} [mass]",
            f"MET↔{ti} [y]",
        ])
        values.extend([mass, y])

    return labels, values


# ----------------- plotting -----------------

def plot_compact_vector(labels, values, event_idx, out_png, title_prefix="RMM-Compact"):
    fig = plt.figure(figsize=(max(12, 0.45 * len(values)), 5))
    ax = plt.gca()
    x = np.arange(len(values))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Value")
    ax.set_title(f"{title_prefix} — Event {event_idx}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser(description="RMM-Compact extractor for V_* schema (20-D, 45-D or 60-D)")
    ap.add_argument("--csv", required=True, help="Path to CSV or CSV.GZ")
    ap.add_argument("--event", type=int, default=None,
                    help="1-based event index (omit to process ALL events)")
    ap.add_argument("--max_types", type=int, default=5,
                    help="Requested number of types (<=5)")
    ap.add_argument("--prefix", default="V_",
                    help="Prefix for matrix columns (default: V_)")
    ap.add_argument("--id_cols", default="Run,Event,Weight,Label",
                    help="Comma-separated metadata columns to ignore when scanning (optional)")
    ap.add_argument("--out", default=None,
                    help="Output CSV path (default: compact{MODE}_all.csv or compact{MODE}_event{N}.csv)")
    ap.add_argument("--mode", choices=["20", "45", "60"], default="60",
                    help="Compact mode: '20' for RMM-C20, '45' for reduced [mass,y] cross-type, '60' for full [mass,y,Et]")
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
        print(f"[info] Requested --max_types={args.max_types}, but (m-1)={m-1} "
              f"is not divisible by it. Using max_types={max_types}.")

    names, slices, maxN = type_slices(m, max_types)

    # Build header
    if args.mode == "20":
        labels, _ = compact20_for_matrix(np.zeros((m, m)), names, slices)
        title_prefix = "RMM-Compact-20"
        default_out = "compact20_all.csv"
        tag = "compact20"
    elif args.mode == "60":
        labels, _ = compact60_for_matrix(np.zeros((m, m)), names, slices)
        title_prefix = "RMM-Compact-60"
        default_out = "compact60_all.csv"
        tag = "compact45"
    else:
        labels, _ = compact45_for_matrix(np.zeros((m, m)), names, slices)
        title_prefix = "RMM-Compact-45"
        default_out = "compact45_all.csv"
        tag = "compact60"

    header = ["event"] + labels

    # Choose rows
    if args.event is not None:
        assert 1 <= args.event <= len(df), f"--event must be in [1..{len(df)}]"
        rows = [args.event - 1]
        out_path = args.out or f"{tag}_event{args.event}.csv"
    else:
        rows = list(range(len(df)))
        out_path = args.out or default_out

    # Compute features
    out_records = []
    last_vals = None
    for ridx in rows:
        A = build_matrix_from_V(df.iloc[ridx], vcols, m)
        if args.mode == "20":
            _, vals = compact20_for_matrix(A, names, slices)
        elif args.mode == "45":
            _, vals = compact45_for_matrix(A, names, slices)
        else:
            _, vals = compact60_for_matrix(A, names, slices)
        out_records.append([ridx + 1] + vals)
        last_vals = vals

    out_df = pd.DataFrame(out_records, columns=header)
    out_df.to_csv(out_path, index=False)

    print(f"Schema: V_* flat | m={m} | max_types={max_types} | maxN={maxN} | types={names}")
    print(f"Mode: {args.mode}  -> {len(labels)} features/event")
    print(f"Processed {len(rows)} event(s). Wrote: {out_path}")
    print("\nPreview:\n", out_df.head(min(5, len(out_df))).to_string(index=False))

    # If single event, also plot the bar chart
    if args.event is not None:
        base_dir = os.path.dirname(out_path) or "."
        out_png = os.path.join(base_dir, f"{tag}_event{args.event}_bar.png")
        plot_compact_vector(labels, last_vals, args.event, out_png, title_prefix=title_prefix)
        print(f"Saved bar chart: {out_png}")


if __name__ == "__main__":
    main()
