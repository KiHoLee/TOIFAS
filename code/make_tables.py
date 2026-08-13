"""Generate the LaTeX rows of the three result tables from the CSVs, so
that every table in the paper is reproducible from data/ (TIFS mandate).
Prints the tabular bodies; paste into main.tex without edits.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"

NAME = {
    "proposed": r"\textbf{Proposed keyed masking}",
    "public_mask": "Public masks",
    "perm_key": r"Global-permutation key~\cite{chen2023shuffling}",
    "index_cipher": "Per-user index cipher",
    "oma_plain": "OMA (no encryption)",
    "random": "Random",
    "hadamard": "Walsh-Hadamard",
    "learned": "Learned",
    "learned_reg": r"Regularized~\eqref{eq:regloss}",
}
RECEIVER = {
    "legit": "Legitimate", "oma": "OMA",
    "insider": "Insider", "eve": "Outsider eavesdropper",
}


def f3(x: str) -> str:
    """Three decimals, or an em-dash for a value that does not apply."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "--"
    return "--" if math.isnan(v) else f"{v:.3f}"


def cell(x: str, bold: bool) -> str:
    s = f3(x)
    if s == "--":
        return "--"
    return rf"$\mathbf{{{s}}}$" if bold else f"${s}$"


def compare_table():
    print("% Table: scheme comparison (from sec_compare.csv)")
    rows = list(csv.DictReader(open(DATA / "sec_compare.csv")))
    order = ["public_mask", "perm_key", "index_cipher", "oma_plain", "proposed"]
    rows.sort(key=lambda r: order.index(r["scheme"]))
    for r in rows:
        b = r["scheme"] == "proposed"
        cells = [cell(r[k], b) for k in
                 ("legit_ser", "eve_out", "eve_in", "jam0_ser")]
        print(f"{NAME[r['scheme']]} & " + " & ".join(cells) + r" \\")


def maskfam_table():
    print("% Table: key families (from sec_maskfam.csv)")
    for r in csv.DictReader(open(DATA / "sec_maskfam.csv")):
        cells = [cell(r[k], False) for k in
                 ("legit_ser", "eve_ser", "eve_ones_ser", "mask_xcorr")]
        print(f"{NAME[r['family']]} & " + " & ".join(cells) + r" \\")


def real_table():
    print("% Table: headline recovery (from real_sec_stats.json)")
    st = json.loads((DATA / "real_sec_stats.json").read_text())
    rec = st["recovery"]
    snrs = sorted(rec, key=float)
    for key in ("legit", "oma", "insider", "eve"):
        cells = " & ".join(f"${rec[s][key]:.3f}$" for s in snrs)
        print(f"{RECEIVER[key]} & {cells}" + r" \\")


if __name__ == "__main__":
    compare_table(); print()
    maskfam_table(); print()
    real_table()
