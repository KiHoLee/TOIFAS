"""Generate the LaTeX rows of the two result tables from the CSVs, so
that every table in the paper is reproducible from data/ (TIFS mandate).
Prints the tabular body; paste into main.tex without edits.
"""
from __future__ import annotations
import csv
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
}


def f3(x: str) -> str:
    try:
        return f"{float(x):.3f}"
    except ValueError:
        return "--"


def compare_table():
    print("% Table: scheme comparison (from sec_compare.csv)")
    rows = list(csv.DictReader(open(DATA / "sec_compare.csv")))
    order = ["public_mask", "perm_key", "index_cipher", "oma_plain", "proposed"]
    rows = sorted(rows, key=lambda r: order.index(r["scheme"]))
    for r in rows:
        cells = [f3(r["legit_ser"]), f3(r["eve_out"]), f3(r["eve_in"]),
                 f3(r["jam0_ser"])]
        if r["scheme"] == "proposed":
            cells = [rf"$\mathbf{{{c}}}$" for c in cells]
        else:
            cells = [f"${c}$" if c != "--" else "--" for c in cells]
        print(f"{NAME[r['scheme']]} & " + " & ".join(cells) + r" \\")


def maskfam_table():
    print("% Table: key families (from sec_maskfam.csv)")
    for r in csv.DictReader(open(DATA / "sec_maskfam.csv")):
        print(f"{NAME[r['family']]} & ${f3(r['legit_ser'])}$ & "
              f"${f3(r['eve_ser'])}$ & ${f3(r['mask_xcorr'])}$" + r" \\")


if __name__ == "__main__":
    compare_table()
    print()
    maskfam_table()
