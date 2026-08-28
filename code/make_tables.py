"""Generate the LaTeX rows of every result table from the CSVs, so
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
    "proposed": r"\textbf{KM (str.)}",
    "proposed_learned": r"\textbf{KM (lrn.)}",
    "public_mask": "Public masks",
    "perm_key": r"Permutation key~\cite{chen2025shufflingtifs}",
    "index_cipher": "Index cipher",
    "oma_plain": "OMA (no encryption)",
    "random": "Random",
    "hadamard": "Structured",
    "learned": "Learned, plain",
    "learned_reg": r"Learned, regularized~\eqref{eq:regloss}",
    "invariant_learned": r"\textbf{Invariant, KM (lrn.)}",
}
RECEIVER = {
    "legit": "Legitimate", "oma": "OMA",
    "insider": "Insider", "eve": "Outsider",
}


def f3(x: str) -> str:
    """Three decimals, or an em-dash for a value that does not apply."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "--"
    return "--" if math.isnan(v) else f"{v:.3f}"


def f4(x: str) -> str:
    """Four decimals, for a column whose values sit against the
    random-guess level and would otherwise all print as 1.000 while the
    body quotes their distance from it in units of 1e-4."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "--"
    return "--" if math.isnan(v) else f"{v:.4f}"


def cell(x: str, bold: bool, wide: bool = False) -> str:
    s = f4(x) if wide else f3(x)
    if s == "--":
        return "--"
    return rf"$\mathbf{{{s}}}$" if bold else f"${s}$"


def compare_table():
    print("% Table: scheme comparison (from sec_compare.csv)")
    rows = list(csv.DictReader(open(DATA / "sec_compare.csv")))
    order = ["public_mask", "perm_key", "index_cipher", "oma_plain",
             "proposed", "proposed_learned"]
    rows.sort(key=lambda r: order.index(r["scheme"]))
    # stage_E does not jam the orthogonal reference, because the jammer an
    # OMA user faces is targeted at public slots rather than mask-matched
    # or mask-blind. stage_L measures that case, so the cell comes from
    # there instead of being left empty.
    jam = {float(r["jsr_db"]): r
           for r in csv.DictReader(open(DATA / "sec_jam_cmp.csv"))}
    oma_jam = jam[0.0]["oma_targeted"]
    for r in rows:
        b = r["scheme"].startswith("proposed")
        if r["scheme"] == "oma_plain" and f3(r["jam0_ser"]) == "--":
            r["jam0_ser"] = oma_jam
        # four decimals would still print 1.0000 here, so the column
        # stays at three and the caption names the chance level
        cells = [cell(r[k], b) for k in
                 ("eve_out", "eve_in", "jam0_ser")]
        print(f"{NAME[r['scheme']]} & " + " & ".join(cells) + r" \\")


def maskfam_table():
    print("% Table: key families (from sec_maskfam.csv)")
    for r in csv.DictReader(open(DATA / "sec_maskfam.csv")):
        # the structured family is the main configuration, so its row is
        # emphasized the same way the proposed row is in the comparison
        b = r["family"] == "hadamard"
        cells = [cell(r[k], b) for k in
                 ("legit_ser", "eve_ser", "mask_xcorr")]
        name = NAME[r["family"]]
        if b:
            name = r"\textbf{" + name + "}"
        print(f"{name} & " + " & ".join(cells) + r" \\")


def refresh_tables():
    print("% Table: key refresh (from refresh_summary.csv)")
    for r in csv.DictReader(open(DATA / "refresh_summary.csv")):
        b = r["scheme"].startswith("Invariant")
        name = (r"\textbf{" + r["scheme"] + "}") if b else r["scheme"]
        f = (lambda t: r"\mathbf{" + t + "}") if b else (lambda t: t)
        print(f"{name} & ${f(format(float(r['legit']), '.3f'))}$ & "
              f"${f(format(float(r['eve']), '.4f'))}$ & "
              # the unit lives in the header, not in every cell
              f"${f(format(float(r['entropy_bits']), '.1f'))}$" + r" \\")
if __name__ == "__main__":
    compare_table(); print()
    maskfam_table(); print()
    refresh_tables()
