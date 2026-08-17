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
    "proposed": r"\textbf{Proposed keyed masking}",
    "public_mask": "Public masks",
    "perm_key": r"Permutation key~\cite{chen2023shuffling}",
    "index_cipher": "Index cipher",
    "oma_plain": "OMA (no encryption)",
    "random": "Random",
    "hadamard": "Walsh-Hadamard",
    "learned": "Learned",
    "learned_reg": r"Regularized~\eqref{eq:regloss}",
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
    # stage_E does not jam the orthogonal reference, because the jammer an
    # OMA user faces is targeted at public slots rather than mask-matched
    # or mask-blind. stage_L measures that case, so the cell comes from
    # there instead of being left empty.
    jam = {float(r["jsr_db"]): r
           for r in csv.DictReader(open(DATA / "sec_jam_cmp.csv"))}
    oma_jam = jam[0.0]["oma_targeted"]
    for r in rows:
        b = r["scheme"] == "proposed"
        if r["scheme"] == "oma_plain" and f3(r["jam0_ser"]) == "--":
            r["jam0_ser"] = oma_jam
        cells = [cell(r[k], b) for k in
                 ("legit_ser", "eve_out", "eve_in", "jam0_ser")]
        print(f"{NAME[r['scheme']]} & " + " & ".join(cells) + r" \\")


def maskfam_table():
    print("% Table: key families (from sec_maskfam.csv)")
    for r in csv.DictReader(open(DATA / "sec_maskfam.csv")):
        # the structured family is the main configuration, so its row is
        # emphasized the same way the proposed row is in the comparison
        b = r["family"] == "hadamard"
        cells = [cell(r[k], b) for k in
                 ("legit_ser", "eve_ser", "eve_ones_ser", "mask_xcorr")]
        name = NAME[r["family"]]
        if b:
            name = r"\textbf{" + name + "}"
        print(f"{name} & " + " & ".join(cells) + r" \\")


def real_table():
    print("% Table: headline recovery (from real_sec_stats.json)")
    st = json.loads((DATA / "real_sec_stats.json").read_text())
    rec = st["recovery"]
    snrs = sorted(rec, key=float)
    for key in ("legit", "oma", "insider", "eve"):
        cells = " & ".join(f"${rec[s][key]:.3f}$" for s in snrs)
        print(f"{RECEIVER[key]} & {cells}" + r" \\")


def refresh_tables():
    print("% Table: key refresh (from refresh_summary.csv)")
    for r in csv.DictReader(open(DATA / "refresh_summary.csv")):
        b = r["scheme"] == "Invariant"
        name = r"\textbf{Invariant}" if b else r["scheme"]
        f = (lambda t: r"\mathbf{" + t + "}") if b else (lambda t: t)
        print(f"{name} & ${f(format(float(r['legit']), '.3f'))}$ & "
              f"${f(format(float(r['eve']), '.4f'))}$ & "
              f"${f(format(float(r['entropy_bits']), '.1f'))}$~bits" + r" \\")
    print()
    print("% Table: known plaintext across a refresh (from refresh_kpa.csv)")
    rows = {r["n_frames"]: r for r in csv.DictReader(open(DATA / "refresh_kpa.csv"))}
    keep = ["2", "8", "64"]
    print("Frames used by the attacker & "
          + " & ".join(f"${k}$" for k in keep) + r" \\")
    for lbl, key in (("Same block", "ser_same_block"),
                     ("Next block", "ser_next_block")):
        print(f"{lbl} & "
              + " & ".join(f"${float(rows[k][key]):.3f}$" for k in keep)
              + r" \\")


if __name__ == "__main__":
    compare_table(); print()
    maskfam_table(); print()
    real_table(); print()
    refresh_tables()
