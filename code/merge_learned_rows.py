# -*- coding: utf-8 -*-
"""Fold the learned-key rows into the two table sources.

Table IV reads sec_compare.csv and Table V reads refresh_summary.csv,
and both are written by the structured stages, which know nothing about
the learned family. Its rows were appended by hand, so a rerun of the
learned stages left the tables behind. This does the fold, so both files
are derived from data/ like every other table source.

Run after code/run_learned_reg.py, before code/make_tables.py.
"""
from __future__ import annotations

import csv
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def read(name):
    with open(DATA / name) as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def write(name, fields, rows):
    with open(DATA / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def upsert(rows, key, value, row):
    """Replace the row carrying key==value, or append it."""
    for i, r in enumerate(rows):
        if r[key] == value:
            rows[i] = row
            return rows
    rows.append(row)
    return rows


def main():
    # Table IV: the learned scheme row, measured by exp_learned.compare
    fields, rows = read("sec_compare.csv")
    _, learned = read("compare_learned.csv")
    assert len(learned) == 1, "compare_learned.csv should carry one row"
    rows = upsert(rows, "scheme", "proposed_learned",
                  {k: learned[0][k] for k in fields})
    write("sec_compare.csv", fields, rows)
    print("sec_compare.csv       proposed_learned  jam0 %s"
          % learned[0]["jam0_ser"])

    # Table V: the learned refresh row, averaged over the blocks that
    # exp_learned.refresh measured, at the same entropy as the
    # structured refresh because the invariance group is the same
    fields, rows = read("refresh_summary.csv")
    _, blocks = read("refresh_learned.csv")
    lg = sum(float(r["legit_ser"]) for r in blocks) / len(blocks)
    ev = sum(float(r["eve_ser"]) for r in blocks) / len(blocks)
    ent = next(r["entropy_bits"] for r in rows
               if r["scheme"] == "Invariant, KM (str.)")
    rows = upsert(rows, "scheme", "Invariant, KM (lrn.)",
                  {"scheme": "Invariant, KM (lrn.)",
                   "legit": "%.6f" % lg, "eve": "%.6f" % ev,
                   "entropy_bits": ent})
    # 9.3: the proposal first, as every figure legend lists it
    rows.sort(key=lambda x: 0 if x["scheme"].startswith("Invariant") else 1)
    write("refresh_summary.csv", fields, rows)
    print("refresh_summary.csv   Invariant, KM (lrn.)  legit %.5f  eve %.5f"
          % (lg, ev))


if __name__ == "__main__":
    main()
