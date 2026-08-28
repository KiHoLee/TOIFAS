# -*- coding: utf-8 -*-
"""Every learned-family number the manuscript quotes, read from data/.

Switching the learned family from the unpenalized keys to the
regularized ones of Section V-C moves every learned value in the paper.
This prints them next to their structured counterparts so the sentences
that carry them can be updated from one place, and so a later rerun can
be checked against what is printed.

Run: python code/report_learned.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def rows(name):
    with open(DATA / name) as f:
        return list(csv.DictReader(f))


def at(rs, key, val, col):
    for r in rs:
        if abs(float(r[key]) - val) < 1e-9:
            return float(r[col])
    raise KeyError("%s=%s not in the sweep" % (key, val))


def main():
    print("== Fig. 2, SER against SNR ==")
    s = rows("sec_snr.csv")
    l = rows("sec_snr_learned.csv")
    ratios = []
    for a, b in zip(s, l):
        r = float(b["legit"]) / float(a["legit"])
        ratios.append(r)
        print("  %5s dB  str %.5f   lrn %.5f   ratio %.3f"
              % (a["snr_db"], float(a["legit"]), float(b["legit"]), r))
    print("  ratio range %.3f to %.3f" % (min(ratios), max(ratios)))

    print("\n== Fig. 3, SER against key length ==")
    s = rows("sec_keylen.csv")
    l = rows("sec_keylen_learned.csv")
    for a, b in zip(s, l):
        print("  L=%-4s str %.5f   lrn %.5f   ratio %.3f   kappa %.5f"
              % (a["L"], float(a["legit_ser"]), float(b["legit_ser"]),
                 float(b["legit_ser"]) / float(a["legit_ser"]),
                 float(b["mask_xcorr"])))
    print("  learned outsider floor %.5f"
          % min(float(r["eve_ser"]) for r in l))

    print("\n== Fig. 4, jamming at JSR 0 dB ==")
    print("  str blind  %.4f" % at(rows("sec_jam.csv"), "jsr_db", 0.0,
                                   "blind"))
    print("  lrn blind  %.4f" % at(rows("sec_jam_learned.csv"), "jsr_db",
                                   0.0, "blind"))

    print("\n== Fig. 6, best of K=1e6 guesses ==")
    print("  str %.4f" % at(rows("sec_brute_cmp.csv"), "K", 1e6, "ser_mask"))
    print("  lrn %.4f" % at(rows("sec_brute_learned.csv"), "K", 1e6,
                            "ser_mask"))

    print("\n== Fig. 7, known plaintext at 10 dB ==")
    for name in ("kpa.csv", "kpa_learned.csv"):
        r = [x for x in rows(name) if float(x["snr_db"]) == 10.0]
        print("  %-16s N=2 %.4f  N=8 %.4f  N=64 %.4f"
              % (name, at(r, "n_frames", 2, "eve_ser"),
                 at(r, "n_frames", 8, "eve_ser"),
                 at(r, "n_frames", 64, "eve_ser")))

    print("\n== Fig. 8, real token streams ==")
    s = rows("real_sec_ter.csv")
    l = rows("real_sec_ter_learned.csv")
    gaps = []
    for a, b in zip(s, l):
        for col in ("ter_eve", "ter_insider"):
            gaps.append(abs(float(a[col]) - float(b[col])))
        print("  %4s dB  legit str %.5f  lrn %.5f   ratio %.3f"
              % (a["snr_db"], float(a["ter_legit"]), float(b["ter_legit"]),
                 float(b["ter_legit"]) / float(a["ter_legit"])))
    print("  largest adversary gap between families %.2e" % max(gaps))
    for name in ("real_sec_stats.json", "real_sec_stats_learned.json"):
        if (DATA / name).exists():
            d = json.loads((DATA / name).read_text())
            print("  %-28s %s" % (name, {k: d[k] for k in list(d)[:6]}))

    print("\n== Table IV, scheme comparison ==")
    for name in ("sec_compare.csv", "compare_learned.csv"):
        for r in rows(name):
            print("  %-20s %s" % (name, dict(r)))

    print("\n== Table V, refresh ==")
    l = rows("refresh_learned.csv")
    print("  lrn legit  %.5f to %.5f"
          % (min(float(r["legit_ser"]) for r in l),
             max(float(r["legit_ser"]) for r in l)))
    print("  lrn eve    %.5f" % (sum(float(r["eve_ser"]) for r in l)
                                 / len(l)))


if __name__ == "__main__":
    main()
