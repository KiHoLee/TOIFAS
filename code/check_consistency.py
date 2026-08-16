# -*- coding: utf-8 -*-
"""Final consistency check: every headline number vs its raw CSV."""
import csv
import math
import re
from pathlib import Path

base = Path(__file__).resolve().parents[1]
# The manuscript is not part of the reproducibility package, so the
# tex-side assertions are skipped when it is absent and the data-side
# assertions still run.
_tex_path = base / "main.tex"
HAVE_TEX = _tex_path.exists()
tex = _tex_path.read_text(encoding="utf-8") if HAVE_TEX else ""


def rows(name):
    with open(base / "data" / name) as f:
        return list(csv.DictReader(f))


def col(name, k):
    return [float(r[k]) for r in rows(name) if r[k] != "nan"]


ok = True


def chk(label, cond, detail, needs_tex=False):
    global ok
    if needs_tex and not HAVE_TEX:
        print("  SKIP  " + label + " :: main.tex not in this package")
        return
    print(("  PASS  " if cond else "  FAIL  ") + label + " :: " + detail)
    if not cond:
        ok = False


print("headline numbers vs raw data")

# 1.27x key-length ratio
k = rows("sec_keylen.csv")
r64 = [x for x in k if int(x["L"]) == 64][0]
ratio = float(r64["oma"]) / float(r64["legit_ser"])
chk("key-length ratio 1.27", round(ratio, 2) == 1.27, "%.4f" % ratio)
chk("1.27 in tex", tex.count("1.27") >= 2, "%d occurrences" % tex.count("1.27"), needs_tex=True)

# blind-jammer gap
g = col("sec_jam_gap.csv", "gap_db")
chk("gap 7.4-8.1 dB", round(min(g), 1) == 7.4 and round(max(g), 1) == 8.1,
    "%.3f to %.3f" % (min(g), max(g)))
chk("no stale 8.5 dB", "8.5$~dB" not in tex, "searched tex", needs_tex=True)
lin = (10 ** (min(g) / 10), 10 ** (max(g) / 10))
chk("about six times power", lin[0] < 6.5 and lin[1] > 5.5,
    "%.2f to %.2f" % lin)

# blind vs permutation
j = rows("sec_jam_cmp.csv")
dmax = max(abs(float(r["blind"]) - float(r["perm_blind"])) for r in j)
chk("within 0.002", dmax <= 0.002, "%.5f" % dmax)
chk("no stale 0.0015 in jamming", "$0.0015$ of the proposed" not in tex, "ok", needs_tex=True)

# brute force
b = rows("sec_brute_cmp.csv")
sm = float(b[-1]["ser_mask"])
chk("brute 0.76 both places", tex.count("$0.76$") >= 2, "%.4f measured" % sm, needs_tex=True)
chk("no stale 0.75 in summary",
    "$0.75$ after $10^{6}$" not in tex, "summary row", needs_tex=True)

# refresh
rs = {r["scheme"]: r for r in rows("refresh_summary.csv")}
chk("refresh 64.8 bits",
    round(float(rs["Invariant"]["entropy_bits"]), 1) == 64.8,
    "%.3f" % float(rs["Invariant"]["entropy_bits"]))
chk("fixed key 15.0 bits",
    round(float(rs["None (fixed key)"]["entropy_bits"]), 1) == 15.0,
    "%.4f" % float(rs["None (fixed key)"]["entropy_bits"]))

# permutation KPA
pk = rows("pkpa.csv")
p6 = float([r for r in pk if r["n_frames"] == "6"][0]["eve_ser"])
chk("perm KPA at N=6 near 0.303", abs(p6 - 0.303) < 0.005, "%.4f" % p6)

# abstract
a = (tex.split(r"\begin{abstract}")[1].split(r"\end{abstract}")[0].strip()
     if HAVE_TEX else "")
w = len(re.split(r"\s+", a)) if a else 0
chk("abstract <= 250 words", w <= 250, "%d words" % w, needs_tex=True)
chk("abstract has no abbreviations",
    not re.findall(r"\b[A-Z]{2,}\b", a), str(re.findall(r"\b[A-Z]{2,}\b", a)), needs_tex=True)

print()
print("ALL CONSISTENT" if ok else "INCONSISTENCIES FOUND")
