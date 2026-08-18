# -*- coding: utf-8 -*-
"""Final consistency check: every headline number vs its raw CSV.

A quoted value that goes stale during a revision is the failure mode this
guards against, so each assertion recomputes from data/ rather than from
another quoted value. The manuscript-side assertions are skipped when
main.tex is absent, which is the case in the reproducibility package.
"""
import csv
import math
import re
from pathlib import Path

base = Path(__file__).resolve().parents[1]
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

# --- Fig. 2: the proposal is below OMA -------------------------------
sn = rows("sec_snr.csv")
lg = [float(x["legit"]) for x in sn]
om = [float(x["oma"]) for x in sn]
rel = [(a - b) / b * 100 for a, b in zip(lg, om)]
chk("legit below OMA at every SNR", max(rel) < 0, "max relative %+.2f%%" % max(rel))
chk("gain 1.3 to 7.9 percent",
    round(-max(rel), 1) == 1.3 and round(-min(rel), 1) == 7.9,
    "%.2f to %.2f percent" % (-max(rel), -min(rel)))
chk("1.3 and 7.9 in tex", "$1.3$ to\n$7.9$~percent" in tex or "$1.3$ to $7.9$~percent" in tex,
    "searched tex", needs_tex=True)
ew = [float(x["eve_wrong"]) for x in sn]
ch = float(sn[0]["chance"])
chk("outsider at chance to 2e-5", max(abs(x - ch) for x in ew) < 2e-5,
    "max deviation %.2e" % max(abs(x - ch) for x in ew))

# --- Fig. 3: key-length ratio ----------------------------------------
k = rows("sec_keylen.csv")
r64 = [x for x in k if int(x["L"]) == 64][0]
ratio = float(r64["oma"]) / float(r64["legit_ser"])
chk("key-length ratio 1.52", round(ratio, 2) == 1.52, "%.4f" % ratio)
chk("1.52 in tex", tex.count("1.52") >= 2, "%d occurrences" % tex.count("1.52"),
    needs_tex=True)
chk("keys exactly orthogonal in the sweep",
    max(float(x["mask_xcorr"]) for x in k) < 1e-6,
    "max xcorr %.2e" % max(float(x["mask_xcorr"]) for x in k))

# --- Fig. 4: jamming --------------------------------------------------
g = col("sec_jam_gap.csv", "gap_db")
chk("gap 5.5-6.3 dB", round(min(g), 1) == 5.5 and round(max(g), 1) == 6.3,
    "%.3f to %.3f" % (min(g), max(g)))
lin = (10 ** (min(g) / 10), 10 ** (max(g) / 10))
chk("about four times power", lin[0] < 4.5 and lin[1] > 3.4,
    "%.2f to %.2f" % lin)
j = rows("sec_jam_cmp.csv")
dmax = max(abs(float(r["blind"]) - float(r["perm_blind"])) for r in j)
chk("within 0.002", dmax <= 0.002, "%.5f" % dmax)
chk("no stale 8.1 dB", "$8.1$~dB" not in tex, "searched tex", needs_tex=True)

# --- Fig. 6: brute force ---------------------------------------------
b = rows("sec_brute_cmp.csv")
sm = float(b[-1]["ser_mask"])
chk("brute 0.59 at 1e6", round(sm, 2) == 0.59, "%.4f" % sm)
chk("0.59 in tex", "$0.59$" in tex, "searched tex", needs_tex=True)
pad0 = next((x["K"] for x in b if float(x["ser_pad"]) < 0.27), None)
chk("index cipher collapses at 65536", pad0 == "65536", str(pad0))

# --- Fig. 7: known plaintext -----------------------------------------
kp = rows("kpa.csv")
legit = float([x for x in k if int(x["L"]) == 16][0]["legit_ser"])
thr = legit * 1.02
first20 = next((x["n_frames"] for x in kp
                if int(x["snr_db"]) == 20 and float(x["eve_ser"]) <= thr), None)
first10 = next((x["n_frames"] for x in kp
                if int(x["snr_db"]) == 10 and float(x["eve_ser"]) <= thr), None)
chk("KPA five frames at 20 dB", first20 == "5", "first N = %s" % first20)
chk("KPA twenty-four frames at 10 dB", first10 == "24", "first N = %s" % first10)
pk = rows("pkpa.csv")
p6 = float([x for x in pk if x["n_frames"] == "6"][0]["eve_ser"])
chk("perm KPA at N=6 near its own 0.258", abs(p6 - 0.258) < 0.005, "%.4f" % p6)

# --- refresh ----------------------------------------------------------
rs = {x["scheme"]: x for x in rows("refresh_summary.csv")}
chk("refresh 64.8 bits",
    round(float(rs["Invariant"]["entropy_bits"]), 1) == 64.8,
    "%.3f" % float(rs["Invariant"]["entropy_bits"]))
chk("fixed key 15.0 bits",
    round(float(rs["None (fixed key)"]["entropy_bits"]), 1) == 15.0,
    "%.4f" % float(rs["None (fixed key)"]["entropy_bits"]))
chk("invariant refresh free",
    abs(float(rs["Invariant"]["legit"]) - float(rs["None (fixed key)"]["legit"]))
    < 0.001, "%.4f vs %.4f" % (float(rs["Invariant"]["legit"]),
                               float(rs["None (fixed key)"]["legit"])))

# --- real tokens ------------------------------------------------------
import json
st = json.loads((base / "data" / "real_sec_stats.json").read_text())
rec = st["recovery"]["28"]
chk("headline recovery 78 vs 76 percent",
    round(rec["legit"] * 100) == 78 and round(rec["oma"] * 100) == 76,
    "%.1f vs %.1f" % (rec["legit"] * 100, rec["oma"] * 100))
chk("legit leads OMA at every point",
    all(st["recovery"][s]["legit"] > st["recovery"][s]["oma"]
        for s in st["recovery"]),
    "checked %d points" % len(st["recovery"]))

# --- the room argument of Fig. 2 and its evidence in Fig. 3 -----------
kl = {int(r["L"]): r for r in rows("sec_keylen.csv")}
r8 = kl[8]
chk("L=8 crowding, proposal behind OMA",
    round(float(r8["legit_ser"]), 3) == 0.949
    and round(float(r8["oma"]), 3) == 0.685,
    "%.3f vs %.3f" % (float(r8["legit_ser"]), float(r8["oma"])))
chk("0.949 and 0.685 in tex", "0.949" in tex and "0.685" in tex,
    "searched tex", needs_tex=True)

# the Fig. 2 inset plots this ratio, so its stated span must hold
sr = rows("sec_snr.csv")
rt = [float(r["oma"]) / float(r["legit"]) for r in sr]
chk("inset ratio spans 1.01 to 1.09", 1.005 < min(rt) and max(rt) < 1.095,
    "%.3f to %.3f" % (min(rt), max(rt)))

# the three secrets named in the setup
chk("secret sizes UL=64, perm 64, pad 16",
    all(t in tex for t in ["$UL=64$ key entries",
                           "one permutation of $64$ positions",
                           "$16$ pad\nbits per user"]),
    "searched tex", needs_tex=True)

# Fig. 5 shows the permutation curve tracking the mask curve
sc = rows("sec_sens_cmp.csv")
dv = max(abs(float(r["ser_mask"]) - float(r["ser_perm"])) for r in sc)
chk("permutation tracks mask in Fig. 5", dv < 0.06, "max gap %.3f" % dv)

# --- abstract ---------------------------------------------------------
a = (tex.split(r"\begin{abstract}")[1].split(r"\end{abstract}")[0].strip()
     if HAVE_TEX else "")
w = len(re.split(r"\s+", a)) if a else 0
chk("abstract <= 250 words", w <= 250, "%d words" % w, needs_tex=True)
chk("abstract has no abbreviations",
    not re.findall(r"\b[A-Z]{2,}\b", a), str(re.findall(r"\b[A-Z]{2,}\b", a)),
    needs_tex=True)

print()
print("ALL CONSISTENT" if ok else "INCONSISTENCIES FOUND")
