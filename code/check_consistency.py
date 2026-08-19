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
chk("gain 24 to 35 percent",
    round(-max(rel)) == 24 and round(-min(rel)) == 35,
    "%.2f to %.2f percent" % (-max(rel), -min(rel)))
chk("24 and 35 in tex", "$24$ to\n$35$~percent" in tex or "$24$ to $35$~percent" in tex,
    "searched tex", needs_tex=True)
ew = [float(x["eve_wrong"]) for x in sn]
ch = float(sn[0]["chance"])
dev = max(abs(x - ch) for x in ew)
chk("outsider at chance to 3.5e-4", dev < 3.6e-4, "max deviation %.2e" % dev)
chk("3.5e-4 in tex", "$3.5\\times10^{-4}$" in tex, "searched tex",
    needs_tex=True)

# the main configuration's legitimate rate, the reference every later
# assertion compares against; taken from the curve the main
# configuration produced rather than looked up by key length
MAIN_LEGIT = [float(x["legit"]) for x in sn if float(x["snr_db"]) == 10][0]
chk("main legitimate 0.053", round(MAIN_LEGIT, 3) == 0.053,
    "%.4f" % MAIN_LEGIT)

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
chk("gap 10.1-11.1 dB", round(min(g), 1) == 10.1 and round(max(g), 1) == 11.1,
    "%.3f to %.3f" % (min(g), max(g)))
lin = (10 ** (min(g) / 10), 10 ** (max(g) / 10))
chk("more than ten times power", lin[0] > 10.0, "%.2f to %.2f" % lin)
j = rows("sec_jam_cmp.csv")
dmax = max(abs(float(r["blind"]) - float(r["perm_blind"])) for r in j)
chk("within 0.002", dmax <= 0.002, "%.5f" % dmax)
chk("no stale 8.1 dB", "$8.1$~dB" not in tex, "searched tex", needs_tex=True)

# --- Fig. 6: brute force ---------------------------------------------
b = rows("sec_brute_cmp.csv")
sm = float(b[-1]["ser_mask"])
chk("brute 0.67 at 1e6", round(sm, 2) == 0.67, "%.4f" % sm)
chk("0.67 in tex", "$0.67$" in tex, "searched tex", needs_tex=True)
closed = (ch - sm) / (ch - MAIN_LEGIT)
chk("brute closes about a third", 0.30 < closed < 0.40, "%.3f" % closed)
bf = float(b[-1]["best_frac"]) * 100
chk("permutation 3.4 percent of positions", round(bf, 1) == 3.4, "%.2f" % bf)
pad0 = next((x["K"] for x in b if float(x["ser_pad"]) < 0.1), None)
chk("index cipher collapses at 65536", pad0 == "65536", str(pad0))

# --- Fig. 7: known plaintext -----------------------------------------
kp = rows("kpa.csv")
legit = MAIN_LEGIT
thr = legit * 1.02
first20 = next((x["n_frames"] for x in kp
                if int(x["snr_db"]) == 20 and float(x["eve_ser"]) <= thr), None)
first10 = next((x["n_frames"] for x in kp
                if int(x["snr_db"]) == 10 and float(x["eve_ser"]) <= thr), None)
chk("KPA three frames at 20 dB", first20 == "3", "first N = %s" % first20)
chk("KPA ten frames at 10 dB", first10 == "10", "first N = %s" % first10)
kp0 = [x for x in kp if int(x["snr_db"]) == 0]
w0 = float(kp0[-1]["eve_ser"]) / legit
chk("0 dB no longer holds", w0 < 1.03, "64 frames reach %.3f of legitimate" % w0)
pk = rows("pkpa.csv")
p6 = float([x for x in pk if x["n_frames"] == "6"][0]["eve_ser"])
chk("perm KPA at N=6 near its own legitimate",
    abs(p6 - MAIN_LEGIT) < 0.005, "%.4f" % p6)

# --- refresh ----------------------------------------------------------
rs = {x["scheme"]: x for x in rows("refresh_summary.csv")}
chk("refresh 364.6 bits",
    round(float(rs["Invariant"]["entropy_bits"]), 1) == 364.6,
    "%.3f" % float(rs["Invariant"]["entropy_bits"]))
chk("fixed key 23.8 bits",
    round(float(rs["None (fixed key)"]["entropy_bits"]), 1) == 23.8,
    "%.4f" % float(rs["None (fixed key)"]["entropy_bits"]))
chk("invariant refresh free",
    abs(float(rs["Invariant"]["legit"]) - float(rs["None (fixed key)"]["legit"]))
    < 0.001, "%.4f vs %.4f" % (float(rs["Invariant"]["legit"]),
                               float(rs["None (fixed key)"]["legit"])))

# --- real tokens ------------------------------------------------------
import json
st = json.loads((base / "data" / "real_sec_stats.json").read_text())
rec = st["recovery"]["28"]
chk("headline recovery 96 vs 93 percent",
    round(rec["legit"] * 100) == 96 and round(rec["oma"] * 100) == 93,
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

# the OMA-to-proposed ratio the narration quotes
sr = rows("sec_snr.csv")
rt = [float(r["oma"]) / float(r["legit"]) for r in sr]
chk("ratio spans 1.32 to 1.54", round(min(rt), 2) == 1.32
    and round(max(rt), 2) == 1.54, "%.3f to %.3f" % (min(rt), max(rt)))

# the three secrets named in the setup
chk("secret sizes: per-user direction, perm 256, pad 16",
    all(t in tex for t in ["length-$64$ key direction per user",
                           "one permutation of $256$",
                           "$16$ pad bits per user"]),
    "searched tex", needs_tex=True)

chk("no stale d=64 configuration in tex",
    "$d=64$ real dimensions" not in tex and "$d/U=16$" not in tex,
    "searched tex", needs_tex=True)

# Fig. 5 shows the permutation curve tracking the mask curve
sc = rows("sec_sens_cmp.csv")
dv = max(abs(float(r["ser_mask"]) - float(r["ser_perm"])) for r in sc)
chk("permutation tracks mask in Fig. 5", dv < 0.06, "max gap %.3f" % dv)

# --- the audit round's corrected quantities ---------------------------
mf = {r["family"]: r for r in rows("sec_maskfam.csv")}
fam_pct = (float(mf["random"]["legit_ser"])
           / float(mf["hadamard"]["legit_ser"]) - 1) * 100
chk("continuous family 29 percent worse", round(fam_pct) == 29,
    "%.1f percent" % fam_pct)
chk("no stale 2.5 factor in tex", "factor of $2.5$" not in tex,
    "searched tex", needs_tex=True)

sc2 = rows("sec_sens_cmp.csv")
worst04 = min(min(float(r["ser_mask"]), float(r["ser_perm"]),
                  float(r["ser_pad"])) for r in sc2
              if float(r["frac"]) <= 0.4)
chk("all three above 0.95 to 40 percent of key", worst04 > 0.95,
    "min %.4f" % worst04)

rf = rows("refresh.csv")
res = max(1 - float(r["eve_invariant"]) for r in rf)
chk("refresh residual below 2.4e-3", res < 2.4e-3, "max %.2e" % res)

rk = rows("refresh_kpa.csv")
nb = max(0.9999847412 - float(r["ser_next_block"]) for r in rk)
chk("next block within 6e-4 of chance", nb < 6e-4, "max %.2e" % nb)

bc = rows("sec_brute_cmp.csv")
pm = min(float(r["ser_perm"]) for r in bc)
chk("permutation floor 0.9996", pm > 0.9996, "min %.5f" % pm)

md = {r["family"]: r for r in rows("maskdegen.csv")}
ks = [int(x) for x in md["learned"]["support99_per_key"].split("/")]
chk("learned keys degenerate: 5 to 8 of 64 entries",
    min(ks) == 5 and max(ks) == 8 and int(md["learned"]["L"]) == 64,
    md["learned"]["support99_per_key"])
chk("learned support overlap 0.10",
    round(float(md["learned"]["mean_overlap"]), 2) == 0.10,
    md["learned"]["mean_overlap"])
chk("degeneracy numbers in tex",
    "$5$ to $8$ of the $64$ entries" in tex and "overlap of\n$0.10$" in tex
    or "$5$ to $8$ of the $64$ entries" in tex and "overlap of $0.10$" in tex,
    "searched tex", needs_tex=True)

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
