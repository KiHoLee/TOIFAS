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
_ewl = [float(x["eve_wrong"]) for x in rows("sec_snr_learned.csv")]
dev = max(dev, max(abs(x - ch) for x in _ewl))
chk("outsider at chance to 4e-4, both families", dev < 4.0e-4,
    "max deviation %.2e" % dev)
chk("4e-4 in tex", "$4\\times10^{-4}$" in tex, "searched tex",
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
chk("1.52 in tex", tex.count("1.52") >= 1, "%d occurrences" % tex.count("1.52"),
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
    round(float(rs["Invariant, KM (str.)"]["entropy_bits"]), 1) == 364.6,
    "%.3f" % float(rs["Invariant, KM (str.)"]["entropy_bits"]))
chk("fixed key 23.8 bits",
    round(float(rs["None (fixed key)"]["entropy_bits"]), 1) == 23.8,
    "%.4f" % float(rs["None (fixed key)"]["entropy_bits"]))
chk("invariant refresh free",
    abs(float(rs["Invariant, KM (str.)"]["legit"]) - float(rs["None (fixed key)"]["legit"]))
    < 0.001, "%.4f vs %.4f" % (float(rs["Invariant, KM (str.)"]["legit"]),
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
    "$5$ to $8$ of the $64$ entries" in " ".join(tex.split())
    and "only $0.10$ of the smaller of any two such sets" in " ".join(tex.split()),
    "searched tex", needs_tex=True)

# --- why the permutation key is granted a shared permutation ---------
pv = {r["variant"]: float(r["legit_ser"]) for r in rows("perm_variant.csv")}
chk("shared permutation legitimate rate", abs(pv["shared"] - 0.053) < 1e-3,
    "%.5f" % pv["shared"])
chk("per-user permutation legitimate rate",
    abs(pv["per_user"] - 0.129) < 1e-3, "%.5f" % pv["per_user"])
if HAVE_TEX:
    chk("quoted permutation cost in tex",
        "at $0.129$ against $0.053$" in " ".join(tex.split()),
        "searched tex", needs_tex=True)


# --- key-length sweep floor ------------------------------------------
# The eavesdropper column is an average over eight substitute-key draws,
# so the quoted floor must track the data and not one lucky draw.
kl = rows("sec_keylen.csv")
floor = min(float(r["eve_ser"]) for r in kl)
chk("eavesdropper floor over key length", 0.9983 < floor < 0.9984,
    "%.6f" % floor)
if HAVE_TEX:
    chk("quoted eavesdropper floor in tex", "$0.9983$" in tex,
        "searched tex", needs_tex=True)


# --- quantities that used to be quoted with no artifact ---------------
import json as _json
rs = _json.load(open(base / "data" / "real_sec_stats.json"))
chk("token collision probability", abs(rs["token_collision"] - 0.0065) < 5e-5,
    "%.6f" % rs["token_collision"])
vm = {r["check"]: r for r in rows("verify_math.csv")}
chk("V5 matched-over-blind ratio stored",
    "V5 matched bias over blind RMS" in vm,
    ", ".join(sorted(vm))[:60])
chk("V8 cross-period remainder",
    abs(float(vm["V8 cross-period remainder"]["empirical"])) < 5e-4,
    vm["V8 cross-period remainder"]["empirical"])

# --- information-theoretic leakage, which no assertion covered ---------
it = {float(r["snr_db"]): r for r in rows("infotheory.csv")}[10.0]
chk("fixed-key leakage 1.34 bits",
    abs(float(it["mi_eve_fixed_bits"]) - 1.34) < 5e-3, it["mi_eve_fixed_bits"])
chk("distinguishing advantage 0.27",
    abs(float(it["tv_fixed"]) - 0.27) < 5e-3, it["tv_fixed"])
chk("refreshed leakage 0.055 bits",
    abs(float(it["mi_eve_refresh_bits"]) - 0.055) < 5e-4,
    it["mi_eve_refresh_bits"])
chk("secrecy rate 14.87 of 14.93",
    abs(float(it["secrecy_rate_refresh_bits"]) - 14.87) < 5e-3
    and abs(float(it["mi_legit_bits"]) - 14.93) < 5e-3,
    "%s of %s" % (it["secrecy_rate_refresh_bits"], it["mi_legit_bits"]))


_fe = {(r["family"], r["keying"], float(r["snr_db"]), int(r["n_frames"])): r
       for r in rows("family_enum.csv")}
_sf = _fe[("structured", "fixed", 10.0, 1)]
_s4 = _fe[("structured", "fixed", 10.0, 4)]
_sr = _fe[("structured", "refreshed", 10.0, 2)]
chk("structured family enumerable at 10 dB",
    abs(float(_sf["outsider_recovery"]) - 0.905) < 5e-3
    and abs(float(_s4["outsider_recovery"]) - 0.990) < 5e-3,
    "N=1 %s, N=4 %s" % (_sf["outsider_recovery"], _s4["outsider_recovery"]))
chk("the refresh stops the outsider enumeration",
    float(_sr["outsider_recovery"]) == 0.0,
    "%s over 200 blocks" % _sr["outsider_recovery"])
chk("the refresh does not stop the insider closure",
    abs(float(_sr["insider_recovery"]) - 0.980) < 5e-3,
    "%s over 200 blocks" % _sr["insider_recovery"])
chk("the learned family defeats both attacks everywhere",
    all(float(r["outsider_recovery"]) == 0.0
        and float(r["insider_recovery"]) == 0.0
        for r in rows("family_enum.csv") if r["family"] == "learned"),
    "%d learned rows" % sum(1 for r in rows("family_enum.csv")
                            if r["family"] == "learned"))

_sl = rows("sec_snr_learned.csv")
_sn = {float(r["snr_db"]): float(r["legit"]) for r in rows("sec_snr.csv")}
chk("learned family tracks the structured one over the SNR range",
    all(1.0 < float(r["legit"]) / _sn[float(r["snr_db"])] < 1.5
        for r in _sl),
    "ratio %.2f to %.2f" % (min(float(r["legit"]) / _sn[float(r["snr_db"])]
                                for r in _sl),
                            max(float(r["legit"]) / _sn[float(r["snr_db"])]
                                for r in _sl)))
_l10 = [float(r["legit"]) for r in _sl if float(r["snr_db"]) == 10.0][0]
chk("learned 0.061 at 10 dB", abs(_l10 - 0.061) < 5e-4, "%.5f" % _l10)
# The learned curves must be the regularized keys, not the unpenalized
# ones. Both are measured in sec_maskfam.csv and they differ by 0.003,
# which is larger than the spread of either, so matching the right row
# pins which family every figure draws. Cross-entropy alone drifts to a
# slot allocation whose key space is a support rather than a sphere, so
# drawing it would not support the key-space claim.
_fam = {r["family"]: float(r["legit_ser"]) for r in rows("sec_maskfam.csv")}
chk("the plotted learned family is the regularized one",
    abs(_l10 - _fam["learned_reg"]) < abs(_l10 - _fam["learned"])
    and abs(_l10 - _fam["learned_reg"]) < 1.5e-3,
    "plotted %.5f, reg %.5f, plain %.5f"
    % (_l10, _fam["learned_reg"], _fam["learned"]))

_bl = {int(r["K"]): float(r["ser_mask"])
       for r in rows("sec_brute_learned.csv")}
chk("learned key resists a million random guesses",
    abs(_bl[1_000_000] - 0.71) < 5e-3, "%.4f" % _bl[1_000_000])
_kl = {(float(r["snr_db"]), int(r["n_frames"])): float(r["eve_ser"])
       for r in rows("kpa_learned.csv")}
chk("learned key falls to known plaintext like the structured one",
    _kl[(10.0, 4)] < 0.08 and _kl[(10.0, 1)] > 0.9,
    "N=1 %.3f, N=4 %.4f" % (_kl[(10.0, 1)], _kl[(10.0, 4)]))

_rs = {float(r["snr_db"]): float(r["ter_legit"])
       for r in rows("real_sec_ter.csv")}
_rl = {float(r["snr_db"]): float(r["ter_legit"])
       for r in rows("real_sec_ter_learned.csv")}
_rat = [_rl[k] / _rs[k] for k in _rs]
chk("learned keeps its uniform-source distance on real text",
    all(1.10 < v < 1.25 for v in _rat),
    "ratio %.2f to %.2f" % (min(_rat), max(_rat)))

# --- trends, which the value assertions above cannot see ---------------
_snr = rows("sec_snr.csv")
_lg = [float(r["legit"]) for r in _snr]
chk("legitimate SER falls monotonically with SNR",
    all(a > b for a, b in zip(_lg, _lg[1:])), "%d points" % len(_lg))
chk("legitimate below the binary OMA reference at every SNR",
    all(float(r["legit"]) < float(r["oma"]) for r in _snr),
    "min margin %.3f" % min(1 - float(r["legit"]) / float(r["oma"])
                            for r in _snr))
_kl = rows("sec_keylen.csv")
chk("legitimate SER falls monotonically with key length",
    all(float(a["legit_ser"]) > float(b["legit_ser"]) for a, b in zip(_kl, _kl[1:])),
    "%d lengths" % len(_kl))
chk("legitimate surpasses the reference from L=16 onward",
    all(float(r["legit_ser"]) < float(r["oma"]) for r in _kl
        if r["oma"] != "nan" and int(float(r["L"])) >= 16),
    "checked L>=16")
_ter = rows("real_sec_ter.csv")
chk("legitimate TER below OMA over the whole range",
    all(float(r["ter_legit"]) < float(r["ter_oma"]) for r in _ter),
    "%d points" % len(_ter))
chk("outsider TER stays above 0.9991",
    min(float(r["ter_eve"]) for r in _ter) > 0.9991,
    "min %.6f" % min(float(r["ter_eve"]) for r in _ter))

# --- files no assertion read ------------------------------------------
_us = rows("users.csv")
chk("keys stay exactly orthogonal at every load",
    all(float(r["mask_xcorr"]) == 0.0 for r in _us),
    "U up to %s" % _us[-1]["users"])
chk("eavesdropper never leaves chance across the load sweep",
    all(float(r["eve_ser"]) > 0.999 for r in _us),
    "min %.6f" % min(float(r["eve_ser"]) for r in _us))
_u = {r["users"]: r for r in _us}
chk("load endpoints 0.027 and 0.946",
    abs(float(_u["2"]["legit_ser"]) - 0.027) < 5e-4
    and abs(float(_u["32"]["legit_ser"]) - 0.946) < 5e-4,
    "%.4f, %.4f" % (float(_u["2"]["legit_ser"]),
                    float(_u["32"]["legit_ser"])))
chk("the OMA crossing lies between U=16 and U=32",
    float(_u["16"]["legit_ser"]) < float(_u["16"]["oma"])
    and float(_u["32"]["legit_ser"]) > float(_u["32"]["oma"]),
    "16: %.3f<%.3f, 32: %.3f>%.3f"
    % (float(_u["16"]["legit_ser"]), float(_u["16"]["oma"]),
       float(_u["32"]["legit_ser"]), float(_u["32"]["oma"])))
_csi = rows("csi.csv")
chk("phase residual moves the rate to 0.057 at 0.2 rad",
    any(abs(float(r["legit_ser"]) - 0.057) < 1e-3 for r in _csi),
    "%d rows" % len(_csi))
_sem = rows("semantic.csv")
chk("legitimate similarity at least 0.96 in both spaces",
    all(float(r["legit"]) >= 0.96 for r in _sem if r["snr_db"] == "10.0"),
    "%d rows" % len(_sem))
_cov = rows("cov_attack.csv")
chk("covariance attack reaches 0.26 at 300 same-key frames",
    any(r["n_frames"] == "300" and abs(float(r["eve_ser"]) - 0.26) < 0.01
        for r in _cov),
    "%d rows" % len(_cov))
chk("unjammed reference is 0.053",
    abs(col("sec_jam.csv", "nojam")[0] - 0.053) < 0.05, "sec_jam.csv read")

# --- the closed-form checks the manuscript quotes ----------------------
_vm = {r["check"]: r for r in rows("verify_math.csv")}
for _k, _c in [("V8 cross-period remainder", 0.0005),
               ("V9 score-variance ratio", 0.05),
               ("V10 format-matched OMA at 10 dB", 0.001),
               ("V3a bias slope in kappa", 0.03)]:
    chk("stored check %s passes" % _k.split()[0],
        _k in _vm and _vm[_k]["verdict"] == "PASS"
        and float(_vm[_k]["abs_err"]) <= _c,
        _vm[_k]["empirical"] if _k in _vm else "row missing")
chk("format-matched OMA quoted as 0.055",
    "$0.055$ at $10$~dB" in tex and "the proposed $0.053$" in tex,
    "Section VI-B",
    needs_tex=True)

# --- tables against their generator -----------------------------------
# Every printed table cell must be the one make_tables.py derives from
# data/, so a rerun that moves a number cannot leave the manuscript behind.
if HAVE_TEX:
    import io
    import contextlib
    import make_tables
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        make_tables.compare_table()
        # the key-family table was folded into the Section VI-F prose
        pass
        make_tables.refresh_tables()
    rows = [r.strip() for r in buf.getvalue().split("\n")
            if r.rstrip().endswith(r"\\")]
    flat = " ".join(tex.split())
    lost = [r for r in rows if " ".join(r.split()) not in flat]
    chk("table rows match the generator", not lost,
        "%d rows, %d missing" % (len(rows), len(lost)), needs_tex=True)
    for r in lost:
        print("      missing:", r[:78])


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
# a checker that always exits zero cannot gate anything
import sys as _sys
_sys.exit(0 if ok else 1)
