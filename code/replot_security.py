"""Canonical replot script for paper 11: regenerates every result figure
from ../data/*.csv and writes paper-ready PDFs to ../fig/. No experiment
is rerun. All result plots share one canvas and axes rectangle (8:6 box).
Label dictionary is fixed here and copied verbatim into tables and prose.

  fig_sec_snr.pdf     : legitimate and eavesdropper SER vs SNR (Fig. 2)
  fig_sec_keylen.pdf  : SER vs key length L (Fig. 3)
  fig_sec_jam.pdf     : target-user SER vs JSR, four schemes (Fig. 4)
  fig_sec_sens.pdf    : eavesdropper SER vs fraction of key held (Fig. 5)
  fig_sec_brute.pdf   : eavesdropper SER vs number of key guesses (Fig. 6)
  fig_sec_kpa.pdf     : eavesdropper SER vs known-plaintext frames (Fig. 7)
  fig_sec_real.pdf    : token error rate on real streams (Fig. 8)

Curves that coincide by construction are drawn deliberately layered, the
lower one wide and semi-transparent and the upper one narrow with open
markers, so every legend entry has a visible curve.
"""
from __future__ import annotations
from pathlib import Path
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "fig"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 7.4,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.4,
    "grid.alpha": 0.6,
    "lines.linewidth": 1.3,
    "lines.markersize": 3.4,
    "figure.figsize": (3.15, 2.36),
    "pdf.fonttype": 42,
})
AXES_RECT = dict(left=0.185, right=0.965, top=0.955, bottom=0.195)

C_LEGIT = "#c0392b"
C_EVE = "#2c5fa8"
C_OMA = "#7f8c8d"
C_CH = "#95a5a6"
C_MATCH = "#8e44ad"
C_PUB = "#16a085"

# fixed label dictionary: tables and prose copy these strings verbatim
LBL = {
    "legit": "Legitimate",
    "oma": "OMA",
    "eve_pub": "Eavesdropper, public masks",
    "eve_key": "Eavesdropper, wrong key",
    "chance": "Random guess",
    "nojam": "No jammer",
    "mask": "Keyed masking",
    "perm": "Permutation key",
    "pad": "Index cipher",
    "insider": "Insider",
    "outsider": "Outsider",
}
# deliberate-layering style for the LOWER of two coinciding curves
UNDER = dict(lw=2.6, ms=7, alpha=0.85)
# and for the curve riding on top of it
OVER = dict(lw=1.2, ms=4.5, mfc="none")


def load(name):
    with open(DATA / name) as f:
        return list(csv.DictReader(f))


def col(rows, k, f=float):
    return [f(r[k]) for r in rows]


def save(fig, name):
    """Write the figure and assert that no axis label is clipped.

    A long y label, or wide minor tick labels such as 6x10^-1 on a log
    axis that spans less than a decade, silently pushes the label off
    the canvas under the fixed axes rectangle. Reading the plotting code
    cannot reveal this, so the check is made on the rendered geometry.
    """
    fig.subplots_adjust(**AXES_RECT)
    fig.canvas.draw()
    fbox = fig.get_window_extent()
    for ax in fig.axes:
        for lbl in (ax.yaxis.label, ax.xaxis.label):
            if not lbl.get_text():
                continue
            b = lbl.get_window_extent()
            if (b.x0 < fbox.x0 or b.y0 < fbox.y0
                    or b.x1 > fbox.x1 or b.y1 > fbox.y1):
                raise RuntimeError(
                    f"{name}: axis label '{lbl.get_text()}' is clipped "
                    f"(label {b} outside figure {fbox}); shorten the "
                    f"label or widen the margin")
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("[OK]", name)


def fig_snr():
    r = load("sec_snr.csv")
    x = col(r, "snr_db")
    fig, ax = plt.subplots()
    # legitimate and OMA coincide by construction; layered deliberately
    ax.semilogy(x, col(r, "legit"), color=C_LEGIT, marker="o", ls="-",
                label=LBL["legit"], **UNDER)
    ax.semilogy(x, col(r, "oma"), color=C_OMA, marker="^", ls=":",
                label=LBL["oma"], **OVER)
    ax.semilogy(x, col(r, "eve_public"), color=C_PUB, marker="v",
                ls="none", markersize=5.2, markerfacecolor="none",
                label=LBL["eve_pub"])
    ax.semilogy(x, col(r, "eve_wrong"), color=C_EVE, marker="s", ls="--",
                label=LBL["eve_key"])
    ax.plot(x, col(r, "chance"), color=C_CH, ls="-.", lw=0.9,
            label=LBL["chance"])
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("SER")
    ax.set_xlim(min(x), max(x))
    ax.legend(loc="lower left")
    save(fig, "fig_sec_snr")


def fig_keylen():
    """The OMA reference is the resource-matched one of oma_ser_keylen,
    which is undefined below L=16 unless 16/L is an integer; those
    rows carry nan and are skipped."""
    r = load("sec_keylen.csv")
    x = col(r, "L", int)
    fig, ax = plt.subplots()
    ax.semilogy(x, col(r, "legit_ser"), color=C_LEGIT, marker="o", ls="-",
                label=LBL["legit"])
    op = [(l, v) for l, v in zip(x, col(r, "oma")) if not math.isnan(v)]
    ax.semilogy([p[0] for p in op], [p[1] for p in op], color=C_OMA,
                marker="^", ls=":", label=LBL["oma"])
    ax.semilogy(x, col(r, "eve_ser"), color=C_EVE, marker="s", ls="--",
                label=LBL["eve_key"])
    ax.set_xlabel("Key length $L$")
    ax.set_ylabel("SER")
    ax.set_xscale("log", base=2)
    ax.legend(loc="center right", bbox_to_anchor=(0.98, 0.72))
    save(fig, "fig_sec_keylen")


def fig_jam():
    """Target-user SER against JSR for four schemes. A linear axis is
    used because the range spans less than one decade, where a log axis
    would print wide minor tick labels that crowd out the y label. The
    no-jammer reference is annotated on the line rather than listed in
    the legend, so the legend never covers it."""
    r = load("sec_jam_cmp.csv")
    x = col(r, "jsr_db")
    me = max(1, len(x) // 8)
    fig, ax = plt.subplots()
    ax.plot(x, col(r, "matched"), color=C_MATCH, marker="P", ls="--",
            markevery=me, label=LBL["mask"] + ", matched")
    ax.plot(x, col(r, "oma_targeted"), color=C_PUB, marker="^", ls=":",
            markevery=me, label=LBL["oma"] + ", targeted")
    # the two blind curves agree to 0.002; deliberate layering
    ax.plot(x, col(r, "blind"), color=C_LEGIT, marker="o", ls="-",
            markevery=me, label=LBL["mask"] + ", blind", **UNDER)
    ax.plot(x, col(r, "perm_blind"), color=C_EVE, marker="s", ls="-.",
            markevery=me, label=LBL["perm"] + ", blind", **OVER)
    nojam = float(load("sec_jam.csv")[0]["nojam"])
    ax.axhline(nojam, color=C_OMA, ls=(0, (1, 3)), lw=0.9)
    ax.text(max(x) - 0.6, nojam + 0.02, LBL["nojam"], ha="right",
            va="bottom", fontsize=7.4, color="#555555")
    ax.set_xlabel("JSR (dB)")
    ax.set_ylabel("SER")
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(0.2, 1.02)
    ax.legend(loc="center right", bbox_to_anchor=(0.985, 0.47))
    save(fig, "fig_sec_jam")


def fig_sens():
    """Key sensitivity of three schemes on one axis, the fraction of the
    key the attacker holds. All three ride the random-guess level over
    most of the range, so the flat region is deliberately layered."""
    r = load("sec_sens_cmp.csv")
    x = col(r, "frac")
    fig, ax = plt.subplots()
    ax.plot(x, col(r, "ser_mask"), color=C_LEGIT, marker="o", ls="-",
            label=LBL["mask"], **UNDER)
    ax.plot(x, col(r, "ser_perm"), color=C_EVE, marker="s", ls="--",
            label=LBL["perm"], **OVER)
    ax.plot(x, col(r, "ser_pad"), color=C_PUB, marker="v", ls="-.",
            lw=1.2, ms=4.5, mfc="none", label=LBL["pad"])
    chance = 1.0 - (1.0 / 16.0) ** 4
    ax.axhline(chance, color=C_CH, ls=":", lw=0.9, label=LBL["chance"])
    ax.set_xlabel("Fraction of the key recovered")
    ax.set_ylabel("Eavesdropper SER")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower left")
    save(fig, "fig_sec_sens")


def fig_brute():
    """Brute-force search against the three keyed schemes at the same
    key length, each mapped through its own sensitivity curve."""
    r = load("sec_brute_cmp.csv")
    x = col(r, "K")
    fig, ax = plt.subplots()
    ax.semilogx(x, col(r, "ser_perm"), color=C_EVE, marker="s", ls="--",
                label=LBL["perm"], **UNDER)
    ax.semilogx(x, col(r, "ser_pad"), color=C_PUB, marker="v", ls="-.",
                label=LBL["pad"], **OVER)
    ax.semilogx(x, col(r, "ser_mask"), color=C_LEGIT, marker="o", ls="-",
                label=LBL["mask"])
    kl = load("sec_keylen.csv")
    legit = float([q for q in kl if int(q["L"]) == 16][0]["legit_ser"])
    ax.axhline(legit, color=C_OMA, ls=":", lw=0.9, label=LBL["legit"])
    ax.set_xlabel("Number of key guesses $K$")
    ax.set_ylabel("Eavesdropper SER")
    ax.set_ylim(0.2, 1.05)
    ax.legend(loc="lower left")
    save(fig, "fig_sec_brute")


def fig_real():
    r = load("real_sec_ter.csv")
    x = col(r, "snr_db")
    fig, ax = plt.subplots()
    # legitimate/OMA and insider/outsider coincide pairwise; layered
    ax.semilogy(x, col(r, "ter_legit"), color=C_LEGIT, marker="o", ls="-",
                label=LBL["legit"], **UNDER)
    ax.semilogy(x, col(r, "ter_oma"), color=C_OMA, marker="^", ls=":",
                label=LBL["oma"], **OVER)
    ax.semilogy(x, col(r, "ter_insider"), color=C_PUB, marker="v", ls="-.",
                lw=2.6, alpha=0.85, ms=7, label=LBL["insider"])
    ax.semilogy(x, col(r, "ter_eve"), color=C_EVE, marker="s", ls="--",
                label=LBL["outsider"], **OVER)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("TER")
    ax.set_xlim(min(x), max(x))
    ax.legend(loc="lower left")
    save(fig, "fig_sec_real")


def fig_kpa():
    """Known-plaintext recovery of the keyed masks at three collection
    SNRs, with the permutation key under the same attack as the linear
    comparison scheme."""
    r = load("kpa.csv")
    fig, ax = plt.subplots()
    sty = {0.0: ("#c0392b", "o"), 10.0: ("#2c5fa8", "s"),
           20.0: ("#16a085", "v")}
    for snr, (c, mk) in sty.items():
        rows = [row for row in r if float(row["snr_db"]) == snr]
        n = [float(row["n_frames"]) for row in rows]
        ser = [float(row["eve_ser"]) for row in rows]
        ax.semilogx(n, ser, color=c, marker=mk, ls="-",
                    label=LBL["mask"] + f", {int(snr)} dB")
    try:
        p = load("pkpa.csv")
        ax.semilogx(col(p, "n_frames"), col(p, "eve_ser"), color=C_MATCH,
                    marker="P", ls="--", label=LBL["perm"] + ", 20 dB")
    except FileNotFoundError:
        print("[skip] pkpa.csv not present yet")
    # legitimate reference measured with the SAME estimator as the
    # eavesdropper curves, namely the four-user average of eval_ser_sse
    # at L=16, taken from sec_keylen.csv rather than from the user-1
    # convention of the scheme-comparison table
    kl = load("sec_keylen.csv")
    legit = float([r for r in kl if int(r["L"]) == 16][0]["legit_ser"])
    ax.axhline(legit, color=C_OMA, ls=":", lw=0.9, label=LBL["legit"])
    ax.set_xlabel("Known-plaintext frames $N$")
    ax.set_ylabel("Eavesdropper SER")
    ax.set_xscale("log", base=2)
    ax.legend(loc="upper right")
    save(fig, "fig_sec_kpa")


def main():
    fig_snr()
    fig_keylen()
    fig_jam()
    try:
        fig_sens()
        fig_brute()
    except FileNotFoundError:
        print("[skip] attack-difficulty CSVs not present yet")
    try:
        fig_real()
    except FileNotFoundError:
        print("[skip] real-token CSV not present yet")
    try:
        fig_kpa()
    except FileNotFoundError:
        print("[skip] known-plaintext CSV not present yet")
    print("[done] figures in", FIG)


if __name__ == "__main__":
    main()
