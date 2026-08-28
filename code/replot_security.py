"""Canonical replot script for paper 11: regenerates every result figure
from ../data/*.csv and writes paper-ready PDFs to ../fig/. No experiment
is rerun. All result plots share one canvas and axes rectangle (8:6 box).
Label dictionary is fixed here and copied verbatim into tables and prose.

  fig_sec_snr.pdf     : legitimate and eavesdropper SER vs SNR (Fig. 2)
  fig_sec_keylen.pdf  : SER vs key length L (Fig. 3)
  fig_sec_jam.pdf     : target-user SER vs JSR, four cases (Fig. 4)
  fig_sec_sens.pdf    : eavesdropper SER vs fraction of key held (Fig. 5)
  fig_sec_brute.pdf   : eavesdropper SER vs number of key guesses (Fig. 6)
  fig_sec_kpa.pdf     : eavesdropper SER vs known-plaintext frames (Fig. 7)
  fig_sec_real.pdf    : token error rate on real streams (Fig. 8)

Curves that coincide by construction are drawn deliberately layered: the
lower one wide and semi-transparent, the upper one narrow with open
markers, and their markers staggered to different sample points through
markevery offsets. Marker size is uniform across every figure, so the
stagger, not the size, is what keeps each legend entry visible.
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
    # The manuscript includes each result figure at 0.74 of a 3.455 in
    # column while the canvas is 3.15 in, a printed scale of 0.812. Every
    # size below is therefore pre-divided by that scale so the PRINTED
    # sizes are 8 pt labels, 7.6 pt ticks and a 6 pt legend at the
    # smallest rung. Change the include width and these must change
    # with it.
    "font.size": 9.9,
    "axes.labelsize": 9.9,
    "legend.fontsize": 9.2,
    "xtick.labelsize": 9.4,
    "ytick.labelsize": 9.4,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.4,
    "grid.alpha": 0.6,
    "lines.linewidth": 1.5,
    "lines.markersize": 5.2,
    "figure.figsize": (3.15, 2.25),   # shorter canvas: same printed width and font size, less page height
    "pdf.fonttype": 42,
})
AXES_RECT = dict(left=0.215, right=0.970, top=0.955, bottom=0.225)

C_LEGIT = "#c0392b"     # KM, structured keys
C_LEARN = "#d98c00"     # KM, learned keys
C_OMA = "#7f8c8d"       # orthogonal multiple access
C_PUB = "#16a085"       # public masks
C_PERM = "#8e44ad"      # permutation key
C_PAD = "#a0522d"       # index cipher
C_EVE = "#2c5fa8"       # an adversary of KM
C_CH = "#95a5a6"        # chance and reference levels
C_MATCH = C_PUB         # the matched jammer is what public masks admit

# One entry per curve the figures draw. Colour identifies the scheme and
# line style the role: solid for a legitimate rate, dashed for an
# adversary, dash-dot for a comparison scheme, dotted for a reference.
# Every figure reads its curves from here, so a reader who learns a
# curve in one figure reads the same curve in the next.
STY = {
    "km_str":  dict(color=C_LEGIT, marker="o", ls="-"),
    "km_lrn":  dict(color=C_LEARN, marker="d", ls="-"),
    "oma":     dict(color=C_OMA, marker="^", ls=":"),
    "pub":     dict(color=C_PUB, marker="v", ls="-."),
    "perm":    dict(color=C_PERM, marker="X", ls="--"),
    "pad":     dict(color=C_PAD, marker="P", ls="-."),
    "eve":     dict(color=C_EVE, marker="s", ls="--"),
    "insider": dict(color=C_EVE, marker="v", ls="-."),
}

# fixed label dictionary: tables and prose copy these strings verbatim
LBL = {
    "legit": "KM (str.)",
    "legit_learned": "KM (lrn.)",
    "oma": "OMA",
    "eve_pub": "Eavesdropper, public masks",
    "eve_key": "Eavesdropper",     # the wrong-key condition is in the caption
    "chance": "Random guess",
    "nojam": "No jammer",
    "mask": "KM (str.)",
    "perm": "Permutation key",
    "pad": "Index cipher",
    "insider": "Insider",
    "outsider": "Outsider",
}
# deliberate-layering style for the LOWER of two coinciding curves
UNDER = dict(lw=2.6, alpha=0.85)          # thick filled line, layered under
# and for the curve riding on top of it
OVER = dict(lw=1.2, mfc="none")           # thin open marker, rides on top


def load(name):
    with open(DATA / name) as f:
        return list(csv.DictReader(f))


def col(rows, k, f=float):
    return [f(r[k]) for r in rows]


def _inflate(box, fig):
    """Grow a bounding box by the marker radius plus the line width, in
    pixels, so a marker whose CENTER clears the box cannot still touch
    its frame."""
    pad = (plt.rcParams["lines.markersize"] / 2.0
           + plt.rcParams["lines.linewidth"]) * fig.dpi / 72.0
    from matplotlib.transforms import Bbox
    return Bbox.from_extents(box.x0 - pad, box.y0 - pad,
                             box.x1 + pad, box.y1 + pad)


def save(fig, name, insets=()):
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
        # No curve may pass under the legend box. Reading the code cannot
        # reveal this, so the check is made on the rendered geometry, the
        # same discipline as the clipping guard above.
        leg = ax.get_legend()
        if leg is not None:
            raw = leg.get_window_extent()
            if (raw.x0 < fbox.x0 or raw.y0 < fbox.y0
                    or raw.x1 > fbox.x1 or raw.y1 > fbox.y1):
                raise RuntimeError(
                    f"{name}: the legend box leaves the canvas "
                    f"({raw} outside {fbox}); narrow or move it")
            lb = _inflate(raw, fig)
            for line in ax.get_lines():
                # full-span reference lines (axhline/axvline) carry axes-
                # fraction endpoints [0,1]; they are not data curves and,
                # spanning the whole axis, would forbid any bottom legend
                xd = list(line.get_xdata())
                if xd == [0, 1] or list(line.get_ydata()) == [0, 1]:
                    continue
                xy = line.get_xydata()
                if len(xy) == 0:
                    continue
                for px, py in ax.transData.transform(xy):
                    if lb.x0 <= px <= lb.x1 and lb.y0 <= py <= lb.y1:
                        raise RuntimeError(
                            f"{name}: a data curve passes under the legend "
                            f"box; move the legend or shrink it")
            for t in ax.texts:
                tb = t.get_window_extent()
                if (lb.x0 < tb.x1 and tb.x0 < lb.x1
                        and lb.y0 < tb.y1 and tb.y0 < lb.y1):
                    raise RuntimeError(
                        f"{name}: the annotation {t.get_text()!r} sits under "
                        f"the legend box; move one of them")
        for t in ax.texts:
            tb = t.get_window_extent()
            for line in ax.get_lines():
                xy = line.get_xydata()
                if len(xy) == 0:
                    continue
                for px, py in ax.transData.transform(xy):
                    if tb.x0 <= px <= tb.x1 and tb.y0 <= py <= tb.y1:
                        raise RuntimeError(
                            f"{name}: a curve is drawn through the "
                            f"annotation {t.get_text()!r}; move it")
    for ins in insets:
        ib = ins.get_window_extent()
        for a in fig.axes:
            if a is ins:
                continue
            for line in a.get_lines():
                xy = line.get_xydata()
                if len(xy) == 0:
                    continue
                for px, py in a.transData.transform(xy):
                    if ib.x0 <= px <= ib.x1 and ib.y0 <= py <= ib.y1:
                        raise RuntimeError(
                            f"{name}: a data curve passes under the inset "
                            f"panel; move or shrink the inset")
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("[OK]", name)


PL_CHOSEN = []      # sizes the sweep settled on, one per figure
PL_FORCED = None    # set by main() on its second pass


def main_legit(snr_db="10"):
    """The legitimate SER of the main configuration, read from the curve
    the main configuration produced rather than looked up by key length."""
    for r in load("sec_snr.csv"):
        if float(r["snr_db"]) == float(snr_db):
            return float(r["legit"])
    raise KeyError("no %s dB row in sec_snr.csv" % snr_db)



# Legend order, applied by place_legend to whatever subset a figure
# draws: the proposal first, then the comparison schemes in the order of
# Table IV, then adversaries, then reference levels. Entries not listed
# keep their plot order after the ranked ones.
LEGEND_ORDER = [
    "KM (str.)", "KM (lrn.)",
    "Public masks", "Permutation key", "Index cipher", "OMA",
    "Eavesdropper", "Eavesdropper, keyed", "Eavesdropper, public masks",
    "Outsider", "Insider",
    "No jammer", "Random guess",
]


def _rank(label):
    """Rank a legend label, matching the collection-SNR variants of
    Fig. 7 on their scheme prefix so they stay together and in order."""
    for i, name in enumerate(LEGEND_ORDER):
        if label == name or label.startswith(name + ","):
            return i
    return len(LEGEND_ORDER)


def place_legend(ax, cands=("lower left", "upper left", "center left",
                           "center right", "lower center", "upper right",
                           "upper center", "center", "lower right"),
                 sizes=(9.2, 8.8, 8.4, 8.0, 7.6, 7.2, 6.8, 6.4), ncol=1):
    """Choose the location and font size whose box the fewest curve points
    fall inside, scored on rendered geometry rather than guessed from the
    data. The size sweep is what makes a long label set placeable: a
    five-entry legend of full scheme names has no clear corner at the
    default size on every figure.

    The axes rectangle is applied first, because save() enforces the same
    test after applying it. Scoring the default layout and then checking
    a different one is how a placement that looked clear here failed
    there.

    When PL_FORCED is set, only that size is tried: the driver runs every
    figure once to learn the smallest size any of them needs, then reruns
    them all at that one size so the legends print uniformly."""
    ax.figure.subplots_adjust(**AXES_RECT)
    if PL_FORCED is not None:
        sizes = (PL_FORCED,)
    best = None
    for size in sizes:
        for loc in cands:
            h, l = ax.get_legend_handles_labels()
            idx = sorted(range(len(l)), key=lambda k: (_rank(l[k]), k))
            h = [h[k] for k in idx]
            l = [l[k] for k in idx]
            leg = ax.legend(h, l, loc=loc, prop={"size": size}, ncol=ncol,
                            handlelength=1.4, columnspacing=0.9,
                            handletextpad=0.5, borderaxespad=0.55,
                            framealpha=1.0)
            ax.figure.canvas.draw()
            lb = _inflate(leg.get_window_extent(), ax.figure)
            hits = 0
            for line in ax.get_lines():
                xy = line.get_xydata()
                if len(xy) == 0:
                    continue
                for px, py in ax.transData.transform(xy):
                    if lb.x0 <= px <= lb.x1 and lb.y0 <= py <= lb.y1:
                        hits += 1
            for t in ax.texts:
                tb = t.get_window_extent()
                if (lb.x0 < tb.x1 and tb.x0 < lb.x1
                        and lb.y0 < tb.y1 and tb.y0 < lb.y1):
                    hits += 50      # an annotation hidden is worse than a
                    #                 few curve points clipped
            if best is None or hits < best[2]:
                best = (loc, size, hits)
            if hits == 0:
                h, l = ax.get_legend_handles_labels()
                idx = sorted(range(len(l)), key=lambda k: (_rank(l[k]), k))
                h = [h[k] for k in idx]
                l = [l[k] for k in idx]
                ax.legend(h, l, loc=loc, prop={"size": size}, ncol=ncol,
                          handlelength=1.4, columnspacing=0.9,
                          handletextpad=0.5, borderaxespad=0.55,
                            framealpha=1.0)
                PL_CHOSEN.append(size)
                return best
    h, l = ax.get_legend_handles_labels()
    idx = sorted(range(len(l)), key=lambda k: (_rank(l[k]), k))
    h = [h[k] for k in idx]
    l = [l[k] for k in idx]
    ax.legend(h, l, loc=best[0], prop={"size": best[1]}, ncol=ncol,
              handlelength=1.4, columnspacing=0.9, handletextpad=0.5,
              borderaxespad=0.55, framealpha=1.0)
    PL_CHOSEN.append(best[1])
    return best


def fig_snr():
    r = load("sec_snr.csv")
    x = col(r, "snr_db")
    fig, ax = plt.subplots()
    # legitimate and the public-mask eavesdropper coincide by
    # construction (same physical layer, public masks decode alike), so
    # the pair is deliberately layered; OMA is separate at this frame
    ax.semilogy(x, col(r, "legit"), **STY["km_str"],
                markevery=(0, 3), label=LBL["legit"], **UNDER)
    # the learned family is the other end of the key-space trade-off,
    # so the figure carries what it costs at every SNR
    rl = load("sec_snr_learned.csv")
    ax.semilogy(col(rl, "snr_db"), col(rl, "legit"), **STY["km_lrn"],
                markevery=(2, 3), label=LBL["legit_learned"])
    ax.semilogy(x, col(r, "oma"), **STY["oma"],
                markevery=(1, 3), label=LBL["oma"], **OVER)
    ax.semilogy(x, col(r, "eve_public"), color=STY["pub"]["color"], marker=STY["pub"]["marker"],
                ls="none", markevery=(2, 3), markerfacecolor="none",
                label=LBL["eve_pub"])
    # this figure carries two eavesdroppers, so the bare label of the
    # key-length figure would not tell them apart
    ax.semilogy(x, col(r, "eve_wrong"), **STY["eve"], label="Eavesdropper, keyed")
    # the chance level lies within 3.5e-4 of the wrong-key curve, so it is
    # drawn for reference but left out of the legend, which the caption
    # names instead; five long entries leave this figure no clear corner
    ax.plot(x, col(r, "chance"), color=C_CH, ls="-.", lw=0.9)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("SER")
    ax.set_xlim(min(x), max(x))
    # most of a decade below the data leaves the lower-left genuinely
    # empty, which is what gives the legend a clear berth
    ax.set_ylim(bottom=2e-4)
    place_legend(ax)
    save(fig, "fig_sec_snr")


def fig_keylen():
    """The OMA reference is the resource-matched one of oma_ser_keylen,
    which is undefined below L=16 unless 16/L is an integer; those
    rows carry nan and are skipped."""
    r = load("sec_keylen.csv")
    x = col(r, "L", int)
    fig, ax = plt.subplots()
    ax.semilogy(x, col(r, "legit_ser"), **STY["km_str"], label=LBL["legit"])
    rl = load("sec_keylen_learned.csv")
    ax.semilogy(col(rl, "L", int), col(rl, "legit_ser"), **STY["km_lrn"], label=LBL["legit_learned"])
    op = [(l, v) for l, v in zip(x, col(r, "oma")) if not math.isnan(v)]
    ax.semilogy([p[0] for p in op], [p[1] for p in op], **STY["oma"], label=LBL["oma"])
    ax.semilogy(x, col(r, "eve_ser"), **STY["eve"], label=LBL["eve_key"])
    ax.set_ylim(top=22.0)    # headroom above the flat eavesdropper curve
    # an error rate cannot exceed one, and the room below the data holds
    # the legend, since every curve decays to the right
    ax.set_ylim(top=1.4, bottom=1.2e-2)
    ax.set_xlabel("Key length $L$")
    ax.set_ylabel("SER")
    ax.set_xscale("log", base=2)
    place_legend(ax)
    save(fig, "fig_sec_keylen")


def fig_jam():
    """Target-user SER against JSR in four cases. A linear axis is used
    because the range spans less than one decade, where a log axis would
    print wide minor tick labels that crowd out the y label. The
    no-jammer reference is named in the caption rather than in the
    legend, which keeps the legend four rows tall."""
    r = load("sec_jam_cmp.csv")
    x = col(r, "jsr_db")
    me = max(1, len(x) // 8)
    fig, ax = plt.subplots()
    rj = load("sec_jam_learned.csv")
    ax.plot(col(rj, "jsr_db"), col(rj, "blind"), **STY["km_lrn"],
            markevery=(1, me), label=LBL["legit_learned"])
    ax.plot(x, col(r, "matched"), **STY["pub"],
            markevery=me, label="Public masks")
    ax.plot(x, col(r, "oma_targeted"), **STY["oma"],
            markevery=me, label=LBL["oma"])
    # the two blind curves agree to 0.002; deliberate layering
    ax.plot(x, col(r, "blind"), **STY["km_str"],
            markevery=(0, me), label=LBL["mask"], **UNDER)
    ax.plot(x, col(r, "perm_blind"), **STY["perm"],
            markevery=(me // 2, me), label=LBL["perm"], **OVER)
    nojam = float(load("sec_jam.csv")[0]["nojam"])
    # the unjammed reference is named in the caption rather than in the
    # legend, which keeps the folded legend two rows tall
    # Behind the legend rather than through it. The placement guard skips
    # axis-spanning lines, so it cannot move the legend off this one, and
    # a reference drawn along the legend frame reads as part of the box.
    ax.axhline(nojam, color=C_OMA, ls=(0, (1, 3)), lw=0.9, zorder=0)
    ax.set_ylim(-0.42, 1.05)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("JSR (dB)")
    ax.set_ylabel("SER")
    ax.set_xlim(min(x), max(x))
    place_legend(ax)
    save(fig, "fig_sec_jam")


def fig_sens():
    """Key sensitivity of three schemes on one axis, the fraction of the
    key the attacker holds. All three ride the random-guess level over
    most of the range, so the flat region is deliberately layered."""
    r = load("sec_sens_cmp.csv")
    x = col(r, "frac")
    fig, ax = plt.subplots()
    ax.plot(x, col(r, "ser_mask"), **STY["km_str"],
            markevery=(0, 3), label=LBL["mask"], **UNDER)
    rs = load("sec_sens_learned.csv")
    ax.plot(col(rs, "frac"), col(rs, "ser_mask"), **STY["km_lrn"],
            markevery=(1, 3), label=LBL["legit_learned"])
    ax.plot(x, col(r, "ser_perm"), **STY["perm"],
            markevery=(1, 3), label=LBL["perm"], **OVER)
    ax.plot(x, col(r, "ser_pad"), **STY["pad"],
            markevery=(2, 3), lw=1.2, mfc="none", label=LBL["pad"])
    # the chance level comes from the stored curve, not from a second
    # copy of the configuration constants
    chance = float(load("sec_snr.csv")[0]["chance"])
    ax.axhline(chance, color=C_CH, ls=":", lw=0.9, label=LBL["chance"])
    ax.set_xlabel("Fraction of the key recovered")
    ax.set_ylabel("Eavesdropper SER")
    ax.set_xlim(0, 1)
    place_legend(ax)
    save(fig, "fig_sec_sens")


def fig_brute():
    """Brute-force search against the three keyed schemes at the same
    key length, each mapped through its own sensitivity curve."""
    r = load("sec_brute_cmp.csv")
    x = col(r, "K")
    fig, ax = plt.subplots()
    ax.semilogx(x, col(r, "ser_perm"), **STY["perm"],
                markevery=(0, 3), label=LBL["perm"], **UNDER)
    ax.semilogx(x, col(r, "ser_pad"), **STY["pad"],
                markevery=(1, 3), label=LBL["pad"], **OVER)
    ax.semilogx(x, col(r, "ser_mask"), **STY["km_str"],
                markevery=(2, 3), label=LBL["mask"])
    rb = load("sec_brute_learned.csv")
    ax.semilogx(col(rb, "K"), col(rb, "ser_mask"), **STY["km_lrn"],
                markevery=(1, 3), label=LBL["legit_learned"])
    ax.set_xlabel("Number of key guesses $K$")
    ax.set_ylabel("Eavesdropper SER")
    ax.set_ylim(0.0, 1.05)      # keep the reference line off the spine
    place_legend(ax)
    save(fig, "fig_sec_brute")


def fig_real():
    r = load("real_sec_ter.csv")
    x = col(r, "snr_db")
    fig, ax = plt.subplots()
    # insider and outsider still nearly coincide and are layered; the
    # legitimate and OMA curves are separate at this frame
    ax.semilogy(x, col(r, "ter_legit"), **STY["km_str"],
                markevery=(0, 2), label=LBL["legit"], **UNDER)
    rt = load("real_sec_ter_learned.csv")
    ax.semilogy(col(rt, "snr_db"), col(rt, "ter_legit"),
                **STY["km_lrn"], markevery=(1, 2),
                label=LBL["legit_learned"])
    ax.semilogy(x, col(r, "ter_oma"), **STY["oma"],
                markevery=(1, 2), label=LBL["oma"], **OVER)
    ax.semilogy(x, col(r, "ter_insider"), **STY["insider"],
                markevery=(0, 2), label=LBL["insider"], **UNDER)
    ax.semilogy(x, col(r, "ter_eve"), **STY["eve"],
                markevery=(1, 2), label=LBL["outsider"], **OVER)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("TER")
    ax.set_xlim(min(x), max(x))
    place_legend(ax)
    save(fig, "fig_sec_real")


def fig_kpa():
    """Known-plaintext recovery of the keyed masks at three collection
    SNRs, with the permutation key under the same attack as the linear
    comparison scheme."""
    r = load("kpa.csv")
    fig, ax = plt.subplots()
    sty = {0.0: (C_LEGIT, "o"), 10.0: (C_EVE, "s"),
           20.0: (C_PUB, "v")}
    for off, (snr, (c, mk)) in enumerate(sty.items()):
        rows = [row for row in r if float(row["snr_db"]) == snr]
        n = [float(row["n_frames"]) for row in rows]
        ser = [float(row["eve_ser"]) for row in rows]
        ax.semilogx(n, ser, color=c, marker=mk, ls="-",
                    markevery=(off, 4), markerfacecolor="none" if off else c,
                    label=f"KM (str.), {int(snr)} dB")
        if snr == 10.0:
            kl = [q for q in load("kpa_learned.csv")
                  if float(q["snr_db"]) == snr]
            ax.semilogx([float(q["n_frames"]) for q in kl],
                        [float(q["eve_ser"]) for q in kl], **STY["km_lrn"],
                        markevery=(2, 4),
                        label=f"KM (lrn.), {int(snr)} dB")
    try:
        p = load("pkpa.csv")
        ax.semilogx(col(p, "n_frames"), col(p, "eve_ser"), color=C_MATCH,
                    marker="P", ls="--", markevery=(3, 4),
                    label=LBL["perm"] + ", 20 dB")
    except FileNotFoundError:
        print("[skip] pkpa.csv not present yet")
    # legitimate reference measured with the SAME estimator as the
    # eavesdropper curves, namely the four-user average of eval_ser_sse
    # in the main configuration, rather than the user-1 convention of the
    # scheme-comparison table
    ax.set_xlabel("Known-plaintext frames $N$")
    ax.set_ylabel("Eavesdropper SER")
    ax.set_xscale("log", base=2)
    # the 0 dB curve sweeps the upper-right, so anchor the legend at the
    # top edge past the steep drops, above every curve at large N
    ax.set_ylim(top=1.18)
    place_legend(ax)
    save(fig, "fig_sec_kpa")


def run_all():
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


def main():
    """Two passes: the first learns the smallest legend size any figure
    needs, the second forces that one size everywhere so the legends
    print uniformly, which the figure standard requires."""
    global PL_FORCED
    PL_FORCED = None
    PL_CHOSEN.clear()
    run_all()
    if PL_CHOSEN:
        PL_FORCED = min(PL_CHOSEN)
        print("[uniform] legend size %.1f pt on every figure" % PL_FORCED)
        PL_CHOSEN.clear()
        run_all()
    print("[done] figures in", FIG)


if __name__ == "__main__":
    main()
