# -*- coding: utf-8 -*-
"""Information-theoretic security metrics for the main configuration.

The evaluation so far reported only the eavesdropper SER. This stage adds
the quantities a physical-layer-security reader expects, all computed
from the SAME Monte Carlo the SER curves use, so no new modelling
assumption enters.

Every metric is derived from the empirical joint law of the transmitted
digit and the DECISION each receiver makes. That decision is a
deterministic function of the received frame, so the data-processing
inequality makes each leakage number a LOWER bound on the true
I(s_u; y_E): what the modelled correlation eavesdropper actually
extracts. Reported per frame, an index carries P digits, so the frame
quantities are P times the per-digit ones under the independent-digit
source the evaluation uses.

  I(s;s_hat)      mutual information between the digit and the decision
  H(s|s_hat)      equivocation, and its ratio to log2(V)
  TV              distinguishing advantage, the average total variation
                  between the decision law given a digit and its marginal
  R_s             secrecy rate, the legitimate information rate minus the
                  eavesdropper one, per frame

Run under WSL. Writes data/infotheory.csv.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sse_lib as L
from sse_lib import DATA, DEVICE, rayleigh_gain, snr_to_sigma2
from exp_full import main_model, eve_wrong_mask
from exp_refresh import kdf_invariant, install

FRAMES = 400_000
CHUNK = 40_000
SNRS = [0.0, 5.0, 10.0, 15.0, 20.0]


@torch.no_grad()
def confusion_refreshed(m, snr_db, sub_key, base_keys, base_book,
                        blocks=64, frames=FRAMES, seed=4242):
    """The same joint counts when the key is redrawn from the invariance
    group every block, against an eavesdropper holding one fixed
    substitute. Each block contributes frames/blocks frames."""
    torch.manual_seed(seed + int(10 * snr_db))
    C = torch.zeros(m.vu, m.vu, dtype=torch.float64, device=DEVICE)
    per = max(CHUNK // 4, frames // blocks)
    for b in range(blocks):
        sg, cp, up = kdf_invariant(5150, b, m.users, m.L)
        install(m, sg * base_keys[up], base_book, colperm=cp)
        C += confusion(m, snr_db, sub_key=sub_key, frames=per,
                       seed=seed + 97 * b)
    install(m, base_keys, base_book)
    return C


@torch.no_grad()
def confusion(m, snr_db, sub_key=None, frames=FRAMES, seed=777):
    """Empirical joint counts of (transmitted digit, decided digit) for
    user 0, pooled over the P periods. sub_key None means the legitimate
    receiver; otherwise the eavesdropper substitutes that key."""
    torch.manual_seed(seed + int(10 * snr_db))
    C = torch.zeros(m.vu, m.vu, dtype=torch.float64, device=DEVICE)
    keys = m.masks() if sub_key is None else sub_key.to(DEVICE)
    done = 0
    while done < frames:
        n = min(CHUNK, frames - done)
        dig = torch.randint(m.vu, (n, m.users, m.P), device=DEVICE)
        Bn = m.unit_codebook()
        e = Bn[dig] / math.sqrt(m.P)
        y = (e * m.masks()[None, :, None, :]).sum(dim=1) / m.c
        h = rayleigh_gain((n, 1), device=DEVICE)
        sig = snr_to_sigma2(torch.full((n,), snr_db), m.d).to(DEVICE).sqrt()
        rx = h[:, :, None, None] * y[:, None] \
            + sig[:, None, None, None] * torch.randn(n, 1, m.P, m.L,
                                                     device=DEVICE)
        r = rx / h[:, :, None, None].clamp_min(1e-6)
        cand = Bn[None, :, :] * keys[:1, None, :]
        dec = torch.einsum("nupl,uvl->nupv", r, cand).argmax(-1)[:, 0]
        idx = dig[:, 0].reshape(-1) * m.vu + dec.reshape(-1)
        C += torch.bincount(idx, minlength=m.vu * m.vu).reshape(
            m.vu, m.vu).to(torch.float64)
        done += n
    return C


def metrics(C, P, V):
    """Mutual information, equivocation and distinguishing advantage from
    a joint count matrix, all in bits."""
    J = C / C.sum()
    px, py = J.sum(1), J.sum(0)
    nz = J > 0
    mi = float((J[nz] * (J[nz] / (px[:, None] * py[None, :])[nz]).log2()).sum())
    hx = float(-(px[px > 0] * px[px > 0].log2()).sum())
    equiv = hx - mi                       # H(digit | decision)
    # distinguishing advantage: E_s || p(dec|s) - p(dec) ||_TV
    cond = J / px[:, None].clamp_min(1e-300)
    tv = float((px * 0.5 * (cond - py[None, :]).abs().sum(1)).sum())
    return {"mi_digit": mi, "equiv_digit": equiv,
            "mi_frame": P * mi, "equiv_frame": P * equiv,
            "equiv_ratio": P * equiv / math.log2(V), "tv": tv}


def main():
    m = main_model()
    m.eval()
    ew = eve_wrong_mask(m.users, m.L, seed=20260813)
    base_keys = m.W.detach().clone().cpu()
    base_book = m.B.detach().clone().cpu()
    rows = []
    for snr in SNRS:
        lg = metrics(confusion(m, snr), m.P, m.V)
        ev = metrics(confusion(m, snr, sub_key=ew), m.P, m.V)
        rf = metrics(confusion_refreshed(m, snr, ew, base_keys, base_book),
                     m.P, m.V)
        rs = max(0.0, lg["mi_frame"] - ev["mi_frame"])
        rs_r = max(0.0, lg["mi_frame"] - rf["mi_frame"])
        rows.append((snr,
                     "%.4f" % lg["mi_frame"], "%.6f" % ev["mi_frame"],
                     "%.6f" % ev["equiv_ratio"], "%.6f" % ev["tv"],
                     "%.4f" % rs,
                     "%.6f" % rf["mi_frame"], "%.6f" % rf["equiv_ratio"],
                     "%.6f" % rf["tv"], "%.4f" % rs_r))
        print("%5.1f dB  legit %6.3f | fixed key: MI %.3f TV %.3f Rs %6.3f "
              "| refreshed: MI %.4f TV %.4f Rs %6.3f"
              % (snr, lg["mi_frame"], ev["mi_frame"], ev["tv"], rs,
                 rf["mi_frame"], rf["tv"], rs_r), flush=True)
    out = DATA / "infotheory.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snr_db", "mi_legit_bits",
                    "mi_eve_fixed_bits", "equiv_ratio_fixed", "tv_fixed",
                    "secrecy_rate_fixed_bits",
                    "mi_eve_refresh_bits", "equiv_ratio_refresh",
                    "tv_refresh", "secrecy_rate_refresh_bits"])
        w.writerows(rows)
    print("[csv]", out)


if __name__ == "__main__":
    main()
