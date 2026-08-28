# -*- coding: utf-8 -*-
"""Key-space attacks against both key families.

The winning correlation is an index-free verifier: with the right key
the winning score is of order 1/c, with a wrong key of order
1/sqrt(L). Two attacks follow, and both need a LIST to rank.

  outsider  Rank the L-1 non-constant Walsh-Hadamard rows and keep the
            U best. Works only if the true keys are in that list.
  insider   A legitimate user holding m_v ranks m_v .* (row). Walsh
            rows are closed under the elementwise product, so this list
            contains every other user's key. The per-block sign draw
            cancels in m_u .* m_v, so the refresh does not remove it.

The structured family is countable and closed under the product, so
both attacks apply to it. A learned mask is a real vector in R^L, so
neither list contains the key and both attacks fail. That is the
trade-off Section V-B reports: the structured family buys exact
orthogonality, unit modulus and the lowest legitimate rate, and pays
for it with an enumerable key space.

Writes data/family_enum.csv.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from exp_full import MAIN_D, base_keys, get_model, main_model
from sse_lib import DEVICE, rayleigh_gain, snr_to_sigma2, write_csv

DATA = Path(__file__).resolve().parents[1] / "data"
TRIALS = 200
SEED = 8131


@torch.no_grad()
def _observe(m, keys, book, snr_db, n, g):
    """n superposed frames under the given keys and codebook, seen by an
    adversary with its own flat-fading gain, which it knows."""
    idx = torch.randint(m.vu, (n, m.users, m.P), generator=g, device=DEVICE)
    e = book[idx] / math.sqrt(m.P)
    y = (e * keys[None, :, None, :]).sum(dim=1) / m.c
    h = rayleigh_gain((n, 1, 1), device=DEVICE)
    sig = float(snr_to_sigma2(torch.tensor(snr_db), m.d).sqrt())
    rx = h * y + sig * torch.randn(n, m.P, m.L, generator=g, device=DEVICE)
    return rx / h


@torch.no_grad()
def _peak_scores(m, r, cand, book):
    """Mean winning per-digit correlation for every candidate key. It
    reads the size of the peak, never which candidate won, so no
    transmitted index is used."""
    out = torch.empty(cand.shape[0])
    for k in range(cand.shape[0]):
        out[k] = torch.einsum("npl,vl->npv", r * cand[k][None, None, :],
                              book).max(dim=2).values.mean()
    return out


def _recovers(rec, target, L):
    return any(float((rec[i] @ target).abs()) / L > 0.99
               for i in range(rec.shape[0]))


@torch.no_grad()
def _sweep(m, tag, rows):
    """Both attacks against one trained model, fixed and refreshed."""
    keys, book0 = m.masks(), m.unit_codebook()
    walsh = base_keys(m.L - 1, m.L).to(DEVICE)
    L, U = m.L, m.users

    for snr in (0.0, 10.0, 20.0):
        for n in (1, 2, 4):
            out = ins = 0
            for t in range(TRIALS):
                g = torch.Generator(device=DEVICE).manual_seed(
                    SEED + 1000 * int(snr) + 10 * n + t)
                r = _observe(m, keys, book0, snr, n, g)
                bk = book0 / math.sqrt(m.P)
                top = _peak_scores(m, r, walsh, bk).topk(U).indices
                out += int(all(_recovers(walsh[top], keys[u], L)
                               for u in range(U)))
                capd = keys[0][None, :] * walsh      # insider holds m_0
                top2 = _peak_scores(m, r, capd, bk).topk(U).indices
                ins += int(_recovers(capd[top2], keys[1], L))
            rows.append((tag, "fixed", snr, n, out / TRIALS, ins / TRIALS))
            print("  %-10s fixed      %4.0f dB N=%d  outsider %.3f  "
                  "insider %.3f" % (tag, snr, n, out / TRIALS, ins / TRIALS))

    # the refresh installs m_u = xi(eps .* m_u^0) and e_i = xi(e_i^0)
    out = ins = 0
    for t in range(TRIALS):
        g = torch.Generator(device=DEVICE).manual_seed(SEED + 77 + t)
        xi = torch.randperm(L, generator=g, device=DEVICE)
        eps = torch.randint(2, (L,), generator=g, device=DEVICE) * 2.0 - 1.0
        rk = (keys * eps[None, :])[:, xi]
        book = book0[:, xi]
        bk = book / math.sqrt(m.P)
        r = _observe(m, rk, book, 10.0, 2, g)
        top = _peak_scores(m, r, walsh, bk).topk(U).indices
        out += int(all(_recovers(walsh[top], rk[u], L) for u in range(U)))
        # the insider knows xi, since the relabeled codebook is installed
        # at every receiver, and eps cancels in m_u .* m_v
        capd = rk[0][None, :] * walsh[:, xi]
        top2 = _peak_scores(m, r, capd, bk).topk(U).indices
        ins += int(_recovers(capd[top2], rk[1], L))
    rows.append((tag, "refreshed", 10.0, 2, out / TRIALS, ins / TRIALS))
    print("  %-10s refreshed  10 dB N=2  outsider %.3f  insider %.3f"
          % (tag, out / TRIALS, ins / TRIALS))


def run():
    torch.manual_seed(SEED)
    rows = []
    _sweep(main_model(), "structured", rows)          # keys frozen to Walsh
    _sweep(get_model(P=4, vu=16, d=MAIN_D, U=4, iters=4000, seed=1),
           "learned", rows)                           # keys trained in R^L
    write_csv(DATA / "family_enum.csv",
              ["family", "keying", "snr_db", "n_frames",
               "outsider_recovery", "insider_recovery"], rows)
    print("[csv]", DATA / "family_enum.csv")


if __name__ == "__main__":
    run()
