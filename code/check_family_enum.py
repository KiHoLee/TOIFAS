# -*- coding: utf-8 -*-
"""Ciphertext-only enumeration of the structured key family.

Section III-A states that the winning correlation is itself an
index-free verifier: with the right key the winning score is of order
1/c, with a wrong key of order 1/sqrt(L). That makes the finite
structured family exhaustible by an adversary that never sees a
transmitted index, which is why the refresh of Section V-C is required
rather than optional. This script is the measurement behind that
claim.

The attack. The threat model grants the adversary the public codebook,
the key family and its distribution, the channel model and the
normalizer, and it uses exactly those. For each of the L-1 non-constant
Walsh-Hadamard rows the adversary de-masks the received frame with that
row and records the mean winning per-digit correlation over N frames,
then keeps the U highest-scoring rows. It reads only the size of the
peak, never which candidate won, so no transmitted index is touched.

It also runs the same attack against a refreshed key. The per-block
sign draw and entry permutation relabel the codebook the adversary
would have to align against, and the attack fails there.

Writes data/family_enum.csv.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from exp_full import base_keys, main_model
from sse_lib import DEVICE, rayleigh_gain, snr_to_sigma2, write_csv

DATA = Path(__file__).resolve().parents[1] / "data"
TRIALS = 200
SEED = 8131


@torch.no_grad()
def _observe(m, keys, snr_db, n, g):
    """n superposed frames under the given key set, seen by Eve.

    Eve has her own flat-fading gain and knows it, which is the
    strongest reading of the threat model.
    """
    Bn = m.unit_codebook()
    idx = torch.randint(m.vu, (n, m.users, m.P), generator=g, device=DEVICE)
    e = Bn[idx] / math.sqrt(m.P)                     # (n,U,P,L)
    y = (e * keys[None, :, None, :]).sum(dim=1) / m.c   # (n,P,L)
    h = rayleigh_gain((n, 1, 1), device=DEVICE)
    sig = float(snr_to_sigma2(torch.tensor(snr_db), m.d).sqrt())
    rx = h * y + sig * torch.randn(n, m.P, m.L, generator=g, device=DEVICE)
    return rx / h


@torch.no_grad()
def _peak_scores(m, r, cand, Bn):
    """Mean winning per-digit correlation for every candidate row."""
    out = torch.empty(cand.shape[0])
    for k in range(cand.shape[0]):
        z = torch.einsum("npl,vl->npv", r * cand[k][None, None, :], Bn)
        out[k] = z.max(dim=2).values.mean()
    return out


def run():
    torch.manual_seed(SEED)
    m = main_model()   # trains, so not under no_grad
    _attack(m)


@torch.no_grad()
def _attack(m):
    keys = m.masks()                                 # (U,L) the true rows
    cand = base_keys(m.L - 1, m.L).to(DEVICE)        # every non-constant row
    Bn = m.unit_codebook() / math.sqrt(m.P)
    rows = []

    for snr in (0.0, 10.0, 20.0):
        for n in (1, 2, 4):
            hit = 0
            for t in range(TRIALS):
                g = torch.Generator(device=DEVICE).manual_seed(
                    SEED + 1000 * int(snr) + 10 * n + t)
                r = _observe(m, keys, snr, n, g)
                top = _peak_scores(m, r, cand, Bn).topk(m.users).indices
                hit += int(set(int(i) for i in top) == set(range(m.users)))
            rows.append((snr, n, "fixed", hit / TRIALS))
            print("  %4.0f dB  N=%d  fixed      recovery %.3f"
                  % (snr, n, hit / TRIALS))

    hit = 0
    for t in range(TRIALS):
        g = torch.Generator(device=DEVICE).manual_seed(SEED + 77 + t)
        perm = torch.randperm(m.L, generator=g, device=DEVICE)
        sign = torch.randint(2, (m.L,), generator=g,
                             device=DEVICE) * 2.0 - 1.0
        rk = (keys * sign[None, :])[:, perm]
        r = _observe(m, rk, 10.0, 2, g)
        top = _peak_scores(m, r, cand, Bn).topk(m.users).indices
        hit += int(set(int(i) for i in top) == set(range(m.users)))
    rows.append((10.0, 2, "refreshed", hit / TRIALS))
    print("    10 dB  N=2  refreshed  recovery %.3f" % (hit / TRIALS))

    write_csv(DATA / "family_enum.csv",
              ["snr_db", "n_frames", "keying", "recovery"], rows)
    print("[csv]", DATA / "family_enum.csv")


if __name__ == "__main__":
    run()
