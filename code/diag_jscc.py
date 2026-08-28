# -*- coding: utf-8 -*-
"""Does learning the mask buy anything, and where would it?

Nothing in the manuscript rests on this; it answers a design question
the paper does not raise. Output in data/jscc.csv.

What the run found, at 10 dB over three training seeds. A: with one
user the mask does not matter, a Walsh row and no mask at all landing
at 0.0133 and 0.0132 against 0.0148 for a learned key, so the mask
carries no part of the source-channel map and only separates users.
B: at the six key lengths where truncated Walsh rows are not exactly
orthogonal, learning wins once, at L=14 with kappa 0.048, and loses at
L=16, 20, 22 and 24 where the rows ARE orthogonal. C: raising the load
to U=20 at L=16 gives learning its second win, by 1e-5 on an error
rate of 0.9998, which is no win at all because both families have
already collapsed. Every other point is a tie inside the seed spread.

So the room for a learned mask is real but narrow, and it is where
exact orthogonality does not exist rather than where the load is high.
A learned mask beating a fixed one wants a loss that is not digit
cross-entropy, or a source that is not uniform, or a channel that is
not a scalar the receiver divides out.

A joint source-channel view says a learned mask should beat a fixed one,
since the fixed one lies inside the search space. It does not here, and
these experiments say why, and where the picture changes.

A. What job does the mask actually do? Run one user. With a single user
   there is nobody to separate from, so if the mask carried any part of
   the source-channel map its choice would still matter. Compare a Walsh
   row, no mask at all, and a learned key, each with the codebook
   trained around it. Equal error rates mean the mask is not part of
   that map: the codebook is, and the mask only separates users. This
   also has a one-line proof. A unit-modulus key has m^2 = 1, so it
   cancels from the signal self-term and from the noise projection
   alike, and the score is unchanged.

B. Where is the structured family no longer optimal? The construction
   supplies exactly orthogonal unit-modulus rows only at the lengths
   where truncation preserves orthogonality. At L = 6, 10, 14, 18, 20
   and 22 the truncated rows correlate, so no exactly orthogonal
   unit-modulus family is available and learning has room to find a
   better packing. Every length is run at several seeds, because a
   single training run is not evidence of a family being better.

C. Overload. Beyond U = L - 1 no orthogonal set of non-constant rows
   exists at all, so the structured family has to reuse rows and the
   comparison is decided by whatever packing learning finds.

Run on a GPU host:  python code/diag_jscc.py
"""
from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_full import (MAIN_D, base_keys, get_model_reg,  # noqa: E402
                      mean_abs_xcorr)
from sse_lib import DATA, DEVICE, SSE, eval_ser_sse, set_seed  # noqa: E402
import sse_lib as L  # noqa: E402

SNR = 10.0
SEEDS = [1, 2, 3]
FRAMES = 400_000


def train_with_key(W0, d, P, vu, U, iters=4000, seed=1, frozen=True):
    """Train the codebook around a given key, optionally holding it."""
    set_seed(seed)
    m = SSE(P=P, vu=vu, d=d, users=U).to(DEVICE)
    with torch.no_grad():
        m.W.copy_(W0.to(DEVICE))
    m.W.requires_grad_(not frozen)
    L.train_sse(m, iters=iters, batch=256, lr=3e-3, seed=seed)
    m.calibrate_power()
    return m


def ms(vals):
    """Mean and, when there is more than one, the sample spread."""
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def part_a(rows):
    """One user: does the choice of mask matter at all?"""
    print("-- A. one user, d=%d, L=%d, %d seeds --"
          % (MAIN_D, MAIN_D // 4, len(SEEDS)))
    Lp = MAIN_D // 4
    cases = [("Walsh row", base_keys(1, Lp), True),
             ("all ones (no mask)", torch.ones(1, Lp), True),
             ("learned, free", base_keys(1, Lp), False)]
    for name, W0, frozen in cases:
        v = [eval_ser_sse(train_with_key(W0, MAIN_D, 4, 16, 1, seed=s,
                                         frozen=frozen),
                          [SNR], frames=FRAMES)[0] for s in SEEDS]
        mu, sd = ms(v)
        print("   %-20s SER %.5f +- %.5f" % (name, mu, sd))
        rows.append(["A one user", name, "%.5f" % mu, "%.5f" % sd, "", ""])


def part_b(rows):
    """Key lengths where no exactly orthogonal unit-modulus family exists."""
    print("\n-- B. key length, U=4, %d seeds --" % len(SEEDS))
    print("   %-4s %-9s %-18s %-18s %s"
          % ("L", "kappa str", "structured SER", "learned SER", "verdict"))
    for Lp in [6, 8, 10, 12, 14, 16, 18, 20, 22, 24]:
        d = 4 * Lp
        try:
            W0 = base_keys(4, Lp)
        except ValueError as e:
            print("   %-4d skipped: %s" % (Lp, e))
            continue
        ks = mean_abs_xcorr(W0)
        vs = [eval_ser_sse(train_with_key(W0, d, 4, 16, 4, seed=s),
                           [SNR], frames=FRAMES)[0] for s in SEEDS]
        vl = [eval_ser_sse(get_model_reg(P=4, vu=16, d=d, U=4, iters=4000,
                                         seed=s),
                           [SNR], frames=FRAMES)[0] for s in SEEDS]
        (mus, sds), (mul, sdl) = ms(vs), ms(vl)
        # a win only counts when it clears the spread of both runs
        win = "learned" if mul + sdl < mus - sds else (
            "structured" if mus + sds < mul - sdl else "tie")
        print("   %-4d %-9.5f %.5f +- %.5f  %.5f +- %.5f  %s"
              % (Lp, ks, mus, sds, mul, sdl, win))
        rows.append(["B key length", "L=%d" % Lp, "%.5f" % mus,
                     "%.5f" % sds, "%.5f" % mul, "%.5f/%s" % (ks, win)])


def part_c(rows):
    """Overload: more users than the construction has orthogonal rows."""
    print("\n-- C. load at L=16, %d seeds --" % len(SEEDS))
    Lp, d = 16, 64
    print("   %-4s %-9s %-18s %-18s %s"
          % ("U", "kappa str", "structured SER", "learned SER", "verdict"))
    for U in [4, 8, 12, 15, 16, 20]:
        try:
            W0 = base_keys(U, Lp)
        except ValueError:
            # beyond the orthogonal rows the construction has to reuse
            # them, which is the honest structured fallback
            H = base_keys(Lp - 1, Lp)
            W0 = H[[i % (Lp - 1) for i in range(U)]]
        ks = mean_abs_xcorr(W0)
        vs = [eval_ser_sse(train_with_key(W0, d, 4, 16, U, seed=s),
                           [SNR], frames=FRAMES)[0] for s in SEEDS]
        vl = [eval_ser_sse(get_model_reg(P=4, vu=16, d=d, U=U, iters=4000,
                                         seed=s),
                           [SNR], frames=FRAMES)[0] for s in SEEDS]
        (mus, sds), (mul, sdl) = ms(vs), ms(vl)
        win = "learned" if mul + sdl < mus - sds else (
            "structured" if mus + sds < mul - sdl else "tie")
        print("   %-4d %-9.5f %.5f +- %.5f  %.5f +- %.5f  %s"
              % (U, ks, mus, sds, mul, sdl, win))
        rows.append(["C load", "U=%d" % U, "%.5f" % mus, "%.5f" % sds,
                     "%.5f" % mul, "%.5f/%s" % (ks, win)])


def main():
    print("device", DEVICE)
    rows = []
    part_a(rows)
    part_b(rows)
    part_c(rows)
    out = DATA / "jscc.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["part", "case", "structured_ser", "structured_sd",
                    "learned_ser", "kappa_and_verdict"])
        w.writerows(rows)
    print("\n[csv]", out)


if __name__ == "__main__":
    main()
