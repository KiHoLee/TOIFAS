# -*- coding: utf-8 -*-
"""Why do structured keys beat learned ones on the legitimate error rate?

The gap is 0.053 against 0.064 at 10 dB, and a reader may reasonably
suspect that the learned keys are handicapped, since they alone are
trained under the channel while the structured ones are fixed by
construction. This measures where the gap comes from.

Three questions, one experiment each.

1. Is the gap a channel-adaptation failure? If it were, the two families
   would differ by more at some channel qualities than at others. The
   ratio across the SNR sweep answers this from data already on disk.

2. Is the structured key a point that training can improve on? Start
   training from the Walsh-Hadamard keys with the masks unfrozen and let
   Adam move them. If the structured point is a genuine optimum, the
   error rate stays or rises; if training is merely under-converged from
   its random start, it falls.

3. What does the learned key lose? Two candidates, measured directly:
   residual cross-user correlation, which the analysis names as the
   first-order leakage and interference term, and departure from unit
   modulus, which spreads the key energy unevenly across the entries so
   that a digit is decided over an effectively shorter support.

Run: python code/diag_whygap.py
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_full import (MAIN_D, base_keys, get_model, main_model,  # noqa: E402
                      mean_abs_xcorr)
from sse_lib import DATA, DEVICE, eval_ser_sse  # noqa: E402
import sse_lib as L  # noqa: E402

FRAMES = 300_000
SNR = 10.0


def modulus_stats(W):
    """How far the key entries are from unit modulus, per user.

    A unit-modulus key puts the same energy on every entry, so the
    signal term does not depend on the key and every entry of the period
    carries its share of the decision. The ratio below is the effective
    fraction of the L entries the key actually uses, by the
    participation ratio (sum a^2)^2 / (L sum a^4) with a the entry
    magnitudes. It is one for a unit-modulus key and 1/L for a key that
    puts everything on one entry.
    """
    a2 = W.pow(2)
    pr = a2.sum(dim=1).pow(2) / (W.shape[1] * a2.pow(2).sum(dim=1))
    return pr


def report(name, model, rows):
    W = model.masks().detach().cpu()
    ser = eval_ser_sse(model, [SNR], frames=FRAMES)[0]
    kap = mean_abs_xcorr(model.masks().detach())
    pr = modulus_stats(W)
    print("%-24s SER %.5f   kappa-bar %.5f   entry use %.3f"
          % (name, ser, kap, float(pr.mean())))
    rows.append([name, "%.5f" % ser, "%.5f" % kap, "%.4f" % float(pr.mean())])
    return ser


def main():
    rows = []
    print("main configuration d=%d, L=%d, 10 dB, %d frames\n"
          % (MAIN_D, MAIN_D // 4, FRAMES))

    print("-- the two families as the paper plots them --")
    fix = main_model()
    s_fix = report("structured (frozen)", fix, rows)
    free = get_model(iters=4000)
    s_free = report("learned (free start)", free, rows)

    print("\n-- question 2: can training improve on the structured key? --")
    # same trainer, same iterations, same seed, but the masks start at
    # the Walsh-Hadamard point and are free to move
    from sse_lib import SSE, set_seed
    set_seed(1)
    m = SSE(P=4, vu=16, d=MAIN_D, users=4).to(DEVICE)
    with torch.no_grad():
        m.W.copy_(base_keys(4, MAIN_D // 4).to(DEVICE))
    m.W.requires_grad_(True)
    L.train_sse(m, iters=4000, batch=256, lr=3e-3, seed=1)
    m.calibrate_power()
    s_warm = report("learned (Walsh start)", m, rows)

    print("\nreading:")
    print("  free start   %+.1f percent against the structured key"
          % (100.0 * (s_free - s_fix) / s_fix))
    print("  Walsh start  %+.1f percent against the structured key"
          % (100.0 * (s_warm - s_fix) / s_fix))
    if s_warm > s_fix:
        print("  training moves off the structured point and pays for it,")
        print("  so the structured key is not a point learning improves on.")
    else:
        print("  training improves on the structured point, so the gap is")
        print("  under-convergence from the random start, not geometry.")

    out = DATA / "whygap.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "legit_ser", "kappa_bar", "entry_use"])
        w.writerows(rows)
    print("\n[csv]", out)


if __name__ == "__main__":
    main()
