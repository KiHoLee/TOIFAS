# -*- coding: utf-8 -*-
"""Does unconstrained key training still degenerate at the main configuration?

The manuscript justifies fixing the keys by a measured failure: with the
keys free, training drives them to disjoint sparse supports, which is an
orthogonal slot allocation rather than a superposition, and which shrinks
the key space to the choice of a support. That was measured at d=64 and
has to be re-measured whenever the configuration moves, because it is
the reason the structured family is the main one.

Reported per user key: the number of entries holding 99 percent of the
energy, and the pairwise overlap of those supports. A dense key spreads
its energy over most of the L entries and the supports coincide; a
degenerate one concentrates on a few and the supports are disjoint.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_full import get_model, main_model, MAIN_D


def support99(w):
    """Smallest set of entries carrying 99 percent of the key energy."""
    e = w.pow(2)
    order = torch.argsort(e, descending=True)
    c = torch.cumsum(e[order], 0) / e.sum()
    k = int((c < 0.99).sum()) + 1
    return set(order[:k].tolist()), k


def describe(name, W):
    L = W.shape[1]
    sups, ks = [], []
    for u in range(W.shape[0]):
        sup, k = support99(W[u])
        sups.append(sup)
        ks.append(k)
    ov = []
    for i in range(len(sups)):
        for j in range(i + 1, len(sups)):
            ov.append(len(sups[i] & sups[j]) / max(1, min(len(sups[i]),
                                                          len(sups[j]))))
    print("%-14s L=%3d  99%%-energy entries per key: %s  "
          "mean pairwise support overlap %.2f"
          % (name, L, ks, sum(ov) / len(ov)))


def main():
    print("main configuration d=%d" % MAIN_D)
    m_free = get_model(iters=4000)            # keys learned, nothing frozen
    describe("learned", m_free.masks().detach().cpu())
    m_fix = main_model()
    describe("Walsh-Hadamard", m_fix.masks().detach().cpu())
    print()
    print("A degenerate key set shows few entries per key and near-zero")
    print("overlap; a dense one shows most entries and overlap near one.")


if __name__ == "__main__":
    main()
