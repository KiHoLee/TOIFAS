"""Decisive noiseless population test of the covariance attack (audit M1).

If the second-order statistics leak the key Gram, they leak it best in
the noiseless infinite-sample limit. This computes the EXACT per-period
second moment E[y_k y_l] over the uniform index distribution with no
channel and no noise, then runs the same recovery, and asks whether the
keys come out. If they do not come out even here, no finite noisy attack
can do better and the leak is not exploitable against this codebook.
"""
from __future__ import annotations
import itertools
import math
import numpy as np
import torch

from sse_lib import DEVICE
from exp_full import get_model, hadamard, eval_ser_eve


def main():
    U, L = 4, 16
    K0 = torch.tensor(hadamard(L)[1:U + 1], dtype=torch.float32)
    m = get_model(iters=4000, freeze_W=K0)
    m.eval()
    Bn = m.unit_codebook().to(DEVICE)          # (Vu,L)
    true_m = m.masks().to(DEVICE)              # (U,L)
    c = m.c

    # exact population second moment of one period, indices uniform
    # y_k = (1/c) sum_u e_{s_u,k} m_{u,k}, s_u iid uniform over Vu
    mu = Bn.mean(0)                            # codebook column mean
    R = (Bn.T @ Bn) / Bn.shape[0]             # E[e_k e_l], (L,L)
    G_true = true_m.T @ true_m                 # (L,L) key Gram, the target
    # E[y_k y_l] = (1/c^2)[ R_kl (M^TM)_kl + (mu_k mu_l)(rowsum_k rowsum_l
    #                       - diag correction) ]; assemble exactly
    rs = true_m.sum(0)                         # sum_u m_{u,k}
    cross = torch.outer(rs, rs) - G_true       # sum_{u!=v} m_uk m_vl
    S = (R * G_true + torch.outer(mu, mu) * cross) / (c * c)

    print(f"codebook column mean |mu|_max = {mu.abs().max():.4f}")
    offdiag = R - torch.diag(torch.diagonal(R))
    print(f"codebook R off-diagonal: max|R_kl| = {offdiag.abs().max():.4f}, "
          f"mean|R_kl| = {offdiag.abs().mean():.4f}")

    # recover G from S using the public R (exactly the attack)
    C = R
    keep = C.abs() > 1e-2
    Ghat = torch.zeros(L, L, device=DEVICE)
    Ghat[keep] = S[keep] * (c * c) / C[keep]
    # how well does the off-diagonal of Ghat match the true key Gram?
    od = ~torch.eye(L, dtype=torch.bool, device=DEVICE)
    usable = keep & od
    if usable.any():
        err = (Ghat[usable] - G_true[usable]).abs().mean()
        rng = G_true[od].abs().mean()
        print(f"usable off-diagonal entries: {int(usable.sum())} of {L*(L-1)}")
        print(f"recovered-Gram error on usable entries: {err:.4f} "
              f"(true off-diag scale {rng:.4f})")
    else:
        print("no usable off-diagonal entries: R is diagonal, zero leak")

    # try to factor and decode from the exact-population Ghat
    Ghat[~keep] = 0.0
    Ghat.fill_diagonal_(float(U))
    Ghat = 0.5 * (Ghat + Ghat.T)
    ev, evec = torch.linalg.eigh(Ghat)
    idx = torch.argsort(ev, descending=True)[:U]
    root = (evec[:, idx] * ev[idx].clamp_min(0).sqrt()).T
    cand = torch.sign(root)
    cand[cand == 0] = 1.0
    best = 1.0
    for perm in itertools.permutations(range(U)):
        for sg in itertools.product([1.0, -1.0], repeat=U):
            Mh = cand[list(perm)] * torch.tensor(sg, device=DEVICE)[:, None]
            best = min(best, eval_ser_eve(m, Mh.cpu(), [10.0],
                                          frames=20_000, seed=13)[0])
    print(f"best eavesdropper SER from EXACT population covariance: {best:.4f}")
    print("chance 0.99998, legitimate ~0.276")


if __name__ == "__main__":
    main()
