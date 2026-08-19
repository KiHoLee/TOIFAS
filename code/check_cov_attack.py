"""Independent check of the ciphertext-only second-order attack (audit M1).

Claim under test: an eavesdropper who observes only received frames (no
known indices) can estimate the key Gram matrix M^T M from the sample
covariance, because per period
    E[y_k y_l] = (1/c^2) * E[h^2] * C_kl * (M^T M)_kl,
where C_kl = (1/Vu) sum_i b_{i,k} b_{i,l} is the PUBLIC codebook column
correlation and the noise touches only the diagonal.

Procedure, using nothing the threat model keeps secret:
  1. collect N received frames y_n = h_n * (1/c) sum_u e_{s_u} ⊙ m_u + noise
  2. form the per-period sample second moment S_kl = mean_n y_{n,k} y_{n,l}
  3. divide the off-diagonal by C_kl (public) to get G_hat ≈ M^T M
  4. set the diagonal of G_hat to U (unit-modulus keys)
  5. factor G_hat = M_hat^T M_hat (rank U), then for Walsh-Hadamard keys
     round to ±1 and search the 2^U U! signed permutations, keeping the
     M_hat that best decodes a handful of the collected frames
  6. report the recovered-entry fraction and the eavesdropper SER, both
     WITHOUT ever using a known index

Run under WSL. Writes data/cov_attack.csv so the manuscript sentence
it supports is traceable to a stored artifact.
"""
from __future__ import annotations
import itertools
import math
from pathlib import Path
import numpy as np
import torch

from sse_lib import rayleigh_gain, DEVICE
from exp_full import main_model, hadamard, eval_ser_eve


def collect_frames(m, n, snr_db, seed):
    """Received frames and the true indices (indices kept only for scoring)."""
    g = torch.Generator().manual_seed(seed)
    Bn = m.unit_codebook()
    true_m = m.masks()
    c = m.c
    sigma = math.sqrt(1.0 / (m.d * 10.0 ** (snr_db / 10.0)))
    digits = torch.randint(m.vu, (n, m.users, m.P), generator=g).to(DEVICE)
    e = Bn[digits] / math.sqrt(m.P)                     # (n,U,P,L)
    y = (e * true_m[None, :, None, :]).sum(dim=1) / c   # (n,P,L)
    h = rayleigh_gain((n,), device=DEVICE)
    y = h[:, None, None] * y + sigma * torch.randn(n, m.P, m.L, device=DEVICE)
    return y, digits, Bn, true_m


def codebook_corr(Bn):
    """Public column correlation C_kl = (1/Vu) sum_i b_ik b_il."""
    return (Bn.T @ Bn) / Bn.shape[0]                    # (L,L)


def attack(m, snr_db, n_frames, seed):
    y, digits, Bn, true_m = collect_frames(m, n_frames, snr_db, seed)
    U, L = m.users, m.L
    yf = y.reshape(-1, L)                               # pool all periods
    S = (yf.T @ yf) / yf.shape[0]                       # (L,L) 2nd moment
    C = codebook_corr(Bn)                               # public
    G = torch.zeros(L, L, device=DEVICE)
    mask = C.abs() > 1e-3
    G[mask] = S[mask] / C[mask]                         # ≈ (1/c^2) M^T M
    scale = float(torch.diagonal(G)[mask.diagonal()].mean()) / U
    G = G / max(scale, 1e-9)                            # normalize so diag≈U
    G.fill_diagonal_(float(U))                          # unit-modulus keys
    # symmetric rank-U factor
    G = 0.5 * (G + G.T)
    evals, evecs = torch.linalg.eigh(G)
    idx = torch.argsort(evals, descending=True)[:U]
    root = evecs[:, idx] * evals[idx].clamp_min(0).sqrt()
    Mhat0 = root.T                                      # (U,L), up to U×U orth
    # for WH keys, snap to ±1 and search signed row permutations
    cand = torch.sign(Mhat0)
    cand[cand == 0] = 1.0
    best = None
    best_ser = 1.0
    val = torch.arange(min(2000, n_frames))
    for perm in itertools.permutations(range(U)):
        for signs in itertools.product([1.0, -1.0], repeat=U):
            Mh = (cand[list(perm)] *
                  torch.tensor(signs, device=DEVICE)[:, None])
            ser = eval_ser_eve(m, Mh.cpu(), [snr_db], frames=20_000,
                               seed=13)[0]
            if ser < best_ser:
                best_ser, best = ser, Mh
    # recovered-entry fraction against the true keys (best sign-aligned)
    tm = torch.sign(true_m).to(DEVICE)
    frac = 0.0
    for perm in itertools.permutations(range(U)):
        for signs in itertools.product([1.0, -1.0], repeat=U):
            Mh = (best[list(perm)] *
                  torch.tensor(signs, device=DEVICE)[:, None])
            frac = max(frac, float((Mh == tm).float().mean()))
    return frac, best_ser


def main():
    import csv
    m = main_model()
    m.eval()
    chance = 1.0 - (1.0 / m.vu) ** m.P
    print(f"chance SER = {chance:.5f} at the main configuration")
    print("ciphertext-only (NO known plaintext):")
    rows = []
    for snr in (10.0, 20.0):
        for nf in (300, 1000, 10000):
            frac, ser = attack(m, snr, nf, seed=1234 + nf)
            rows.append((snr, nf, "%.4f" % frac, "%.4f" % ser))
            print(f"  {snr:4.0f} dB  N={nf:6d}  "
                  f"key-entry recovery={frac:.3f}  eve SER={ser:.4f}")
    out = Path(__file__).resolve().parents[1] / "data" / "cov_attack.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snr_db", "n_frames", "entry_recovery", "eve_ser"])
        w.writerows(rows)
    print("[csv]", out)


if __name__ == "__main__":
    main()
