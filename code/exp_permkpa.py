"""Known-plaintext attack on the global-permutation key (run under WSL).

The permutation scheme keeps the masks public and protects the frame
with one secret permutation of the d entries shared by all users. Like
the keyed masking, the protection is linear, so an attacker that knows
the indices a few frames carried can recover the secret. This stage
measures how many known frames the recovery needs, mirroring the grid
of exp_kpa.py so the two curves share one figure.

Attack: with N known frames the attacker knows the pre-permutation
signal x_n and observes y_n = h_n * perm(x_n) + noise at the collection
SNR. The cross-correlation matrix C[i, j] = sum_n y_n[i] x_n[j] peaks at
j = perm(i) because h_n > 0, so the permutation is the assignment that
maximizes the total correlation, solved by the Hungarian method. The
recovered permutation then decodes user 1 at 10 dB, the convention of
exp_kpa.py.

Writes data/pkpa.csv. Fixed seeds: permutation 11 (the stage-I secret),
collection 909.
"""
from __future__ import annotations
import math
import numpy as np
import torch

from sse_lib import write_csv, set_seed, DATA, DEVICE
from exp_full import main_model, eval_scheme_permuted_eve, rayleigh_gain

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:                                    # greedy fallback
    def linear_sum_assignment(cost):
        c = cost.copy()
        n = c.shape[0]
        rows = np.empty(n, dtype=int)
        cols = np.empty(n, dtype=int)
        for k in range(n):
            i, j = np.unravel_index(np.argmin(c), c.shape)
            rows[k], cols[k] = i, j
            c[i, :] = np.inf
            c[:, j] = np.inf
        order = np.argsort(rows)
        return rows[order], cols[order]

COLLECT_DB = 20.0
DECODE_DB = 10.0
# The curve's variance is dominated by WHICH positions the recovered
# permutation gets wrong, not by the SER estimate inside one trial: the
# within-trial standard deviation at 50k frames is 2e-3 while the
# trial-to-trial spread is ~1.6e-2. Averaging over many independent
# collections is therefore what smooths the curve, so trials are raised
# and per-trial frames lowered at roughly constant total cost.
TRIALS = 120
EVAL_FRAMES = 50_000


def main():
    m = main_model()          # training needs grad
    m.eval()
    _run(m)


@torch.no_grad()
def _run(m):
    d = m.P * m.L
    Bn = m.unit_codebook()
    true_m = m.masks()
    c = m.c
    gp = torch.Generator().manual_seed(11)
    gperm = torch.randperm(d, generator=gp)
    perms = gperm[None].repeat(m.users, 1)
    sigma = math.sqrt(1.0 / (d * 10.0 ** (COLLECT_DB / 10.0)))

    print(f"[P] permutation known-plaintext, collect {COLLECT_DB:.0f} dB, "
          f"decode {DECODE_DB:.0f} dB ...")
    rows = []
    for nf in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32, 48, 64]:
        fr, sr = [], []
        for t in range(TRIALS):
            g = torch.Generator().manual_seed(909 + 1000 * t + nf)
            digits = torch.randint(m.vu, (nf, m.users, m.P), generator=g)
            e = Bn[digits.to(DEVICE)] / math.sqrt(m.P)
            x = (e * true_m[None, :, None, :]).sum(dim=1) / c   # (nf,P,L)
            xf = x.reshape(nf, d)
            h = rayleigh_gain((nf,), device=DEVICE)
            noise = sigma * torch.randn(nf, d, device=DEVICE)
            yf = h[:, None] * xf[:, gperm.to(DEVICE)] + noise
            C = (yf.T @ xf).cpu().numpy()                       # (d,d)
            _, est = linear_sum_assignment(-C)
            est_t = torch.tensor(est, dtype=torch.long)
            fr.append(float((est_t == gperm).float().mean()))
            sr.append(eval_scheme_permuted_eve(
                m, DECODE_DB, EVAL_FRAMES, perms,
                eve_perms=est_t[None].repeat(m.users, 1),
                seed=777 + 31 * t))
        frac = sum(fr) / len(fr)
        ser = sum(sr) / len(sr)
        rows.append((nf, frac, ser))
        print(f"   N={nf:3d} frac={frac:.4f} eve={ser:.4f}")
    write_csv(DATA / "pkpa.csv", ["n_frames", "perm_frac", "eve_ser"], rows)
    print("[done] pkpa.csv")


if __name__ == "__main__":
    main()
