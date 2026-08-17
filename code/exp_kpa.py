"""Stage H: known-plaintext attack on the keyed masking.

The masking is linear in the keys, so an attacker who knows the indices
carried by some frames can write one linear equation per dimension per
frame and solve for the keys by least squares. This script measures how
much known plaintext the attacker needs before the recovered key is good
enough to decode, and how receiver noise slows that recovery down.

Per dimension k the observation of frame n is
    y_k(n) = (1/c) sum_u e_{s_u(n),k} m_{u,k} + noise,
so stacking N frames gives A m_k = y_k with A(n,u) = e_{s_u(n),k}/c, an
N-by-U system that is solvable once N >= U in the noiseless case. The
attacker solves it per dimension, then correlates the estimate with the
true key and runs the correlation receiver with the estimated key.

Outputs:
  kpa.csv : key correlation and eavesdropper SER against the number of
            known-plaintext frames, at several SNRs
"""
from __future__ import annotations
import math

import numpy as np
import torch

from sse_lib import (DATA, DEVICE, SSE, rayleigh_gain, snr_to_sigma2,
                     set_seed, write_csv, eval_ser_sse)
from exp_full import main_model, eval_ser_eve

SNRS = [0.0, 10.0, 20.0]
NFRAMES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64]
SEED = 4242
EVAL_FRAMES = 50_000
# The spread across independent key-recovery attempts dominates the
# spread across channel realizations within one attempt, so the curve is
# smoothed by drawing many attempts rather than by lengthening each one.
TRIALS = 40


@torch.no_grad()
def collect_known_plaintext(model: SSE, n_frames: int, snr_db: float,
                            gen: torch.Generator):
    """Return (digits, raw observations, channel gains) for an attacker
    that knows the transmitted indices.

    The raw observation is returned rather than an equalized one. A
    maximum-likelihood attacker keeps the channel gain in the design
    matrix instead of dividing by it, which weights every frame by its
    own quality and is the strongest use of the collected material. It
    also avoids the numerical blow-up that equalizing a deep fade would
    cause.
    """
    digits = torch.randint(model.vu, (n_frames, model.users, model.P),
                           generator=gen).to(DEVICE)
    Bn = model.unit_codebook()
    m = model.masks()
    e = Bn[digits] / math.sqrt(model.P)
    y = (e * m[None, :, None, :]).sum(dim=1) / model.c      # (N,P,L)
    h = rayleigh_gain((n_frames,), device=DEVICE)
    sigma = snr_to_sigma2(snr_db, model.d).to(DEVICE).sqrt()
    noise = torch.randn(n_frames, model.P, model.L, device=DEVICE)
    obs = h[:, None, None] * y + sigma * noise
    return digits, obs, h


@torch.no_grad()
def solve_keys(model: SSE, digits, obs, h):
    """Maximum-likelihood key estimate from known plaintext.

    Each period of each frame is an independent observation of the same
    per-period key, so the P periods multiply the effective number of
    equations. For entry l the system is A x = b with
    A[(n,p), u] = h(n) e_{digit(n,u,p), l} / (c sqrt(P)) and b the raw
    observation, so a frame in a deep fade contributes a small row on
    both sides and is downweighted rather than amplified.
    """
    Bn = model.unit_codebook()          # (Vu, L)
    N, U, P = digits.shape
    L = model.L
    c = float(model.c)
    est = torch.zeros(U, L, device=DEVICE)
    for l in range(L):
        # design matrix over all (frame, period) pairs
        A = Bn[digits, l] / (c * math.sqrt(P))          # (N,U,P)
        A = A * h[:, None, None]
        A = A.permute(0, 2, 1).reshape(N * P, U)        # (N*P, U)
        b = obs[:, :, l].reshape(N * P, 1)              # (N*P, 1)
        # A pseudo-inverse with an absolute tolerance is used instead of
        # a least-squares driver. Training can leave a codebook entry
        # numerically dead, with every codeword value below the smallest
        # normal float. That entry carries no information about the
        # digit, and inverting its system would amplify noise without
        # bound, so the absolute tolerance discards it and the estimate
        # for that entry stays at zero, which is what a careful attacker
        # would do.
        sol = torch.linalg.pinv(A.double(), atol=1e-12, rtol=0.0) @ b.double()
        est[:, l] = sol[:, 0].float()
    est = torch.nan_to_num(est)
    # normalize to the key norm convention
    est = est / est.norm(dim=1, keepdim=True).clamp_min(1e-9) * math.sqrt(L)
    return est


def key_correlation(est: torch.Tensor, true: torch.Tensor) -> float:
    """Mean absolute normalized correlation over the users."""
    a = est / est.norm(dim=1, keepdim=True).clamp_min(1e-9)
    b = true / true.norm(dim=1, keepdim=True).clamp_min(1e-9)
    return float((a * b).sum(dim=1).abs().mean())


def main():
    set_seed(SEED)
    model = main_model()
    model.eval()
    true_m = model.masks().detach()
    legit = eval_ser_sse(model, [10.0], frames=200_000)[0]
    print(f"[kpa] legitimate SER at 10 dB = {legit:.4g}, U={model.users}, "
          f"L={model.L}")

    # Nested known-plaintext sets with common random numbers. Within a
    # trial the attacker collects one pool of frames and the N-frame
    # estimate uses the first N of them, so more material can only help,
    # exactly as an attacker accumulating traffic would experience. The
    # evaluation noise is also shared across N within a trial. Both
    # choices remove the between-point variance that would otherwise make
    # the averaged curve jagged, without changing what is being measured.
    nmax = max(NFRAMES)
    rows = []
    for snr in SNRS:
        acc = {n: [[], []] for n in NFRAMES}
        for t in range(TRIALS):
            gen = torch.Generator(device="cpu").manual_seed(
                SEED + int(snr) + 1000 * t)
            digits, obs, h = collect_known_plaintext(model, nmax, snr, gen)
            eval_seed = 777 + 31 * t + int(snr)
            for n in NFRAMES:
                est = solve_keys(model, digits[:n], obs[:n], h[:n])
                acc[n][0].append(key_correlation(est, true_m))
                acc[n][1].append(eval_ser_eve(model, est.cpu(), [10.0],
                                              frames=EVAL_FRAMES,
                                              seed=eval_seed)[0])
        for n in NFRAMES:
            ks, ss = acc[n]
            kappa = sum(ks) / len(ks)
            ser = sum(ss) / len(ss)
            rows.append((snr, n, kappa, ser))
            print(f"   snr={snr:4.1f} N={n:5d} kappa={kappa:.4f} "
                  f"eve_ser={ser:.4f}")
    write_csv(DATA / "kpa.csv",
              ["snr_db", "n_frames", "kappa", "eve_ser"], rows)
    print("[done] kpa.csv")


if __name__ == "__main__":
    main()
