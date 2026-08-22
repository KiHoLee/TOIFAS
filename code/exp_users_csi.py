# -*- coding: utf-8 -*-
"""Two robustness sweeps the evaluation was missing.

Users. Every other stage fixes U=4. The structured family admits U up to
L-1, and as U approaches L the frame fills with cross-user patterns, so
this sweep asks what the load costs the legitimate users and whether the
confidentiality survives it.

Channel estimation. Every other stage equalizes with the exact gain.
Here the receiver divides by an estimate h+e with e zero mean and
variance sigma_e^2 relative to the gain, so the residual phase and
amplitude error enters the correlation the same way a key mismatch
would, and the question is how much of the legitimate margin it costs.

Run under WSL. Writes data/users.csv and data/csi.csv.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sse_lib as L
from sse_lib import DATA, DEVICE, rayleigh_gain, snr_to_sigma2, eval_ser_sse
from exp_full import (main_model, base_keys, get_model, eve_wrong_mask,
                      eval_ser_eve, mean_abs_xcorr, MAIN_D)

SNR = 10.0
FRAMES = 400_000
CHUNK = 40_000
USERS = [2, 4, 8, 16, 32, 48]
CSI = [0.0, 1e-3, 1e-2, 3e-2, 1e-1]
PHASE = [0.0, 0.02, 0.05, 0.10, 0.20]      # residual phase error, radians rms


@torch.no_grad()
def ser_with_csi_error(m, snr_db, nmse, frames=FRAMES, seed=606):
    """Legitimate SER when the receiver equalizes with a noisy estimate."""
    torch.manual_seed(seed + int(1e4 * nmse))
    wrong = tot = 0
    while tot < frames:
        n = min(CHUNK, frames - tot)
        dig = torch.randint(m.vu, (n, m.users, m.P), device=DEVICE)
        Bn = m.unit_codebook()
        e = Bn[dig] / math.sqrt(m.P)
        y = (e * m.masks()[None, :, None, :]).sum(dim=1) / m.c
        h = rayleigh_gain((n, m.users), device=DEVICE)
        sig = snr_to_sigma2(torch.full((n,), snr_db), m.d).to(DEVICE).sqrt()
        rx = h[:, :, None, None] * y[:, None] \
            + sig[:, None, None, None] * torch.randn(n, m.users, m.P, m.L,
                                                     device=DEVICE)
        # estimate with a zero-mean error of the stated relative variance
        hhat = h + math.sqrt(nmse) * h.abs() * torch.randn_like(h)
        r = rx / hhat[:, :, None, None].clamp_min(1e-6)
        cand = Bn[None, :, :] * m.masks()[:, None, :]
        dec = torch.einsum("nupl,uvl->nupv", r, cand).argmax(-1)
        wrong += int((dec != dig).any(dim=-1).sum())
        tot += n * m.users
    return wrong / tot


@torch.no_grad()
def ser_with_phase_error(m, snr_db, rms, frames=FRAMES, seed=707):
    """Legitimate SER under a residual phase error. Entries 2n-1 and 2n
    are the I and Q of one complex channel use, so an uncompensated
    phase rotates that pair. Unlike an amplitude error, this is not a
    common scale and the argmax is not invariant to it."""
    torch.manual_seed(seed + int(1e3 * rms))
    wrong = tot = 0
    half = m.L // 2
    while tot < frames:
        n = min(CHUNK, frames - tot)
        dig = torch.randint(m.vu, (n, m.users, m.P), device=DEVICE)
        Bn = m.unit_codebook()
        e = Bn[dig] / math.sqrt(m.P)
        y = (e * m.masks()[None, :, None, :]).sum(dim=1) / m.c
        h = rayleigh_gain((n, m.users), device=DEVICE)
        sig = snr_to_sigma2(torch.full((n,), snr_db), m.d).to(DEVICE).sqrt()
        noise = torch.randn(n, m.users, m.P, m.L, device=DEVICE)
        rx = h[:, :, None, None] * y[:, None] + sig[:, None, None, None] * noise
        r = rx / h[:, :, None, None].clamp_min(1e-6)
        if rms > 0:                       # rotate each I/Q pair
            th = rms * torch.randn(n, m.users, 1, half, device=DEVICE)
            v = r.reshape(n, m.users, m.P, half, 2)
            i, q = v[..., 0], v[..., 1]
            c_, s_ = th.cos(), th.sin()      # broadcast over periods
            r = torch.stack([i * c_ - q * s_, i * s_ + q * c_],
                            dim=-1).reshape(n, m.users, m.P, m.L)
        cand = Bn[None, :, :] * m.masks()[:, None, :]
        dec = torch.einsum("nupl,uvl->nupv", r, cand).argmax(-1)
        wrong += int((dec != dig).any(dim=-1).sum())
        tot += n * m.users
    return wrong / tot


def main():
    # --- users -------------------------------------------------------
    rows = []
    print("user load at %g dB, L=%d" % (SNR, MAIN_D // 4), flush=True)
    for U in USERS:
        Lp = MAIN_D // 4
        if U > Lp - 1:
            print("  U=%d exceeds L-1, skipped" % U, flush=True)
            continue
        m = get_model(d=MAIN_D, U=U, iters=4000, freeze_W=base_keys(U, Lp))
        m.eval()
        lg = eval_ser_sse(m, [SNR], frames=FRAMES)[0]
        ew = eve_wrong_mask(U, Lp, seed=20260813)
        ev = eval_ser_eve(m, ew, [SNR], frames=FRAMES)[0]
        xc = mean_abs_xcorr(m.masks().detach())
        rows.append((U, "%.6f" % lg, "%.6f" % ev, "%.6f" % xc))
        print("  U=%2d  legit %.4f  eve %.5f  xcorr %.2e"
              % (U, lg, ev, xc), flush=True)
    with open(DATA / "users.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["users", "legit_ser", "eve_ser", "mask_xcorr"])
        w.writerows(rows)
    print("[csv]", DATA / "users.csv", flush=True)

    # --- channel estimation error ------------------------------------
    m = main_model()
    m.eval()
    rows = []
    print("channel estimation error at %g dB" % SNR, flush=True)
    for nmse in CSI:
        s = ser_with_csi_error(m, SNR, nmse)
        rows.append(("%g" % nmse, "%.6f" % s))
        print("  nmse %-6g legit %.4f" % (nmse, s), flush=True)
    print("residual phase error at %g dB" % SNR, flush=True)
    prows = []
    for rms in PHASE:
        s_ = ser_with_phase_error(m, SNR, rms)
        prows.append(("%g" % rms, "%.6f" % s_))
        print("  phase rms %-5g legit %.4f" % (rms, s_), flush=True)
    with open(DATA / "csi.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["impairment", "level", "legit_ser"])
        w.writerows([("amplitude_nmse",) + r for r in rows]
                    + [("phase_rms_rad",) + r for r in prows])
    print("[csv]", DATA / "csi.csv")


if __name__ == "__main__":
    main()
