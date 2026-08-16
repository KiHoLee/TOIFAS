"""Feasibility study for paper 11 (TIFS): the per-user mask as a
physical-layer key.

Three questions, all under the shared-embedding multiple-access model of
sse_lib.py (real-vector convention, flat Rayleigh fading):

  Q1 (encryption): a legitimate receiver knows its mask mu_u; an
     eavesdropper (Eve) does not. How far above chance can Eve decode?
     We measure the legitimate symbol error rate (SER) against Eve's SER
     when Eve applies (a) a wrong mask drawn from the same distribution,
     (b) no mask (mu = 1), (c) the average mask. Chance level is
     (Vu-1)/Vu per digit, 1-(1/Vu)^P per frame.

  Q2 (key entropy vs dimension): as the per-period length L grows, two
     independently drawn unit-norm masks become more nearly orthogonal,
     so Eve's residual after de-masking with a wrong key grows. We sweep
     L and report Eve's SER and the mean absolute mask cross-correlation.

  Q3 (jamming robustness): a jammer adds h_J * w to the frame, where w is
     an arbitrary unit waveform (worst case: aligned with the victim's
     masked codeword direction; and random). We sweep the
     jammer-to-signal ratio (JSR) and report the legitimate SER, to show
     the mask spreads a mismatched jammer and bounds its effect.

This is a CPU-sized feasibility run (small V), not the final experiment.
Seeds fixed; results written to ../data as CSV.
"""
from __future__ import annotations

import math
import numpy as np
import torch

import sse_lib as L
from sse_lib import SSE, rayleigh_gain, snr_to_sigma2, write_csv, set_seed, DATA, DEVICE


# ----------------------------------------------------------------------
# Eve: apply a chosen (wrong) set of masks to the SAME received frame the
# legitimate users see, then run the correlation receiver.
# ----------------------------------------------------------------------
@torch.no_grad()
def eval_ser_eve(model: SSE, eve_masks: torch.Tensor, snr_list,
                 frames: int = 400_000, chunk: int = 50_000, seed: int = 777):
    """eve_masks: (U, L) the masks Eve uses in place of the true ones.
    Eve observes the same physically transmitted frame (true masks used at
    the transmitter) but correlates with eve_masks."""
    model.eval().to(DEVICE)
    Bn = model.unit_codebook()
    true_m = model.masks()
    eve_masks = eve_masks.to(DEVICE)
    c = model.c
    out = []
    for snr_db in snr_list:
        g = torch.Generator(device="cpu").manual_seed(seed + int(10 * snr_db))
        err = tot = 0
        for n0 in range(0, frames, chunk):
            n = min(chunk, frames - n0)
            digits = torch.randint(model.vu, (n, model.users, model.P),
                                   generator=g).to(DEVICE)
            # transmit with the TRUE masks
            e = Bn[digits] / math.sqrt(model.P)
            y = (e * true_m[None, :, None, :]).sum(dim=1) / c        # (n,P,L)
            h = rayleigh_gain((n, model.users), device=DEVICE)
            sigma = snr_to_sigma2(snr_db).to(DEVICE).sqrt()
            noise = torch.randn(n, model.users, model.P, model.L, device=DEVICE)
            y_rx = h[:, :, None, None] * y[:, None] + sigma * noise
            r = y_rx / h[:, :, None, None].clamp_min(1e-6)           # (n,U,P,L)
            # Eve correlates with her (wrong) masks
            cand = Bn[None, :, :] * eve_masks[:, None, :]            # (U,Vu,L)
            scores = torch.einsum("nupl,uvl->nupv", r, cand)
            wrong = (scores.argmax(-1) != digits).any(dim=2)
            err += int(wrong.sum()); tot += n * model.users
        out.append(err / tot)
    return out


@torch.no_grad()
def eval_ser_jam(model: SSE, snr_db, jsr_db_list, frames: int = 400_000,
                 chunk: int = 50_000, seed: int = 777, mode: str = "aligned"):
    """Legitimate SER with an added jammer h_J * sqrt(JSR) * w.
    mode='aligned': w points along user 0's masked mean codeword direction
    (a structured, mask-matched worst case for user 0).
    mode='random': w is an isotropic random unit frame each transmission."""
    model.eval().to(DEVICE)
    Bn = model.unit_codebook()
    true_m = model.masks()
    c = model.c
    sigma = snr_to_sigma2(snr_db).to(DEVICE).sqrt()
    # aligned jammer direction: mask-0 applied to a fixed unit codeword,
    # i.e. what an attacker would build if it copied the public codebook
    # but guessed the (secret) mask wrong -> here we give it mask 0 exactly
    # as the strongest realistic structured jammer.
    w_fixed = (Bn[0][None, :] * true_m[0][None, :]).repeat(model.P, 1)  # (P,L)
    w_fixed = w_fixed / w_fixed.norm()
    out = []
    for jsr_db in jsr_db_list:
        jsr = 10.0 ** (jsr_db / 10.0)
        g = torch.Generator(device="cpu").manual_seed(seed + int(10 * jsr_db))
        err = tot = 0
        for n0 in range(0, frames, chunk):
            n = min(chunk, frames - n0)
            digits = torch.randint(model.vu, (n, model.users, model.P),
                                   generator=g).to(DEVICE)
            e = Bn[digits] / math.sqrt(model.P)
            y = (e * true_m[None, :, None, :]).sum(dim=1) / c        # (n,P,L)
            h = rayleigh_gain((n, model.users), device=DEVICE)
            hJ = rayleigh_gain((n,), device=DEVICE)
            if mode == "aligned":
                w = w_fixed[None].expand(n, model.P, model.L)
            else:
                w = torch.randn(n, model.P, model.L, device=DEVICE)
                w = w / w.reshape(n, -1).norm(dim=1)[:, None, None].clamp_min(1e-8)
            jam = (hJ * math.sqrt(jsr))[:, None, None] * w           # (n,P,L)
            noise = torch.randn(n, model.users, model.P, model.L, device=DEVICE)
            y_rx = (h[:, :, None, None] * y[:, None]
                    + h[:, :, None, None] * 0  # keep shape clarity
                    + jam[:, None] + sigma * noise)
            r = y_rx / h[:, :, None, None].clamp_min(1e-6)
            cand = Bn[None, :, :] * true_m[:, None, :]
            scores = torch.einsum("nupl,uvl->nupv", r, cand)
            wrong = (scores.argmax(-1) != digits).any(dim=2)
            err += int(wrong.sum()); tot += n * model.users
        out.append(err / tot)
    return out


def mean_abs_cross_corr(masks: torch.Tensor) -> float:
    """Mean |<mu_i, mu_j>| / (||mu_i|| ||mu_j||) over i<j."""
    m = masks / masks.norm(dim=1, keepdim=True).clamp_min(1e-8)
    G = (m @ m.T).abs()
    U = m.shape[0]
    off = G[~torch.eye(U, dtype=torch.bool, device=G.device)]
    return float(off.mean())


def main():
    set_seed(1)
    # CPU-sized feasibility configuration: V = Vu^P = 16^2 = 256
    P, VU, D, U = 2, 16, 64, 4
    snr_eval = [0.0, 5.0, 10.0, 15.0, 20.0]
    chance_frame = 1.0 - (1.0 / VU) ** P

    model = SSE(P=P, vu=VU, d=D, users=U).to(DEVICE)
    print(f"[train] SSE P={P} Vu={VU} d={D} U={U} V={model.V} on {DEVICE}")
    L.TRAIN_SNR_DB = (0.0, 20.0)
    model_iters = 1500
    curve = L.train_sse(model, iters=model_iters, batch=256, lr=3e-3,
                        log_every=0, seed=1)
    model.calibrate_power()

    legit = L.eval_ser_sse(model, snr_eval, frames=400_000)
    print("[Q1] legitimate SER:", [f"{v:.3g}" for v in legit])

    # Eve variants
    set_seed(20260813)
    eve_wrong = torch.randn(U, model.L) / math.sqrt(model.L)
    eve_wrong = eve_wrong / eve_wrong.norm(dim=1, keepdim=True) * math.sqrt(model.L)
    eve_none = torch.ones(U, model.L)
    eve_avg = model.masks().mean(dim=0, keepdim=True).repeat(U, 1).cpu()

    eve_w = eval_ser_eve(model, eve_wrong, snr_eval, frames=400_000)
    eve_n = eval_ser_eve(model, eve_none, snr_eval, frames=400_000)
    eve_a = eval_ser_eve(model, eve_avg, snr_eval, frames=400_000)
    print("[Q1] Eve wrong-mask SER:", [f"{v:.3g}" for v in eve_w])
    print("[Q1] Eve no-mask   SER:", [f"{v:.3g}" for v in eve_n])
    print(f"[Q1] chance frame SER = {chance_frame:.4f}")

    write_csv(DATA / "pilot" / "feas_q1_eavesdrop.csv",
              ["snr_db", "legit", "eve_wrong", "eve_none", "eve_avg", "chance"],
              [(s, legit[i], eve_w[i], eve_n[i], eve_a[i], chance_frame)
               for i, s in enumerate(snr_eval)])

    # Q2: key entropy vs per-period length L (grow d at fixed P)
    print("[Q2] sweeping period length L ...")
    q2_rows = []
    for d in [16, 32, 64, 128, 256]:
        set_seed(1)
        mdl = SSE(P=P, vu=VU, d=d, users=U).to(DEVICE)
        L.train_sse(mdl, iters=model_iters, batch=256, lr=3e-3, seed=1)
        mdl.calibrate_power()
        set_seed(20260813)
        ew = torch.randn(U, mdl.L) / math.sqrt(mdl.L)
        ew = ew / ew.norm(dim=1, keepdim=True) * math.sqrt(mdl.L)
        lg = L.eval_ser_sse(mdl, [10.0], frames=300_000)[0]
        ev = eval_ser_eve(mdl, ew, [10.0], frames=300_000)[0]
        xc = mean_abs_cross_corr(mdl.masks().detach().cpu())
        q2_rows.append((mdl.L, d, lg, ev, xc))
        print(f"   L={mdl.L:4d}  legit={lg:.3g}  eve={ev:.3g}  |xcorr|={xc:.3f}")
    write_csv(DATA / "pilot" / "feas_q2_keyentropy.csv",
              ["L", "d", "legit_ser", "eve_ser", "mask_xcorr"], q2_rows)

    # Q3: jamming robustness at SNR=10 dB
    print("[Q3] jamming sweep at SNR=10 dB ...")
    jsr = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
    jam_al = eval_ser_jam(model, 10.0, jsr, frames=300_000, mode="aligned")
    jam_rd = eval_ser_jam(model, 10.0, jsr, frames=300_000, mode="random")
    print("[Q3] aligned-jammer SER:", [f"{v:.3g}" for v in jam_al])
    print("[Q3] random-jammer  SER:", [f"{v:.3g}" for v in jam_rd])
    write_csv(DATA / "pilot" / "feas_q3_jamming.csv",
              ["jsr_db", "ser_aligned", "ser_random"],
              [(j, jam_al[i], jam_rd[i]) for i, j in enumerate(jsr)])

    print("\n[done] feasibility CSVs written to", DATA)


if __name__ == "__main__":
    main()
