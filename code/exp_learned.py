# -*- coding: utf-8 -*-
"""Learned-key counterparts of the structured-key result stages.

Keyed masking is realized two ways, with structured Walsh-Hadamard keys
and with keys learned in R^L. The two differ in key space, so the paper
reports both wherever a figure or table carries a keyed-masking result.
This script produces the learned side of the key-length sweep, the
jamming sweep, the known-plaintext attack, the scheme comparison and
the refresh, writing files named *_learned.csv next to the structured
ones.

Every evaluation mirrors its structured counterpart exactly: same SNR,
same frame counts, same seeds, same evaluators. Only the key family
differs.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

import exp_kpa
from exp_full import (MAIN_D, eval_ser_eve, eval_ser_jam, eve_wrong_mask,
                      get_model, mean_abs_xcorr, oma_ser_keylen)
from sse_lib import DATA, DEVICE, eval_ser_sse, write_csv

SEED = 1


def learned_model(d=MAIN_D, P=4, vu=16, U=4, iters=4000, seed=SEED):
    """The learned counterpart of main_model: same everything, keys free."""
    return get_model(P=P, vu=vu, d=d, U=U, iters=iters, seed=seed)


def keylen():
    """Fig. 3's learned curve."""
    print("[learned] key length ...")
    rows = []
    for d in [32, 48, 64, 80, 96, 128, 192, 256]:
        m = learned_model(d=d)
        lg = eval_ser_sse(m, [10.0], frames=500_000)[0]
        ev = sum(eval_ser_eve(
                    m, eve_wrong_mask(m.users, m.L,
                                      seed=20260813 + 101 * k).to(DEVICE),
                    [10.0], frames=500_000 // 8)[0]
                 for k in range(8)) / 8.0
        rows.append((m.L, d, lg, ev, mean_abs_xcorr(m.masks().detach()),
                     oma_ser_keylen(m.L, 10.0)))
        print("   L=%3d  legit %.4f  eve %.4f" % (m.L, lg, ev))
    write_csv(DATA / "sec_keylen_learned.csv",
              ["L", "d", "legit_ser", "eve_ser", "mask_xcorr", "oma"], rows)


def jamming():
    """Fig. 4's learned curves."""
    print("[learned] jamming ...")
    m = learned_model()
    jsr = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
    blind = eval_ser_jam(m, 10.0, jsr, frames=500_000, mode="blind", target=0)
    matched = eval_ser_jam(m, 10.0, jsr, frames=500_000, mode="matched",
                           target=0)
    nojam = eval_ser_jam(m, 10.0, [-40.0], frames=500_000, mode="blind",
                         target=0)[0]
    write_csv(DATA / "sec_jam_learned.csv",
              ["jsr_db", "blind", "matched", "nojam"],
              [(j, blind[i], matched[i], nojam) for i, j in enumerate(jsr)])
    print("   blind  :", ["%.3f" % v for v in blind])


def kpa():
    """Fig. 7's learned curve. The attack is linear algebra on the key,
    so it applies to a real-valued key exactly as to a sign pattern."""
    print("[learned] known plaintext ...")
    m = learned_model()
    m.eval()
    true_m = m.masks().detach()
    nmax = max(exp_kpa.NFRAMES)
    rows = []
    for snr in exp_kpa.SNRS:
        acc = {n: [[], []] for n in exp_kpa.NFRAMES}
        for t in range(exp_kpa.TRIALS):
            gen = torch.Generator(device="cpu").manual_seed(
                exp_kpa.SEED + int(snr) + 1000 * t)
            digits, obs, h = exp_kpa.collect_known_plaintext(m, nmax, snr, gen)
            eval_seed = 777 + 31 * t + int(snr)
            for n in exp_kpa.NFRAMES:
                est = exp_kpa.solve_keys(m, digits[:n], obs[:n], h[:n])
                acc[n][0].append(exp_kpa.key_correlation(est, true_m))
                acc[n][1].append(eval_ser_eve(m, est.cpu(), [10.0],
                                              frames=exp_kpa.EVAL_FRAMES,
                                              seed=eval_seed)[0])
        for n in exp_kpa.NFRAMES:
            ks, ss = acc[n]
            rows.append((snr, n, sum(ks) / len(ks), sum(ss) / len(ss)))
        print("   %4.0f dB done" % snr)
    write_csv(DATA / "kpa_learned.csv",
              ["snr_db", "n_frames", "kappa", "eve_ser"], rows)


def refresh():
    """Table VI's learned rows: the invariance refresh acts through
    eps^2 = 1 and a relabeling, so it is available to any real key."""
    print("[learned] refresh ...")
    m = learned_model()
    W0, B0 = m.W.detach().clone(), m.B.detach().clone()
    base = eval_ser_sse(m, [10.0], frames=300_000)[0]
    out = []
    for b in range(8):
        g = torch.Generator(device=DEVICE).manual_seed(5150 + b)
        xi = torch.randperm(m.L, generator=g, device=DEVICE)
        eps = torch.randint(2, (m.L,), generator=g, device=DEVICE) * 2.0 - 1.0
        tau = torch.randperm(m.users, generator=g, device=DEVICE)
        with torch.no_grad():
            m.W.copy_((W0[tau] * eps[None, :])[:, xi])
            m.B.copy_(B0[:, xi])
        lg = eval_ser_sse(m, [10.0], frames=300_000)[0]
        ev = eval_ser_eve(m, eve_wrong_mask(m.users, m.L,
                                            seed=20260813).to(DEVICE),
                          [10.0], frames=300_000)[0]
        out.append((b, lg, ev))
    with torch.no_grad():
        m.W.copy_(W0); m.B.copy_(B0)
    write_csv(DATA / "refresh_learned.csv",
              ["block", "legit_ser", "eve_ser"], out)
    print("   unrefreshed %.5f  refreshed %.5f..%.5f"
          % (base, min(r[1] for r in out), max(r[1] for r in out)))


def compare():
    """Table IV's learned row: the same four columns as the structured
    scheme, under the same jammer at a JSR of 0 dB."""
    print("[learned] scheme comparison ...")
    m = learned_model()
    F = 300_000
    legit = eval_ser_sse(m, [10.0], frames=F)[0]
    out = eval_ser_eve(m, eve_wrong_mask(m.users, m.L,
                                         seed=20260813).to(DEVICE),
                       [10.0], frames=F)[0]
    ins = eval_ser_eve(m, m.masks().detach().roll(1, 0), [10.0], frames=F)[0]
    jam = eval_ser_jam(m, 10.0, [0.0], frames=F, mode="blind", target=0)[0]
    write_csv(DATA / "compare_learned.csv",
              ["scheme", "legit_ser", "eve_out", "eve_in", "jam0_ser"],
              [("proposed_learned", legit, out, ins, jam)])
    print("   legit %.4f  out %.4f  in %.4f  jam %.4f"
          % (legit, out, ins, jam))


def main():
    keylen()
    jamming()
    kpa()
    refresh()
    compare()
    print("[done] learned-key CSVs in", DATA)


if __name__ == "__main__":
    main()


def sens():
    """Fig. 5's learned curve: eavesdropper SER against the fraction of
    the key the attacker holds. correlated_masks builds a substitute at
    a prescribed correlation to any real key, so the sweep applies to a
    learned key exactly as to a sign pattern."""
    from exp_full import correlated_masks
    print("[learned] key sensitivity ...")
    m = learned_model()
    F, TR = 600_000, 12
    true_m = m.masks().detach().cpu()
    gen = torch.Generator().manual_seed(31)
    fracs = [0.0, 0.2, 0.4, 0.6, 0.75, 0.85, 0.9, 0.92, 0.94, 0.955,
             0.97, 0.985, 1.0]
    rows = []
    for f in fracs:
        acc = [eval_ser_eve(m, correlated_masks(true_m, f, gen), [10.0],
                            frames=F // TR, seed=777 + 17 * t)[0]
               for t in range(TR)]
        rows.append((f, sum(acc) / len(acc)))
    write_csv(DATA / "sec_sens_learned.csv", ["frac", "ser_mask"], rows)
    print("   f=0 %.4f  f=1 %.4f" % (rows[0][1], rows[-1][1]))


def brute():
    """Fig. 6's learned curve. The best-of-K correlation is a property of
    the key space, which both realizations share at the same L, so only
    the sensitivity mapping differs and it is re-read from the learned
    sweep."""
    import csv as _csv
    import numpy as np
    print("[learned] brute-force search ...")
    with open(DATA / "sec_sens_learned.csv") as f:
        cmp_rows = list(_csv.DictReader(f))
    f_arr = np.array([float(r["frac"]) for r in cmp_rows])
    mask_arr = np.array([float(r["ser_mask"]) for r in cmp_rows])
    L = MAIN_D // 4
    ks = [1, 3, 10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 65_536,
          100_000, 300_000, 1_000_000]
    rng = np.random.default_rng(2026)
    rows = []
    for K in ks:
        best = np.sqrt(rng.beta(0.5, (L - 1) / 2.0, size=(400, K)).max(1))
        rows.append((K, float(np.mean(np.interp(best, f_arr, mask_arr)))))
    write_csv(DATA / "sec_brute_learned.csv", ["K", "ser_mask"], rows)
    print("   K=1e6 %.4f" % rows[-1][1])


def real():
    """Fig. 8's learned curves.

    exp_real_sec writes fixed file names, so the structured artifacts are
    held aside, the run is repeated with the learned model, its output is
    copied to *_learned names, and the originals are put back. A failure
    anywhere restores them.
    """
    import shutil
    import exp_real_sec as R
    print("[learned] real token streams ...")
    names = ("real_sec_ter.csv", "real_sec_stats.json")
    saved = {n: (DATA / n).read_bytes() for n in names
             if (DATA / n).exists()}
    orig = R.main_model
    try:
        R.main_model = lambda **kw: learned_model(
            d=kw.get("d", MAIN_D), P=kw.get("P", 4),
            vu=kw.get("vu", 16), U=kw.get("U", 4))
        R.main()
        for n in names:
            if (DATA / n).exists():
                shutil.copyfile(DATA / n,
                                DATA / n.replace(".", "_learned.", 1))
    finally:
        R.main_model = orig
        for n, blob in saved.items():
            (DATA / n).write_bytes(blob)
    print("   learned artifacts written, structured ones restored")
