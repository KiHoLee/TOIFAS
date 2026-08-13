"""Stage K: the key-refresh layer, implemented and evaluated.

Section VI shows that a few known-plaintext frames recover a fixed key,
so the key has to be refreshed every coherence block. Refreshing is not
as simple as drawing new keys. The decision statistic of a legitimate
receiver contains a signal term that does not depend on the key, because
a unit-modulus key satisfies m_{u,k}^2 = 1, and a cross-user term that
depends on the sign patterns m_v .* m_u. A codebook trained with one key
set adapts to those particular patterns, so installing an unrelated key
set destroys the separation even when the new keys are exactly
orthogonal. Two constructions are compared here.

  Naive refresh: draw a fresh orthogonal key set every block, namely a
  fresh selection of Walsh-Hadamard rows. This changes the cross-user
  patterns and is measured below to fail.

  Invariant refresh: draw only from the transformations that leave every
  cross-user pattern intact, so the legitimate performance is unchanged
  by construction while the transmitted material changes. Three such
  transformations exist and they compose:

    1. a global sign for each of the L frame entries, applied to every
       user, which leaves m_v .* m_u unchanged because the two signs
       cancel,                                        L bits
    2. a permutation of the L frame entries applied to the keys and to
       the codebook together, which is a relabeling,  log2(L!) bits
    3. a permutation of which user holds which row,   log2(U!) bits

  At L=16 and U=4 that is 16 + 44.25 + 4.58 = 64.8 bits per block, and
  each transformation is verified below to leave the legitimate error
  rate unchanged.

The evaluation asks three questions:
  K1 does the legitimate receiver survive a refreshed key,
  K2 does the eavesdropper stay at the random-guess level,
  K3 does a key recovered by known plaintext in one block decode the
     next block.

Outputs: refresh.csv, refresh_kpa.csv
"""
from __future__ import annotations
import math

import numpy as np
import torch

from sse_lib import DATA, DEVICE, SSE, write_csv, eval_ser_sse
from exp_full import hadamard, get_model, eval_ser_eve, eve_wrong_mask
from exp_kpa import collect_known_plaintext, solve_keys

SEED = 5150
BLOCKS = 24
FRAMES = 300_000


def base_keys(U: int, Lp: int) -> torch.Tensor:
    """The fixed orthogonal key set the codebook is trained around. Row 0
    of the Sylvester construction is the all-ones vector, which any
    adversary can write down, so the users take rows 1 to U."""
    return torch.tensor(hadamard(Lp)[1:U + 1], dtype=torch.float32)


def kdf_invariant(seed: int, block: int, U: int, Lp: int):
    """Derive one block's key material from the invariance group."""
    rng = np.random.default_rng([seed, block])
    signs = torch.tensor(rng.choice([-1.0, 1.0], size=(1, Lp)),
                         dtype=torch.float32)
    colperm = torch.tensor(rng.permutation(Lp), dtype=torch.long)
    userperm = torch.tensor(rng.permutation(U), dtype=torch.long)
    return signs, colperm, userperm


def kdf_naive(seed: int, block: int, U: int, Lp: int) -> torch.Tensor:
    """Fresh orthogonal rows every block, which changes the cross-user
    patterns the codebook was trained for."""
    rng = np.random.default_rng([seed, 10_000 + block])
    rows = rng.choice(np.arange(1, Lp), size=U, replace=False)
    return torch.tensor(hadamard(Lp)[rows], dtype=torch.float32)


def entropy_bits(U: int, Lp: int) -> float:
    return (Lp + math.lgamma(Lp + 1) / math.log(2.0)
            + math.lgamma(U + 1) / math.log(2.0))


def install(model: SSE, keys: torch.Tensor, codebook: torch.Tensor,
            colperm=None):
    """Install one block's key material. A column permutation relabels
    the frame entries of the keys and the codebook together."""
    with torch.no_grad():
        if colperm is None:
            model.W.copy_(keys.to(DEVICE))
            model.B.copy_(codebook.to(DEVICE))
        else:
            model.W.copy_(keys[:, colperm].to(DEVICE))
            model.B.copy_(codebook[:, colperm].to(DEVICE))
    model.calibrate_power()


def main():
    P, VU, D, U = 4, 16, 64, 4
    Lp = D // P
    print(f"[K] refresh: L={Lp}, U={U}, "
          f"{entropy_bits(U, Lp):.1f} bits per block from the invariance group")

    K0 = base_keys(U, Lp)
    m = get_model(P=P, vu=VU, d=D, U=U, iters=4000, freeze_W=K0)
    m.eval()
    B0 = m.B.detach().clone().cpu()
    ew = eve_wrong_mask(U, Lp, seed=20260813)

    rows = []
    for t in range(BLOCKS):
        signs, colperm, userperm = kdf_invariant(SEED, t, U, Lp)
        install(m, (K0 * signs)[userperm], B0, colperm)
        lg = eval_ser_sse(m, [10.0], frames=FRAMES)[0]
        ev = eval_ser_eve(m, ew, [10.0], frames=FRAMES)[0]
        install(m, kdf_naive(SEED, t, U, Lp), B0)
        lg_naive = eval_ser_sse(m, [10.0], frames=FRAMES)[0]
        rows.append((t, lg, lg_naive, ev))
        if t < 3 or t == BLOCKS - 1:
            print(f"   block {t:3d} invariant={lg:.4f} naive={lg_naive:.4f} "
                  f"eve={ev:.4f}")
    write_csv(DATA / "refresh.csv",
              ["block", "legit_invariant", "legit_naive", "eve_ser"], rows)
    inv = [r[1] for r in rows]; nai = [r[2] for r in rows]
    ev = [r[3] for r in rows]
    print(f"   invariant refresh: mean={np.mean(inv):.4f} "
          f"min={min(inv):.4f} max={max(inv):.4f}")
    print(f"   naive refresh    : mean={np.mean(nai):.4f}")
    print(f"   eavesdropper     : mean={np.mean(ev):.5f}")

    print("[K] known plaintext across a refresh ...")
    kpa_rows = []
    for nf in [2, 4, 8, 16, 32, 64]:
        same, nxt = [], []
        for t in range(8):
            s1, c1, u1 = kdf_invariant(SEED, t, U, Lp)
            install(m, (K0 * s1)[u1], B0, c1)
            gen = torch.Generator(device="cpu").manual_seed(SEED + 100 * t + nf)
            digits, obs, h = collect_known_plaintext(m, nf, 20.0, gen)
            est = solve_keys(m, digits, obs, h)
            same.append(eval_ser_eve(m, est.cpu(), [10.0], frames=100_000)[0])
            s2, c2, u2 = kdf_invariant(SEED, t + 1, U, Lp)
            install(m, (K0 * s2)[u2], B0, c2)
            nxt.append(eval_ser_eve(m, est.cpu(), [10.0], frames=100_000)[0])
        kpa_rows.append((nf, float(np.mean(same)), float(np.mean(nxt))))
        print(f"   N={nf:3d} same block={kpa_rows[-1][1]:.4f} "
              f"next block={kpa_rows[-1][2]:.4f}")
    write_csv(DATA / "refresh_kpa.csv",
              ["n_frames", "ser_same_block", "ser_next_block"], kpa_rows)
    print("[done] refresh.csv, refresh_kpa.csv")


if __name__ == "__main__":
    main()
