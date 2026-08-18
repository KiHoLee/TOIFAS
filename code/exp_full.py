"""Full-scale security evaluation for paper 11 (run under WSL CUDA).

Reuses the SSE transmit/receive core from sse_lib.py and adds an
eavesdropper receiver, a jammer channel, and structured mask families.
Main configuration d=256, P=4, Vu=16 (V=Vu^P=65,536), U=4 users, matching
the language-model token vocabulary scale.

Stages (each writes a CSV to ../data; figures come from replot_security.py
and the two result tables from make_tables.py):
  A  security vs SNR       -> sec_snr.csv      (Fig. 2)
  B  key length            -> sec_keylen.csv   (Fig. 3)
  C  jamming vs JSR        -> sec_jam.csv      (Fig. 4)
  D  mask families         -> sec_maskfam.csv  (key-family table)
  E  scheme comparison     -> sec_compare.csv  (comparison table)
  F  attack difficulty     -> sec_sens.csv, sec_brute.csv (Figs. 5-6)

Experiment scripts write CSV only, never draw. Fixed seeds.
"""
from __future__ import annotations
import math
import numpy as np
import torch

import sse_lib as L
from sse_lib import (SSE, rayleigh_gain, snr_to_sigma2, write_csv, set_seed,
                     eval_ser_sse, oma_ser, DATA, DEVICE)


# ----------------------------------------------------------------------
# eavesdropper: correlate the transmitted (true-mask) frame with a
# substitute mask the eavesdropper does not truly hold.
# ----------------------------------------------------------------------
@torch.no_grad()
def eval_ser_eve(model: SSE, eve_masks: torch.Tensor, snr_list,
                 frames: int, chunk: int = 100_000, seed: int = 777):
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
            e = Bn[digits] / math.sqrt(model.P)
            y = (e * true_m[None, :, None, :]).sum(dim=1) / c
            h = rayleigh_gain((n, model.users), device=DEVICE)
            sigma = snr_to_sigma2(snr_db, model.d).to(DEVICE).sqrt()
            noise = torch.randn(n, model.users, model.P, model.L, device=DEVICE)
            y_rx = h[:, :, None, None] * y[:, None] + sigma * noise
            r = y_rx / h[:, :, None, None].clamp_min(1e-6)
            cand = Bn[None, :, :] * eve_masks[:, None, :]
            scores = torch.einsum("nupl,uvl->nupv", r, cand)
            wrong = (scores.argmax(-1) != digits).any(dim=2)
            err += int(wrong.sum()); tot += n * model.users
        out.append(err / tot)
    return out


@torch.no_grad()
def eval_ser_jam(model: SSE, snr_db, jsr_db_list, frames: int,
                 chunk: int = 100_000, seed: int = 777, mode: str = "blind",
                 target: int = 0):
    """Returns the target-user SER (user `target`, the user a mask-matched
    jammer aims at). The mask-matched jammer aligns with the target key,
    which a mask-blind jammer cannot do. Reporting the target-user SER,
    rather than the user average, isolates how efficiently each jammer can
    degrade a chosen victim (Proposition 2)."""
    model.eval().to(DEVICE)
    Bn = model.unit_codebook()
    true_m = model.masks()
    c = model.c
    sigma = snr_to_sigma2(snr_db, model.d).to(DEVICE).sqrt()
    # matched jammer aligns with the target user's masked codeword mean
    # direction (needs that user's secret key)
    w_fixed = (Bn[target][None, :] * true_m[target][None, :]).repeat(model.P, 1)
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
            y = (e * true_m[None, :, None, :]).sum(dim=1) / c
            h = rayleigh_gain((n, model.users), device=DEVICE)
            hJ = rayleigh_gain((n,), device=DEVICE)
            if mode == "matched":
                w = w_fixed[None].expand(n, model.P, model.L)
            else:
                w = torch.randn(n, model.P, model.L, device=DEVICE)
                w = w / w.reshape(n, -1).norm(dim=1)[:, None, None].clamp_min(1e-8)
            jam = (hJ * math.sqrt(jsr))[:, None, None] * w
            noise = torch.randn(n, model.users, model.P, model.L, device=DEVICE)
            y_rx = h[:, :, None, None] * y[:, None] + jam[:, None] + sigma * noise
            r = y_rx / h[:, :, None, None].clamp_min(1e-6)
            cand = Bn[None, :, :] * true_m[:, None, :]
            scores = torch.einsum("nupl,uvl->nupv", r, cand)
            wrong = (scores.argmax(-1) != digits).any(dim=2)  # (n,U)
            err += int(wrong[:, target].sum()); tot += n
        out.append(err / tot)
    return out


def hadamard(n: int) -> np.ndarray:
    """Sylvester construction, n a power of two."""
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def mean_abs_xcorr(masks: torch.Tensor) -> float:
    m = masks / masks.norm(dim=1, keepdim=True).clamp_min(1e-8)
    G = (m @ m.T).abs()
    U = m.shape[0]
    off = G[~torch.eye(U, dtype=torch.bool, device=G.device)]
    return float(off.mean())


def random_mask(U, Lp):
    W = torch.randn(U, Lp)
    return W / W.norm(dim=1, keepdim=True) * math.sqrt(Lp)


def eve_wrong_mask(U, Lp, seed):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(U, Lp, generator=g)
    return W / W.norm(dim=1, keepdim=True) * math.sqrt(Lp)


MAIN_D = 256      # embedding dimension of the main configuration


def get_model(P=4, vu=16, d=MAIN_D, U=4, iters=4000, seed=1,
              freeze_W=None, tag=""):
    """Train an SSE model, optionally with fixed (frozen) masks."""
    set_seed(seed)
    m = SSE(P=P, vu=vu, d=d, users=U).to(DEVICE)
    if freeze_W is not None:
        with torch.no_grad():
            m.W.copy_(freeze_W.to(DEVICE))
        m.W.requires_grad_(False)
    L.train_sse(m, iters=iters, batch=256, lr=3e-3, seed=seed)
    m.calibrate_power()
    return m


def base_keys(U: int, Lp: int) -> torch.Tensor:
    """The structured key family: U non-constant rows of a Walsh-Hadamard
    matrix, truncated to Lp entries.

    Row 0 of the Sylvester construction is the all-ones vector, which any
    adversary can write down without searching, so the users take rows
    1..U. The construction exists at power-of-two orders, so for other
    key lengths the next power-of-two order is truncated to Lp entries.
    That truncation keeps the entries unit modulus and, at every length
    the evaluation uses, keeps the rows exactly orthogonal as well; the
    measured cross-correlation is reported alongside every sweep point.
    Requires U <= Lp - 1 non-constant rows to exist."""
    n = 1 << max(math.ceil(math.log2(max(Lp, U + 1))), 1)
    H = hadamard(n)
    if H.shape[0] - 1 < U:
        raise ValueError(f"key length {Lp} admits only {H.shape[0]-1} "
                         f"non-constant rows, fewer than U={U}")
    return torch.tensor(H[1:U + 1, :Lp].copy(), dtype=torch.float32)


def main_model(iters=4000, P=4, vu=16, d=MAIN_D, U=4):
    """The main configuration used by every stage below.

    The keys are frozen to the structured Walsh-Hadamard family rather
    than learned. Unconstrained mask training converges to disjoint
    sparse supports, that is, to an orthogonal slot allocation, which
    collapses the superposition into OMA and leaves the key space far
    smaller than a dense direction in R^L. The structured family is
    dense, exactly orthogonal, and unit modulus, which is also the
    condition the key-refresh invariance argument requires."""
    return get_model(P=P, vu=vu, d=d, U=U, iters=iters,
                     freeze_W=base_keys(U, d // P))


def stage_A():
    print("[A] security vs SNR (V=65536) ...")
    m = main_model()
    snr = [float(v) for v in range(0, 21, 2)]
    frames = 800_000
    legit = eval_ser_sse(m, snr, frames=frames)
    ew = eve_wrong_mask(m.users, m.L, seed=20260813).to(DEVICE)
    eve_w = eval_ser_eve(m, ew, snr, frames=frames)
    eve_n = eval_ser_eve(m, torch.ones(m.users, m.L), snr, frames=frames)
    # conventional public-mask scheme: the eavesdropper holds the same
    # (public) masks and decodes exactly like a legitimate user
    eve_p = eval_ser_eve(m, m.masks().detach().cpu(), snr, frames=frames)
    oma = [oma_ser_keylen(m.L, s, bits=int(math.log2(m.V))) for s in snr]
    chance = 1.0 - (1.0 / m.vu) ** m.P
    write_csv(DATA / "sec_snr.csv",
              ["snr_db", "legit", "eve_wrong", "eve_none", "eve_public",
               "oma", "chance"],
              [(s, legit[i], eve_w[i], eve_n[i], eve_p[i], oma[i], chance)
               for i, s in enumerate(snr)])
    print("   legit:", [f"{v:.2e}" for v in legit])
    print("   eve  :", [f"{v:.3f}" for v in eve_w])


def train_sse_reg(m: SSE, iters=4000, batch=256, lr=3e-3, seed=1,
                  lam_orth=1.0, lam_flat=0.1):
    """Regularized key learning for improved spreading and de-spreading.
    Adds to the digit-wise cross entropy (i) an orthogonality penalty on
    the off-diagonal key Gram entries, which reduces cross-user
    interference and residual leakage, and (ii) a constant-modulus
    penalty that flattens the key spectrum, which maximizes the spreading
    of a mask-blind jammer (Proposition 2: the jammer concentration on
    candidate i is sum_k w_k^2 e_{i,k}^2 weighted through the key, and a
    flat key removes any low-energy entries a jammer could exploit)."""
    set_seed(seed)
    m.to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    ce = torch.nn.CrossEntropyLoss()
    for it in range(1, iters + 1):
        digits = torch.randint(m.vu, (batch, m.users, m.P), device=DEVICE)
        snr = torch.empty(batch).uniform_(0.0, 20.0)
        m.calibrate_power(8192)
        scores = m(digits, snr) * m.logit_scale.exp()
        loss = ce(scores.reshape(-1, m.vu), digits.reshape(-1))
        mk = m.masks()
        G = (mk @ mk.T) / m.L
        off = G - torch.eye(m.users, device=G.device)
        loss = loss + lam_orth * off.pow(2).sum()
        loss = loss + lam_flat * (mk.pow(2) - 1.0).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    m.calibrate_power()
    return m


def get_model_reg(P=4, vu=16, d=MAIN_D, U=4, iters=4000, seed=1):
    set_seed(seed)
    m = SSE(P=P, vu=vu, d=d, users=U).to(DEVICE)
    train_sse_reg(m, iters=iters, seed=seed)
    return m


def oma_ser_keylen(L, snr_db, bits=16, n_grid=200_000):
    """Resource-matched OMA reference for the key-length sweep.

    The OMA user owns d/U = L exclusive real dimensions for its 16 index
    bits at the same per-dimension SNR. For L >= 16 the best use of the
    allocation is antipodal signaling on 16 dimensions with the frame
    energy concentrated on them, an energy gain of L/16 per bit. For
    L < 16 the user must pack 16/L bits per dimension, a 2^(16/L)-ary
    pulse-amplitude constellation, defined when 16/L is an integer and
    reported as nan otherwise.
    """
    import numpy as np
    if L >= bits:
        return oma_ser([snr_db + 10.0 * math.log10(L / bits)], bits=bits)[0]
    if bits % L:
        return float("nan")
    M = 2 ** (bits // L)
    # M-PAM levels +-A, +-3A, ..., +-(M-1)A with unit AVERAGE symbol energy
    # give A^2 = 3/(M^2-1), so the distance to the decision boundary is A
    # and the Q-function argument is h*sqrt(3*g/(M^2-1)). Using 6 instead
    # of 3 would assume an average energy of two per dimension.
    x = (np.arange(n_grid) + 0.5) / n_grid
    h = np.sqrt(-np.log(1.0 - x))
    g = 10.0 ** (snr_db / 10.0)
    arg = np.clip(h * math.sqrt(3.0 * g / (M * M - 1.0)), 0, 38)
    q = (1.0 - 1.0 / M) * np.array([math.erfc(v / math.sqrt(2.0))
                                    for v in arg])
    q = np.clip(q, 0.0, 1.0)
    return float(np.mean(1.0 - (1.0 - q) ** L))


def stage_B():
    print("[B] key length (dense grid so the curve is smooth) ...")
    rows = []
    # L = d/P. Lengths 6, 10 and 14 are dropped because the
    # truncated Walsh-Hadamard rows are not exactly orthogonal
    # there, and L=4 admits only three non-constant rows for
    # U=4 users.
    for d in [32, 48, 64, 80, 96, 128, 192, 256]:
        m = main_model(d=d)          # same structured family as Fig. 2
        lg = eval_ser_sse(m, [10.0], frames=500_000)[0]
        ew = eve_wrong_mask(m.users, m.L, seed=20260813).to(DEVICE)
        ev = eval_ser_eve(m, ew, [10.0], frames=500_000)[0]
        xc = mean_abs_xcorr(m.masks().detach())
        oma = oma_ser_keylen(m.L, 10.0)
        rows.append((m.L, d, lg, ev, xc, oma))
        print(f"   L={m.L:4d} legit={lg:.2e} eve={ev:.3f} xcorr={xc:.4f} "
              f"oma={oma:.4f}")
    write_csv(DATA / "sec_keylen.csv",
              ["L", "d", "legit_ser", "eve_ser", "mask_xcorr", "oma"], rows)


def stage_C():
    print("[C] jamming vs JSR ...")
    m = main_model()
    jsr = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
    blind = eval_ser_jam(m, 10.0, jsr, frames=500_000, mode="blind", target=0)
    matched = eval_ser_jam(m, 10.0, jsr, frames=500_000, mode="matched", target=0)
    # target-user SER with no jammer, for the reference line
    nojam = eval_ser_jam(m, 10.0, [-40.0], frames=500_000, mode="blind",
                         target=0)[0]
    write_csv(DATA / "sec_jam.csv",
              ["jsr_db", "blind", "matched", "nojam"],
              [(j, blind[i], matched[i], nojam) for i, j in enumerate(jsr)])
    print(f"   target no-jam={nojam:.2e}")
    print("   blind  :", [f"{v:.3f}" for v in blind])
    print("   matched:", [f"{v:.3f}" for v in matched])


def stage_D():
    print("[D] mask families ...")
    P, vu, d, U = 4, 16, MAIN_D, 4
    Lp = d // P
    fams = {}
    # random fixed masks
    set_seed(7); fams["random"] = random_mask(U, Lp)
    # Walsh-Hadamard rows (orthogonal). Row 0 of the Sylvester
    # construction is the all-ones vector, which any adversary can write
    # down, so it is excluded and the users take rows 1 to U.
    Hd = base_keys(U, Lp)          # the main configuration's key family
    fams["hadamard"] = Hd
    ones = torch.ones(U, Lp)          # the cheapest possible guess
    rows = []
    for name, W in fams.items():
        m = get_model(P=P, vu=vu, d=d, U=U, iters=4000, freeze_W=W)
        lg = eval_ser_sse(m, [10.0], frames=500_000)[0]
        ew = eve_wrong_mask(U, Lp, seed=20260813).to(DEVICE)
        ev = eval_ser_eve(m, ew, [10.0], frames=500_000)[0]
        ev1 = eval_ser_eve(m, ones, [10.0], frames=500_000)[0]
        xc = mean_abs_xcorr(m.masks().detach())
        rows.append((name, lg, ev, ev1, xc))
        print(f"   {name:9s} legit={lg:.2e} eve={ev:.3f} ones={ev1:.3f} "
              f"xcorr={xc:.4f}")
    # learned masks (plain cross entropy)
    m = get_model(P=P, vu=vu, d=d, U=U, iters=4000)
    lg = eval_ser_sse(m, [10.0], frames=500_000)[0]
    ew = eve_wrong_mask(U, Lp, seed=20260813).to(DEVICE)
    ev = eval_ser_eve(m, ew, [10.0], frames=500_000)[0]
    ev1 = eval_ser_eve(m, ones, [10.0], frames=500_000)[0]
    xc = mean_abs_xcorr(m.masks().detach())
    rows.append(("learned", lg, ev, ev1, xc))
    print(f"   {'learned':9s} legit={lg:.2e} eve={ev:.3f} ones={ev1:.3f} "
          f"xcorr={xc:.4f}")
    # regularized key learning (orthogonality + constant modulus)
    mr = get_model_reg(P=P, vu=vu, d=d, U=U, iters=4000)
    lgr = eval_ser_sse(mr, [10.0], frames=500_000)[0]
    evr = eval_ser_eve(mr, ew, [10.0], frames=500_000)[0]
    evr1 = eval_ser_eve(mr, ones, [10.0], frames=500_000)[0]
    xcr = mean_abs_xcorr(mr.masks().detach())
    rows.append(("learned_reg", lgr, evr, evr1, xcr))
    print(f"   {'learn_reg':9s} legit={lgr:.2e} eve={evr:.3f} ones={evr1:.3f} "
          f"xcorr={xcr:.4f}")
    # jamming robustness of plain vs regularized keys (blind jammer)
    jsr = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
    jb_plain = eval_ser_jam(m, 10.0, jsr, frames=300_000, mode="blind")
    jb_reg = eval_ser_jam(mr, 10.0, jsr, frames=300_000, mode="blind")
    write_csv(DATA / "sec_regjam.csv",
              ["jsr_db", "plain", "regularized"],
              [(j, jb_plain[i], jb_reg[i]) for i, j in enumerate(jsr)])
    write_csv(DATA / "sec_maskfam.csv",
              ["family", "legit_ser", "eve_ser", "eve_ones_ser",
               "mask_xcorr"], rows)


@torch.no_grad()
def eval_scheme(model: SSE, snr_db, frames, *, rx_masks=None, perms=None,
                jam_w=None, jsr_db=None, target=0, chunk=100_000, seed=777,
                decode_user=0):
    """Generic evaluator for the comparison schemes.
    rx_masks: masks used at the decoding receiver (None = true masks).
    perms:    (U,d-index) per-user secret permutations applied at tx to
              x_u; the decoder for `decode_user` inverse-permutes first.
              rx side without the permutation just decodes raw.
    jam_w:    None or 'matched'/'blind' jammer aimed at `target`.
    Returns SER of `decode_user` (frame error over its P digits)."""
    model.eval().to(DEVICE)
    Bn = model.unit_codebook()
    true_m = model.masks()
    c = model.c
    sigma = snr_to_sigma2(snr_db, model.d).to(DEVICE).sqrt()
    d = model.P * model.L
    if perms is not None:
        inv = torch.argsort(perms, dim=1)
    if jam_w == "matched":
        wf = (Bn[target][None, :] * true_m[target][None, :]).repeat(model.P, 1)
        if perms is not None:
            wfl = wf.reshape(-1)[perms[target]]
            wf = wfl.reshape(model.P, model.L)
        wf = wf / wf.norm()
    jsr = 10.0 ** (jsr_db / 10.0) if jsr_db is not None else 0.0
    g = torch.Generator(device="cpu").manual_seed(seed + int(10 * snr_db))
    err = tot = 0
    for n0 in range(0, frames, chunk):
        n = min(chunk, frames - n0)
        digits = torch.randint(model.vu, (n, model.users, model.P),
                               generator=g).to(DEVICE)
        e = Bn[digits] / math.sqrt(model.P)
        x = e * true_m[None, :, None, :]                  # (n,U,P,L)
        if perms is not None:
            xf = x.reshape(n, model.users, d)
            xf = torch.stack([xf[:, u][:, perms[u]] for u in range(model.users)], 1)
            x = xf.reshape(n, model.users, model.P, model.L)
        y = x.sum(dim=1) / c                              # (n,P,L)
        h = rayleigh_gain((n,), device=DEVICE)            # decode_user channel
        y_rx = h[:, None, None] * y                       # (n,P,L)
        if jam_w is not None:
            hJ = rayleigh_gain((n,), device=DEVICE)
            if jam_w == "matched":
                w = wf[None].expand(n, model.P, model.L)
            else:
                w = torch.randn(n, model.P, model.L, device=DEVICE)
                w = w / w.reshape(n, -1).norm(dim=1)[:, None, None].clamp_min(1e-8)
            y_rx = y_rx + (hJ * math.sqrt(jsr))[:, None, None] * w
        y_rx = y_rx + sigma * torch.randn(n, model.P, model.L, device=DEVICE)
        r = y_rx / h[:, None, None].clamp_min(1e-6)
        if perms is not None:
            rf = r.reshape(n, d)[:, inv[decode_user]]
            r = rf.reshape(n, model.P, model.L)
        m_rx = true_m if rx_masks is None else rx_masks.to(DEVICE)
        cand = Bn * m_rx[decode_user][None, :]            # (Vu,L)
        scores = torch.einsum("npl,vl->npv", r, cand)
        wrong = (scores.argmax(-1) != digits[:, decode_user]).any(dim=1)
        err += int(wrong.sum()); tot += n
    return err / tot


def stage_E():
    """Comparison across five schemes at 10 dB, V=65,536, user-0 metrics.
    Columns: legitimate SER; outsider-eavesdropper SER; insider SER (a
    curious legitimate user of the SAME system decoding user 0 with its
    own credentials); target-user SER under the strongest jammer the
    attacker can BUILD from public knowledge at JSR 0 dB (matched if the
    masks are public, blind if the PHY structure is secret)."""
    print("[E] scheme comparison ...")
    m = main_model()
    F = 400_000
    d = m.P * m.L
    set_seed(20260813)
    ew = eve_wrong_mask(m.users, m.L, seed=20260813)
    # shuffling-style multi-user adaptation: one GLOBAL secret permutation
    # shared by all users (per-user permutations break the trained
    # multi-user separation, so the shared key is the fair extension)
    gp = torch.Generator().manual_seed(11)
    gperm = torch.randperm(d, generator=gp)
    perms = gperm[None].repeat(m.users, 1)

    chance = 1.0 - (1.0 / m.vu) ** m.P
    insider_masks = torch.roll(m.masks().detach().cpu(), 1, 0)  # user 1's key

    rows = []
    # S1 proposed keyed masking: per-user secret masks
    lg = eval_scheme(m, 10.0, F)
    ev = eval_scheme(m, 10.0, F, rx_masks=ew)
    ins = eval_scheme(m, 10.0, F, rx_masks=insider_masks)
    jm = eval_scheme(m, 10.0, F, jam_w="blind", jsr_db=0.0)
    rows.append(("proposed", lg, ev, ins, jm))
    # S2 public-mask superposition (no key): everyone decodes, attacker
    # builds the matched jammer
    jm2 = eval_scheme(m, 10.0, F, jam_w="matched", jsr_db=0.0)
    rows.append(("public_mask", lg, lg, lg, jm2))
    # S3 global permutation key over public masks (shuffling-style): the
    # outsider lacks the permutation, but every insider holds it and the
    # masks are public, so insiders decode each other
    lg3 = eval_scheme(m, 10.0, F, perms=perms)
    ev3 = eval_scheme_permuted_eve(m, 10.0, F, perms)
    jm3 = eval_scheme(m, 10.0, F, perms=perms, jam_w="blind", jsr_db=0.0)
    rows.append(("perm_key", lg3, ev3, lg3, jm3))
    # S4 per-user index cipher (one-time pad on the digits) over public
    # masks: content protected from outsiders and insiders, but the PHY
    # is public so the matched jammer remains buildable
    rows.append(("index_cipher", lg, chance, chance, jm2))
    # S5 OMA digital, no encryption: open to everyone
    lg5 = oma_ser_keylen(m.L, 10.0, bits=int(math.log2(m.V)))
    rows.append(("oma_plain", lg5, lg5, lg5, float("nan")))

    write_csv(DATA / "sec_compare.csv",
              ["scheme", "legit_ser", "eve_out", "eve_in", "jam0_ser"], rows)
    for r in rows:
        print("   ", r)


@torch.no_grad()
def eval_scheme_permuted_eve(model: SSE, snr_db, frames, perms,
                             chunk=100_000, seed=777, eve_perms=None):
    """Eve for S3: sees the per-user permuted tx, holds the PUBLIC masks
    but not the permutation, decodes user 0 raw. When eve_perms is given,
    Eve first undoes the permutation she believes was used, which models
    an attacker holding a partially recovered permutation key."""
    model.eval().to(DEVICE)
    Bn = model.unit_codebook()
    true_m = model.masks()
    c = model.c
    d = model.P * model.L
    sigma = snr_to_sigma2(snr_db, model.d).to(DEVICE).sqrt()
    g = torch.Generator(device="cpu").manual_seed(seed + int(10 * snr_db))
    err = tot = 0
    for n0 in range(0, frames, chunk):
        n = min(chunk, frames - n0)
        digits = torch.randint(model.vu, (n, model.users, model.P),
                               generator=g).to(DEVICE)
        e = Bn[digits] / math.sqrt(model.P)
        x = e * true_m[None, :, None, :]
        xf = x.reshape(n, model.users, d)
        xf = torch.stack([xf[:, u][:, perms[u]] for u in range(model.users)], 1)
        y = xf.reshape(n, model.users, model.P, model.L).sum(dim=1) / c
        h = rayleigh_gain((n,), device=DEVICE)
        y_rx = h[:, None, None] * y + sigma * torch.randn(
            n, model.P, model.L, device=DEVICE)
        r = y_rx / h[:, None, None].clamp_min(1e-6)
        if eve_perms is not None:
            inv = torch.argsort(eve_perms[0]).to(r.device)
            r = r.reshape(n, d)[:, inv].reshape(n, model.P, model.L)
        cand = Bn * true_m[0][None, :]
        scores = torch.einsum("npl,vl->npv", r, cand)
        wrong = (scores.argmax(-1) != digits[:, 0]).any(dim=1)
        err += int(wrong.sum()); tot += n
    return err / tot


def correlated_masks(true_m: torch.Tensor, rho: float, gen: torch.Generator):
    """Substitute masks with prescribed normalized correlation rho to the
    true keys: mtil = rho*m + sqrt(1-rho^2)*m_perp, ||mtil|| = ||m||."""
    U, Lp = true_m.shape
    out = torch.empty_like(true_m)
    for u in range(U):
        m = true_m[u]
        p = torch.randn(Lp, generator=gen)
        p = p - (p @ m) / (m @ m) * m
        p = p / p.norm() * m.norm()
        out[u] = rho * m + math.sqrt(max(0.0, 1 - rho * rho)) * p
    return out


def stage_F():
    """Attack difficulty in the style of standard security evaluations.
    (i) Key sensitivity: Eve SER against the correlation rho between her
        guess and the true key (avalanche-style curve).
    (ii) Brute-force key search: expected Eve SER against the number of
        random key guesses K, where for each trial the attacker keeps the
        guess with the LARGEST correlation to the true key (a genie-aided
        upper bound on any selection rule). The best-guess correlation
        rho_max(K, L) is sampled by Monte Carlo and mapped through the
        measured sensitivity curve of (i)."""
    print("[F] attack difficulty ...")
    m = main_model()
    F = 200_000
    true_m = m.masks().detach().cpu()
    gen = torch.Generator().manual_seed(31)

    # (i) sensitivity curve, densest where the curve falls steeply
    rhos = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75,
            0.8, 0.84, 0.88, 0.90, 0.92, 0.94, 0.96, 0.97, 0.98,
            0.99, 0.995, 1.0]
    sens = []
    for rho in rhos:
        mt = correlated_masks(true_m, rho, gen)
        ser = eval_ser_eve(m, mt, [10.0], frames=F)[0]
        sens.append((rho, ser))
        print(f"   rho={rho:.2f} eve_ser={ser:.4f}")
    write_csv(DATA / "sec_sens.csv", ["rho", "eve_ser"], sens)

    # (ii) brute-force: sample rho_max(K, L) and interpolate SER(rho)
    import numpy as np
    r_arr = np.array([r for r, _ in sens])
    s_arr = np.array([s for _, s in sens])

    def ser_of_rho(r):
        return float(np.interp(abs(r), r_arr, s_arr))

    ks = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
    rows = []
    rng = np.random.default_rng(2026)
    for Lp in [8, 16, 32, 64]:
        for K in ks:
            trials = 400
            # rho of a random unit guess vs a fixed key in R^L is the
            # first coordinate of a random unit vector; sample K per trial
            best = np.empty(trials)
            for t in range(trials):
                g = rng.standard_normal((K, Lp))
                g /= np.linalg.norm(g, axis=1, keepdims=True)
                best[t] = np.abs(g[:, 0]).max()
            ser_est = float(np.mean([ser_of_rho(b) for b in best]))
            rows.append((Lp, K, float(best.mean()), ser_est))
        print(f"   L={Lp} done")
    write_csv(DATA / "sec_brute.csv",
              ["L", "K", "best_rho", "eve_ser"], rows)


def partial_perm(true_perm: torch.Tensor, frac: float, gen: torch.Generator):
    """A permutation that agrees with true_perm on a fraction frac of the
    positions and is scrambled on the rest, which is what an attacker
    holding part of a permutation key would have."""
    d = true_perm.numel()
    k = int(round(frac * d))
    idx = torch.randperm(d, generator=gen)
    keep, rest = idx[:k], idx[k:]
    out = true_perm.clone()
    if rest.numel() > 1:
        out[rest] = true_perm[rest][torch.randperm(rest.numel(), generator=gen)]
    return out


TRIALS_PERM = 60


def stage_I():
    """Key sensitivity of three schemes on one axis.

    The axis is the fraction of the key the attacker has recovered. For
    the proposed scheme that fraction is the normalized correlation
    between the guessed and the true mask. For the permutation scheme it
    is the fraction of positions the guessed permutation places
    correctly. For the index cipher it is the fraction of pad bits the
    attacker knows. Its error rate is the closed form
    1 - (1-p_ch) 2^{-(1-f) log2 V}, the probability of decoding the
    ciphered index over the channel times the probability that the
    unknown pad bits, which stay uniform, are all guessed right.
    """
    print("[I] key sensitivity across schemes ...")
    m = main_model()
    F = 600_000          # more frames per point for a smooth curve
    TRIALS_MASK = 12     # independent substitute keys per point
    d = m.P * m.L
    true_m = m.masks().detach().cpu()
    gen = torch.Generator().manual_seed(31)
    gp = torch.Generator().manual_seed(11)
    gperm = torch.randperm(d, generator=gp)
    perms = gperm[None].repeat(m.users, 1)
    # a marker grid comparable to the other result figures, with the
    # spacing tightened only where the curves fall
    fracs = [0.0, 0.2, 0.4, 0.6, 0.75, 0.85, 0.9, 0.92, 0.94, 0.955,
             0.97, 0.985, 1.0]
    bits = math.log2(m.V)
    # the channel success of a public-mask receiver, which the cipher
    # cannot exceed even with the full pad
    lg1 = eval_scheme(m, 10.0, 200_000)
    rows = []
    for f in fracs:
        acc_m = []
        for t in range(TRIALS_MASK):
            mt = correlated_masks(true_m, f, gen)
            acc_m.append(eval_ser_eve(m, mt, [10.0],
                                      frames=F // TRIALS_MASK,
                                      seed=777 + 17 * t)[0])
        ser_mask = sum(acc_m) / len(acc_m)
        # a partial permutation is combinatorially lumpy, so the point
        # is averaged over independent draws of which positions the
        # attacker holds
        acc = []
        for t in range(TRIALS_PERM):
            pp = partial_perm(gperm, f, gen)
            pperms = pp[None].repeat(m.users, 1)
            acc.append(eval_scheme_permuted_eve(m, 10.0, F // TRIALS_PERM,
                                                perms, eve_perms=pperms,
                                                seed=777 + 13 * t))
        ser_perm = sum(acc) / len(acc)
        ser_pad = 1.0 - (1.0 - lg1) * 2.0 ** (-(1.0 - f) * bits)
        rows.append((f, ser_mask, ser_perm, ser_pad))
        print(f"   f={f:.3f} mask={ser_mask:.4f} perm={ser_perm:.4f} "
              f"pad={ser_pad:.4f}")
    write_csv(DATA / "sec_sens_cmp.csv",
              ["frac", "ser_mask", "ser_perm", "ser_pad"], rows)


def stage_J():
    """Brute-force search against three schemes at the same key length.

    Keyed masking: K random unit keys, keep the best correlation, map it
    through the measured sensitivity curve of stage I.
    Permutation key: K random permutations of the d positions, keep the
    one that places the most positions correctly, map the resulting
    fraction through the same sensitivity curve.
    Index cipher: K random pads out of the 2^{log2 V} possible pads, so
    the attacker holds the right pad with probability K/V and still has
    to decode the ciphered index over the channel.
    """
    print("[J] brute-force search across schemes ...")
    import numpy as np
    cmp_rows = list(csv_rows(DATA / "sec_sens_cmp.csv"))
    f_arr = np.array([float(r["frac"]) for r in cmp_rows])
    mask_arr = np.array([float(r["ser_mask"]) for r in cmp_rows])
    perm_arr = np.array([float(r["ser_perm"]) for r in cmp_rows])

    d, L, V = MAIN_D, MAIN_D // 4, 65536
    # channel floor of the cipher receiver, read from the stage-I curve
    # at a fully known pad so both figures share one source
    lg1 = 1.0 - (1.0 - float(cmp_rows[-1]["ser_pad"]))
    ks = [1, 3, 10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 65_536,
          100_000, 300_000, 1_000_000]
    rng = np.random.default_rng(2026)
    trials = 400
    rows = []
    for K in ks:
        # keyed masking: best |first coordinate| of K random unit vectors
        best_kappa = np.empty(trials)
        best_frac = np.empty(trials)
        for t in range(trials):
            # |first coordinate| of a uniform random unit vector in R^L:
            # its square is Beta(1/2, (L-1)/2), so the best of K draws
            # needs K scalars rather than K*L Gaussians
            best_kappa[t] = np.sqrt(rng.beta(0.5, (L - 1) / 2.0,
                                             size=K).max())
            # permutation: fraction of fixed points, Binomial(d, 1/d) per
            # draw, so the best of K draws is the max of K such counts
            best_frac[t] = rng.binomial(d, 1.0 / d, size=K).max() / d
        ser_mask = float(np.mean(np.interp(best_kappa, f_arr, mask_arr)))
        ser_perm = float(np.mean(np.interp(best_frac, f_arr, perm_arr)))
        ser_pad = 1.0 - min(1.0, K / V) * (1.0 - lg1)
        rows.append((K, ser_mask, ser_perm, ser_pad,
                     float(best_kappa.mean()), float(best_frac.mean())))
        print(f"   K={K:8d} mask={ser_mask:.4f} perm={ser_perm:.4f} "
              f"pad={ser_pad:.4f}")
    write_csv(DATA / "sec_brute_cmp.csv",
              ["K", "ser_mask", "ser_perm", "ser_pad",
               "best_kappa", "best_frac"], rows)


def csv_rows(path):
    import csv as _csv
    with open(path) as f:
        yield from _csv.DictReader(f)


def oma_ser_jammed(snr_db, jsr_db_list, bits=16, U=4, d=256, n_grid=4096):
    """OMA under a jammer that concentrates on the victim's slots.

    An OMA user occupies L = d/U exclusive real dimensions that are
    public, and drives its 16 index bits on 16 of them with the whole
    allocation energy, an amplitude gain of sqrt(L/bits) per bit. A
    jammer needs no key to put all of its power on those same public
    dimensions. With unit energy per real dimension and a total jammer
    energy of rho times the frame energy, concentrating on bits of the d
    dimensions gives a per-dimension jammer variance of (d/bits)*rho.

    The jammer reaches the victim through its own Rayleigh channel, the
    same convention eval_scheme uses for every simulated scheme, so the
    victim sees an effective noise variance of 1/snr + U*rho*hJ**2 with
    E[hJ**2]=1. Averaging over the independent signal and jammer gains
    uses a product of exponential quantile grids.
    """
    q = (torch.arange(n_grid, dtype=torch.float64) + 0.5) / n_grid
    h2 = -torch.log1p(-q)                      # |h|^2 ~ Exp(1)
    hj2 = h2.clone()                           # |hJ|^2 ~ Exp(1), independent
    h = h2.sqrt()[:, None]                     # (n,1) signal amplitude
    snr = 10.0 ** (snr_db / 10.0)
    gain = math.sqrt((d / U) / bits)                      # antipodal amplitude
    out = []
    for jsr_db in jsr_db_list:
        rho = 10.0 ** (jsr_db / 10.0)
        var = (1.0 / snr + (d / bits) * rho * hj2)[None, :]        # (1,n)
        arg = (h * gain / var.sqrt()).clamp(0, 38)
        pe = 0.5 * torch.erfc(arg / math.sqrt(2.0))       # per-bit error
        out.append(float((1.0 - (1.0 - pe) ** bits).mean()))
    return out


def stage_L():
    """Jamming comparison across schemes at 10 dB.

    proposed blind   : the strongest jammer the proposed scheme admits
                       while the key stays secret
    public matched   : the jammer a public-mask scheme always faces
    permutation blind: the shuffling-style scheme, whose secret
                       permutation also denies the jammer a target
    OMA targeted     : the jammer an orthogonal scheme faces, since its
                       slot assignment is public and needs no key
    """
    print("[L] jamming across schemes ...")
    m = main_model()
    F = 300_000
    d = m.P * m.L
    gp = torch.Generator().manual_seed(11)
    perms = torch.randperm(d, generator=gp)[None].repeat(m.users, 1)
    jsr = [float(v) for v in range(-10, 21, 2)]
    oma = oma_ser_jammed(10.0, jsr, bits=int(math.log2(m.V)),
                         U=m.users, d=m.d)
    rows = []
    for i, j in enumerate(jsr):
        blind = eval_scheme(m, 10.0, F, jam_w="blind", jsr_db=j)
        matched = eval_scheme(m, 10.0, F, jam_w="matched", jsr_db=j)
        perm = eval_scheme(m, 10.0, F, perms=perms, jam_w="blind", jsr_db=j)
        rows.append((j, blind, matched, perm, oma[i]))
        print(f"   JSR={j:6.1f} blind={blind:.4f} matched={matched:.4f} "
              f"perm={perm:.4f} oma={oma[i]:.4f}")
    write_csv(DATA / "sec_jam_cmp.csv",
              ["jsr_db", "blind", "matched", "perm_blind", "oma_targeted"],
              rows)
    stage_L_gap(rows)


def stage_L_gap(rows):
    """Store the blind-vs-matched power gap as a raw artifact.

    For every error level both curves reach, the gap is the extra JSR the
    blind jammer needs to inflict it. Both curves are interpolated on the
    dense grid, so the quoted range comes from a stored file rather than
    from a hand interpolation.
    """
    import numpy as np
    j = np.array([r[0] for r in rows])
    blind = np.array([r[1] for r in rows])
    matched = np.array([r[2] for r in rows])
    lo = max(blind.min(), matched.min())
    hi = min(blind.max(), matched.max())
    ser = np.linspace(lo, hi, 200)
    jb = np.interp(ser, blind, j)
    jm = np.interp(ser, matched, j)
    gap = jb - jm
    write_csv(DATA / "sec_jam_gap.csv", ["ser", "gap_db"],
              list(zip(ser.tolist(), gap.tolist())))
    print(f"   gap: {gap.min():.2f} to {gap.max():.2f} dB "
          f"over SER {lo:.3f} to {hi:.3f}")


def main():
    print(f"device={DEVICE}")
    stage_A()
    stage_B()
    stage_C()
    stage_D()
    stage_E()
    stage_F()
    stage_I()
    stage_J()
    stage_L()
    print("[done] full-scale security CSVs in", DATA)


if __name__ == "__main__":
    main()
