# -*- coding: utf-8 -*-
"""Where does the legitimate advantage over OMA go?

An ideal M-ary receiver at the main configuration should reach 0.199 at
10 dB against the 0.275 of resource-matched OMA, a factor of 1.38, while
the system measures 0.257, a factor of 1.07. This script splits the
shortfall into its two causes: residual multi-user interference, which
orthogonal keys do not remove because masking is elementwise, and the
distance the trained unit codebook falls short of an orthogonal set.
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sse_lib as L
from sse_lib import DEVICE, snr_to_sigma2, rayleigh_gain
from exp_full import main_model

SNR_DB = 10.0
FRAMES = 400_000
CH = 40_000


def ser(m, snr_db, frames, solo=False):
    """SER of user 0. With solo=True the other users transmit nothing,
    while the power normalizer c is left at its four-user value so that
    user 0 keeps exactly the energy it has in the real system."""
    tot = wrong = 0
    with torch.no_grad():
        while tot < frames:
            n = min(CH, frames - tot)
            dig = torch.randint(m.vu, (n, m.users, m.P), device=DEVICE)
            Bn = m.unit_codebook()
            e = Bn[dig] / math.sqrt(m.P)
            mk = m.masks()
            x = e * mk[None, :, None, :]
            if solo:
                x = x[:, :1]
            y = x.sum(dim=1) / m.c
            h = rayleigh_gain((n, 1), device=DEVICE)
            sig = snr_to_sigma2(torch.full((n,), snr_db), m.d).to(DEVICE).sqrt()
            rx = h[:, :, None, None] * y[:, None] \
                + sig[:, None, None, None] * torch.randn(n, 1, m.P, m.L,
                                                         device=DEVICE)
            r = rx / h[:, :, None, None].clamp_min(1e-6)
            cand = Bn[None, :, :] * mk[:1, None, :]
            sc = torch.einsum("nupl,uvl->nupv", r, cand)
            bad = (sc.argmax(-1)[:, 0] != dig[:, 0]).any(dim=-1)
            wrong += int(bad.sum())
            tot += n
    return wrong / tot


def main():
    m = main_model()
    with torch.no_grad():
        Bn = m.unit_codebook()
        G = Bn @ Bn.T
        off = G - torch.diag(torch.diag(G))
        mk = m.masks()
        Gm = mk @ mk.T / m.L
        offm = Gm - torch.diag(torch.diag(Gm))

    print("main configuration: d=%d P=%d L=%d Vu=%d U=%d"
          % (m.d, m.P, m.L, m.vu, m.users))
    print("key cross-correlation, max |off-diagonal| : %.2e"
          % offm.abs().max())
    print("codebook Gram, max |off-diagonal|         : %.4f"
          % off.abs().max())
    print("codebook Gram, rms off-diagonal           : %.4f"
          % off.pow(2).sum().div(m.vu * (m.vu - 1)).sqrt())
    print("(an orthogonal set of %d codewords in %d dims would read 0)"
          % (m.vu, m.L))
    print()
    four = ser(m, SNR_DB, FRAMES, solo=False)
    solo = ser(m, SNR_DB, FRAMES, solo=True)
    print("user-0 SER, all four users transmitting   : %.4f" % four)
    print("user-0 SER, other users silent            : %.4f" % solo)
    print("OMA, resource matched (closed form)       : %.4f"
          % L.oma_ser([SNR_DB])[0])
    print("ideal 16-ary orthogonal (separate MC)     : 0.1986")


if __name__ == "__main__":
    main()
