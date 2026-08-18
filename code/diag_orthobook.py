# -*- coding: utf-8 -*-
"""Does an orthogonal unit codebook recover the shortfall?

diag_interference shows the gap to the single-user M-ary bound is not
multi-user interference but the geometry of the trained unit codebook,
whose Gram matrix carries a large root-mean-square off-diagonal where an
orthogonal set would carry zero. Vu <= L admits an exactly orthogonal
set, so this measures what installing one buys.

Two orthogonal sets are tried, because the choice is not free. The
Walsh-Hadamard set collides with the keys: the rows are closed under the
elementwise product, so masking a Hadamard codeword by a Hadamard key
returns another Hadamard codeword and every user ends up with the same
candidate set. A random orthogonal set carries no such group structure,
and masking by a unit-modulus key preserves its orthogonality exactly.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sse_lib as L
from sse_lib import DEVICE, SSE
from exp_full import hadamard, base_keys, oma_ser_keylen, MAIN_D
from diag_interference import ser

SNR = [0.0, 10.0, 20.0]
FRAMES = 400_000


def fixed_model(B, P=4, vu=16, d=64, U=4):
    L.set_seed(1)
    m = SSE(P=P, vu=vu, d=d, users=U).to(DEVICE)
    with torch.no_grad():
        m.B.copy_(B.to(DEVICE))
        m.W.copy_(base_keys(U, d // P).to(DEVICE))
    m.calibrate_power()
    return m


def hadamard_book(vu=16, Lp=MAIN_D // 4):
    B = torch.zeros(vu, Lp)
    B[:, :vu] = torch.tensor(hadamard(vu).copy(), dtype=torch.float32)
    return B


def random_ortho_book(vu=16, Lp=MAIN_D // 4, seed=7):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(Lp, Lp, generator=g)
    Q, _ = torch.linalg.qr(A)
    return Q[:vu].contiguous()


def report(name, m):
    with torch.no_grad():
        Bn = m.unit_codebook()
        G = Bn @ Bn.T
        off = (G - torch.diag(torch.diag(G))).abs().max()
    row = [name, "%.2e" % off]
    for s in SNR:
        row.append("%.4f" % ser(m, s, FRAMES))
    print("%-22s %-10s %-9s %-9s %-9s" % tuple(row))


def main():
    print("%-22s %-10s %-9s %-9s %-9s"
          % ("unit codebook", "max|off|", "0 dB", "10 dB", "20 dB"))
    report("Walsh-Hadamard", fixed_model(hadamard_book()))
    report("random orthogonal", fixed_model(random_ortho_book()))
    from exp_full import main_model
    report("trained", main_model())
    Lp = MAIN_D // 4
    print("%-22s %-10s %-9s %-9s %-9s"
          % ("OMA, resource matched", "-",
             "%.4f" % oma_ser_keylen(Lp, 0.0),
             "%.4f" % oma_ser_keylen(Lp, 10.0),
             "%.4f" % oma_ser_keylen(Lp, 20.0)))





def solo_check():
    """Splitting each candidate set from the superposition it must live
    in. Orthogonal codewords are ideal for one user alone and are what
    the single-user bound assumes, but the masked sets of different users
    are then far from orthogonal to each other."""
    print()
    print("%-22s %-12s %-12s" % ("unit codebook", "solo 10 dB", "4-user 10 dB"))
    for name, B in (("random orthogonal", random_ortho_book()),
                    ("trained (retrain)", None)):
        if B is None:
            from exp_full import main_model
            m = main_model()
        else:
            m = fixed_model(B)
        print("%-22s %-12.4f %-12.4f"
              % (name, ser(m, 10.0, FRAMES, solo=True),
                 ser(m, 10.0, FRAMES, solo=False)))
    print("(solo isolates the candidate set from the superposition)")


if __name__ == "__main__":
    main()
    solo_check()
