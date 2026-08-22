# -*- coding: utf-8 -*-
"""Semantic leakage: does a wrong index still carry the meaning?

The symbol error rate counts any wrong index as a total failure, which
is the right accounting for a bit pipe and the wrong one for a semantic
pipe: a token decoded as a near synonym has leaked the meaning even
though the index is wrong. This stage measures what the SER cannot see,
on two semantic scales.

  codeword cosine  cos(e_shat, e_s) between the embedding a receiver
                   reconstructs and the transmitted one, uniform indices
  BERT cosine      cos of the BERT input embeddings of the decoded and
                   the transmitted token, on the AG News stream, which
                   is semantic similarity in the space the vocabulary
                   was built for

Each is reported for the legitimate receiver, the outsider and the
insider, against the chance level of two independently drawn tokens.
A scheme leaks semantically if the adversary's similarity sits above
that chance level.

Run under WSL. Writes data/semantic.csv.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sse_lib as L
from sse_lib import DATA, DEVICE, rayleigh_gain, snr_to_sigma2
from exp_full import main_model, eve_wrong_mask

FRAMES = 200_000
CHUNK = 20_000
SNRS = [0.0, 10.0, 20.0]
REAL_SNRS = [0.0, 10.0, 20.0, 28.0]


@torch.no_grad()
def decide(m, dig, snr_db, keys, gen=None):
    """Decisions of a receiver holding `keys`, for the frames carrying
    `dig`. Returns the decided digits of user 0."""
    n = dig.shape[0]
    Bn = m.unit_codebook()
    e = Bn[dig] / math.sqrt(m.P)
    y = (e * m.masks()[None, :, None, :]).sum(dim=1) / m.c
    h = rayleigh_gain((n, 1), device=DEVICE)
    sig = snr_to_sigma2(torch.full((n,), snr_db), m.d).to(DEVICE).sqrt()
    rx = h[:, :, None, None] * y[:, None] \
        + sig[:, None, None, None] * torch.randn(n, 1, m.P, m.L,
                                                 device=DEVICE)
    r = rx / h[:, :, None, None].clamp_min(1e-6)
    cand = Bn[None, :, :] * keys[:1, None, :]
    return torch.einsum("nupl,uvl->nupv", r, cand).argmax(-1)[:, 0]


def frame_embedding(m, digits):
    """The d-dimensional embedding an index maps to, digits (N,P)."""
    Bn = m.unit_codebook()
    return (Bn[digits] / math.sqrt(m.P)).reshape(digits.shape[0], -1)


@torch.no_grad()
def codeword_cosine(m, snr_db, keys, seed):
    """Mean cosine between the reconstructed and the true embedding."""
    torch.manual_seed(seed + int(10 * snr_db))
    tot, done = 0.0, 0
    while done < FRAMES:
        n = min(CHUNK, FRAMES - done)
        dig = torch.randint(m.vu, (n, m.users, m.P), device=DEVICE)
        dec = decide(m, dig, snr_db, keys)
        c = F.cosine_similarity(frame_embedding(m, dec),
                                frame_embedding(m, dig[:, 0]), dim=1)
        tot += float(c.sum())
        done += n
    return tot / done


@torch.no_grad()
def codeword_chance(m, seed=99):
    """Cosine between two independently drawn indices."""
    torch.manual_seed(seed)
    a = torch.randint(m.vu, (FRAMES, m.P), device=DEVICE)
    b = torch.randint(m.vu, (FRAMES, m.P), device=DEVICE)
    return float(F.cosine_similarity(frame_embedding(m, a),
                                     frame_embedding(m, b), dim=1).mean())


def load_bert_embeddings():
    """BERT input embedding matrix, the space AG News tokens live in."""
    from transformers import AutoModel
    mdl = AutoModel.from_pretrained("bert-base-uncased")
    return mdl.get_input_embeddings().weight.detach().to(DEVICE)


@torch.no_grad()
def real_semantic(m, emb, ids_all, snr_db, keys, seed):
    """Mean BERT cosine between the decoded and the transmitted token of
    user 0. ids_all is (N,U): every user carries its OWN stream, so an
    insider decoding user 0 gains nothing from its own traffic."""
    torch.manual_seed(seed + int(10 * snr_db))
    n = ids_all.shape[0]
    dig = torch.stack([(ids_all // (m.vu ** p)) % m.vu
                       for p in range(m.P)], -1).to(DEVICE)   # (N,U,P)
    tot, done = 0.0, 0
    while done < n:
        k = min(CHUNK, n - done)
        dec = decide(m, dig[done:done + k], snr_db, keys)
        rec = sum(dec[:, p] * (m.vu ** p) for p in range(m.P))
        true = ids_all[done:done + k, 0].to(DEVICE)
        rec = rec.clamp(max=emb.shape[0] - 1)
        tot += float(F.cosine_similarity(emb[rec], emb[true], dim=1).sum())
        done += k
    return tot / n


def main():
    m = main_model()
    m.eval()
    ew = eve_wrong_mask(m.users, m.L, seed=20260813)
    insider = m.masks()[1:2].detach()          # user 2 attacking user 1
    rows = []

    chance = codeword_chance(m)
    print("codeword chance cosine %.4f" % chance, flush=True)
    for snr in SNRS:
        lg = codeword_cosine(m, snr, m.masks(), 5150)
        ev = codeword_cosine(m, snr, ew.to(DEVICE), 5151)
        ins = codeword_cosine(m, snr, insider, 5152)
        rows.append(("codeword", snr, "%.4f" % lg, "%.4f" % ev,
                     "%.4f" % ins, "%.4f" % chance))
        print("codeword %4.0f dB  legit %.4f  outsider %.4f  insider %.4f"
              % (snr, lg, ev, ins), flush=True)

    # real token streams in the BERT embedding space
    try:
        from exp_real_sec import load_streams
        streams, _bounds, _vocab = load_streams()
        nmin = min(len(x) for x in streams)
        ids = torch.stack([torch.as_tensor(x[:nmin], dtype=torch.long)
                           for x in streams], dim=1)[:100_000]   # (N,U)
        emb = load_bert_embeddings()
        rnd = torch.randint(0, emb.shape[0], (ids.shape[0],))
        ch = float(F.cosine_similarity(emb[ids[:, 0].to(DEVICE)],
                                       emb[rnd.to(DEVICE)], dim=1).mean())
        print("BERT chance cosine %.4f" % ch, flush=True)
        for snr in REAL_SNRS:
            lg = real_semantic(m, emb, ids, snr, m.masks(), 5160)
            ev = real_semantic(m, emb, ids, snr, ew.to(DEVICE), 5161)
            ins = real_semantic(m, emb, ids, snr, insider, 5162)
            rows.append(("bert", snr, "%.4f" % lg, "%.4f" % ev,
                         "%.4f" % ins, "%.4f" % ch))
            print("bert     %4.0f dB  legit %.4f  outsider %.4f  insider %.4f"
                  % (snr, lg, ev, ins), flush=True)
    except Exception as exc:                    # keep the codeword rows
        print("[skip] real-token semantic stage: %s: %s"
              % (type(exc).__name__, exc), flush=True)

    out = DATA / "semantic.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["space", "snr_db", "legit", "outsider", "insider",
                    "chance"])
        w.writerows(rows)
    print("[csv]", out)


if __name__ == "__main__":
    main()
