"""Stage G: security on real language-model token streams.

AG News test headlines are tokenized with the bert-base-uncased
WordPiece tokenizer (vocabulary 30,522). Four users carry four disjoint
headline streams, each frame transmits one token per user, and the token
identifier is carried by its base-16 digits, so the digit space
16^4 = 65,536 covers the vocabulary. The keys and codebook trained on
uniform indices are reused unchanged, so this stage tests the design on
a real, highly non-uniform source without retraining.

Two metrics are reported. The token error rate is the symbol-level
measure used in the rest of the paper. The headline recovery rate is a
meaning-level measure: the fraction of complete headlines a receiver
reconstructs without a single token error, which is what an
eavesdropper actually needs to read the message.

Outputs:
  real_sec_ter.csv   : token error rate vs SNR for legitimate, outsider
                       eavesdropper, insider, and OMA
  real_sec_stats.json: stream statistics and headline recovery rates
"""
from __future__ import annotations

import json
import math

import torch

import sse_lib as L
from sse_lib import (DATA, DEVICE, SSE, rayleigh_gain, snr_to_sigma2,
                     set_seed, write_csv)
from exp_full import get_model, eve_wrong_mask

SNR_GRID = [0, 4, 8, 12, 16, 20, 24, 28]
# headline recovery is meaningful only where the legitimate user clears
# most tokens, since a headline averages tens of tokens and needs every
# one of them correct
REC_SNR = (20, 24, 28)
N_TEXTS = 2000
REPEATS = 8
REC_RUNS = 4
SEED_EVAL = 777
P_MAX, VU, U = 4, 16, 4


def load_streams():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds = load_dataset("fancyzhx/ag_news", split="test")
    texts = [ds[i]["text"] for i in range(N_TEXTS)]
    streams = [[] for _ in range(U)]
    bounds = [[] for _ in range(U)]     # (start, end) per headline
    for i, t in enumerate(texts):
        ids = tok(t, add_special_tokens=False)["input_ids"]
        u = i % U
        s = len(streams[u])
        streams[u].extend(ids)
        bounds[u].append((s, s + len(ids)))
    n = min(len(s) for s in streams)
    streams = [s[:n] for s in streams]
    bounds = [[(a, b) for (a, b) in bu if b <= n] for bu in bounds]
    return streams, bounds, tok.vocab_size


def ids_to_digits(ids: torch.Tensor) -> torch.Tensor:
    """(N,U) token ids -> (N,U,P) base-16 digits, most significant first."""
    d, x = [], ids.clone()
    for _ in range(P_MAX):
        d.append(x % VU)
        x = x // VU
    return torch.stack(d[::-1], dim=-1)


@torch.no_grad()
def wrong_keyed(model: SSE, digits_all, snr_db, seed, rx_masks=None,
                chunk=50_000):
    """Per-frame per-user error indicator (N,U) for a receiver that
    correlates with rx_masks. rx_masks=None means the legitimate keys."""
    torch.manual_seed(seed)
    Bn = model.unit_codebook()
    true_m = model.masks()
    rx = true_m if rx_masks is None else rx_masks.to(DEVICE)
    c = model.c
    N = digits_all.shape[0]
    wrong = torch.zeros(N, U, dtype=torch.bool)
    sigma = snr_to_sigma2(snr_db, model.d).to(DEVICE).sqrt()
    for n0 in range(0, N, chunk):
        dg = digits_all[n0:n0 + chunk].to(DEVICE)
        n = dg.shape[0]
        e = Bn[dg] / math.sqrt(model.P)
        y = (e * true_m[None, :, None, :]).sum(dim=1) / c
        h = rayleigh_gain((n, U), device=DEVICE)
        noise = torch.randn(n, U, model.P, model.L, device=DEVICE)
        y_rx = h[:, :, None, None] * y[:, None] + sigma * noise
        r = y_rx / h[:, :, None, None].clamp_min(1e-6)
        cand = Bn[None, :, :] * rx[:, None, :]
        sc = torch.einsum("nupl,uvl->nupv", r, cand)
        wrong[n0:n0 + chunk] = (sc.argmax(-1) != dg).any(dim=2).cpu()
    return wrong


@torch.no_grad()
def wrong_oma(ids_all, snr_db, seed, bits=16):
    """Antipodal signaling on the actual token bits, same frame energy."""
    torch.manual_seed(seed)
    N, Uu = ids_all.shape
    b = ((ids_all[..., None] >> torch.arange(bits)) & 1).float() * 2 - 1
    b = b.to(DEVICE)
    sigma = math.sqrt(1.0 / (10.0 ** (snr_db / 10.0)))
    h = rayleigh_gain((N, Uu, 1))
    y = h * b + sigma * torch.randn(N, Uu, bits, device=DEVICE)
    return ((y * b) < 0).any(dim=2).cpu()


def headline_recovery(wrong: torch.Tensor, bounds) -> tuple[int, int]:
    """A headline counts as recovered only if every token is correct."""
    ok = tot = 0
    for u in range(U):
        wu = wrong[:, u]
        for (a, b) in bounds[u]:
            tot += 1
            ok += int(not bool(wu[a:b].any()))
    return ok, tot


def main():
    set_seed(SEED_EVAL)
    streams, bounds, vocab = load_streams()
    ids_all = torch.tensor(list(zip(*streams)), dtype=torch.long)   # (N,U)
    digits_all = ids_to_digits(ids_all)
    N = ids_all.shape[0]
    assert int(ids_all.max()) < VU ** P_MAX

    print(f"[real] {N} frames, {int(torch.unique(ids_all).numel())} "
          f"distinct tokens, max id {int(ids_all.max())}")

    # keys and codebook trained on uniform indices, reused unchanged
    model = get_model(P=P_MAX, vu=VU, d=64, U=U, iters=4000)
    model.eval()

    eve_m = eve_wrong_mask(U, model.L, seed=20260813)          # outsider
    ins_m = model.masks().detach().roll(1, 0).cpu()            # insider

    schemes = {
        "legit": lambda s, k: wrong_keyed(model, digits_all, s, k),
        "eve": lambda s, k: wrong_keyed(model, digits_all, s, k, eve_m),
        "insider": lambda s, k: wrong_keyed(model, digits_all, s, k, ins_m),
        "oma": lambda s, k: wrong_oma(ids_all, s, k),
    }

    rows = []
    for s in SNR_GRID:
        ter = {}
        for name, fn in schemes.items():
            e = 0
            for r in range(REPEATS):
                e += int(fn(s, SEED_EVAL + 1000 * r + int(10 * s)).sum())
            ter[name] = e / (N * U * REPEATS)
        rows.append((s, ter["legit"], ter["eve"], ter["insider"], ter["oma"]))
        print("[real]", [f"{v:.4g}" for v in rows[-1]])
    write_csv(DATA / "real_sec_ter.csv",
              ["snr_db", "ter_legit", "ter_eve", "ter_insider", "ter_oma"],
              rows)

    rec = {}
    for s in REC_SNR:
        rec[str(s)] = {}
        for name, fn in schemes.items():
            ok = tot = 0
            for r in range(REC_RUNS):
                w = fn(s, SEED_EVAL + 5000 * r + int(10 * s))
                o, t = headline_recovery(w, bounds)
                ok += o; tot += t
            rec[str(s)][name] = ok / tot
            print(f"[rec] {s} dB {name}: {ok}/{tot} = {ok/tot:.4g}")

    stats = {
        "vocab_size": vocab,
        "n_texts": N_TEXTS,
        "frames": N,
        "repeats": REPEATS,
        "decisions_per_point": N * U * REPEATS,
        "distinct_tokens": int(torch.unique(ids_all).numel()),
        "max_token_id": int(ids_all.max()),
        "headlines_scored": sum(len(b) for b in bounds),
        "headline_runs": REC_RUNS,
        "recovery": rec,
    }
    (DATA / "real_sec_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
