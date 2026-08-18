"""Shared library for the scalable shared embedding (SSE) letter.

System model (real-vector convention, declared in the paper):
  d-dimensional real embedding frame, U users, single transmitter.
  Proposed SSE: the frame is split into P periods of length L = d/P and
  one unit codebook B in R^{Vu x L} is reused in every period, so the
  vocabulary size is V = Vu^P while the codebook stores only Vu*L
  numbers. Index v maps to base-Vu digits (i_1,...,i_P).
  Per-user periodic masks mu_u in R^L are repeated over the P periods.
  Tx: y = (1/c) * sum_u e(s_u) .* m_u, c fixes unit average frame power.
  Channel: user u sees h_u * y + n, h_u^2 ~ Exp(1) (Rayleigh magnitude,
  known at the receiver), n ~ N(0, sigma^2 I), sigma^2 = 1/(d*snr).
  Rx u: equalize by h_u, per period correlate with the masked unit
  codewords b_i .* mu_u and take the argmax digit.

Device: CUDA when available (run under WSL), CPU fallback.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "fig"
DATA.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------
# global configuration
# ----------------------------------------------------------------------
D = 256         # embedding dimension (real)
U = 4           # users
VU = 16         # unit codebook size
P_MAX = 4       # periods for the main configuration, V = 16^4 = 65536
SEED = 1

TRAIN_ITERS = 4000
TRAIN_BATCH = 256
TRAIN_SNR_DB = (0.0, 20.0)
LR = 3e-3


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def snr_to_sigma2(snr_db: torch.Tensor | float,
                  d: int = D) -> torch.Tensor | float:
    """Per-dimension noise variance for unit frame power and E[h^2]=1.
    Pass the model's actual frame dimension d when it differs from the
    module default, otherwise the effective SNR shifts with d."""
    snr = 10.0 ** (torch.as_tensor(snr_db, dtype=torch.float64) / 10.0)
    return (1.0 / (d * snr)).float()


def rayleigh_gain(shape, device=DEVICE) -> torch.Tensor:
    """|g| with g ~ CN(0,1): h^2 ~ Exp(1), E[h^2] = 1."""
    u = torch.rand(shape, device=device).clamp_min(1e-12)
    return torch.sqrt(-torch.log(u))


# ----------------------------------------------------------------------
# proposed scalable shared embedding
# ----------------------------------------------------------------------
class SSE(torch.nn.Module):
    """Periodic unit-codebook shared embedding with per-user periodic masks."""

    def __init__(self, P: int = P_MAX, vu: int = VU, d: int = D, users: int = U):
        super().__init__()
        assert d % P == 0, "embedding dimension must split into P periods"
        self.P, self.vu, self.d, self.users = P, vu, d, users
        self.L = d // P
        self.B = torch.nn.Parameter(torch.randn(vu, self.L) / math.sqrt(self.L))
        self.W = torch.nn.Parameter(torch.randn(users, self.L) / math.sqrt(self.L))
        self.logit_scale = torch.nn.Parameter(torch.tensor(2.0))
        # transmit power normalizer, calibrated after training (buffer)
        self.register_buffer("c", torch.tensor(1.0))

    @property
    def V(self) -> int:
        return self.vu ** self.P

    def unit_codebook(self) -> torch.Tensor:
        return self.B / self.B.norm(dim=1, keepdim=True).clamp_min(1e-8)

    def masks(self) -> torch.Tensor:
        return self.W / self.W.norm(dim=1, keepdim=True).clamp_min(1e-8) * math.sqrt(self.L)

    def tx_frame(self, digits: torch.Tensor) -> torch.Tensor:
        """digits: (N, U, P) ints -> unnormalized tx frame (N, P, L)."""
        Bn = self.unit_codebook()
        e = Bn[digits] / math.sqrt(self.P)          # (N,U,P,L)
        m = self.masks()                            # (U,L)
        x = e * m[None, :, None, :]
        return x.sum(dim=1)                         # (N,P,L)

    def calibrate_power(self, n: int = 65536) -> None:
        with torch.no_grad():
            digits = torch.randint(self.vu, (n, self.users, self.P), device=self.B.device)
            y = self.tx_frame(digits)
            self.c.fill_(float(y.pow(2).sum(dim=(1, 2)).mean().sqrt()))

    def scores(self, r: torch.Tensor) -> torch.Tensor:
        """r: equalized rx frame (N,P,L) -> scores (N,U,P,Vu)."""
        Bn = self.unit_codebook()                   # (Vu,L)
        m = self.masks()                            # (U,L)
        cand = Bn[None, :, :] * m[:, None, :]       # (U,Vu,L)
        return torch.einsum("npl,uvl->nupv", r, cand)

    def forward(self, digits: torch.Tensor, snr_db: torch.Tensor,
                h: torch.Tensor | None = None):
        """digits (N,U,P), snr_db (N,) -> per-user scores and rx frames."""
        N = digits.shape[0]
        y = self.tx_frame(digits) / self.c          # (N,P,L)
        if h is None:
            h = rayleigh_gain((N, self.users), device=y.device)
        sigma = snr_to_sigma2(snr_db, self.d).to(y.device).sqrt()  # (N,)
        n = torch.randn(N, self.users, self.P, self.L, device=y.device)
        y_rx = h[:, :, None, None] * y[:, None] + sigma[:, None, None, None] * n
        r = y_rx / h[:, :, None, None].clamp_min(1e-6)     # equalized (N,U,P,L)
        Bn = self.unit_codebook()
        m = self.masks()
        cand = Bn[None, :, :] * m[:, None, :]              # (U,Vu,L)
        scores = torch.einsum("nupl,uvl->nupv", r, cand)
        return scores

    def n_params(self) -> int:
        return self.B.numel() + self.W.numel()


# ----------------------------------------------------------------------
# conventional scheme: full unstructured codebook (prior shared embedding)
# ----------------------------------------------------------------------
class FullCodebook(torch.nn.Module):
    """Directly trained V x d codebook with full-d per-user masks."""

    def __init__(self, V: int, d: int = D, users: int = U):
        super().__init__()
        self.Vc, self.d, self.users = V, d, users
        self.E = torch.nn.Parameter(torch.randn(V, d) / math.sqrt(d))
        self.W = torch.nn.Parameter(torch.randn(users, d) / math.sqrt(d))
        self.logit_scale = torch.nn.Parameter(torch.tensor(2.0))
        self.register_buffer("c", torch.tensor(1.0))

    @property
    def V(self) -> int:
        return self.Vc

    def codebook(self) -> torch.Tensor:
        return self.E / self.E.norm(dim=1, keepdim=True).clamp_min(1e-8)

    def masks(self) -> torch.Tensor:
        return self.W / self.W.norm(dim=1, keepdim=True).clamp_min(1e-8) * math.sqrt(self.d)

    def tx_frame(self, idx: torch.Tensor) -> torch.Tensor:
        En = self.codebook()
        e = En[idx]                                 # (N,U,d)
        m = self.masks()
        return (e * m[None]).sum(dim=1)             # (N,d)

    def calibrate_power(self, n: int = 65536) -> None:
        with torch.no_grad():
            idx = torch.randint(self.Vc, (n, self.users), device=self.E.device)
            y = self.tx_frame(idx)
            self.c.fill_(float(y.pow(2).sum(dim=1).mean().sqrt()))

    def forward(self, idx: torch.Tensor, snr_db: torch.Tensor,
                h: torch.Tensor | None = None):
        N = idx.shape[0]
        y = self.tx_frame(idx) / self.c             # (N,d)
        if h is None:
            h = rayleigh_gain((N, self.users), device=y.device)
        sigma = snr_to_sigma2(snr_db).to(y.device).sqrt()
        n = torch.randn(N, self.users, self.d, device=y.device)
        y_rx = h[:, :, None] * y[:, None] + sigma[:, None, None] * n
        r = y_rx / h[:, :, None].clamp_min(1e-6)    # (N,U,d)
        En = self.codebook()
        m = self.masks()
        cand = En[None, :, :] * m[:, None, :]       # (U,V,d)
        scores = torch.einsum("nud,uvd->nuv", r, cand)
        return scores

    def n_params(self) -> int:
        return self.E.numel() + self.W.numel()


# ----------------------------------------------------------------------
# training
# ----------------------------------------------------------------------
def train_sse(model: SSE, iters: int = TRAIN_ITERS, batch: int = TRAIN_BATCH,
              lr: float = LR, log_every: int = 0, eval_snr: float = 10.0,
              eval_frames: int = 20000, seed: int = SEED):
    """Digit-wise cross entropy under superposition; cost independent of V."""
    set_seed(seed)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = torch.nn.CrossEntropyLoss()
    curve = []
    t0 = time.time()
    for it in range(1, iters + 1):
        digits = torch.randint(model.vu, (batch, model.users, model.P), device=DEVICE)
        snr = torch.empty(batch).uniform_(*TRAIN_SNR_DB)
        model.calibrate_power(8192)
        scores = model(digits, snr) * model.logit_scale.exp()
        loss = ce(scores.reshape(-1, model.vu), digits.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if log_every and (it % log_every == 0 or it == 1):
            ser = eval_ser_sse(model, [eval_snr], frames=eval_frames)[0]
            curve.append((it, time.time() - t0, float(loss.detach()), ser))
    model.calibrate_power()
    return curve


def train_full(model: FullCodebook, iters: int = TRAIN_ITERS,
               batch: int = TRAIN_BATCH, lr: float = LR, log_every: int = 0,
               eval_snr: float = 10.0, eval_frames: int = 20000,
               seed: int = SEED):
    set_seed(seed)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = torch.nn.CrossEntropyLoss()
    curve = []
    t0 = time.time()
    for it in range(1, iters + 1):
        idx = torch.randint(model.Vc, (batch, model.users), device=DEVICE)
        snr = torch.empty(batch).uniform_(*TRAIN_SNR_DB)
        model.calibrate_power(8192)
        scores = model(idx, snr) * model.logit_scale.exp()
        loss = ce(scores.reshape(-1, model.Vc), idx.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if log_every and (it % log_every == 0 or it == 1):
            ser = eval_ser_full(model, [eval_snr], frames=eval_frames)[0]
            curve.append((it, time.time() - t0, float(loss.detach()), ser))
    model.calibrate_power()
    return curve


# ----------------------------------------------------------------------
# evaluation (Monte Carlo)
# ----------------------------------------------------------------------
@torch.no_grad()
def eval_ser_sse(model: SSE, snr_list, frames: int = 2_000_000,
                 chunk: int = 100_000, seed: int = 777):
    """Frame (vocabulary-symbol) error rate: any wrong digit is an error."""
    model.eval().to(DEVICE)
    out = []
    for snr_db in snr_list:
        g = torch.Generator(device="cpu").manual_seed(seed + int(10 * snr_db))
        err = tot = 0
        for n0 in range(0, frames, chunk):
            n = min(chunk, frames - n0)
            digits = torch.randint(model.vu, (n, model.users, model.P),
                                   generator=g).to(DEVICE)
            snr = torch.full((n,), float(snr_db))
            scores = model(digits, snr)
            wrong = (scores.argmax(-1) != digits).any(dim=2)   # (n,U)
            err += int(wrong.sum()); tot += n * model.users
        out.append(err / tot)
    return out


@torch.no_grad()
def eval_ser_full(model: FullCodebook, snr_list, frames: int = 2_000_000,
                  chunk: int = 50_000, seed: int = 777):
    model.eval().to(DEVICE)
    if model.Vc >= 4096:
        chunk = max(512, (1 << 23) // model.Vc)
    out = []
    for snr_db in snr_list:
        g = torch.Generator(device="cpu").manual_seed(seed + int(10 * snr_db))
        err = tot = 0
        for n0 in range(0, frames, chunk):
            n = min(chunk, frames - n0)
            idx = torch.randint(model.Vc, (n, model.users), generator=g).to(DEVICE)
            snr = torch.full((n,), float(snr_db))
            scores = model(idx, snr)
            err += int((scores.argmax(-1) != idx).sum()); tot += n * model.users
        out.append(err / tot)
    return out


# ----------------------------------------------------------------------
# conventional digital scheme: OMA with BPSK per real dimension (QPSK
# per complex subcarrier under the declared real-imaginary stacking)
# ----------------------------------------------------------------------
def oma_ser(snr_db_list, bits: int = 16, n_grid: int = 200_000):
    """Semi-analytic expression under the real-dimension convention.
    All bits of a vocabulary symbol share the same flat-fading gain, so
    SER = E_h[1 - (1 - Q(h sqrt(snr)))^bits] with h^2 ~ Exp(1); the
    expectation is evaluated by numerical integration on a dense grid."""
    x = (np.arange(n_grid) + 0.5) / n_grid          # uniform quantiles
    h = np.sqrt(-np.log(1.0 - x))                   # inverse-CDF transform
    out = []
    for s in snr_db_list:
        g = 10.0 ** (s / 10.0)
        q = 0.5 * np.array([math.erfc(v / math.sqrt(2.0)) for v in
                            np.clip(h * math.sqrt(g), 0, 38)])
        out.append(float(np.mean(1.0 - (1.0 - q) ** bits)))
    return out


@torch.no_grad()
def oma_ser_mc(snr_db_list, bits: int = 16, frames: int = 2_000_000,
               chunk: int = 200_000, seed: int = 777):
    """Monte Carlo check of the closed form (same channel conventions)."""
    out = []
    for s in snr_db_list:
        sigma = math.sqrt(1.0 / (10.0 ** (s / 10.0)))  # per-dim, unit Es
        err = tot = 0
        torch.manual_seed(seed + int(10 * s))
        for n0 in range(0, frames, chunk):
            n = min(chunk, frames - n0)
            h = rayleigh_gain((n, 1))
            b = torch.randint(0, 2, (n, bits), device=DEVICE) * 2.0 - 1.0
            y = h * b + sigma * torch.randn(n, bits, device=DEVICE)
            wrong = ((y * b) < 0).any(dim=1)
            err += int(wrong.sum()); tot += n
        out.append(err / tot)
    return out


# ----------------------------------------------------------------------
# analysis: union bound with residual-interference Gaussian approximation
# ----------------------------------------------------------------------
@torch.no_grad()
def sse_union_bound(model: SSE, snr_db_list, n_mc_h: int = 200_000):
    """P_e <= 1 - (1 - min(1, sum_pairs)) ... digit union bound averaged
    over the Rayleigh gain, with measured residual interference treated
    as additional Gaussian noise (approximation stated in the paper)."""
    model.eval().to(DEVICE)
    Bn = model.unit_codebook()
    m = model.masks()
    c = float(model.c)
    # residual interference power per dimension at user u (measured)
    digits = torch.randint(model.vu, (65536, model.users, model.P), device=DEVICE)
    e = Bn[digits] / math.sqrt(model.P)
    x = e * m[None, :, None, :]                     # (N,U,P,L)
    y = x.sum(dim=1) / c                            # (N,P,L)
    # per-user: signal = x_u/c, interference = (y - x_u/c)
    interf_pw = []
    dmin2 = []
    for u in range(model.users):
        su = x[:, u] / c                            # (N,P,L)
        iu = y - su
        # project interference onto the normalized candidate directions
        cand = Bn * m[u][None, :]                   # (Vu,L)
        cn = cand / cand.norm(dim=1, keepdim=True).clamp_min(1e-8)
        proj = torch.einsum("npl,vl->npv", iu, cn)
        interf_pw.append(float(proj.pow(2).mean()))
        # pairwise distances of the scaled candidate set (tx side scaling)
        cs = cand / (c * math.sqrt(model.P))
        dd = torch.cdist(cs, cs)
        dmin2.append(float((dd + torch.eye(model.vu, device=DEVICE) * 1e9).min() ** 2))
    res = []
    g = rayleigh_gain(n_mc_h)
    for s in snr_db_list:
        sig2 = float(snr_to_sigma2(torch.tensor(s)))
        pe_users = []
        for u in range(model.users):
            # effective noise per dim after equalization: sig2/h^2 + interf
            sig_eff2 = sig2 / g.pow(2) + interf_pw[u]
            arg = torch.sqrt(torch.clamp(torch.tensor(dmin2[u], device=DEVICE)
                                         / (4.0 * sig_eff2), min=0.0))
            q = 0.5 * torch.erfc(arg / math.sqrt(2.0))
            p_digit = torch.clamp((model.vu - 1) * q, max=1.0)
            p_frame = 1.0 - (1.0 - p_digit) ** model.P
            pe_users.append(float(p_frame.mean()))
        res.append(sum(pe_users) / len(pe_users))
    return res


def write_csv(path: Path, header: list[str], rows) -> None:
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(f"{v:.10g}" if isinstance(v, float) else str(v)
                             for v in row) + "\n")
    print("[csv]", path)
