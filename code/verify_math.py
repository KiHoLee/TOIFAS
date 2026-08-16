"""Numerical verification of the closed forms in Sections IV and V of
paper 11 (mask-as-key encryption and jamming robustness).

Every claim that enters the manuscript is checked here against Monte
Carlo, with a PASS/FAIL verdict and the achieved agreement level printed.
Real-vector convention, dimension d, U users, codebook of V unit-norm
codewords, per-user masks with zero-mean entries normalized to
||m||^2 = d (so E[m_k^2] = 1).

Notation matches the tex:
  y = (1/c) sum_u e_{s_u} .* m_u        transmit frame
  legit score  z_{u,i}   = r_u^T (e_i .* m_u),   r_u = y + n_u/h_u
  eve score    zE_{u,i}  = (y_E/h_E)^T (e_i .* mtil_u)
  jammer adds  h_J sqrt(rho) w   to the victim observation

Run on CPU (NumPy); no training involved, pure algebra checks.
"""
from __future__ import annotations
import numpy as np

RNG = np.random.default_rng(2026)
D, U, V = 64, 4, 256


def unit_codebook(V, d, rng):
    E = rng.standard_normal((V, d))
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def masks(U, d, rng):
    """Zero-mean entries normalized so ||m_u||^2 = d."""
    M = rng.standard_normal((U, d))
    return M / np.linalg.norm(M, axis=1, keepdims=True) * np.sqrt(d)


ROWS = []                      # (tag, claim, emp, err, tol, verdict)


def report(tag, claim, emp, tol, extra=""):
    err = abs(claim - emp)
    ok = err <= tol
    ROWS.append((tag, claim, emp, err, tol, "PASS" if ok else "FAIL"))
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}: claim={claim:.5g} "
          f"emp={emp:.5g} |err|={err:.2g} tol={tol:g} {extra}")
    return ok


def v1_legit_self_alignment():
    """Claim: E[e_s^T diag(m^2) e_s] = 1 (signal self-correlation)."""
    E = unit_codebook(V, D, RNG)
    vals = []
    for _ in range(4000):
        m = masks(1, D, RNG)[0]
        s = RNG.integers(V)
        vals.append(float((E[s] ** 2) @ (m ** 2)))
    return report("V1 legit self-alignment", 1.0, float(np.mean(vals)), 2e-2)


def v2_eve_uninformed():
    """Claim: with an independent substitute mask, the eavesdropper's
    correct-index correlation has the same mean as any wrong index, so
    the mean advantage is zero and the eavesdropper SER = (V-1)/V,
    independent of SNR."""
    E = unit_codebook(V, D, RNG)
    # mean advantage of the true index over the wrong indices, noiseless
    adv = []
    ser_by_snr = {}
    for snr_db in [0.0, 10.0, 20.0, 80.0]:   # 80 dB stands in for noiseless
        sigma = np.sqrt(1.0 / (D * 10 ** (snr_db / 10.0)))
        err = 0
        trials = 6000
        for _ in range(trials):
            s = RNG.integers(V, size=U)
            M = masks(U, D, RNG)
            c = 1.0  # scale-invariant for argmax
            y = np.zeros(D)
            for u in range(U):
                y += E[s[u]] * M[u]
            y /= np.sqrt(U)  # any fixed scale
            # eavesdropper targets user 0 with an independent wrong mask
            mtil = masks(1, D, RNG)[0]
            hE = np.sqrt(-np.log(RNG.random()))
            rE = y + (sigma / hE) * RNG.standard_normal(D)
            scores = (E * mtil) @ rE          # (V,)
            if snr_db == 80.0:
                adv.append(scores[s[0]] - scores.mean())
            if scores.argmax() != s[0]:
                err += 1
        ser_by_snr[snr_db] = err / trials
    chance = (V - 1) / V
    ok1 = report("V2a eve mean advantage", 0.0, float(np.mean(adv)), 3e-3)
    ok2 = True
    for snr_db, ser in ser_by_snr.items():
        tag = f"V2b eve SER @ {int(snr_db)}dB"
        ok2 &= report(tag, chance, ser, 1.5e-2)
    return ok1 and ok2


def v3_leakage_vs_correlation():
    """Claim: if the substitute mask has normalized correlation
    rho = <m,mtil>/d with the true mask, the eavesdropper's true-index
    bias grows linearly in rho; independent random masks give
    E|rho| = O(1/sqrt(d)); orthogonal masks give rho = 0."""
    E = unit_codebook(V, D, RNG)
    # (a) bias vs prescribed rho
    slopes = []
    for rho in [0.0, 0.25, 0.5, 0.75, 1.0]:
        bias = []
        for _ in range(3000):
            m = masks(1, D, RNG)[0]
            mp = masks(1, D, RNG)[0]
            mp = mp - (mp @ m) / (m @ m) * m          # orthogonalize
            mp = mp / np.linalg.norm(mp) * np.sqrt(D)
            mtil = rho * m + np.sqrt(1 - rho ** 2) * mp
            s = RNG.integers(V)
            # noiseless single-user useful alignment for the true index
            bias.append(float((E[s] ** 2) @ (m * mtil)))
        slopes.append((rho, float(np.mean(bias))))
    # claim: bias(rho) = rho * bias(1); check linearity
    b1 = slopes[-1][1]
    lin_ok = all(abs(b - rho * b1) <= 3e-2 for rho, b in slopes)
    print(f"[{'PASS' if lin_ok else 'FAIL'}] V3a bias linear in rho: "
          + ", ".join(f"rho={r:.2f}->{b:.3f}" for r, b in slopes))
    # (b) random independent mask correlation: E|corr| = sqrt(2/(pi d))
    # (the folded-normal mean of a N(0, 1/d) variable)
    corrs = []
    for _ in range(5000):
        m = masks(1, D, RNG)[0]
        mt = masks(1, D, RNG)[0]
        corrs.append(abs((m @ mt) / D))
    emp = float(np.mean(corrs))
    claim = float(np.sqrt(2.0 / (np.pi * D)))
    ok_b = report("V3b random mask E|corr|", claim, emp, 0.1 * claim)
    return lin_ok and ok_b


def v4_blind_jammer_spread():
    """Claim: a mask-blind jammer (w independent of m_u) contributes a
    zero-mean term to every candidate score with variance
    (hJ^2/hu^2) rho * sum_k w_k^2 e_{i,k}^2, i.e. it is spread with no
    systematic bias toward any index."""
    E = unit_codebook(V, D, RNG)
    rho = 1.0
    # (a) structural claim: the mask projection g_i = (w .* m)^T e_i is
    # zero-mean, decoupled from the positive gain factor hJ/hu.
    g, var_emp, var_cl = [], [], []
    for _ in range(20000):
        m = masks(1, D, RNG)[0]
        w = RNG.standard_normal(D); w /= np.linalg.norm(w)
        i = RNG.integers(V)
        gi = (w * m) @ E[i]
        g.append(gi)
        var_emp.append(gi ** 2)
        var_cl.append(np.sum(w ** 2 * E[i] ** 2))
    ok1 = report("V4a blind jammer projection mean", 0.0,
                 float(np.mean(g)), 3e-3)
    # (b) variance of the projection matches sum_k w_k^2 e_{i,k}^2; the
    # full contribution scales this by (hJ^2/hu^2) rho.
    ok2 = report("V4b blind jammer projection variance",
                 float(np.mean(var_cl)), float(np.mean(var_emp)),
                 0.05 * float(np.mean(var_cl)))
    return ok1 and ok2


def v5_matched_jammer_concentrates():
    """Claim: a mask-matched jammer aligned with the victim key for a
    target index t creates a bias of order one on index t, while the
    blind-jammer projection has zero mean and RMS of order 1/sqrt(d).
    The physically meaningful separation is matched bias over blind RMS,
    which is sqrt(d) (the sample mean of the blind bias estimates zero
    and is pure Monte Carlo noise, so it is NOT a valid denominator)."""
    E = unit_codebook(V, D, RNG)
    bias_matched, blind_sq = [], []
    for _ in range(3000):
        m = masks(1, D, RNG)[0]
        t = RNG.integers(V)
        wm = E[t] * m; wm /= np.linalg.norm(wm)      # matched (needs m)
        wb = RNG.standard_normal(D); wb /= np.linalg.norm(wb)  # blind
        bias_matched.append(float((wm * m) @ E[t]))
        blind_sq.append(float(((wb * m) @ E[t]) ** 2))
    bm = float(np.mean(bias_matched))
    brms = float(np.sqrt(np.mean(blind_sq)))
    ratio = bm / brms
    ok = abs(ratio - np.sqrt(D)) <= 0.25 * np.sqrt(D) and bm > 0.9
    print(f"[{'PASS' if ok else 'FAIL'}] V5 matched bias / blind RMS: "
          f"matched={bm:.3f} blind_rms={brms:.4f} ratio={ratio:.1f} "
          f"(claim sqrt(d)={np.sqrt(D):.1f})")
    return ok


def main():
    print(f"config d={D} U={U} V={V}\n")
    results = {
        "V1": v1_legit_self_alignment(),
        "V2": v2_eve_uninformed(),
        "V3": v3_leakage_vs_correlation(),
        "V4": v4_blind_jammer_spread(),
        "V5": v5_matched_jammer_concentrates(),
    }
    print("\nsummary:", {k: ("PASS" if v else "FAIL") for k, v in results.items()})
    print("ALL PASS" if all(results.values()) else "SOME FAILED")
    # stored artifact so every quoted verification number has a raw file
    import csv as _csv
    from pathlib import Path as _Path
    data = _Path(__file__).resolve().parents[1] / "data"
    with open(data / "verify_math.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["check", "claim", "empirical", "abs_err", "tol",
                    "verdict"])
        w.writerows(ROWS)
    print("[csv]", data / "verify_math.csv")


if __name__ == "__main__":
    main()
