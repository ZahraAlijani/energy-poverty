"""
empirical_lambda.py -- calibrates the self-correction rates lambda of the
simplified energy-poverty model on Czech EU-SILC microdata and quantifies the
impact of lambda on the stability of the system.

Data: dom_all_echudoba.RDS (household-level EU-SILC, 2005-2025). The file already
contains the fuzzy indicators of the paper:
    I1_f      fuzzy heating/cooling indicator      -> state h
    I2_f      fuzzy debt indicator                 -> state d
    bydzat_f  fuzzy energy-burden membership       -> state b
    chudoba24_f  fuzzy poverty membership          -> state p
Household weights are in PKOEF.

Method. We form the weighted annual prevalence of each dimension, obtaining a
short multivariate time series x_t=(h,d,b,p) for 2009-2025. The simplified model
xdot = A x (+ c) is fitted with the sparsity pattern of the paper (each equation
regresses the annual change of one dimension on itself and its two drivers),
using ordinary least squares with an intercept c that represents a constant
exogenous inflow and shifts the equilibrium to the observed non-zero level. The
diagonal of A yields the empirical self-correction rates lambda_i=-A_ii; the
eigenvalues of A decide whether deprivation is self-correcting (all Re<0, stable)
or self-sustaining (some Re>0, poverty trap). Finally we sweep lambda_b and
locate the critical value at which the dominant eigenvalue changes sign.

    python empirical_lambda.py      # prints the calibration and writes the figure

Requires numpy, pandas, matplotlib, pyreadr.
"""

from pathlib import Path

import numpy as np
import numpy.linalg as la
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyreadr

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
RDS_PATH = Path(__file__).with_name("dom_all_echudoba.RDS")
OUTDIR = Path(__file__).with_name("img")
WEIGHT_COL = "PKOEF"

# Mapping: model state -> fuzzy-indicator column in the data.
STATE_COLS = {"h": "I1_f", "d": "I2_f", "b": "bydzat_f", "p": "chudoba24_f"}
STATE_LABELS = {"h": r"$h$ (thermal)", "d": r"$d$ (arrears)",
                "b": r"$b$ (burden)", "p": r"$p$ (poverty)"}
STATE_COLORS = {"h": "red", "d": "blue", "b": "darkorange", "p": "darkviolet"}

# Regressors of each equation, following the sparsity of matrix A in the paper:
#   hdot = -lh h + a_hb b + a_hp p
#   ddot = -ld d + a_db b + a_dp p
#   bdot = -lb b + a_bd d + a_bp p
#   pdot = -lp p + a_pd d + a_ph h
EQUATIONS = {"h": ["h", "b", "p"], "d": ["d", "b", "p"],
             "b": ["b", "d", "p"], "p": ["p", "d", "h"]}

LAMBDA_B_GRID = np.linspace(-0.20, 0.80, 401)

mpl.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300,
                     "savefig.bbox": "tight", "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3, "lines.linewidth": 1.8})


# ---------------------------------------------------------------------------
# Data -> weighted annual prevalence series
# ---------------------------------------------------------------------------
def load_series():
    """Weighted annual prevalence of each state variable, indexed by year."""
    df = pyreadr.read_r(str(RDS_PATH))[None].copy()
    df["rok"] = df["rok"].astype(int)
    w = pd.to_numeric(df[WEIGHT_COL], errors="coerce")

    def wmean(col):
        x = pd.to_numeric(df[col], errors="coerce")
        m = x.notna() & w.notna()
        num = (x[m] * w[m]).groupby(df["rok"][m]).sum()
        den = w[m].groupby(df["rok"][m]).sum()
        return num / den

    series = {s: wmean(col) for s, col in STATE_COLS.items()}
    return pd.DataFrame(series).dropna()


# ---------------------------------------------------------------------------
# Structured OLS estimation of A
# ---------------------------------------------------------------------------
def estimate_A(S, intercept=True):
    """Fit xdot = A x (+ c) with the paper's sparsity; return A, lambdas, R2."""
    order = list(STATE_COLS)                 # h, d, b, p
    idx = {s: i for i, s in enumerate(order)}
    X = S[order].values
    dX = np.diff(X, axis=0)                   # annual change ~ xdot (dt = 1 year)
    Xl = X[:-1]
    n = len(Xl)

    A = np.zeros((4, 4))
    lambdas, r2 = {}, {}
    for s, regs in EQUATIONS.items():
        cols = [idx[r] for r in regs]
        Z = Xl[:, cols]
        if intercept:
            Z = np.hstack([Z, np.ones((n, 1))])
        y = dX[:, idx[s]]
        beta, *_ = la.lstsq(Z, y, rcond=None)
        for r, coef in zip(regs, beta):       # first len(regs) coeffs are A-entries
            A[idx[s], idx[r]] = coef
        lambdas[s] = -A[idx[s], idx[s]]
        yhat = Z @ beta
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2[s] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return A, lambdas, r2


def dominant_eig(A):
    return max(np.linalg.eigvals(A).real)


def lambda_b_threshold(A):
    """Value of lambda_b at which the dominant eigenvalue crosses zero."""
    i = list(STATE_COLS).index("b")
    dom = np.array([dominant_eig(_with_lambda_b(A, i, lb)) for lb in LAMBDA_B_GRID])
    sign_change = np.where(np.diff(np.sign(dom)))[0]
    if len(sign_change) == 0:
        return None, LAMBDA_B_GRID, dom
    k = sign_change[0]
    # linear interpolation of the zero crossing
    lb0, lb1, d0, d1 = (LAMBDA_B_GRID[k], LAMBDA_B_GRID[k + 1], dom[k], dom[k + 1])
    thr = lb0 - d0 * (lb1 - lb0) / (d1 - d0)
    return thr, LAMBDA_B_GRID, dom


def _with_lambda_b(A, i, lb):
    Ab = A.copy()
    Ab[i, i] = -lb
    return Ab


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(S, A, lambdas, thr, grid, dom, savepath=None):
    OUTDIR.mkdir(exist_ok=True)
    if savepath is None:
        savepath = OUTDIR / "fig7_empirical_lambda.png"

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.5, 3.6))

    # (a) observed weighted annual prevalence
    for s in STATE_COLS:
        axa.plot(S.index, S[s], marker="o", ms=3,
                 color=STATE_COLORS[s], label=STATE_LABELS[s])
    axa.set_title("(a) Weighted annual prevalence, Czech EU--SILC")
    axa.set_xlabel("year")
    axa.set_ylabel("fuzzy deprivation share")
    axa.set_ylim(0, 1)
    axa.legend(ncol=2, fontsize=8, framealpha=0.85)

    # (b) impact of lambda_b on the dominant eigenvalue
    lam_b = lambdas["b"]
    axb.plot(grid, dom, color="black")
    axb.axhline(0.0, color="gray", lw=0.8)
    axb.fill_between(grid, dom, 0, where=(dom > 0), color="red", alpha=0.12)
    axb.fill_between(grid, dom, 0, where=(dom <= 0), color="green", alpha=0.10)
    if thr is not None:
        axb.axvline(thr, color="gray", ls=":", lw=1)
        axb.annotate(rf"$\lambda_b^*\approx{thr:.2f}$", (thr, axb.get_ylim()[1] * 0.6),
                     ha="left", fontsize=8)
    axb.axvline(lam_b, color="darkorange", ls="--", lw=1.4)
    axb.annotate(rf"empirical $\lambda_b\approx{lam_b:.2f}$",
                 (lam_b, axb.get_ylim()[0] * 0.7), ha="right", fontsize=8,
                 color="darkorange")
    axb.set_title(r"(b) Dominant eigenvalue vs. $\lambda_b$")
    axb.set_xlabel(r"burden self-correction rate $\lambda_b$")
    axb.set_ylabel(r"$\max_i \operatorname{Re}\lambda_i(A)$")
    axb.text(grid[5], axb.get_ylim()[1] * 0.75, "trap", color="darkred", fontsize=8)
    axb.text(grid[-40], axb.get_ylim()[0] * 0.75, "stable", color="darkgreen",
             fontsize=8)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"fig7_empirical_lambda.{ext}")
    plt.close(fig)
    print(f"  wrote img/fig7_empirical_lambda.png and .pdf")


# ---------------------------------------------------------------------------
def report(S, A, lambdas, r2, thr):
    order = list(STATE_COLS)
    print(f"Years: {S.index.min()}-{S.index.max()}  (n = {len(S)} annual points)")
    print("\nEstimated self-correction rates (per year):")
    for s in order:
        print(f"   lambda_{s} = {lambdas[s]:+.3f}   (equation R^2 = {r2[s]:.2f})")
    ev = np.linalg.eigvals(A)
    print("\nEigenvalues of A:", ", ".join(f"{e.real:+.3f}{e.imag:+.3f}i" for e in ev))
    dom = dominant_eig(A)
    regime = "poverty trap (Re>0)" if dom > 0 else "stable / near-stable (Re<0)"
    print(f"Dominant eigenvalue: {dom:+.3f}  ->  {regime}")
    if thr is not None:
        print(f"\nCritical burden rate: lambda_b* = {thr:.3f}")
        print(f"Empirical lambda_b   = {lambdas['b']:.3f} "
              f"({'above' if lambdas['b'] > thr else 'below'} threshold)")


def main():
    S = load_series()
    A, lambdas, r2 = estimate_A(S, intercept=True)
    thr, grid, dom = lambda_b_threshold(A)
    report(S, A, lambdas, r2, thr)
    make_figure(S, A, lambdas, thr, grid, dom)


if __name__ == "__main__":
    main()
