"""
empirical_lambda.py -- calibrates the self-correction rates lambda of the
simplified energy-poverty model on Czech EU-SILC microdata and quantifies the
impact of lambda on the stability of the system.

Data: dom_all_echudoba.RDS (household-level EU-SILC, 2005-2025). The fuzzy
memberships of the four dimensions are *not* taken from the "_f" columns of
that file -- those are either binary or piecewise linear -- but are rebuilt
from the underlying continuous variables with the C^1-smooth cosine
transitions of the paper; see silc_fuzzy.py for the construction and for the
threshold parameters:

    h  thermal deprivation   1{DOST_VYTAP == 2}                    (binary item)
    d  arrears               1{DLUH_PLATB == 1}                    (binary item)
    b  energy burden         Psi_{0->1}(pene; 0.15, 0.25)
    p  income poverty        Psi_{1->0}(rezsj24 / line_y; 0.8, 1.2)

Method. We form the weighted annual prevalence of each dimension, obtaining a
short multivariate time series x_t=(h,d,b,p) for 2009-2025. The simplified model
xdot = A x (+ c) is fitted with the sparsity pattern of the paper (each equation
regresses the annual change of one dimension on itself and its two drivers),
using ordinary least squares with an intercept c that represents a constant
exogenous inflow and shifts the equilibrium to the observed non-zero level. The
diagonal of A yields the empirical self-correction rates lambda_i=-A_ii; the
eigenvalues of A decide whether deprivation is self-correcting (all Re<0, stable)
or self-sustaining (some Re>0, poverty trap). We then sweep lambda_b and locate
the critical value at which the dominant eigenvalue changes sign, and finally
repeat the whole calibration for other transition widths and for the graded
variant of the arrears dimension, to show that the conclusion does not rest on
the particular fuzzification parameters.

Two fits are reported: unconstrained OLS, and the same fit restricted to the
sign pattern that the model imposes on A (lambda_i >= 0, alpha_ij >= 0). The
restriction matters here, because with sixteen annual differences and strongly
collinear series the unconstrained fit leaves the class of matrices the theory
describes.

    python empirical_lambda.py      # prints the calibration and writes the figure

Requires numpy, pandas, scipy, matplotlib, pyreadr.
"""

from pathlib import Path

import numpy as np
import numpy.linalg as la
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear

import silc_fuzzy as sf

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
OUTDIR = Path(__file__).with_name("img")

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

# Specification used for the figure and for the sensitivity table.
#   D_SOURCE         "utility" = the paper's I2 (binary EU-SILC item),
#                    "count"   = graded over the four arrears domains.
#   SIGN_CONSTRAINED restrict the fit to lambda_i >= 0 and alpha_ij >= 0, i.e.
#                    to the sign pattern the model imposes on A.
D_SOURCE = "utility"
SIGN_CONSTRAINED = True

# Fuzzification variants used for the robustness check: (label, thresholds,
# arrears variant). The widths Delta_b, Delta_p stay centred on the official
# cut-offs, so all variants reproduce the same crisp classification at the
# 1/2-cut and differ only in how much gradedness they admit.
VARIANTS = [
    ("baseline   Delta_b=0.10, Delta_p=0.40, d binary",
     sf.SilcThresholds(), "utility"),
    ("narrow     Delta_b=0.05, Delta_p=0.20, d binary",
     sf.SilcThresholds(Lb=0.175, Ub=0.225, Lp=0.90, Up=1.10), "utility"),
    ("wide       Delta_b=0.20, Delta_p=0.60, d binary",
     sf.SilcThresholds(Lb=0.10, Ub=0.30, Lp=0.70, Up=1.30), "utility"),
    ("near-crisp Delta_b=0.002,Delta_p=0.01, d binary",
     sf.SilcThresholds(Lb=0.199, Ub=0.201, Lp=0.995, Up=1.005), "utility"),
    ("baseline widths, d graded over arrears domains",
     sf.SilcThresholds(), "count"),
]

mpl.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300,
                     "savefig.bbox": "tight", "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3, "lines.linewidth": 1.8})


# ---------------------------------------------------------------------------
# Structured OLS estimation of A
# ---------------------------------------------------------------------------
def estimate_A(S, intercept=True, sign_constrained=False):
    """
    Fit xdot = A x (+ c) with the paper's sparsity; return A, lambdas, R2.

    With sign_constrained=True the estimation is restricted to the sign pattern
    the model itself imposes -- self-correction rates lambda_i = -A_ii >= 0 and
    cross-effects alpha_ij >= 0 -- so that the fitted matrix belongs to the class
    of matrices the theory is about. Unconstrained OLS is free to leave that
    class, and on these short and strongly collinear series it does.
    """
    order = list(sf.STATES)                   # h, d, b, p
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

        if sign_constrained:
            # own effect A_ss <= 0, cross effects >= 0, intercept unrestricted
            lo = np.array([-np.inf] + [0.0] * (len(regs) - 1) + ([-np.inf] if intercept else []))
            hi = np.array([0.0] + [np.inf] * (len(regs) - 1) + ([np.inf] if intercept else []))
            beta = lsq_linear(Z, y, bounds=(lo, hi)).x
        else:
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
    i = list(sf.STATES).index("b")
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


def critical_rates(A, grid=None):
    """
    Critical value of each self-correction rate: the level of lambda_i at which
    the dominant eigenvalue changes sign, all other entries held fixed. Returns
    None for a dimension whose rate cannot flip the sign over the grid.
    """
    if grid is None:
        grid = np.linspace(-0.40, 3.60, 801)
    out = {}
    for i, s in enumerate(sf.STATES):
        dom = np.array([dominant_eig(_with_lambda_b(A, i, l)) for l in grid])
        k = np.where(np.diff(np.sign(dom)))[0]
        if len(k) == 0:
            out[s] = None
        else:
            j = k[0]
            out[s] = grid[j] - dom[j] * (grid[j + 1] - grid[j]) / (dom[j + 1] - dom[j])
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(S, A, lambdas, thr, grid, dom, savepath=None):
    OUTDIR.mkdir(exist_ok=True)
    if savepath is None:
        savepath = OUTDIR / "fig7_empirical_lambda.png"

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.5, 3.6))

    # (a) observed weighted annual prevalence
    for s in sf.STATES:
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
    print("  wrote img/fig7_empirical_lambda.png and .pdf")


# ---------------------------------------------------------------------------
def report(S, A, lambdas, r2, thr, title="Calibration"):
    order = list(sf.STATES)
    print(f"\n{title}: years {S.index.min()}-{S.index.max()}  "
          f"(n = {len(S)} annual points)")
    print("Fitted A (rows = equations h, d, b, p):")
    for i, s in enumerate(order):
        print("   " + " ".join(f"{A[i, j]:+8.4f}" for j in range(4)) + f"   [{s}]")
    print("\nEstimated self-correction rates (per year):")
    for s in order:
        print(f"   lambda_{s} = {lambdas[s]:+.3f}   (equation R^2 = {r2[s]:.2f})")
    ev = np.linalg.eigvals(A)
    print("\nEigenvalues of A:", ", ".join(f"{e.real:+.3f}{e.imag:+.3f}i" for e in ev))
    dom = dominant_eig(A)
    regime = "poverty trap (Re>0)" if dom > 0 else "stable / near-stable (Re<0)"
    print(f"Dominant eigenvalue: {dom:+.3f}  ->  {regime}")
    crit = critical_rates(A)
    print("Critical rates (dominant eigenvalue changes sign at):")
    for s in order:
        c = crit[s]
        margin = "" if c is None else f", estimated {lambdas[s]:+.3f}"
        print(f"   lambda_{s}* = " + ("none in [-0.4, 3.6]" if c is None
                                      else f"{c:+.3f}") + margin)
    if thr is not None:
        print(f"Empirical lambda_b = {lambdas['b']:.3f} "
              f"({'above' if lambdas['b'] > thr else 'below'} lambda_b* = {thr:.3f})")


def sensitivity(hh, sign_constrained=False):
    """Re-calibrate under the alternative fuzzification parameters of VARIANTS."""
    print("\nSensitivity to the fuzzification parameters"
          f" ({'sign-constrained' if sign_constrained else 'OLS'})")
    print(f"{'variant':<48} {'mean b':>7} {'mean p':>7} {'lambda_b':>9} "
          f"{'dom eig':>8} {'lambda_b*':>10}")
    for label, thr, d_source in VARIANTS:
        S = sf.annual_series(sf.add_memberships(hh, thr, d_source))
        A, lambdas, _ = estimate_A(S, sign_constrained=sign_constrained)
        crit, _, _ = lambda_b_threshold(A)
        print(f"{label:<48} {S['b'].mean():>7.3f} {S['p'].mean():>7.3f} "
              f"{lambdas['b']:>9.3f} {dominant_eig(A):>8.3f} "
              + ("        --" if crit is None else f"{crit:>10.3f}"))


def main(sign_constrained=SIGN_CONSTRAINED):
    hh = sf.load_households()
    hh = sf.add_memberships(hh, sf.THRESHOLDS, d_source=D_SOURCE)
    sf.audit(hh)

    S = sf.annual_series(hh)
    print("\nWeighted annual prevalence")
    print(S.to_string(float_format="%.4f"))

    # Both fits are reported: the sign-constrained one quoted in the text, and
    # unconstrained OLS as the identification diagnostic.
    for constrained in (False, True):
        A, lambdas, r2 = estimate_A(S, intercept=True, sign_constrained=constrained)
        thr, _, _ = lambda_b_threshold(A)
        report(S, A, lambdas, r2, thr,
               title="Sign-constrained fit" if constrained else "Unconstrained OLS")

    sensitivity(hh, sign_constrained=sign_constrained)

    A, lambdas, r2 = estimate_A(S, intercept=True, sign_constrained=sign_constrained)
    thr, grid, dom = lambda_b_threshold(A)
    make_figure(S, A, lambdas, thr, grid, dom)


if __name__ == "__main__":
    main()
