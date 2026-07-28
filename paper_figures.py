"""
Regenerates every figure of "Dynamical Modeling of Energy Poverty"
(PovertyDynamics.tex) from the model itself -- no figure is drawn by hand.

    python paper_figures.py            # writes img/*.png and img/*.pdf

Figures produced (numbering as in the paper):

    Fig. 1  img/fig1_trap_trajectories   Sec. 5.6.1  poverty-trap regime, lambda_b = 0
    Fig. 2  img/fig2_stable_trajectories Sec. 5.6.2  stable regime, lambda_b > 0
    Fig. 3  img/fig3_heating_cooling     Sec. 6.1.1  heating/cooling indicator I1
    Fig. 4  img/fig4_debt_indicator      Sec. 6.1.2  debt indicator I2 and its sensitivity
    Fig. 5  img/fig5_expenses_income     Sec. 6.1.3  burden, poverty, and I3 = mu_b (x) mu_p
    Fig. 6  img/fig6_fuzzy_index         Sec. 6.2.1  fuzzy indicators and index along trajectories

All parameter values are collected in the CONFIG block below and are the ones
quoted in the captions. The fuzzy operators and transitions are imported from
FuzzyEnergyPoverty.py, so the paper, this script and that module cannot drift
apart. Running the script also prints the eigenvalues of the system matrix in
both regimes, i.e. the numbers cited in Sections 5.6.1 and 5.6.2.

Requires numpy, scipy and matplotlib (all available in Google Colab).
"""

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import FuzzyEnergyPoverty as fz

# ---------------------------------------------------------------------------
# CONFIG - every number that appears in a caption of the paper
# ---------------------------------------------------------------------------
OUTDIR = Path(__file__).with_name("img")

# Coupling coefficients of the simplified system, eqs. (12)-(15).
ALPHA = dict(hb=0.2, hp=0.3, db=0.2, dp=0.2, bd=0.3, bp=0.2, pd=0.2, ph=0.2)

# Self-damping rates per regime: (lambda_h, lambda_d, lambda_b, lambda_p).
REGIMES = {
    "trap":   (1.0, 1.0, 0.0, 1.0),   # Fig. 1: no direct damping of the burden
    "stable": (1.0, 1.0, 1.2, 1.0),   # Fig. 2: burden damped directly
}

# Initial conditions of the four panels; all inside the state space [0,1]^4.
INITIAL_CONDITIONS = [
    (1.0, 0.2, 0.5, 0.4),
    (0.2, 1.0, 0.8, 0.3),
    (0.8, 0.8, 0.8, 0.8),
    (1.0, 0.5, 0.2, 1.0),
]

T_SPAN = (0.0, 20.0)
T_EVAL = np.linspace(*T_SPAN, 601)

# Colors and labels of the four state variables (same order everywhere).
STATE_COLORS = ["red", "blue", "darkorange", "darkviolet"]
STATE_LABELS = [r"$h(t)$", r"$d(t)$", r"$b(t)$", r"$p(t)$"]

# Fig. 3: heating/cooling transitions in degrees Celsius.
TEMP_RANGE = (10.0, 35.0)
LH, UH = 18.0, 21.0          # heating transition
LC, UC = 25.0, 28.0          # cooling transition

# Fig. 4: debt indicator, crisp threshold tau_2 with three transition widths.
TAU_D = 0.5
DEBT_WIDTHS = [0.1, 0.3, 0.6]
DEBT_COLORS = ["darkorange", "red", "darkviolet"]

# Fig. 5: burden and income transitions bracketing tau_b = 0.20 and tau_p = 0.60.
B_L, B_U = 0.15, 0.25        # (L_b, U_b) bracket the burden threshold tau_b
Y_L, Y_U = 0.50, 0.70        # (L_p, U_p) bracket the income threshold tau_p
TAU_B, TAU_P = 0.20, 0.60
B_RANGE = (0.0, 0.4)
Y_RANGE = (0.0, 1.2)

# Fig. 6: transitions used along the trajectories, equal weights.
BAND = (0.3, 0.7)            # [L, U] in every dimension
WEIGHTS = (1 / 3, 1 / 3, 1 / 3)
X0_INDEX = (0.8, 0.8, 0.8, 0.8)

mpl.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.8,
})


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def system_matrix(lambdas):
    """System matrix A of eqs. (12)-(15) for (lambda_h, lambda_d, lambda_b, lambda_p)."""
    lh, ld, lb, lp = lambdas
    a = ALPHA
    return np.array([
        [-lh,     0.0,     a["hb"], a["hp"]],
        [0.0,     -ld,     a["db"], a["dp"]],
        [0.0,     a["bd"], -lb,     a["bp"]],
        [a["ph"], a["pd"], 0.0,     -lp],
    ], dtype=float)


def solve(lambdas, x0, t_eval=T_EVAL):
    """Integrate x' = A x on T_SPAN and return (t, h, d, b, p)."""
    A = system_matrix(lambdas)
    sol = solve_ivp(lambda t, x: A @ x, T_SPAN, np.asarray(x0, dtype=float),
                    t_eval=t_eval, rtol=1e-8, atol=1e-10)
    return sol.t, *sol.y


def save(fig, name):
    """Write a figure as PNG and PDF into img/."""
    OUTDIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote img/{name}.png and img/{name}.pdf")


# ---------------------------------------------------------------------------
# Figures 1 and 2: state trajectories in the two regimes
# ---------------------------------------------------------------------------
def figure_trajectories(regime, name, ymax):
    """2x2 panel of trajectories, one panel per initial condition."""
    lambdas = REGIMES[regime]
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.4), sharex=True, sharey=True)

    for ax, x0 in zip(axes.flat, INITIAL_CONDITIONS):
        t, *states = solve(lambdas, x0)
        for series, color, label in zip(states, STATE_COLORS, STATE_LABELS):
            ax.plot(t, series, color=color, label=label)
        ax.set_title(r"$x_0=(%s)$" % ",\\,".join(f"{v:.1f}" for v in x0))
        ax.set_xlim(*T_SPAN)
        ax.set_ylim(0, ymax)

    axes[0, 0].legend(ncol=2, framealpha=0.85)
    for ax in axes[1, :]:
        ax.set_xlabel("time $t$")
    for ax in axes[:, 0]:
        ax.set_ylabel("state value")

    fig.tight_layout()
    save(fig, name)


# ---------------------------------------------------------------------------
# Figure 3: heating and cooling indicator against indoor temperature
# ---------------------------------------------------------------------------
def figure_heating_cooling():
    h = np.linspace(*TEMP_RANGE, 600)
    mu_H = fz.leq(h, LH, UH)                  # (h <~ [LH, UH]), falling
    mu_C = fz.geq(h, LC, UC)                  # (h >~ [LC, UC]), rising
    I1 = fz.prob_sum(mu_H, mu_C)              # eq. (29)

    fig, (ax_h, ax_i, ax_c) = plt.subplots(1, 3, figsize=(6.3, 2.3), sharey=True)

    ax_h.plot(h, mu_H, color="red")
    for x, lbl, ha in ((LH, "$L_h$", "right"), (UH, "$U_h$", "left")):
        ax_h.axvline(x, color="gray", ls=":", lw=0.8)
        ax_h.annotate(lbl, (x, 0.12), ha=ha, fontsize=8)
    ax_h.set_title("(a) heating adequacy")

    ax_i.plot(h, I1, color="black")
    ax_i.axvspan(UH, LC, color="green", alpha=0.12)
    ax_i.text((UH + LC) / 2, 0.5, "comfort\nregion", ha="center", va="center",
              fontsize=8, color="darkgreen")
    ax_i.set_title("(b) fuzzy index")

    ax_c.plot(h, mu_C, color="blue")
    for x, lbl, ha in ((LC, "$L_c$", "right"), (UC, "$U_c$", "left")):
        ax_c.axvline(x, color="gray", ls=":", lw=0.8)
        ax_c.annotate(lbl, (x, 0.12), ha=ha, fontsize=8)
    ax_c.set_title("(c) cooling adequacy")

    for ax in (ax_h, ax_i, ax_c):
        ax.set_xlabel(r"$h(t)$  [$^\circ$C]")
        ax.set_xlim(*TEMP_RANGE)
        ax.set_ylim(-0.03, 1.05)
    ax_h.set_ylabel("membership degree")

    fig.tight_layout()
    save(fig, "fig3_heating_cooling")


# ---------------------------------------------------------------------------
# Figure 4: debt indicator and its sensitivity
# ---------------------------------------------------------------------------
def figure_debt():
    d = np.linspace(0.0, 1.0, 800)
    fig, (ax_i, ax_s) = plt.subplots(1, 2, figsize=(6.3, 2.5))

    ax_i.step([0, TAU_D, TAU_D, 1], [0, 0, 1, 1], where="post",
              color="gray", ls="--", lw=1.2,
              label=r"crisp $\mathbf{1}\{d\geq\tau_d\}$")
    for width, color in zip(DEBT_WIDTHS, DEBT_COLORS):
        L, U = TAU_D - width / 2, TAU_D + width / 2
        ax_i.plot(d, fz.psi_up(d, L, U), color=color,
                  label=rf"$\Delta_d={width}$")
        ax_s.plot(d, fz.psi_up_derivative(d, L, U), color=color,
                  label=rf"$\Delta_d={width}$")

    ax_i.axvline(TAU_D, color="gray", ls=":", lw=0.8)
    ax_i.annotate(r"$\tau_d$", (TAU_D, 0.07), ha="left", fontsize=8)
    ax_i.set_ylim(-0.03, 1.05)
    ax_i.set_ylabel("membership degree")
    ax_i.set_title(r"(a) debt indicator $\widetilde I_2$")
    ax_i.legend(loc="upper left", framealpha=0.85)

    ax_s.set_ylim(-0.5, np.pi / (2 * min(DEBT_WIDTHS)) * 1.08)
    ax_s.set_ylabel(r"$\partial\widetilde I_2/\partial d$")
    ax_s.set_title("(b) sensitivity")
    ax_s.legend(loc="upper right", framealpha=0.85)

    for ax in (ax_i, ax_s):
        ax.set_xlabel("$d(t)$")
        ax.set_xlim(0, 1)

    fig.tight_layout()
    save(fig, "fig4_debt_indicator")


# ---------------------------------------------------------------------------
# Figure 5: energy-expenses-and-income indicator
# ---------------------------------------------------------------------------
def figure_expenses_income():
    b = np.linspace(*B_RANGE, 600)
    y = np.linspace(*Y_RANGE, 600)

    fig, (ax_b, ax_y, ax_i) = plt.subplots(1, 3, figsize=(6.3, 2.4))

    ax_b.plot(b, fz.geq(b, B_L, B_U), color="red")
    for x, lbl, ha in ((B_L, "$L_b$", "right"), (B_U, "$U_b$", "left")):
        ax_b.axvline(x, color="gray", ls=":", lw=0.8)
        ax_b.annotate(lbl, (x, 0.88), ha=ha, fontsize=8)
    ax_b.axvline(TAU_B, color="k", ls="--", lw=0.9)
    ax_b.annotate(r"$\tau_b$", (TAU_B, 0.07), ha="center", fontsize=8)
    ax_b.set_xlim(*B_RANGE)
    ax_b.set_xlabel("$b(t)$")
    ax_b.set_ylabel("membership degree")
    ax_b.set_title(r"(a) burden $\mu_b$")

    ax_y.plot(y, fz.leq(y, Y_L, Y_U), color="blue")
    for x, lbl, ha in ((Y_L, "$L_p$", "right"), (Y_U, "$U_p$", "left")):
        ax_y.axvline(x, color="gray", ls=":", lw=0.8)
        ax_y.annotate(lbl, (x, 0.88), ha=ha, fontsize=8)
    ax_y.axvline(TAU_P, color="k", ls="--", lw=0.9)
    ax_y.annotate(r"$\tau_p$", (TAU_P, 0.07), ha="center", fontsize=8)
    ax_y.set_xlim(*Y_RANGE)
    ax_y.set_xlabel("$y(t)$")
    ax_y.set_title(r"(b) income $\mu_p$")

    for ax in (ax_b, ax_y):
        ax.set_ylim(-0.03, 1.05)

    # I3 = mu_b (x) mu_p over the (b, y) plane, eq. (31).
    B, Y = np.meshgrid(np.linspace(*B_RANGE, 240), np.linspace(*Y_RANGE, 240))
    I3 = fz.t_norm(fz.geq(B, B_L, B_U), fz.leq(Y, Y_L, Y_U))

    mesh = ax_i.pcolormesh(B, Y, I3, shading="gouraud", cmap="viridis",
                           vmin=0.0, vmax=1.0)
    # Boundary of the crisp deprivation region {b >= tau_b} and {y <= tau_p}.
    ax_i.plot([TAU_B, TAU_B], [Y_RANGE[0], TAU_P], color="white", ls="--", lw=1.2)
    ax_i.plot([TAU_B, B_RANGE[1]], [TAU_P, TAU_P], color="white", ls="--", lw=1.2)
    ax_i.set_xlabel("$b(t)$")
    ax_i.set_ylabel("$y(t)$")
    ax_i.set_title(r"(c) $\widetilde I_3=\mu_b\otimes\mu_p$")
    ax_i.grid(False)
    fig.colorbar(mesh, ax=ax_i, fraction=0.046, pad=0.04)

    fig.tight_layout()
    save(fig, "fig5_expenses_income")


# ---------------------------------------------------------------------------
# Figure 6: fuzzy indicators and composite index along trajectories
# ---------------------------------------------------------------------------
def fuzzy_index_along(lambdas, x0=X0_INDEX):
    """Evaluate I1, I2, I3 and P along a trajectory, eqs. (30)-(32), (38)."""
    L, U = BAND
    t, h, d, b, p = solve(lambdas, x0)

    I1 = fz.psi_up(h, L, U)                       # intensity reading, eq. (33)
    I2 = fz.psi_up(d, L, U)                       # eq. (30)
    I3 = fz.t_norm(fz.psi_up(b, L, U), fz.psi_up(p, L, U))   # eqs. (31)-(32)

    w1, w2, w3 = WEIGHTS
    P = fz.prob_sum_all(w1 * I1, w2 * I2, w3 * I3)
    return t, I1, I2, I3, P


def figure_fuzzy_index():
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6), sharey=True)
    panels = [("stable", r"(a) stable regime ($\lambda_b=1.2$)"),
              ("trap", r"(b) poverty-trap regime ($\lambda_b=0$)")]

    for ax, (regime, title) in zip(axes, panels):
        t, I1, I2, I3, P = fuzzy_index_along(REGIMES[regime])
        ax.plot(t, I1, color="red", label=r"$\widetilde I_1$")
        ax.plot(t, I2, color="blue", label=r"$\widetilde I_2$")
        ax.plot(t, I3, color="darkorange", label=r"$\widetilde I_3$")
        ax.plot(t, P, color="black", lw=2.4, label=r"$\widetilde P$")
        ax.set_title(title)
        ax.set_xlabel("time $t$")
        ax.set_xlim(*T_SPAN)
        ax.set_ylim(-0.03, 1.05)

    axes[0].set_ylabel("membership degree")
    axes[0].legend(ncol=2, framealpha=0.85)
    fig.tight_layout()
    save(fig, "fig6_fuzzy_index")


# ---------------------------------------------------------------------------
# Numbers quoted in the text
# ---------------------------------------------------------------------------
def report():
    """Print the eigenvalues and index levels cited in Sections 5.6 and 6.2."""
    print("\nEigenvalues of A:")
    for regime, lambdas in REGIMES.items():
        eig = np.linalg.eigvals(system_matrix(lambdas))
        dominant = eig[np.argmax(eig.real)]
        print(f"  {regime:6s}: " + ", ".join(f"{v.real:+.4f}" for v in sorted(eig.real))
              + f"   dominant = {dominant.real:+.4f}")

    print("\nComposite index along the trajectory from x0 =", X0_INDEX)
    for regime in REGIMES:
        t, _, _, _, P = fuzzy_index_along(REGIMES[regime])
        print(f"  {regime:6s}: P(0) = {P[0]:.4f}, "
              f"min P = {P.min():.4f} at t = {t[P.argmin()]:.2f}, "
              f"P(20) = {P[-1]:.4f}")


def main():
    print("Generating figures into", OUTDIR)
    figure_trajectories("trap", "fig1_trap_trajectories", ymax=5.5)
    figure_trajectories("stable", "fig2_stable_trajectories", ymax=1.05)
    figure_heating_cooling()
    figure_debt()
    figure_expenses_income()
    figure_fuzzy_index()
    report()


if __name__ == "__main__":
    main()
