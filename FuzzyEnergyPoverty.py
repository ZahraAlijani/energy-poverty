"""
Fuzzy representation of energy-poverty deprivation dimensions.

Implements Section 6 of "Dynamics of Energy Poverty":
    - Fuzzy operators: product t-norm and probabilistic sum         (eqs. below Fig. 2)
    - C^1-smooth cosine threshold transitions Psi_{0->1}, Psi_{1->0} (eq. 27-28)
    - Fuzzy deprivation indicators I1 (heating/cooling), I2 (debt),
      I3 (energy-expenses & income)                                 (eq. 29-32)
    - Composite fuzzy energy-poverty index P                        (Sec. 6.2)

The continuous states come from the simplified dynamical system
    x(t) = [h(t), d(t), b(t), p(t)]  in  [0, 1]^4
(the ODE solved in TrajectoriesSimulation.py). This module maps those
states to fuzzy membership degrees in [0, 1] and aggregates them into a
single fuzzy poverty score P(t).

All functions are vectorized: pass scalars or NumPy arrays (e.g. a whole
trajectory sol.y[k]) interchangeably.
"""

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Fuzzy logic operators
# ---------------------------------------------------------------------------
def t_norm(x, y):
    """Product t-norm (fuzzy AND):  x (x) y = x * y."""
    return np.asarray(x, dtype=float) * np.asarray(y, dtype=float)


def prob_sum(x, y):
    """Probabilistic sum (fuzzy OR):  x (+) y = x + y - x*y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return x + y - x * y


def prob_sum_all(*args):
    """Fold the probabilistic sum over several membership values."""
    acc = args[0]
    for a in args[1:]:
        acc = prob_sum(acc, a)
    return acc


# ---------------------------------------------------------------------------
# Smooth cosine threshold transitions (eq. 27-28)
# ---------------------------------------------------------------------------
def psi_up(x, L, U):
    """
    Rising C^1-smooth transition  Psi_{0->1}(x; L, U):
        0                       for x <= L
        (1 - cos(pi (x-L)/(U-L))) / 2   for L < x < U
        1                       for x >= U
    Zero slope at both endpoints. Relational notation: (x >~ [L,U]).
    """
    if U <= L:
        raise ValueError(f"Require L < U, got L={L}, U={U}.")
    x = np.asarray(x, dtype=float)
    s = np.clip((x - L) / (U - L), 0.0, 1.0)
    return (1.0 - np.cos(np.pi * s)) / 2.0


def psi_up_derivative(x, L, U):
    """
    Derivative of the rising transition (eq. 34 in the paper):
        pi / (2 (U-L)) * sin(pi (x-L)/(U-L))   for L < x < U,   0 otherwise.
    Vanishes at both endpoints; peaks at the midpoint with value pi/(2 (U-L)).
    """
    if U <= L:
        raise ValueError(f"Require L < U, got L={L}, U={U}.")
    x = np.asarray(x, dtype=float)
    inside = (x > L) & (x < U)
    out = np.zeros_like(x, dtype=float)
    out[inside] = (np.pi / (2.0 * (U - L))
                   * np.sin(np.pi * (x[inside] - L) / (U - L)))
    return out


def psi_down(x, L, U):
    """
    Falling C^1-smooth transition  Psi_{1->0}(x; L, U) = 1 - Psi_{0->1}(x; L, U):
        1  for x <= L,  0  for x >= U,  smooth in between.
    Relational notation: (x <~ [L,U]).
    """
    return 1.0 - psi_up(x, L, U)


# Relational aliases matching the paper's notation.
def geq(x, L, U):
    """(x >~ [L,U]) : degree to which x is 'sufficiently high'."""
    return psi_up(x, L, U)


def leq(x, L, U):
    """(x <~ [L,U]) : degree to which x is 'sufficiently low'."""
    return psi_down(x, L, U)


# ---------------------------------------------------------------------------
# Threshold parameters for the fuzzy indicators
# ---------------------------------------------------------------------------
@dataclass
class FuzzyThresholds:
    """
    Lower/upper thresholds (L < U) for each smooth transition.

    Defaults are chosen for normalized states in [0, 1]. Adjust to match the
    physical interpretation (e.g. indoor-temperature thresholds hH, hC for the
    heating/cooling indicator, as in Section 6.1.1).
    """
    # I1 - heating & cooling indicator, temperature reading (eq. 29)
    LH: float = 0.10   # heating: full demand for h <= LH
    UH: float = 0.45   # heating: no demand for  h >= UH
    LC: float = 0.55   # cooling: no demand for  h <= LC
    UC: float = 0.90   # cooling: full demand for h >= UC

    # I1 - one-sided intensity reading (eq. 33), used along ODE trajectories
    Lh: float = 0.30   # below Lh: thermally adequate
    Uh: float = 0.70   # above Uh: fully inadequate

    # I2 - debt / arrears indicator (state d)
    Ld: float = 0.30   # below Ld: no debt deprivation
    Ud: float = 0.70   # above Ud: full debt deprivation

    # I3 - energy-expenses & income indicator (states b, p)
    bL: float = 0.30   # burden: below bL -> not burdened
    bU: float = 0.70   # burden: above bU -> fully burdened
    pL: float = 0.30   # poverty intensity: below pL -> not poor
    pU: float = 0.70   # poverty intensity: above pU -> fully poor

    # Income form of the poverty membership (eq. 31), in units of the income
    # variable y (e.g. residual income relative to its median). Used instead of
    # (pL, pU) when I3 is evaluated on EU-SILC income rather than on the state p.
    yL: float = 0.50   # below yL -> clearly poor
    yU: float = 0.70   # above yU -> clearly not poor


# ---------------------------------------------------------------------------
# Fuzzy deprivation indicators (eq. 29-33)
# ---------------------------------------------------------------------------
def indicator_I1(h, thr=FuzzyThresholds()):
    """
    Heating & cooling indicator (eq. 29):
        I1 = mu_H(h) (+) mu_C(h),
    where mu_H = (h <~ [LH, UH]) is the heating-demand degree and
          mu_C = (h >~ [LC, UC]) is the cooling-demand degree.
    High I1 => thermal discomfort (heating or cooling needed).
    """
    mu_H = leq(h, thr.LH, thr.UH)
    mu_C = geq(h, thr.LC, thr.UC)
    return prob_sum(mu_H, mu_C)


def indicator_I1_intensity(h, thr=FuzzyThresholds()):
    """
    One-sided form of the thermal indicator (eq. 33):  I1 = (h >~ [Lh, Uh]).
    Used when h(t) is the *intensity* of thermal inadequacy (the state of the
    dynamical system), which already aggregates insufficient heating and
    insufficient cooling into one deprivation-increasing variable. Use
    indicator_I1() instead when h(t) is an indoor temperature.
    """
    return geq(h, thr.Lh, thr.Uh)


def indicator_I2(d, thr=FuzzyThresholds()):
    """Debt / arrears indicator (eq. 30):  I2 = (d >~ [Ld, Ud])."""
    return geq(d, thr.Ld, thr.Ud)


def poverty_membership(p, thr=FuzzyThresholds()):
    """
    Poverty membership in state-space form (eq. 32):  mu_p = (p >~ [pL, pU]).
    Here p is the poverty *intensity* of the dynamical system: larger p means
    more deprivation, so the transition is increasing.
    """
    return geq(p, thr.pL, thr.pU)


def poverty_membership_from_income(y, thr=FuzzyThresholds()):
    """
    Poverty membership in income form (eq. 31):  mu_p = (y <~ [yL, yU]).
    Here y is an income variable: larger y means less deprivation, so the
    transition is decreasing. Equivalent to poverty_membership() up to the
    reparametrization of the thresholds.
    """
    return leq(y, thr.yL, thr.yU)


def indicator_I3(b, p, thr=FuzzyThresholds(), income=False):
    """
    Energy-expenses & income indicator (eq. 31-32):
        I3 = (b >~ [bL, bU]) (x) mu_p.
    High burden AND poverty => high deprivation; neither alone suffices
    (conjunctive, no compensation).

    With income=False (default), the second argument is the poverty intensity
    p(t) of the dynamical system and mu_p = (p >~ [pL, pU]).
    With income=True, it is an income variable y(t) and mu_p = (y <~ [yL, yU]).
    """
    burden_high = geq(b, thr.bL, thr.bU)
    mu_p = (poverty_membership_from_income(p, thr) if income
            else poverty_membership(p, thr))
    return t_norm(burden_high, mu_p)


# ---------------------------------------------------------------------------
# Composite fuzzy energy-poverty index (eq. 38)
# ---------------------------------------------------------------------------
def fuzzy_poverty_index(h, d, b, p, weights=(1 / 3, 1 / 3, 1 / 3),
                        thr=FuzzyThresholds()):
    """
    Composite fuzzy energy-poverty index (eq. 38):
        P = w1 I1 (+) w2 I2 (+) w3 I3,   with  sum(w_i) = 1, w_i >= 0.

    Returns (P, I1, I2, I3) so the components can be inspected/plotted.
    """
    w1, w2, w3 = weights
    if not np.isclose(w1 + w2 + w3, 1.0):
        raise ValueError(f"Weights must sum to 1, got {w1 + w2 + w3}.")

    I1 = indicator_I1(h, thr)
    I2 = indicator_I2(d, thr)
    I3 = indicator_I3(b, p, thr)

    P = prob_sum_all(w1 * I1, w2 * I2, w3 * I3)
    return P, I1, I2, I3


# ---------------------------------------------------------------------------
# Figure 3 (Section 6.1.1): heating/cooling indicator vs. indoor temperature
# ---------------------------------------------------------------------------
def make_heating_cooling_figure(LH=18.0, UH=21.0, LC=25.0, UC=28.0,
                                temp_range=(10.0, 35.0),
                                savepath="fuzzy_heating_cooling_indicator.png"):
    """
    Reproduce Figure 3: heating adequacy, the resulting fuzzy index, and
    cooling adequacy as functions of indoor temperature (degrees Celsius).

        (a) mu_H(h) = (h <~ [LH, UH])   -- falling: full heating demand when cold
        (b) I1(h)   = mu_H (+) mu_C      -- probabilistic sum (eq. 29)
        (c) mu_C(h) = (h >~ [LC, UC])   -- rising: full cooling demand when warm

    Cooling sits at higher temperatures than heating, leaving an intermediate
    comfort region where both memberships are ~0.
    """
    import matplotlib.pyplot as plt

    h = np.linspace(temp_range[0], temp_range[1], 600)
    mu_H = leq(h, LH, UH)
    mu_C = geq(h, LC, UC)
    I1 = prob_sum(mu_H, mu_C)

    fig, (ax_h, ax_i, ax_c) = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    # (a) Heating adequacy
    ax_h.plot(h, mu_H, color="tab:red", lw=2)
    ax_h.axvspan(temp_range[0], LH, color="tab:red", alpha=0.06)
    for x in (LH, UH):
        ax_h.axvline(x, color="gray", ls=":", lw=1)
    ax_h.annotate(r"$L_H$", (LH, 0.02), ha="center", fontsize=9)
    ax_h.annotate(r"$U_H$", (UH, 0.02), ha="center", fontsize=9)
    ax_h.set_title(r"(a) Heating adequacy $\mu_H(h)$")

    # (b) Fuzzy heating-and-cooling index
    ax_i.plot(h, I1, color="black", lw=2.5)
    ax_i.axvspan(UH, LC, color="tab:green", alpha=0.08)
    ax_i.text((UH + LC) / 2, 0.5, "comfort\nregion", ha="center", va="center",
              fontsize=9, color="tab:green")
    ax_i.set_title(r"(b) Fuzzy index $\tilde I_1(h)=\mu_H\oplus\mu_C$")

    # (c) Cooling adequacy
    ax_c.plot(h, mu_C, color="tab:blue", lw=2)
    ax_c.axvspan(UC, temp_range[1], color="tab:blue", alpha=0.06)
    for x in (LC, UC):
        ax_c.axvline(x, color="gray", ls=":", lw=1)
    ax_c.annotate(r"$L_C$", (LC, 0.02), ha="center", fontsize=9)
    ax_c.annotate(r"$U_C$", (UC, 0.02), ha="center", fontsize=9)
    ax_c.set_title(r"(c) Cooling adequacy $\mu_C(h)$")

    for ax in (ax_h, ax_i, ax_c):
        ax.set_xlabel("indoor temperature $h$  [$^\\circ$C]")
        ax.set_ylim(-0.03, 1.05)
        ax.grid(True, alpha=0.3)
    ax_h.set_ylabel("membership degree")

    fig.suptitle("Figure 3: Heating adequacy, fuzzy index, and cooling "
                 "adequacy vs. indoor temperature", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(savepath, dpi=300)
    return fig


# ---------------------------------------------------------------------------
# Demonstration: couple the fuzzy layer to the ODE trajectories
# ---------------------------------------------------------------------------
def _demo():
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp

    # --- Simplified dynamical system (stable regime, lambda_b > 0) -----------
    alpha_hb, alpha_hp = 0.2, 0.3
    alpha_db, alpha_dp = 0.2, 0.2
    alpha_bd, alpha_bp = 0.3, 0.2
    alpha_ph, alpha_pd = 0.2, 0.2
    lambda_h, lambda_d, lambda_b, lambda_p = 1.0, 1.0, 1.2, 1.0

    A = np.array([
        [-lambda_h, 0.0,       alpha_hb,  alpha_hp],
        [0.0,       -lambda_d, alpha_db,  alpha_dp],
        [0.0,       alpha_bd,  -lambda_b, alpha_bp],
        [alpha_ph,  alpha_pd,  0.0,       -lambda_p],
    ], dtype=float)

    t_span = (0, 20)
    t_eval = np.linspace(*t_span, 500)
    x0 = np.array([0.8, 0.8, 0.8, 0.8])

    sol = solve_ivp(lambda t, x: A @ x, t_span, x0, t_eval=t_eval)
    h, d, b, p = sol.y

    # --- Fuzzy layer --------------------------------------------------------
    thr = FuzzyThresholds()
    P, I1, I2, I3 = fuzzy_poverty_index(h, d, b, p, weights=(0.4, 0.3, 0.3),
                                        thr=thr)

    # --- Figure 3: heating/cooling indicator vs. indoor temperature ---------
    make_heating_cooling_figure()

    # --- Panel 1: static membership functions -------------------------------
    grid = np.linspace(0, 1, 400)
    fig1, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(grid, leq(grid, thr.LH, thr.UH), label=r"$\mu_H$ (heating)")
    ax[0].plot(grid, geq(grid, thr.LC, thr.UC), label=r"$\mu_C$ (cooling)")
    ax[0].plot(grid, indicator_I1(grid, thr), "k--", label=r"$\tilde I_1$")
    ax[0].set_title(r"$\tilde I_1$: heating & cooling")
    ax[1].plot(grid, indicator_I2(grid, thr), label=r"$\tilde I_2$")
    ax[1].set_title(r"$\tilde I_2$: debt / arrears")
    ax[2].plot(grid, geq(grid, thr.bL, thr.bU), label=r"$\mu_b$ (burden high)")
    ax[2].plot(grid, poverty_membership(grid, thr), label=r"$\mu_p$ (poverty)")
    ax[2].set_title(r"$\tilde I_3$ components")
    for a in ax:
        a.set_xlabel("state value")
        a.set_ylabel("membership")
        a.grid(True)
        a.legend()
    fig1.suptitle("Fuzzy membership functions (C$^1$-smooth cosine transitions)")
    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    fig1.savefig("fuzzy_membership_functions.png", dpi=300)

    # --- Panel 2: fuzzy indicators & index along the trajectory -------------
    fig2, (axl, axr) = plt.subplots(1, 2, figsize=(13, 5))
    for series, lbl in zip((h, d, b, p), ("h", "d", "b", "p")):
        axl.plot(sol.t, series, label=f"${lbl}(t)$")
    axl.set_title("State trajectories (stable regime)")
    axl.set_xlabel("Time $t$")
    axl.set_ylabel("state value")
    axl.grid(True)
    axl.legend()

    axr.plot(sol.t, I1, label=r"$\tilde I_1$ heating/cooling")
    axr.plot(sol.t, I2, label=r"$\tilde I_2$ debt")
    axr.plot(sol.t, I3, label=r"$\tilde I_3$ expenses/income")
    axr.plot(sol.t, P, "k", lw=2.5, label=r"$\tilde P$ composite")
    axr.set_title("Fuzzy energy-poverty index along trajectory")
    axr.set_xlabel("Time $t$")
    axr.set_ylabel("membership degree")
    axr.grid(True)
    axr.legend()
    fig2.tight_layout()
    fig2.savefig("fuzzy_poverty_index_trajectory.png", dpi=300)

    plt.show()
    print(f"Composite fuzzy poverty index P: start={P[0]:.3f}, end={P[-1]:.3f}")


if __name__ == "__main__":
    _demo()
