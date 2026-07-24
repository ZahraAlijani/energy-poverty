"""
Trajectory simulation of a 4-state linear energy-poverty model.

State vector x = [h, d, b, p]:
    h - household health / well-being
    d - disposable income (or debt, sign-dependent)
    b - energy burden
    p - poverty level

Dynamics: x'(t) = A x(t), a linear ODE integrated with scipy.solve_ivp.
Three parameter regimes are simulated:
    1. Baseline sweep over several initial conditions.
    2. Poverty-trap (unstable) regime: lambda_b = 0, no direct damping of burden.
    3. Stable regime: lambda_b > 0, direct damping of burden.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Shared coupling coefficients (off-diagonal interaction strengths)
# ---------------------------------------------------------------------------
alpha_hb = 0.2
alpha_hp = 0.3
alpha_db = 0.2
alpha_dp = 0.2
alpha_bd = 0.3
alpha_bp = 0.2
alpha_pd = 0.2
alpha_ph = 0.2


def system_matrix(lambda_h, lambda_d, lambda_b, lambda_p):
    """Build the 4x4 system matrix A for the given self-decay rates."""
    return np.array([
        [-lambda_h, 0.0,       alpha_hb,  alpha_hp],
        [0.0,       -lambda_d, alpha_db,  alpha_dp],
        [0.0,       alpha_bd,  -lambda_b, alpha_bp],
        [alpha_ph,  alpha_pd,  0.0,       -lambda_p],
    ], dtype=float)


def make_rhs(A):
    """Return the right-hand side function x' = A x for a given matrix A."""
    def rhs(t, x):
        return A @ x
    return rhs


# ---------------------------------------------------------------------------
# Shared simulation settings
# ---------------------------------------------------------------------------
t_span = (0, 20)
t_eval = np.linspace(t_span[0], t_span[1], 500)

initial_conditions = [
    np.array([1.0, 0.2, 0.5, 0.4]),
    np.array([0.2, 1.0, 0.8, 0.3]),
    np.array([0.8, 0.8, 0.8, 0.8]),
    np.array([1.5, 0.5, 0.2, 1.0]),
]
labels = [r"$h(t)$", r"$d(t)$", r"$b(t)$", r"$p(t)$"]


# ---------------------------------------------------------------------------
# 1. Baseline: one figure per initial condition
# ---------------------------------------------------------------------------
def simulate_baseline():
    """Simulate the baseline regime, one plot per initial condition."""
    A = system_matrix(lambda_h=1.0, lambda_d=1.0, lambda_b=0.0, lambda_p=1.0)
    rhs = make_rhs(A)

    for i, x0 in enumerate(initial_conditions, start=1):
        sol = solve_ivp(rhs, t_span, x0, t_eval=t_eval)

        plt.figure(figsize=(8, 5))
        for j in range(4):
            plt.plot(sol.t, sol.y[j], label=labels[j])
        plt.title(f"Trajectory simulation, initial condition {i}: x0 = {x0}")
        plt.xlabel("t")
        plt.ylabel("state value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# 2. + 3. Regime comparison in a 2x2 grid of initial conditions
# ---------------------------------------------------------------------------
def simulate_regime(A, suptitle, savepath, print_label):
    """Simulate one regime across all initial conditions in a 2x2 grid."""
    rhs = make_rhs(A)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, x0 in zip(axes, initial_conditions):
        sol = solve_ivp(rhs, t_span, x0, t_eval=t_eval)
        for i in range(4):
            ax.plot(sol.t, sol.y[i], label=labels[i])

        x0_str = ",\\,".join([f"{v:.1f}" for v in x0])
        ax.set_title(rf"Initial condition $x_0=({x0_str})$")
        ax.grid(True)

    axes[0].legend()
    for ax in axes[2:]:
        ax.set_xlabel("Time $t$")
    for ax in axes[::2]:
        ax.set_ylabel("State value")

    plt.suptitle(suptitle)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(savepath, dpi=300)
    plt.show()

    print(f"Eigenvalues ({print_label}):")
    print(np.linalg.eigvals(A))


def simulate_poverty_trap():
    """Unstable poverty-trap regime: lambda_b = 0 (no damping of burden)."""
    A = system_matrix(lambda_h=1.0, lambda_d=1.0, lambda_b=0.0, lambda_p=1.0)
    simulate_regime(
        A,
        suptitle=r"Poverty-trap regime ($\lambda_b = 0$): persistent energy burden",
        savepath="unstable_regime_trajectories.png",
        print_label="poverty-trap regime",
    )


def simulate_stable():
    """Stable regime: lambda_b > 0 (direct damping of burden)."""
    A = system_matrix(lambda_h=1.0, lambda_d=1.0, lambda_b=1.2, lambda_p=1.0)
    simulate_regime(
        A,
        suptitle=r"Stable regime ($\lambda_b > 0$): convergence to zero",
        savepath="stable_regime_trajectories.png",
        print_label="stable regime",
    )


if __name__ == "__main__":
    simulate_baseline()
    simulate_poverty_trap()
    simulate_stable()
