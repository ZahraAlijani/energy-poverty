# Energy Poverty — Dynamical Modeling

Code accompanying the paper *Dynamical Modeling of Energy Poverty*
(`PovertyDynamics.tex`). Every figure and every numerical value quoted in the
paper is produced from the model itself by the scripts in this repository.

## Model in one paragraph

Four observable dimensions of energy poverty — inability to heat or cool the
dwelling `h(t)`, arrears on utility bills `d(t)`, energy burden `b(t)`, and
financial poverty `p(t)` — evolve on `[0,1]^4` under the linear system
`x'(t) = A x(t)`, where the diagonal terms `-λ_i` represent recovery and the
off-diagonal terms the mutual reinforcement between dimensions. The EU-SILC
indicators are then obtained from the state by smooth (C¹, cosine) threshold
transitions instead of crisp cut-offs, and aggregated into a fuzzy energy
poverty index with the probabilistic sum.

## Files

| File | Contents |
|---|---|
| `FuzzyEnergyPoverty.py` | Fuzzy layer: product t-norm, probabilistic sum, cosine transitions and their derivative, the indicators `I1`, `I2`, `I3`, and the composite index `P`. |
| `silc_fuzzy.py` | Household-level memberships from the raw Czech EU-SILC variables in `dom_all_echudoba.RDS`: yearly at-risk-of-poverty line, cosine transitions on the energy expenditure share and on relative residual income, all threshold parameters in one dataclass, plus an audit of gradedness and of consistency with the official binary flags. |
| `empirical_lambda.py` | Empirical calibration of Section 7: weighted annual prevalence series, estimation of `λ_i` under the model's sign restrictions and by plain OLS, the `λ_b` sweep, the robustness check over transition widths, and figure 7. |
| `paper_figures.py` | Integrates the system and regenerates every figure of the paper into `img/`; also prints the eigenvalues and index levels cited in the text. |
| `img/` | The figures as included by the paper (PDF and PNG). |

## Reproducing the figures

```bash
pip install numpy scipy matplotlib
python paper_figures.py
```

This writes `img/figN_*.png` and `img/figN_*.pdf`, the exact filenames the paper
includes, and prints the eigenvalues of the system matrix in both regimes.

| Figure | Section | Function |
|---|---|---|
| 1 | 5.6.1 | `figure_trajectories("trap", …)` — poverty-trap regime, `λ_b = 0` |
| 2 | 5.6.2 | `figure_trajectories("stable", …)` — stable regime, `λ_b > 0` |
| 3 | 6.1.1 | `figure_heating_cooling()` — heating/cooling indicator |
| 4 | 6.1.2 | `figure_debt()` — debt indicator and its sensitivity |
| 5 | 6.1.3 | `figure_expenses_income()` — burden, poverty, and their t-norm |
| 6 | 6.2.1 | `figure_fuzzy_index()` — indicators and index along trajectories |
| — | 5.6, 6.2.1 | `report()` — eigenvalues and index levels |

All parameters that appear in a caption (coupling coefficients, damping rates,
initial conditions, thresholds, weights) are collected in the `CONFIG` block at
the top of `paper_figures.py`.

## Data

The indicators follow the Czech operationalization of energy poverty based on
EU-SILC (inability to heat adequately; arrears on utility bills; high energy
burden combined with income poverty), but the framework is transferable: the
thresholds in `FuzzyThresholds` can be replaced by locally relevant ones without
changing the dynamics.

The empirical part reads `dom_all_echudoba.RDS` (household-level Czech EU-SILC,
2005–2025; not redistributable, hence gitignored):

```bash
pip install numpy scipy pandas matplotlib pyreadr
python silc_fuzzy.py        # memberships + audit of the fuzzification
python empirical_lambda.py  # calibration, robustness check, img/fig7_*
```

The memberships are **not** taken from the `*_f` columns of that file. `I1_f`
and `I2_f` are binary copies of the survey items `DOST_VYTAP` and `DLUH_PLATB`,
`bydzat_f` grades the *subjective housing-cost* burden rather than the energy
burden, and `chudoba24_f` is a composite poverty score fuzzified by a piecewise
linear ramp. `silc_fuzzy.py` therefore rebuilds `b` from `pene` (energy
expenditure over net income, cut-off 20 %) and `p` from `rezsj24` relative to
the yearly poverty line, in both cases with the C¹ cosine transition centred on
the official cut-off, so that the ½-cut reproduces `nad20` and `ch_prij24`
exactly.
