"""
yearly_models.py -- the yearly persistence models of the energy-poverty
measures: estimation, fit, the 2025 hold-out prediction and the rolling
one-year-ahead evaluation, for the model with intercept and for the model
without intercept.

Data: dom_all_echudoba.csv (household-level Czech EU-SILC, 2005-2025). The two
measures are the weighted yearly means, with the household weight PKOEF, of

    prevalence EPI   ech_dynamic   (binary, 0/1)
    average FEPI     fech_dynamic  (graded, [0,1])

Models. The scalar model dP/dt = -lambda P observed once a year gives
P_t = e^{-lambda} P_{t-1}. Two regressions of P_t on P_{t-1} are estimated by
OLS on the training years 2006-2024:

    with intercept     P_t = a + c P_{t-1} + eps_t     (paper eq. yearly-pure)
    without intercept  P_t =     c P_{t-1} + eps_t     (paper eq. yearly-noint)

with lambda = -ln c and half-life ln 2 / lambda. R^2 is computed around the
mean in both cases, so the two models are comparable. The rolling evaluation
re-estimates each model on 2006..(2017+i) and predicts 2017+i+1, i=0..7.
Errors are actual minus predicted.

    python yearly_models.py      # prints the values of all tables

Requires numpy, pandas, scipy.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA = Path(__file__).with_name("dom_all_echudoba.csv")
MEASURES = {"ech_dynamic": "prevalence EPI", "fech_dynamic": "average FEPI"}
TRAIN = (2006, 2024)
TEST_YEAR = 2025
ROLLING_LAST_TRAIN = range(2017, 2025)   # predicts 2018..2025
LEVEL = 0.95


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
def yearly_series(path=DATA):
    """Weighted yearly means of the two measures, indexed by year."""
    df = pd.read_csv(path, usecols=["rok", "PKOEF", *MEASURES], low_memory=False)
    out = {}
    for col in MEASURES:
        d = df[["rok", "PKOEF", col]].copy()
        # binary columns come as TRUE/FALSE, graded ones as numbers
        d[col] = pd.to_numeric(d[col].astype(str).str.upper().replace({"TRUE": "1", "FALSE": "0"}),
                               errors="coerce")
        d = d.dropna()
        d["wx"] = d[col] * d["PKOEF"]
        g = d.groupby("rok")
        out[col] = g["wx"].sum() / g["PKOEF"].sum()
    series = pd.DataFrame(out)
    series.index = series.index.astype(int)
    return series


# ---------------------------------------------------------------------------
# ESTIMATION
# ---------------------------------------------------------------------------
def fit(y, x, intercept):
    """OLS of y on x, with or without intercept; the slope is the last coefficient."""
    X = np.column_stack([np.ones_like(x), x]) if intercept else x[:, None]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    n, k = X.shape
    dof = n - k
    s2 = res @ res / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(xtx_inv) * s2)
    return {
        "beta": beta, "se": se, "dof": dof, "n": n, "xtx_inv": xtx_inv,
        "p": 2 * stats.t.sf(np.abs(beta / se), dof),
        "rse": np.sqrt(s2),
        "r2": 1 - (res @ res) / ((y - y.mean()) @ (y - y.mean())),
    }


def predict(model, x0, intercept, level=LEVEL):
    """Point prediction and prediction interval at P_{t-1} = x0."""
    v = np.array([1.0, x0]) if intercept else np.array([x0])
    yhat = v @ model["beta"]
    half = (stats.t.ppf((1 + level) / 2, model["dof"])
            * model["rse"] * np.sqrt(1 + v @ model["xtx_inv"] @ v))
    return yhat, yhat - half, yhat + half


def fit_years(P, first, last, intercept):
    years = np.arange(first, last + 1)
    return fit(P.loc[years].values, P.loc[years - 1].values, intercept)


def rolling(P, intercept):
    """One-year-ahead predictions, re-estimating on a growing window."""
    rows = []
    for last in ROLLING_LAST_TRAIN:
        model = fit_years(P, TRAIN[0], last, intercept)
        yhat, _, _ = predict(model, P.loc[last], intercept)
        rows.append((last + 1, P.loc[last + 1], yhat, P.loc[last + 1] - yhat))
    return pd.DataFrame(rows, columns=["year", "actual", "predicted", "difference"])


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------
def report(series):
    tcrit = None
    for intercept, name in [(True, "WITH intercept (eq. yearly-pure)"),
                            (False, "WITHOUT intercept (eq. yearly-noint)")]:
        print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
        for col, label in MEASURES.items():
            P = series[col]
            m = fit_years(P, *TRAIN, intercept)
            c, se_c = m["beta"][-1], m["se"][-1]
            tcrit = stats.t.ppf((1 + LEVEL) / 2, m["dof"])
            ci = (c - tcrit * se_c, c + tcrit * se_c)
            lam = -np.log(c)
            p_c1 = 2 * stats.t.sf(abs((c - 1) / se_c), m["dof"])
            print(f"\n--- {label}  (n = {m['n']}, training {TRAIN[0]}-{TRAIN[1]})")
            if intercept:
                a = m["beta"][0]
                print(f"intercept a = {a:.4f} (p = {m['p'][0]:.3f}),  "
                      f"long-run level a/(1-c) = {a / (1 - c):.4f}")
            print(f"c = {c:.4f} (SE {se_c:.4f}, p = {m['p'][-1]:.2g}),  "
                  f"{LEVEL:.0%} CI [{ci[0]:.4f}, {ci[1]:.4f}],  p(c=1) = {p_c1:.3f}")
            print(f"lambda = {lam:.4f},  half-life = {np.log(2) / lam:.2f} years")
            print(f"R^2 = {m['r2']:.3f},  RSE = {m['rse']:.4f}")
            yhat, lo, hi = predict(m, P.loc[TEST_YEAR - 1], intercept)
            print(f"{TEST_YEAR}: observed {P.loc[TEST_YEAR]:.4f}, predicted {yhat:.4f}, "
                  f"error {P.loc[TEST_YEAR] - yhat:.4f}, {LEVEL:.0%} PI [{lo:.4f}, {hi:.4f}]")
            r = rolling(P, intercept)
            print(r.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
            e = r["difference"].values
            print(f"mean {e.mean():.4f}  MAE {np.abs(e).mean():.4f}  "
                  f"RMSE {np.sqrt((e ** 2).mean()):.4f}  max |e| {np.abs(e).max():.4f}")


def rolling_slopes(P, intercept):
    """Estimated c on each rolling training window."""
    return np.array([fit_years(P, TRAIN[0], last, intercept)["beta"][-1]
                     for last in ROLLING_LAST_TRAIN])


def compare(series):
    """Numbers quoted in the comparison of the two models."""
    print(f"\n{'=' * 72}\nCOMPARISON: with intercept vs without intercept\n{'=' * 72}")
    for col, label in MEASURES.items():
        P = series[col]
        e_with = np.abs(rolling(P, True)["difference"].values)
        e_without = np.abs(rolling(P, False)["difference"].values)
        e_with_r, e_without_r = e_with.round(4), e_without.round(4)   # as in the tables
        mae = e_with.mean(), e_without.mean()
        rmse = np.sqrt((e_with ** 2).mean()), np.sqrt((e_without ** 2).mean())
        c_with, c_without = rolling_slopes(P, True), rolling_slopes(P, False)
        print(f"\n--- {label}")
        print(f"years closer without intercept: {(e_without_r < e_with_r).sum()} of {len(e_with)}, "
              f"ties {(e_without_r == e_with_r).sum()}, "
              f"closer with intercept {(e_without_r > e_with_r).sum()}")
        print(f"MAE  reduction: {1 - mae[1] / mae[0]:.1%}")
        print(f"RMSE reduction: {1 - rmse[1] / rmse[0]:.1%}")
        print(f"rolling c, with intercept:    {c_with.min():.3f} .. {c_with.max():.3f}  "
              f"({(c_with > 1).sum()} windows with c > 1)")
        print(f"rolling c, without intercept: {c_without.min():.3f} .. {c_without.max():.3f}")


def main():
    series = yearly_series()
    print("Weighted yearly series:")
    print(series.rename(columns=MEASURES).round(4).to_string())
    report(series)
    compare(series)


if __name__ == "__main__":
    main()
