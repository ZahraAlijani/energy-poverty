"""
silc_fuzzy.py -- household-level fuzzy memberships for the four deprivation
dimensions, built from the *raw* Czech EU-SILC variables with the C^1-smooth
cosine transitions of Section 6 (eq. 27-28), not from the pre-computed columns
of the data file.

Why this module exists
----------------------
The file dom_all_echudoba.RDS ships a set of columns whose names end in "_f"
and which look like fuzzy memberships, but they are not the memberships of the
paper:

    I1_f          == 1{DOST_VYTAP == 2}                  -> values {0, 1}
    I2_f          == 1{DLUH_PLATB == 1}                  -> values {0, 1}
    bydzat_f      graded subjective *housing*-cost burden -> values {0, .5, 1}
    chudoba24_f   composite poverty score, and its income
                  part ch_prij24_f is a *linear* ramp

so (i) the first two carry no gradedness at all, (ii) bydzat is the subjective
burden of total housing costs, not the energy burden b of the model, (iii) the
poverty column is the composite deprivation measure rather than the income
poverty p of the model, and (iv) where the file is graded, the transition is
piecewise linear, hence only C^0 -- it has the kinks at the thresholds that the
cosine transition was introduced to remove.

This module therefore rebuilds the memberships from the underlying continuous
variables:

    b  energy burden       Psi_{0->1}(pene; L_b, U_b)
                           pene = monthly energy expenditure / monthly net
                           income; the crisp counterpart is nad20 = 1{pene>=.20}
    p  income poverty      Psi_{1->0}(rezsj24 / line_y; L_p, U_p)
                           rezsj24 = residual income per consumption unit,
                           line_y = 0.6 x weighted median of rezsj24 in year y;
                           the crisp counterpart is ch_prij24 = 1{rezsj24<line_y}
    d  arrears            "utility": 1{DLUH_PLATB == 1}, the paper's I2, crisp;
                          "count":   Psi_{0->1}(number of arrears domains among
                                     utility bills, rent, mortgage, loans;
                                     L_d, U_d)
    h  thermal deprivation 1{DOST_VYTAP == 2} -- "cannot afford to keep the
                           dwelling adequately warm"

For h the EU-SILC questionnaire records a yes/no answer only: there is no
continuous quantity to compare with a threshold, so the cosine transition
degenerates to the crisp indicator (the Delta -> 0 limit discussed after
eq. 33). The same holds for d in its "utility" variant; the "count" variant
recovers gradedness from the number of payment domains in arrears, whose crisp
counterpart is "in arrears in at least one domain".

All transitions are centred on the crisp cut-off, tau = (L + U) / 2, so that
thresholding a membership at 1/2 reproduces exactly the official binary
classification; the width Delta = U - L is the only free parameter and is
reported in THRESHOLDS below.

Requires numpy, pandas, pyreadr.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

from FuzzyEnergyPoverty import psi_up, psi_down, t_norm

# ---------------------------------------------------------------------------
# Raw EU-SILC columns used here
# ---------------------------------------------------------------------------
RDS_PATH = Path(__file__).with_name("dom_all_echudoba.RDS")

COL_YEAR = "rok"            # survey year, stored as a factor with labels 2005..2025
COL_WEIGHT = "PKOEF"        # household cross-sectional weight
COL_WARM = "DOST_VYTAP"     # 1 = can keep the dwelling adequately warm, 2 = cannot
COL_ARREARS = ["DLUH_PLATB",   # utility bills   (1 = in arrears)
               "DLUH_NAJEM",   # rent
               "DLUH_HYPOT",   # mortgage
               "DLUH_PUJC"]    # loans / instalments
COL_BURDEN = "pene"         # energy expenditure / net monthly income, a ratio
COL_RESID = "rezsj24"       # residual income per consumption unit, CZK / month

# Crisp indicators of the data provider, kept for validation only.
COL_CRISP = {"h": "I1_f", "d": "I2_f", "b": "nad20", "p": "ch_prij24"}

STATES = ["h", "d", "b", "p"]


# ---------------------------------------------------------------------------
# Fuzzification parameters
# ---------------------------------------------------------------------------
@dataclass
class SilcThresholds:
    """
    Bracketing thresholds L < U of the cosine transitions, in the units of the
    EU-SILC variable each transition is applied to. Every pair is centred on the
    crisp cut-off tau it replaces, tau = (L + U) / 2.

    Energy burden b: pene is a share of net monthly income, tau_b = 0.20 is the
    Czech "high energy burden" cut-off (nad20). Delta_b = 0.10 admits the
    gradedness of the 15-25 % band, where a household is neither clearly
    burdened nor clearly safe.

    Income poverty p: applied to the residual income *relative to the poverty
    line*, r = rezsj24 / line_y, so tau_p = 1 for every year and the transition
    does not have to be re-scaled as incomes grow. Delta_p = 0.40 corresponds to
    the 80-120 % band around the line.

    Arrears d ("count" variant): applied to the number of payment domains in
    arrears, tau_d = 1 domain, so one domain gives exactly 1/2 and two or more
    give 1.

    Thermal deprivation h: the source item is binary, no transition applies;
    Delta_h = 0 is recorded here only to make the degeneracy explicit.
    """
    Lb: float = 0.15
    Ub: float = 0.25
    Lp: float = 0.80
    Up: float = 1.20
    Ld: float = 0.0
    Ud: float = 2.0
    at_risk_rate: float = 0.60   # poverty line = 0.6 x weighted median residual income

    @property
    def table(self):
        """Parameters as a DataFrame, for the parameter table of the paper."""
        rows = [
            ("b", "pene (share of net income)", self.Lb, self.Ub),
            ("p", "rezsj24 / poverty line", self.Lp, self.Up),
            ("d", "number of arrears domains", self.Ld, self.Ud),
            ("h", "binary survey item", np.nan, np.nan),
        ]
        df = pd.DataFrame(rows, columns=["state", "variable", "L", "U"])
        df["tau"] = (df.L + df.U) / 2
        df["Delta"] = df.U - df.L
        return df


THRESHOLDS = SilcThresholds()


# ---------------------------------------------------------------------------
# Loading and the yearly poverty line
# ---------------------------------------------------------------------------
def load_households(path=RDS_PATH):
    """Raw household records with the columns used below, year as an integer."""
    df = pyreadr.read_r(str(path))[None]
    cols = ([COL_YEAR, COL_WEIGHT, COL_WARM, COL_BURDEN, COL_RESID]
            + COL_ARREARS + list(COL_CRISP.values()))
    out = df[cols].copy()
    out[COL_YEAR] = out[COL_YEAR].astype(int)     # factor labels 2005..2025
    for c in cols[1:]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def weighted_median(x, w):
    """Lower weighted median: first value whose cumulative weight reaches 50 %."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w)
    x, w = x[m], w[m]
    if x.size == 0:
        return np.nan
    o = np.argsort(x)
    x, w = x[o], w[o]
    i = int(np.searchsorted(np.cumsum(w), 0.5 * w.sum()))
    return float(x[min(i, x.size - 1)])


def poverty_lines(df, thr=THRESHOLDS):
    """
    At-risk-of-poverty line per year: 0.6 x weighted median residual income per
    consumption unit. Reproduces the crisp flag ch_prij24 of the data file
    exactly, which is what audit() checks.
    """
    return pd.Series({
        y: thr.at_risk_rate * weighted_median(g[COL_RESID], g[COL_WEIGHT])
        for y, g in df.groupby(COL_YEAR)})


# ---------------------------------------------------------------------------
# Household-level memberships
# ---------------------------------------------------------------------------
def add_memberships(df, thr=THRESHOLDS, d_source="utility"):
    """
    Append the fuzzy memberships h, d, b, p and their crisp counterparts
    h0, d0, b0, p0. d_source selects the arrears variant ("utility" = the
    paper's I2, crisp; "count" = graded over the four payment domains).
    """
    out = df.copy()

    # --- b: energy burden, cosine transition on the expenditure share --------
    out["b"] = psi_up(out[COL_BURDEN].values, thr.Lb, thr.Ub)
    out.loc[out[COL_BURDEN].isna(), "b"] = np.nan
    out["b0"] = (out[COL_BURDEN] >= (thr.Lb + thr.Ub) / 2).astype(float)
    out.loc[out[COL_BURDEN].isna(), "b0"] = np.nan

    # --- p: income poverty, cosine transition on residual income / line -----
    line = poverty_lines(out, thr)
    out["poverty_line"] = out[COL_YEAR].map(line)
    rel = out[COL_RESID] / out["poverty_line"]
    out["rel_income"] = rel
    out["p"] = psi_down(rel.values, thr.Lp, thr.Up)
    out.loc[rel.isna(), "p"] = np.nan
    out["p0"] = (rel < (thr.Lp + thr.Up) / 2).astype(float)
    out.loc[rel.isna(), "p0"] = np.nan

    # --- h: thermal deprivation, binary survey item (degenerate transition) --
    out["h"] = (out[COL_WARM] == 2).astype(float)
    out["h0"] = out["h"]

    # --- d: arrears ----------------------------------------------------------
    n_domains = sum((out[c] == 1).astype(float) for c in COL_ARREARS)
    out["arrears_domains"] = n_domains
    if d_source == "utility":
        out["d"] = (out[COL_ARREARS[0]] == 1).astype(float)
        out["d0"] = out["d"]
    elif d_source == "count":
        out["d"] = psi_up(n_domains.values, thr.Ld, thr.Ud)
        out["d0"] = (n_domains >= (thr.Ld + thr.Ud) / 2).astype(float)
    else:
        raise ValueError(f"d_source must be 'utility' or 'count', got {d_source!r}")

    # --- the conjunctive indicator I3 = (b high) (x) (income low), eq. 31 ----
    out["I3"] = t_norm(out["b"].values, out["p"].values)
    return out


def annual_series(df, cols=STATES, weight=COL_WEIGHT, year=COL_YEAR):
    """Weighted annual prevalence of each column, indexed by year."""
    w = df[weight]

    def wmean(col):
        x = df[col]
        m = x.notna() & w.notna()
        return (x[m] * w[m]).groupby(df[year][m]).sum() / w[m].groupby(df[year][m]).sum()

    return pd.DataFrame({c: wmean(c) for c in cols}).dropna()


def build_series(path=RDS_PATH, thr=THRESHOLDS, d_source="utility", households=None):
    """Convenience wrapper: raw file -> (annual series, household frame)."""
    hh = add_memberships(load_households(path) if households is None else households,
                         thr, d_source)
    return annual_series(hh), hh


# ---------------------------------------------------------------------------
# Diagnostics: does the fuzzification actually do anything?
# ---------------------------------------------------------------------------
def audit(hh, thr=THRESHOLDS):
    """
    Report, per dimension: the share of households in the open interval (0,1)
    -- i.e. those the crisp definition classifies with a hard yes/no and the
    fuzzy one does not -- and the agreement between the 1/2-cut of the
    membership and the provider's own crisp indicator.
    """
    print("Fuzzification parameters")
    print(thr.table.to_string(index=False, na_rep="--", float_format="%.3f"))

    print("\nGradedness and consistency with the crisp EU-SILC definitions")
    print(f"{'state':>6} {'mean fuzzy':>11} {'mean crisp':>11} "
          f"{'in (0,1)':>9} {'1/2-cut == official':>20}")
    for s in STATES:
        f = hh[s]
        c = hh[COL_CRISP[s]]
        m = f.notna() & c.notna()
        strict = ((f > 0) & (f < 1)).sum() / max(f.notna().sum(), 1)
        agree = ((f[m] >= 0.5).astype(float) == c[m]).mean()
        print(f"{s:>6} {f.mean():>11.4f} {hh[s + '0'].mean():>11.4f} "
              f"{strict:>8.1%} {agree:>19.1%}")

    print("\nRecovered at-risk-of-poverty line (CZK per consumption unit and month)")
    line = poverty_lines(hh, thr).dropna()
    print("  " + "  ".join(f"{y}:{v:,.0f}" for y, v in line.items()))


def main():
    hh = load_households()
    hh = add_memberships(hh, THRESHOLDS)
    audit(hh)
    S = annual_series(hh)
    print("\nWeighted annual prevalence")
    print(S.to_string(float_format="%.4f"))


if __name__ == "__main__":
    main()
