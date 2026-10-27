"""
Reconciliation of risk data with accounting data (BCBS 239 principle 3: risk data reconciled with sources,
including accounting data). Each difference is broken down into explained items; the residual is tested against
a tolerance.

Loans (entity × currency, original currency):
    GL gross carrying amount − risk on-balance outstanding
      = accrued interest (in the GL, not in the risk exposure)
      + records rejected by blocking data-quality rules (in the GL, quarantined in risk)
      + residual (unexplained).
Derivatives (entity, EUR): GL net fair value − risk MtM.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import sources as S

TOL_REL, TOL_ABS = 0.001, 1.0e6       # residual tolerated: max(0.1 % of the GL balance, € 1 m equivalent)


def loans(src: S.Sources, model) -> pd.DataFrame:
    gl = src.gl[src.gl["account"] == "Loans and advances – gross carrying amount"].set_index(["entity", "currency"])["balance"]
    cr = model.credit[model.credit["on_balance"]]
    risk = cr.groupby(["entity", "currency"])["outstanding"].sum()
    acc = cr.groupby(["entity", "currency"])["accrued_interest"].sum()
    L = model.rejected_loans
    rej = L[L["currency"].notna() & (L["product"] != "Guarantee")]
    # rejected records the GL holds: positive amounts (negative outstanding was excluded at source) incl. accrued interest
    rej = rej[rej["outstanding"] > 0]
    rej_amt = (rej["outstanding"] + rej["accrued_interest"]).groupby([rej["entity"], rej["currency"]]).sum()
    t = pd.DataFrame({"GL": gl, "risk outstanding": risk, "accrued interest": acc, "rejected by DQ": rej_amt}).fillna(0.0)
    t["difference"] = t["GL"] - t["risk outstanding"]
    t["unexplained"] = t["difference"] - t["accrued interest"] - t["rejected by DQ"]
    fx = pd.Series([S.FX_TO_EUR[c] for c in t.index.get_level_values(1)], index=t.index)
    tol = np.maximum(TOL_REL * t["GL"].abs(), TOL_ABS / fx)
    t["tolerance"] = tol
    t["status"] = np.where(t["unexplained"].abs() <= tol, "reconciled", "break")
    t["unexplained (EUR)"] = t["unexplained"] * fx
    return t


def derivatives(src: S.Sources, model) -> pd.DataFrame:
    gl = src.gl[src.gl["account"].str.startswith("Trading")].copy()
    gl["GL (EUR)"] = gl["balance"] * gl["currency"].map(S.FX_TO_EUR)
    g = gl.set_index("entity")["GL (EUR)"]
    r = model.market.groupby("entity")["mtm_eur"].sum()
    t = pd.DataFrame({"GL (EUR)": g, "risk MtM (EUR)": r}).fillna(0.0)
    t["difference"] = t["GL (EUR)"] - t["risk MtM (EUR)"]
    t["status"] = np.where(t["difference"].abs() <= np.maximum(TOL_REL * t["GL (EUR)"].abs(), TOL_ABS), "reconciled", "break")
    return t
