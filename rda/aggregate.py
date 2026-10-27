"""
Risk aggregation on the integrated data model.

* Credit: EAD, RWA (standardised), expected loss and provisions along any combination of axes; large exposures
  by group of connected clients against Tier 1 (CRR art. 392 and 395: reporting at 10 %, limit at 25 %);
  concentration indices.
* Market: VaR and expected shortfall aggregated from P&L vectors (never by adding VaRs), by desk, entity and
  group, with the diversification benefit; sensitivities by risk factor; limit utilisation.
* Liquidity: simplified LCR and NSFR (Basel III run-off, inflow, ASF and RSF factors).
* Operational: net losses by Basel event type and business line, last 12 months, largest events.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import sources as S

TIER1_EUR = 6.0e9
DESK_VAR_LIMITS = {"Rates": 1.0e6, "Credit trading": 1.6e6, "FX": 0.5e6, "Equity derivatives": 0.5e6, "Treasury ALM": 0.6e6}

LCR_OUT = {"Retail deposits – stable": 0.05, "Retail deposits – less stable": 0.10, "Operational deposits": 0.25,
           "Non-operational deposits – corporates": 0.40, "Non-operational deposits – financials": 1.00,
           "Committed facilities – corporates (undrawn)": 0.10}
LCR_IN = {"Inflows – performing loans < 30 days": 0.50, "Inflows – reverse repos < 30 days": 1.00}
HQLA_HC = {"HQLA Level 1": 1.0, "HQLA Level 2A": 0.85, "HQLA Level 2B": 0.50}
ASF = {"Stable funding – equity and long-term debt": 1.0, "Retail deposits – stable": 0.95, "Retail deposits – less stable": 0.90,
       "Operational deposits": 0.50, "Non-operational deposits – corporates": 0.50, "Non-operational deposits – financials": 0.0}
RSF = {"HQLA Level 1": 0.05, "HQLA Level 2A": 0.15, "HQLA Level 2B": 0.50, "Loans to non-financials > 1 year": 0.85,
       "Inflows – performing loans < 30 days": 0.50, "Inflows – reverse repos < 30 days": 0.10, "Other assets": 1.0,
       "Committed facilities – corporates (undrawn)": 0.05}


def credit(credit_df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = credit_df.groupby(by, dropna=False).agg(loans=("loan_id", "count"), counterparties=("lei", "nunique"),
                                                 outstanding=("outstanding_eur", "sum"), EAD=("ead_eur", "sum"),
                                                 RWA=("rwa_eur", "sum"), EL=("el_eur", "sum"), provisions=("provision_eur", "sum"))
    g["RW density"] = g["RWA"] / g["EAD"]
    g["share of EAD"] = g["EAD"] / g["EAD"].sum()
    return g.sort_values("EAD", ascending=False)


def large_exposures(credit_df: pd.DataFrame, trades: pd.DataFrame, cpty: pd.DataFrame, tier1: float = TIER1_EUR) -> pd.DataFrame:
    """Exposure to groups of connected clients: loans EAD + positive MtM of trades (simplified counterparty exposure)."""
    tr = trades[trades["cpty_lei"] != "UNKNOWN"].merge(cpty[["lei", "group"]], left_on="cpty_lei", right_on="lei", how="left")
    tr = tr.groupby("group")["mtm_eur"].apply(lambda x: x.clip(lower=0).sum())
    ln = credit_df.groupby("group")["ead_eur"].sum()
    g = pd.DataFrame({"loans EAD": ln, "derivatives (positive MtM)": tr}).fillna(0.0)
    g["exposure"] = g.sum(axis=1)
    g["% of Tier 1"] = g["exposure"] / tier1
    g["status"] = np.select([g["% of Tier 1"] > 0.25, g["% of Tier 1"] >= 0.10], ["limit breach (> 25 %)", "large exposure (≥ 10 %)"], "")
    return g.sort_values("exposure", ascending=False)


def hhi(x) -> float:
    w = np.asarray(x, float)
    w = w / w.sum()
    return float((w ** 2).sum())


def var_es(pnl: np.ndarray, q: float = 0.99, window: int = 250) -> tuple[float, float]:
    x = np.asarray(pnl)[-window:]
    v = -np.quantile(x, 1 - q)
    es = -x[x <= np.quantile(x, 1 - 0.975)].mean()
    return float(v), float(es)


def market(model, by: str = "desk", q: float = 0.99) -> pd.DataFrame:
    tk = model.market
    rows = []
    for k, g in tk.groupby(by):
        p = model.pnl[g["trade_id"]].sum(axis=1).to_numpy()
        v, e = var_es(p, q)
        rows.append({by: k, "trades": len(g), "MtM": g["mtm_eur"].sum(), "VaR 99 % 1d": v, "ES 97.5 % 1d": e})
    out = pd.DataFrame(rows).set_index(by)
    tot = model.pnl.sum(axis=1).to_numpy()
    v, e = var_es(tot, q)
    out.attrs["total VaR"] = v
    out.attrs["total ES"] = e
    out.attrs["sum of VaRs"] = out["VaR 99 % 1d"].sum()
    if by == "desk":
        out["limit"] = out.index.map(DESK_VAR_LIMITS)
        out["utilisation"] = out["VaR 99 % 1d"] / out["limit"]
    return out


def sensitivities(model) -> pd.DataFrame:
    tk = model.market
    sc = [c for c in tk.columns if c.startswith("sens_")]
    s = tk.groupby("desk")[sc].sum().T
    s.index = [c[5:] for c in s.index]
    return s


def liquidity(model) -> pd.DataFrame:
    Y = model.liquidity
    rows = []
    for ent, g in list(Y.groupby("entity")) + [("Group", Y)]:
        a = g.groupby("category")["amount_eur"].sum()
        l1 = a.get("HQLA Level 1", 0)
        l2a = a.get("HQLA Level 2A", 0) * HQLA_HC["HQLA Level 2A"]
        l2b = a.get("HQLA Level 2B", 0) * HQLA_HC["HQLA Level 2B"]
        l2b = min(l2b, 15 / 85 * (l1 + l2a))                          # simplified caps
        l2 = min(l2a + l2b, 2 / 3 * l1)
        hqla = l1 + l2
        out = sum(a.get(k, 0) * f for k, f in LCR_OUT.items())
        inn = min(sum(a.get(k, 0) * f for k, f in LCR_IN.items()), 0.75 * out)
        asf = sum(a.get(k, 0) * f for k, f in ASF.items())
        rsf = sum(a.get(k, 0) * f for k, f in RSF.items())
        rows.append({"entity": ent, "HQLA": hqla, "outflows": out, "inflows (capped)": inn, "LCR": hqla / (out - inn),
                     "ASF": asf, "RSF": rsf, "NSFR": asf / rsf})
    return pd.DataFrame(rows).set_index("entity")


def oprisk(model, months: int = 12) -> dict:
    O = model.oploss
    recent = O[O["event_date"] > S.ASOF - pd.DateOffset(months=months)]
    by_type = recent.pivot_table(index="event_type", columns="business_line", values="net_loss_eur", aggfunc="sum", fill_value=0.0)
    annual = O.assign(year=O["event_date"].dt.year).groupby("year")["net_loss_eur"].agg(["sum", "count"])
    return {"recent": recent, "matrix": by_type, "annual": annual, "top": recent.nlargest(10, "net_loss_eur")}
