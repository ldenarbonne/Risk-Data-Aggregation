"""
Pipeline from the source systems to the integrated risk data model, with record-level lineage.

Stages: ingestion → data-quality controls → rejection of blocking failures (quarantine) → enrichment with the
reference data (counterparty, group, sector, country, rating, book, desk, business line) → currency conversion to EUR →
derived measures (EAD, standardised RWA, P&L vectors) → risk data model. Every loaded record carries its source
system, source key and the list of transformations applied to it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import dq
from . import sources as S

# CRE20.42: corporate exposures, standardised risk weights by external rating
RW_SA = {"AAA": 0.20, "AA": 0.20, "A": 0.50, "BBB": 0.75, "BB": 1.00, "B": 1.50, "CCC": 1.50, "Unrated": 1.00}
PD_BY_RATING = {"AAA": 0.0003, "AA": 0.0005, "A": 0.0008, "BBB": 0.002, "BB": 0.008, "B": 0.03, "CCC": 0.2}


@dataclass
class Model:
    credit: pd.DataFrame
    market: pd.DataFrame
    pnl: pd.DataFrame            # daily P&L vectors per trade (hypothetical, from sensitivities × market moves)
    liquidity: pd.DataFrame
    oploss: pd.DataFrame
    quarantine: pd.DataFrame
    rejected_loans: pd.DataFrame  # staging rows of LOANS rejected by blocking rules (duplicates excluded)
    run_log: pd.DataFrame
    dq_results: pd.DataFrame
    dq_fails: dict


def _blocking(res, fails, ds):
    ids = res[(res["dataset"] == ds) & (res["severity"] == "blocking")]["rule"]
    f = fails[ds][list(ids)]
    return f.any(axis=1), f.apply(lambda r: ", ".join(f.columns[r.to_numpy()]), axis=1)


def _warnings(res, fails, ds):
    ids = res[(res["dataset"] == ds) & (res["severity"] == "warning")]["rule"]
    f = fails[ds][list(ids)]
    return f.apply(lambda r: ", ".join(f.columns[r.to_numpy()]), axis=1)


def run(src: S.Sources) -> Model:
    res, fails = dq.run(src)
    log, quarantine = [], []
    cp = src.ref_cpty.set_index("lei")
    code2lei = dict(zip(src.ref_map["local_code"], src.ref_map["lei"]))

    # ------------------------------------------------------------------ credit
    L = src.loans.copy()
    block, why = _blocking(res, fails, "LOANS")
    warn = _warnings(res, fails, "LOANS")
    q = L[block].assign(dataset="LOANS", reason=why[block], key=L.loc[block, "loan_id"])
    quarantine.append(q[["dataset", "key", "reason"]])
    rejected_loans = L[block & ~fails["LOANS"]["L05"]]
    ok = L[~block].copy()
    ok["warnings"] = warn[~block]
    ok["lei"] = ok["cpty_code"].map(code2lei)
    for c in ["name", "group", "country", "sector", "rating"]:
        ok[c] = ok["lei"].map(cp[c])
    pd_fix = ~ok["pd"].between(1e-6, 1)
    ok.loc[pd_fix, "pd"] = ok.loc[pd_fix, "rating"].map(PD_BY_RATING)
    fx = ok["currency"].map(S.FX_TO_EUR)
    ok["fx_rate"] = fx
    ok["outstanding_eur"] = ok["outstanding"] * fx
    ok["undrawn_eur"] = ok["undrawn"] * fx
    ok["ccf"] = np.where(ok["product"] == "Guarantee", 1.0, ok["ccf"])
    ok["on_balance"] = ok["product"] != "Guarantee"
    ok["ead_eur"] = ok["outstanding_eur"] + ok["ccf"] * ok["undrawn_eur"]
    ok["rw"] = np.where(ok["stage"] == 3, 1.5, ok["rating"].map(RW_SA).fillna(1.0))
    ok["rwa_eur"] = ok["ead_eur"] * ok["rw"]
    ok["el_eur"] = ok["ead_eur"] * ok["pd"] * ok["lgd"]
    ok["provision_eur"] = ok["provision"] * fx
    ok["source"] = "LOANS"
    ok["lineage"] = ("LOANS." + ok["loan_id"] + " → map " + ok["cpty_code"] + "→LEI " + ok["lei"].str[:8] + "… → REF (group, sector, rating) → FX "
                     + ok["currency"] + "→EUR @" + fx.round(4).astype(str) + " → EAD = drawn + CCF × undrawn → RW " + ok["rw"].astype(str)
                     + np.where(pd_fix, " → PD replaced by rating PD (rule L03)", ""))
    log.append(("LOANS", len(L), int(fails["LOANS"]["L05"].sum()), int(block.sum()), len(ok), int((ok["warnings"] != "").sum())))

    # ------------------------------------------------------------------ market
    T = src.trades.copy()
    block, why = _blocking(res, fails, "TRADES")
    warn = _warnings(res, fails, "TRADES")
    q = T[block].assign(dataset="TRADES", reason=why[block], key=T.loc[block, "trade_id"])
    quarantine.append(q[["dataset", "key", "reason"]])
    tk = T[~block].merge(src.ref_books, on="book", how="left")
    tk["warnings"] = warn[~block].to_numpy()
    tk["cpty_lei"] = tk["cpty_lei"].fillna("UNKNOWN")
    tk["fx_rate"] = tk["currency"].map(S.FX_TO_EUR)
    tk["mtm_eur"] = tk["mtm"] * tk["fx_rate"]
    tk["notional_eur"] = tk["notional"] * tk["fx_rate"]
    tk["source"] = "TRADES"
    tk["lineage"] = ("TRADES." + tk["trade_id"] + " → REF book " + tk["book"] + " (desk " + tk["desk"] + ", entity " + tk["entity"]
                     + ") → FX " + tk["currency"] + "→EUR → P&L vector = sensitivities × MKT moves (last 500 days)")
    scols = [c for c in tk.columns if c.startswith("sens_")]
    M = src.mkt[[c[5:] for c in scols]].to_numpy()
    pnl = pd.DataFrame(M @ tk[scols].to_numpy().T * tk["fx_rate"].to_numpy(), index=src.mkt.index, columns=tk["trade_id"])
    log.append(("TRADES", len(T), int(fails["TRADES"]["T05"].sum()), int(block.sum()), len(tk), int((tk["warnings"] != "").sum())))

    # ------------------------------------------------------------------ liquidity
    Y = src.treasury.copy()
    block, why = _blocking(res, fails, "TREASURY")
    quarantine.append(Y[block].assign(dataset="TREASURY", reason=why[block], key=Y.loc[block, "category"])[["dataset", "key", "reason"]])
    Y = Y[~block].assign(source="TREASURY")
    log.append(("TREASURY", len(src.treasury), 0, int(block.sum()), len(Y), 0))

    # ------------------------------------------------------------------ operational losses
    O = src.oploss.copy()
    block, why = _blocking(res, fails, "OPLOSS")
    warn = _warnings(res, fails, "OPLOSS")
    quarantine.append(O[block].assign(dataset="OPLOSS", reason=why[block], key=O.loc[block, "event_id"])[["dataset", "key", "reason"]])
    O = O[~block].copy()
    O["warnings"] = warn[~block]
    O["recovery_eur"] = np.minimum(O["recovery_eur"], O["gross_loss_eur"])
    O["net_loss_eur"] = O["gross_loss_eur"] - O["recovery_eur"]
    O["source"] = "OPLOSS"
    log.append(("OPLOSS", len(src.oploss), 0, int(block.sum()), len(O), int((O["warnings"] != "").sum())))

    run_log = pd.DataFrame(log, columns=["dataset", "ingested", "duplicates", "rejected (blocking)", "loaded", "loaded with warnings"])
    return Model(ok, tk, pnl, Y, O, pd.concat(quarantine, ignore_index=True), rejected_loans, run_log, res, fails)
