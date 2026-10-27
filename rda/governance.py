"""
BCBS 239 governance: self-assessment of the 14 principles with evidence computed from the run (data-quality
scores, reconciliation, timeliness, lineage coverage) and a data dictionary of critical data elements (CDEs).
Ratings: 4 fully compliant, 3 largely compliant, 2 materially non-compliant, 1 non-compliant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRINCIPLES = [
    (1, "Governance", "I. Overarching governance and infrastructure"),
    (2, "Data architecture and IT infrastructure", "I. Overarching governance and infrastructure"),
    (3, "Accuracy and integrity", "II. Risk data aggregation capabilities"),
    (4, "Completeness", "II. Risk data aggregation capabilities"),
    (5, "Timeliness", "II. Risk data aggregation capabilities"),
    (6, "Adaptability", "II. Risk data aggregation capabilities"),
    (7, "Accuracy (of risk reports)", "III. Risk reporting practices"),
    (8, "Comprehensiveness", "III. Risk reporting practices"),
    (9, "Clarity and usefulness", "III. Risk reporting practices"),
    (10, "Frequency", "III. Risk reporting practices"),
    (11, "Distribution", "III. Risk reporting practices"),
    (12, "Review", "IV. Supervisory review, tools and cooperation"),
    (13, "Remedial actions and supervisory measures", "IV. Supervisory review, tools and cooperation"),
    (14, "Home / host cooperation", "IV. Supervisory review, tools and cooperation"),
]


def _rate(x, good, ok, poor):
    return 4 if x >= good else 3 if x >= ok else 2 if x >= poor else 1


def assess(model, recon_loans: pd.DataFrame, recon_deriv: pd.DataFrame, feeds: pd.DataFrame, asof, adhoc_seconds: float | None = None) -> pd.DataFrame:
    res = model.dq_results
    acc = res[res["dimension"].isin(["Accuracy", "Integrity", "Consistency"])]["pass rate"].mean()
    comp = res[res["dimension"].isin(["Completeness", "Uniqueness"])]["pass rate"].mean()
    loaded = model.run_log["loaded"].sum() / model.run_log["ingested"].sum()
    late = int((feeds["latest as-of"] < asof).sum())
    rec_ok = float((recon_loans["status"] == "reconciled").mean())
    der_ok = float((recon_deriv["status"] == "reconciled").mean())
    lineage = float(model.credit["lineage"].notna().mean() * 0.5 + model.market["lineage"].notna().mean() * 0.5)
    blocking_fail = int((res["status"] == "fail").sum())
    ev = {
        1: ("Data owners named for every source; DQ thresholds and escalation defined", 3, "manual"),
        2: (f"Integrated data model; record-level lineage on {lineage:.0%} of loaded records", _rate(lineage, 0.999, 0.95, 0.8), "automatic"),
        3: (f"Accuracy / integrity / consistency rules: {acc:.2%} average pass rate; {blocking_fail} blocking rules failing", _rate(acc, 0.999, 0.99, 0.97), "automatic"),
        4: (f"Completeness / uniqueness: {comp:.2%}; {loaded:.1%} of ingested records loaded", _rate(min(comp, loaded), 0.999, 0.98, 0.95), "automatic"),
        5: (f"{late} source feed(s) behind the reporting date", 4 if late == 0 else 3 if late == 1 else 2, "automatic"),
        6: (f"Ad hoc report produced in {adhoc_seconds:.2f} s" if adhoc_seconds is not None else "Ad hoc report capability available (see Ad hoc tab)", 4 if (adhoc_seconds or 0) < 5 else 3, "automatic"),
        7: (f"Risk / finance reconciliation: {rec_ok:.0%} of loan cells and {der_ok:.0%} of derivative cells within tolerance", _rate(min(rec_ok, der_ok), 1.0, 0.9, 0.75), "automatic"),
        8: ("Credit, market, liquidity and operational risk covered at group and entity level", 4, "automatic"),
        9: ("Dashboard with drill-down to the transaction and lineage", 3, "manual"),
        10: ("Daily market risk, monthly credit / liquidity / operational risk; stress-time frequency increase possible", 3, "manual"),
        11: ("Distribution list and access rights to be documented", 2, "manual"),
        12: ("Supervisory review – not assessed by the bank", np.nan, "n/a"),
        13: ("Supervisory measures – not assessed by the bank", np.nan, "n/a"),
        14: ("Home / host cooperation – supervisory principle", np.nan, "n/a"),
    }
    rows = []
    for num, title, block in PRINCIPLES:
        e, r, src = ev[num]
        rows.append({"principle": num, "title": title, "block": block, "evidence": e, "rating": r, "basis": src})
    return pd.DataFrame(rows).set_index("principle")


DICTIONARY = pd.DataFrame([
    ("Exposure at default (EAD)", "Drawn amount plus credit conversion factor × undrawn commitment, in EUR", "LOANS.outstanding, LOANS.undrawn, LOANS.ccf", "FX to EUR; EAD = drawn + CCF × undrawn", "Head of credit risk", "L01, L02, L04, L05", "Credit dashboard, large exposures, RWA", True),
    ("Risk-weighted assets (RWA)", "EAD × standardised risk weight by counterparty rating (CRE20)", "Model.ead_eur, REF.rating", "RW lookup; stage 3 = 150 %", "Head of credit risk", "L02, R02", "Credit dashboard, capital", True),
    ("Counterparty LEI", "ISO 17442 legal entity identifier of the obligor", "REF code mapping, TRADES.cpty_lei", "local code → LEI", "Data office", "L02, T01, T06, R01, R03", "All counterparty aggregations", True),
    ("Group of connected clients", "Ultimate parent group of the counterparty", "REF counterparties.group", "LEI → group", "Data office", "R02", "Large exposures", True),
    ("Mark-to-market (MtM)", "Fair value of trading positions in EUR", "TRADES.mtm", "FX to EUR", "Head of market risk", "T02, T03, T05", "Market dashboard, reconciliation", True),
    ("Risk sensitivities", "P&L for a standard move of each risk factor", "TRADES.sens_*", "FX to EUR; × market moves → P&L vector", "Head of market risk", "T02, T04", "VaR, ES, limits", True),
    ("Value-at-Risk (VaR 99 %, 1 day)", "Historical-simulation quantile of the aggregated P&L vector (250 days)", "Model.pnl", "Sum of P&L vectors, then quantile", "Head of market risk", "T02, T03, T04", "Market dashboard", True),
    ("HQLA", "High-quality liquid assets after haircuts and caps", "TREASURY.amount_eur", "Level 2A × 85 %, 2B × 50 %, caps", "Treasurer", "Y01, Y02", "LCR", True),
    ("Net operational loss", "Gross loss minus recoveries", "OPLOSS.gross_loss_eur, OPLOSS.recovery_eur", "recovery capped at gross loss", "Head of operational risk", "O01, O02, O03", "Operational risk dashboard", True),
    ("Gross carrying amount", "Accounting balance of loans and advances", "GL", "none", "Chief accounting officer", "G01", "Risk / finance reconciliation", False),
], columns=["term", "definition", "source fields", "transformation", "data owner", "DQ rules", "used in", "critical data element"])
