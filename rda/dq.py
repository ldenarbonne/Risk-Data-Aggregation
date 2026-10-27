"""
Data-quality rule catalogue and engine. Dimensions follow BCBS 239 (principles 3 to 5) and the ECB guide on
risk data aggregation and risk reporting (2024): completeness, accuracy / validity, consistency / integrity,
timeliness and uniqueness. A blocking rule rejects the failing records from the risk data model; a warning keeps
them, flagged.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import sources as S

LEI_RE = re.compile(r"^[A-Z0-9]{18}[0-9]{2}$")


def rules(src: S.Sources) -> list[dict]:
    asof = S.ASOF
    codes = set(src.ref_map["local_code"])
    books = set(src.ref_books["book"])
    leis = set(src.ref_cpty["lei"])
    R = []

    def add(rid, ds, dim, desc, sev, thr, fn, idcol):
        R.append({"rule": rid, "dataset": ds, "dimension": dim, "description": desc, "severity": sev, "threshold": thr,
                  "fn": fn, "id": idcol})

    add("L01", "LOANS", "Completeness", "Currency is populated", "blocking", 0.999, lambda d: d["currency"].notna(), "loan_id")
    add("L02", "LOANS", "Integrity", "Counterparty code maps to an LEI in the reference data", "blocking", 0.999,
        lambda d: d["cpty_code"].isin(codes), "loan_id")
    add("L03", "LOANS", "Accuracy", "PD within (0, 1]", "warning", 0.995, lambda d: (d["pd"] > 0) & (d["pd"] <= 1), "loan_id")
    add("L04", "LOANS", "Accuracy", "Outstanding amount is not negative", "blocking", 0.999, lambda d: d["outstanding"] >= 0, "loan_id")
    add("L05", "LOANS", "Uniqueness", "Loan identifier is unique", "blocking", 1.0, lambda d: ~d.duplicated("loan_id", keep="first"), "loan_id")
    add("L06", "LOANS", "Consistency", "Maturity date not before the reporting date", "warning", 0.99, lambda d: d["maturity_date"] >= asof, "loan_id")
    add("L07", "LOANS", "Accuracy", "LGD within [0, 1]", "warning", 0.999, lambda d: d["lgd"].between(0, 1), "loan_id")
    add("L08", "LOANS", "Timeliness", "As-of date equals the reporting date", "warning", 1.0, lambda d: d["asof"] == asof, "loan_id")
    add("T01", "TRADES", "Completeness", "Counterparty LEI is populated", "warning", 0.995, lambda d: d["cpty_lei"].notna(), "trade_id")
    add("T02", "TRADES", "Integrity", "Book exists in the reference data", "blocking", 1.0, lambda d: d["book"].isin(books), "trade_id")
    add("T03", "TRADES", "Accuracy", "|MtM| ≤ 50 % of notional (unit-error check)", "blocking", 0.999,
        lambda d: d["mtm"].abs() <= 0.5 * d["notional"], "trade_id")
    add("T04", "TRADES", "Timeliness", "As-of date equals the reporting date", "warning", 1.0, lambda d: d["asof"] == asof, "trade_id")
    add("T05", "TRADES", "Uniqueness", "Trade identifier is unique", "blocking", 1.0, lambda d: ~d.duplicated("trade_id"), "trade_id")
    add("T06", "TRADES", "Integrity", "Counterparty LEI exists in the reference data (when populated)", "warning", 0.999,
        lambda d: d["cpty_lei"].isna() | d["cpty_lei"].isin(leis), "trade_id")
    add("O01", "OPLOSS", "Accuracy", "Event date not after the reporting date", "blocking", 1.0, lambda d: d["event_date"] <= asof, "event_id")
    add("O02", "OPLOSS", "Consistency", "Recovery not above the gross loss", "warning", 0.999, lambda d: d["recovery_eur"] <= d["gross_loss_eur"], "event_id")
    add("O03", "OPLOSS", "Completeness", "Basel event type is valid", "blocking", 1.0, lambda d: d["event_type"].isin(S.BASEL_EVENT_TYPES), "event_id")
    add("R01", "REF counterparties", "Accuracy", "LEI follows the ISO 17442 format", "warning", 1.0,
        lambda d: d["lei"].astype(str).str.match(LEI_RE), "lei")
    add("R02", "REF counterparties", "Completeness", "Sector, country and rating populated", "warning", 1.0,
        lambda d: d[["sector", "country", "rating"]].notna().all(axis=1), "lei")
    add("R03", "REF counterparties", "Uniqueness", "LEI is unique", "blocking", 1.0, lambda d: ~d.duplicated("lei"), "lei")
    add("Y01", "TREASURY", "Accuracy", "Amounts not negative", "blocking", 1.0, lambda d: d["amount_eur"] >= 0, "category")
    add("Y02", "TREASURY", "Completeness", "Every entity reports HQLA and deposits", "warning", 1.0,
        lambda d: d.groupby("entity")["category"].transform("count") >= 10, "category")
    add("G01", "GL", "Integrity", "Entity exists in the legal-entity reference", "blocking", 1.0,
        lambda d: d["entity"].isin(S.ENTITIES["entity"]), "account")
    return R


DATASETS = {"LOANS": "loans", "TRADES": "trades", "OPLOSS": "oploss", "REF counterparties": "ref_cpty", "TREASURY": "treasury", "GL": "gl"}


def run(src: S.Sources) -> tuple[pd.DataFrame, dict]:
    """Returns the rule results and, per dataset, a boolean frame of failures (rows × rules)."""
    out, fails = [], {}
    for r in rules(src):
        d = getattr(src, DATASETS[r["dataset"]])
        ok = np.asarray(r["fn"](d), bool)
        f = fails.setdefault(r["dataset"], pd.DataFrame(index=d.index))
        f[r["rule"]] = ~ok
        n = len(d)
        out.append({k: r[k] for k in ("rule", "dataset", "dimension", "description", "severity", "threshold")} |
                   {"tested": n, "failed": int((~ok).sum()), "pass rate": float(ok.mean()) if n else 1.0})
    res = pd.DataFrame(out)
    res["status"] = np.where(res["pass rate"] >= res["threshold"], "pass", np.where(res["severity"] == "blocking", "fail", "warning"))
    return res, fails


def scores(res: pd.DataFrame) -> pd.DataFrame:
    """Data-quality score (mean pass rate) by dataset and dimension."""
    return res.pivot_table(index="dataset", columns="dimension", values="pass rate", aggfunc="mean")


def failing_records(src: S.Sources, fails: dict, rule_id: str) -> pd.DataFrame:
    for ds, f in fails.items():
        if rule_id in f:
            d = getattr(src, DATASETS[ds])
            return d.loc[f[rule_id].to_numpy()]
    return pd.DataFrame()
