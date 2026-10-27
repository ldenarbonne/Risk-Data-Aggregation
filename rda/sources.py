"""
Synthetic source systems of a banking group, each with its own conventions, and injected data-quality defects.

* REF – reference data: counterparties (LEI, group parent, country, sector, rating), legal entities, books and desks,
  FX rates, mapping of local counterparty codes to LEIs.
* LOANS – credit system (per entity, local currency, local counterparty codes).
* TRADES – front-office trading system (trades with mark-to-market and risk sensitivities).
* MKT – market data history (daily factor returns) used by the market risk engine.
* TREASURY – liquidity positions (HQLA, deposits, contractual inflows and outflows by bucket).
* OPLOSS – operational loss events.
* GL – general ledger (finance) balances by entity, account and currency.

Every injected defect is recorded in `defects` (dataset, record, type), the ground truth against which the
data-quality rules are tested.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ASOF = pd.Timestamp("2026-09-30")
ENTITIES = pd.DataFrame([
    ("BNKFR", "Banque Exemple SA (Paris)", "FR", "EUR", None),
    ("BNKUK", "Exemple Bank plc (London)", "GB", "GBP", "BNKFR"),
    ("BNKLU", "Exemple Luxembourg SA", "LU", "EUR", "BNKFR"),
    ("BNKUS", "Exemple Securities Inc (New York)", "US", "USD", "BNKFR"),
], columns=["entity", "name", "country", "currency", "parent"])
FX_TO_EUR = {"EUR": 1.0, "USD": 1 / 1.08, "GBP": 1 / 0.84, "CHF": 1 / 0.94, "JPY": 1 / 162.0}
SECTORS = ["Manufacturing", "Real estate", "Financials", "Energy", "Retail trade", "Transport", "Information & communication",
           "Public administration", "Health", "Construction"]
COUNTRIES = ["FR", "DE", "IT", "ES", "GB", "US", "LU", "NL", "BE", "CH"]
RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
BUSINESS_LINES = ["Corporate banking", "Real estate finance", "Global markets", "Treasury", "Retail banking"]
BASEL_EVENT_TYPES = ["Internal fraud", "External fraud", "Employment practices", "Clients, products & business practices",
                     "Damage to physical assets", "Business disruption & system failures", "Execution, delivery & process management"]


@dataclass
class Sources:
    ref_cpty: pd.DataFrame
    ref_map: pd.DataFrame
    ref_books: pd.DataFrame
    loans: pd.DataFrame
    trades: pd.DataFrame
    mkt: pd.DataFrame
    treasury: pd.DataFrame
    oploss: pd.DataFrame
    gl: pd.DataFrame
    feeds: pd.DataFrame
    defects: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["dataset", "record", "type"]))

    def tables(self) -> dict:
        return {"REF counterparties": self.ref_cpty, "REF code mapping": self.ref_map, "REF books": self.ref_books,
                "LOANS": self.loans, "TRADES": self.trades, "TREASURY": self.treasury, "OPLOSS": self.oploss, "GL": self.gl}


def _lei(rng) -> str:
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join(rng.choice(list(chars), 18)) + "".join(rng.choice(list("0123456789"), 2))


def generate(seed: int = 7, defect_rate: float = 0.02, late_feed: bool = True) -> Sources:
    rng = np.random.default_rng(seed)
    defects = []
    # ------------------------------------------------------------------ reference data
    n_c = 220
    groups = [f"GRP{i:03d}" for i in range(60)]
    cpty = pd.DataFrame({
        "lei": [_lei(rng) for _ in range(n_c)],
        "name": [f"Counterparty {i:03d}" for i in range(n_c)],
        "group": rng.choice(groups, n_c),
        "country": rng.choice(COUNTRIES, n_c, p=[0.32, 0.12, 0.09, 0.08, 0.1, 0.12, 0.04, 0.05, 0.04, 0.04]),
        "sector": rng.choice(SECTORS, n_c),
        "rating": rng.choice(RATINGS, n_c, p=[0.03, 0.08, 0.2, 0.32, 0.2, 0.13, 0.04]),
    })
    # a few large groups (concentration)
    cpty.loc[:8, "group"] = "GRP900"
    cpty.loc[9:14, "group"] = "GRP901"
    codes = []
    for ent in ENTITIES["entity"]:
        sub = cpty.sample(frac=0.6, random_state=int(rng.integers(1e6)))
        for lei in sub["lei"]:
            codes.append((ent, f"{ent[-2:]}{rng.integers(10000, 99999)}", lei))
    rmap = pd.DataFrame(codes, columns=["entity", "local_code", "lei"]).drop_duplicates("local_code")
    books = pd.DataFrame([
        ("RATES-EUR-01", "Rates", "BNKFR", "Global markets"), ("RATES-USD-01", "Rates", "BNKUS", "Global markets"),
        ("CREDIT-01", "Credit trading", "BNKFR", "Global markets"), ("FX-01", "FX", "BNKUK", "Global markets"),
        ("EQD-01", "Equity derivatives", "BNKFR", "Global markets"), ("ALM-01", "Treasury ALM", "BNKFR", "Treasury"),
        ("CREDIT-UK", "Credit trading", "BNKUK", "Global markets"),
    ], columns=["book", "desk", "entity", "business_line"])

    # ------------------------------------------------------------------ loans (credit system)
    n_l = 1500
    rmap_e = rmap.groupby("entity")
    rows = []
    code2lei = dict(zip(rmap["local_code"], rmap["lei"]))
    lei2rating = dict(zip(cpty["lei"], cpty["rating"]))
    codes_by_ent = {e: g["local_code"].to_numpy() for e, g in rmap_e}
    for i in range(n_l):
        ent = rng.choice(ENTITIES["entity"], p=[0.55, 0.2, 0.12, 0.13])
        code = rng.choice(codes_by_ent[ent])
        r = lei2rating[code2lei[code]]
        ccy = {"BNKUK": "GBP", "BNKUS": "USD"}.get(ent, "EUR") if rng.random() < 0.8 else rng.choice(["EUR", "USD", "CHF"])
        out = float(np.round(np.exp(rng.normal(np.log(4e6), 1.1)), -3))
        und = float(np.round(out * rng.choice([0, 0, 0.2, 0.5, 1.0]), -3))
        pdv = {"AAA": 0.0003, "AA": 0.0005, "A": 0.0008, "BBB": 0.002, "BB": 0.008, "B": 0.03, "CCC": 0.2}[r] * rng.lognormal(0, 0.2)
        rows.append({"loan_id": f"L{ent[-2:]}{i:05d}", "entity": ent, "cpty_code": code, "product": rng.choice(
            ["Term loan", "Revolving facility", "Mortgage (CRE)", "Overdraft", "Guarantee"], p=[0.45, 0.2, 0.15, 0.1, 0.1]),
            "business_line": rng.choice(["Corporate banking", "Real estate finance", "Retail banking"], p=[0.6, 0.25, 0.15]),
            "currency": ccy, "outstanding": out, "undrawn": und, "ccf": 0.4 if und > 0 else 0.0, "pd": round(min(pdv, 0.99), 5),
            "lgd": round(float(rng.uniform(0.2, 0.6)), 3), "maturity_date": ASOF + pd.Timedelta(days=int(rng.integers(30, 3650))),
            "stage": rng.choice([1, 2, 3], p=[0.88, 0.1, 0.02]), "asof": ASOF})
    loans = pd.DataFrame(rows)
    loans["provision"] = np.round(loans["outstanding"] * loans["pd"] * loans["lgd"] * np.where(loans["stage"] == 3, 1 / loans["pd"], 1.0) * np.where(loans["stage"] == 2, 3.0, 1.0), -2)
    loans["accrued_interest"] = np.round(loans["outstanding"] * rng.uniform(0.0, 0.01, n_l), -2)
    # a large exposure to the large group
    big = cpty.loc[cpty["group"] == "GRP900", "lei"].iloc[:3]
    for k, lei in enumerate(big):
        code = rmap.loc[(rmap["lei"] == lei) & (rmap["entity"] == "BNKFR"), "local_code"]
        if len(code):
            loans.loc[k, ["entity", "cpty_code", "outstanding", "currency"]] = ["BNKFR", code.iloc[0], 3.5e8, "EUR"]

    def inject(df, name, idcol, n, typ, fn):
        idx = rng.choice(df.index, n, replace=False)
        for i in idx:
            fn(df, i)
            defects.append((name, df.loc[i, idcol] if idcol in df else i, typ))

    k = max(int(defect_rate * n_l), 1)
    inject(loans, "LOANS", "loan_id", k, "unmapped counterparty code", lambda d, i: d.__setitem__("cpty_code", d["cpty_code"].where(d.index != i, f"XX{rng.integers(10000, 99999)}")))
    inject(loans, "LOANS", "loan_id", k, "missing currency", lambda d, i: d.__setitem__("currency", d["currency"].where(d.index != i, None)))
    inject(loans, "LOANS", "loan_id", k, "PD out of range", lambda d, i: d.__setitem__("pd", d["pd"].where(d.index != i, 1.7)))
    inject(loans, "LOANS", "loan_id", k, "negative outstanding", lambda d, i: d.__setitem__("outstanding", d["outstanding"].where(d.index != i, -abs(d.loc[i, "outstanding"]))))
    inject(loans, "LOANS", "loan_id", k, "maturity before reporting date", lambda d, i: d.__setitem__("maturity_date", d["maturity_date"].where(d.index != i, ASOF - pd.Timedelta(days=40))))
    dup = loans.sample(k, random_state=int(rng.integers(1e6)))
    for lid in dup["loan_id"]:
        defects.append(("LOANS", lid, "duplicate record"))
    loans = pd.concat([loans, dup], ignore_index=True)

    # ------------------------------------------------------------------ market data and trades
    n_days = 500
    factors = ["EUR 2Y", "EUR 10Y", "USD 10Y", "Credit IG", "Credit HY", "EURUSD", "GBPEUR", "EURO STOXX 50", "S&P 500"]
    vols = np.array([4, 5, 6, 3, 9, 0.006, 0.005, 0.012, 0.011])            # bp / day or relative
    C = np.full((9, 9), 0.25); np.fill_diagonal(C, 1.0)
    C[0, 1] = C[1, 0] = 0.8; C[1, 2] = C[2, 1] = 0.6; C[3, 4] = C[4, 3] = 0.7; C[7, 8] = C[8, 7] = 0.75
    C[3, 7] = C[7, 3] = -0.5; C[4, 7] = C[7, 4] = -0.6; C[4, 8] = C[8, 4] = -0.5
    w, v = np.linalg.eigh(C); C = v @ np.diag(np.clip(w, 1e-3, None)) @ v.T; d = np.sqrt(np.diag(C)); C = C / np.outer(d, d)
    Lc = np.linalg.cholesky(C)
    Z = rng.standard_t(4, (n_days, 9)) / np.sqrt(2) @ Lc.T
    mkt = pd.DataFrame(Z * vols, columns=factors, index=pd.bdate_range(end=ASOF, periods=n_days))
    n_t = 900
    tr = []
    for i in range(n_t):
        b = books.iloc[int(rng.integers(len(books)))]
        lei = rng.choice(cpty["lei"])
        prod = {"Rates": ["IRS", "Swaption", "Government bond"], "Credit trading": ["Corporate bond", "CDS"], "FX": ["FX forward", "FX option"],
                "Equity derivatives": ["Equity option", "Equity swap"], "Treasury ALM": ["Government bond", "IRS"]}[b["desk"]]
        sens = dict.fromkeys(factors, 0.0)
        if b["desk"] in ("Rates", "Treasury ALM"):
            sens[rng.choice(["EUR 2Y", "EUR 10Y", "USD 10Y"])] = float(rng.normal(0, 8000))      # € per bp
        elif b["desk"] == "Credit trading":
            sens[rng.choice(["Credit IG", "Credit HY"])] = float(rng.normal(-400, 2500))
        elif b["desk"] == "FX":
            sens[rng.choice(["EURUSD", "GBPEUR"])] = float(rng.normal(0, 2e6))                  # € per 100 %
        else:
            sens[rng.choice(["EURO STOXX 50", "S&P 500"])] = float(rng.normal(0, 1.5e6))
        ccy = "USD" if b["entity"] == "BNKUS" else ("GBP" if b["entity"] == "BNKUK" else "EUR")
        tr.append({"trade_id": f"T{i:06d}", "book": b["book"], "cpty_lei": lei, "product": rng.choice(prod),
                   "notional": float(np.round(np.exp(rng.normal(np.log(2e7), 0.8)), -4)), "currency": ccy,
                   "mtm": float(np.round(rng.normal(0, 4e5), -2)), "trade_date": ASOF - pd.Timedelta(days=int(rng.integers(1, 1500))),
                   **{f"sens_{f}": s for f, s in sens.items()}, "asof": ASOF})
    trades = pd.DataFrame(tr)
    kt = max(int(defect_rate * n_t), 1)
    inject(trades, "TRADES", "trade_id", kt, "missing LEI", lambda d, i: d.__setitem__("cpty_lei", d["cpty_lei"].where(d.index != i, None)))
    inject(trades, "TRADES", "trade_id", kt, "unknown book", lambda d, i: d.__setitem__("book", d["book"].where(d.index != i, "OLD-BOOK-99")))
    inject(trades, "TRADES", "trade_id", max(kt // 2, 1), "MtM unit error (×1000)", lambda d, i: d.__setitem__("mtm", d["mtm"].where(d.index != i, d.loc[i, "mtm"] * 1000)))
    if late_feed:
        trades.loc[trades["book"] == "FX-01", "asof"] = ASOF - pd.Timedelta(days=3)
        for tid in trades.loc[trades["book"] == "FX-01", "trade_id"]:
            defects.append(("TRADES", tid, "stale as-of date"))

    # ------------------------------------------------------------------ treasury (liquidity)
    tre = []
    for ent in ENTITIES["entity"]:
        scale = {"BNKFR": 1.0, "BNKUK": 0.35, "BNKLU": 0.2, "BNKUS": 0.3}[ent]
        for cat, amt in [("HQLA Level 1", 9e9), ("HQLA Level 2A", 1.5e9), ("HQLA Level 2B", 0.6e9),
                         ("Retail deposits – stable", 14e9), ("Retail deposits – less stable", 6e9),
                         ("Operational deposits", 5e9), ("Non-operational deposits – corporates", 7e9),
                         ("Non-operational deposits – financials", 3e9), ("Committed facilities – corporates (undrawn)", 6e9),
                         ("Inflows – performing loans < 30 days", 4e9), ("Inflows – reverse repos < 30 days", 2.5e9),
                         ("Stable funding – equity and long-term debt", 12e9), ("Loans to non-financials > 1 year", 30e9),
                         ("Other assets", 6e9)]:
            tre.append({"entity": ent, "category": cat, "amount_eur": float(np.round(amt * scale * rng.uniform(0.85, 1.15), -6)), "asof": ASOF})
    treasury = pd.DataFrame(tre)

    # ------------------------------------------------------------------ operational losses
    n_o = 260
    op = pd.DataFrame({
        "event_id": [f"OP{i:05d}" for i in range(n_o)],
        "entity": rng.choice(ENTITIES["entity"], n_o, p=[0.55, 0.2, 0.1, 0.15]),
        "business_line": rng.choice(BUSINESS_LINES, n_o),
        "event_type": rng.choice(BASEL_EVENT_TYPES, n_o, p=[0.05, 0.25, 0.08, 0.17, 0.03, 0.07, 0.35]),
        "event_date": [ASOF - pd.Timedelta(days=int(x)) for x in rng.integers(0, 3 * 365, n_o)],
        "gross_loss_eur": np.round(np.exp(rng.normal(np.log(60e3), 1.6, n_o)), -2),
    })
    op["recovery_eur"] = np.round(op["gross_loss_eur"] * rng.choice([0, 0, 0, 0.3, 0.8], n_o), -2)
    op.loc[0, "gross_loss_eur"] = 18e6
    op.loc[0, "event_type"] = "Clients, products & business practices"
    ko = max(int(defect_rate * n_o), 1)
    inject(op, "OPLOSS", "event_id", ko, "event date in the future", lambda d, i: d.__setitem__("event_date", d["event_date"].where(d.index != i, ASOF + pd.Timedelta(days=60))))
    inject(op, "OPLOSS", "event_id", ko, "recovery above gross loss", lambda d, i: d.__setitem__("recovery_eur", d["recovery_eur"].where(d.index != i, d.loc[i, "gross_loss_eur"] * 1.5)))

    # ------------------------------------------------------------------ general ledger (finance)
    clean = loans.drop_duplicates("loan_id")
    clean = clean[clean["currency"].notna() & (clean["outstanding"] > 0)]
    gl = []
    for (ent, ccy), g in clean.groupby(["entity", "currency"]):
        on = g.loc[g["product"] != "Guarantee"]
        bal = on["outstanding"].sum() + on["accrued_interest"].sum()
        if ent == "BNKLU" and ccy == "EUR":
            bal += 12.5e6                                                    # unexplained break
            defects.append(("GL", f"{ent}/{ccy}", "unexplained reconciliation break"))
        gl.append({"entity": ent, "account": "Loans and advances – gross carrying amount", "currency": ccy, "balance": round(bal, -2)})
        gl.append({"entity": ent, "account": "Loss allowance", "currency": ccy, "balance": -round(on["provision"].sum(), -2)})
        gl.append({"entity": ent, "account": "Off-balance sheet guarantees", "currency": ccy,
                   "balance": round(g.loc[g["product"] == "Guarantee", "outstanding"].sum(), -2)})
    t_ok = trades[trades["book"].isin(books["book"])]
    for (ent), g in t_ok.merge(books, on="book").groupby("entity"):
        ccy = "USD" if ent == "BNKUS" else ("GBP" if ent == "BNKUK" else "EUR")
        gl.append({"entity": ent, "account": "Trading derivatives and securities – net fair value", "currency": ccy,
                   "balance": round(g.loc[g["mtm"].abs() < 5e7, "mtm"].sum() * 1.0, -2)})
    gl = pd.DataFrame(gl)

    feeds = pd.DataFrame([
        ("REF", "Reference data hub", "Data office", ASOF, "06:00"), ("LOANS", "Credit system (per entity)", "Credit risk IT", ASOF, "07:30"),
        ("TRADES", "Front-office trading platform", "Markets IT", ASOF - pd.Timedelta(days=3) if late_feed else ASOF, "08:15"),
        ("MKT", "Market data service", "Market data team", ASOF, "05:45"), ("TREASURY", "ALM / treasury system", "Treasury", ASOF, "09:00"),
        ("OPLOSS", "Operational loss database", "Operational risk", ASOF, "10:30"), ("GL", "General ledger", "Finance", ASOF, "11:00"),
    ], columns=["source", "system", "data owner", "latest as-of", "delivery time"])
    return Sources(cpty, rmap, books, loans, trades, mkt, treasury, op, gl, feeds,
                   pd.DataFrame(defects, columns=["dataset", "record", "type"]))
