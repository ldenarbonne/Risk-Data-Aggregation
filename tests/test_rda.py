import numpy as np
import pandas as pd
import pytest

from rda import aggregate as A
from rda import dq
from rda import governance as G
from rda import io as IO
from rda import pipeline as PL
from rda import recon as R
from rda import sources as S


@pytest.fixture(scope="module")
def run():
    src = S.generate(7, 0.02, True)
    return src, PL.run(src)


DEFECT_RULE = {"unmapped counterparty code": "L02", "missing currency": "L01", "PD out of range": "L03",
               "negative outstanding": "L04", "maturity before reporting date": "L06", "duplicate record": "L05",
               "missing LEI": "T01", "unknown book": "T02", "MtM unit error (×1000)": "T03", "stale as-of date": "T04",
               "event date in the future": "O01", "recovery above gross loss": "O02"}


def test_every_injected_defect_is_detected(run):
    src, m = run
    for (ds, typ), g in src.defects.groupby(["dataset", "type"]):
        if typ not in DEFECT_RULE:
            continue
        fr = dq.failing_records(src, m.dq_fails, DEFECT_RULE[typ])
        idcol = {"LOANS": "loan_id", "TRADES": "trade_id", "OPLOSS": "event_id"}[ds]
        assert set(g["record"]) <= set(fr[idcol]), typ


def test_clean_sources_pass(run):
    src = S.generate(7, 0.0, False)
    res, _ = dq.run(src)
    # with no injected defect only the (single, forced) minimum of one defect per type can fail
    assert res["pass rate"].min() > 0.98


def test_record_counts_balance(run):
    _, m = run
    lg = m.run_log
    assert (lg["ingested"] == lg["rejected (blocking)"] + lg["loaded"]).all()
    assert m.credit["loan_id"].is_unique and m.credit["lineage"].notna().all()


def test_ead_rwa(run):
    _, m = run
    c = m.credit
    assert np.allclose(c["ead_eur"], c["outstanding_eur"] + c["ccf"] * c["undrawn_eur"])
    assert np.allclose(c["rwa_eur"], c["ead_eur"] * c["rw"])
    assert (c["pd"].between(0, 1)).all()


def test_reconciliation_finds_only_the_injected_break(run):
    src, m = run
    rl = R.loans(src, m)
    br = rl[rl["status"] == "break"]
    assert list(br.index) == [("BNKLU", "EUR")] and abs(br["unexplained"].iloc[0] - 12.5e6) < 1.0
    assert (R.derivatives(src, m)["status"] == "reconciled").all()


def test_var_aggregation_subadditive(run):
    _, m = run
    mk = A.market(m)
    assert mk.attrs["total VaR"] <= mk.attrs["sum of VaRs"]
    p = m.pnl.sum(axis=1).to_numpy()
    assert np.isclose(mk.attrs["total VaR"], -np.quantile(p[-250:], 0.01))


def test_large_exposures_and_liquidity(run):
    src, m = run
    le = A.large_exposures(m.credit, m.market, src.ref_cpty, 6e9)
    assert le["% of Tier 1"].iloc[0] > 0.25 and le.iloc[0]["status"].startswith("limit")
    lq = A.liquidity(m)
    assert (lq["LCR"] > 0).all() and "Group" in lq.index


def test_governance_and_import(run):
    src, m = run
    ga = G.assess(m, R.loans(src, m), R.derivatives(src, m), src.feeds, S.ASOF, 0.5)
    assert len(ga) == 14 and ga["rating"].dropna().between(1, 4).all()
    new = IO.replace_table(src, "LOANS", src.loans.head(300).astype({"outstanding": str}))
    m2 = PL.run(new)
    assert m2.run_log.set_index("dataset").loc["LOANS", "ingested"] == 300
    with pytest.raises(ValueError):
        IO.replace_table(src, "LOANS", src.loans.drop(columns=["pd"]))


def test_app_runs():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("../app.py", default_timeout=900)
    at.run()
    assert not at.exception
