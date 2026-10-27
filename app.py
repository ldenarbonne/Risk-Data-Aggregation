"""Risk Data Aggregator – BCBS 239 risk data aggregation, data quality, reconciliation and risk reporting."""
from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rda import aggregate as A
from rda import dq
from rda import governance as G
from rda import io as IO
from rda import pipeline as PL
from rda import recon as R
from rda import sources as S

st.set_page_config(page_title="Risk Data Aggregator", page_icon="🗂️", layout="wide")
BLUE, ORANGE, RED, GREEN, GREY = "#1f6f8b", "#d4892b", "#b8433a", "#3b8c5a", "#9aa5b1"
M = 1e6


def h(df):
    return min(35 * (len(df) + 1) + 3, 900)


def text(df: pd.DataFrame, fmt: dict) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        f = fmt.get(c)
        if f is not None:
            out[c] = [("–" if (v is None or (isinstance(v, float) and np.isnan(v))) else f.format(v)) for v in df[c]]
    return out


@st.cache_data(show_spinner="Generating the source systems…")
def make_sources(seed, rate, late):
    return S.generate(seed, rate, late)


@st.cache_data(show_spinner="Running the pipeline…")
def run_pipeline(_src, key):
    return PL.run(_src)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Source systems")
    seed = st.number_input("Seed", 0, 9999, 7)
    rate = st.slider("Injected defect rate", 0.0, 0.08, 0.02, 0.005, format="%.3f")
    late = st.checkbox("Trading feed delivered late (FX book, 3 days old)", value=True)
    st.header("Reporting")
    tier1 = st.number_input("Tier 1 capital (€ bn)", 1.0, 50.0, 6.0, 0.5) * 1e9
    st.caption(f"Reporting date: {S.ASOF:%d %B %Y}")

src = make_sources(seed, rate, late)
key = ("synthetic", seed, rate, late)
if "imported" in st.session_state and st.session_state.get("imp_base") == key:
    for name, df in st.session_state["imported"].items():
        src = IO.replace_table(src, name, df)
    key = key + tuple(sorted(st.session_state["imported"]))
model = run_pipeline(src, key)
rl, rd = R.loans(src, model), R.derivatives(src, model)

st.title("Risk Data Aggregator")
st.caption("Aggregation of credit, market, liquidity and operational risk data from several source systems into one "
           "risk data model — with lineage, data-quality controls, reconciliation to the general ledger, drill-down "
           "reporting and a BCBS 239 self-assessment.")

tabs = st.tabs(["Overview", "Pipeline & lineage", "Data quality", "Reconciliation", "Credit", "Market", "Liquidity",
                "Operational", "Ad hoc report", "BCBS 239", "Import CSV", "Methodology"])

# ---------------------------------------------------------------- overview
with tabs[0]:
    cr = model.credit
    mk = A.market(model)
    lq = A.liquidity(model)
    op = A.oprisk(model)
    le = A.large_exposures(cr, model.market, src.ref_cpty, tier1)
    dq_score = model.dq_results["pass rate"].mean()
    c = st.columns(4)
    c[0].metric("Credit EAD", f"€{cr['ead_eur'].sum() / 1e9:,.2f} bn", f"RWA €{cr['rwa_eur'].sum() / 1e9:,.2f} bn", delta_color="off")
    c[1].metric("Market VaR 99 % 1d", f"€{mk.attrs['total VaR'] / M:,.2f} m", f"sum of desks €{mk.attrs['sum of VaRs'] / M:,.2f} m", delta_color="off")
    c[2].metric("Group LCR", f"{lq.loc['Group', 'LCR']:.0%}", f"NSFR {lq.loc['Group', 'NSFR']:.0%}", delta_color="off")
    c[3].metric("Op. losses (12 m, net)", f"€{op['recent']['net_loss_eur'].sum() / M:,.1f} m")
    c = st.columns(4)
    c[0].metric("Data-quality score", f"{dq_score:.2%}", f"{int((model.dq_results['status'] == 'fail').sum())} blocking rules failing",
                delta_color="off")
    c[1].metric("Reconciliation", f"{int((rl['status'] == 'reconciled').sum())} / {len(rl)} cells",
                f"{int((rl['status'] == 'break').sum())} break(s)", delta_color="off")
    c[2].metric("Records quarantined", f"{len(model.quarantine):,}", f"of {model.run_log['ingested'].sum():,} ingested", delta_color="off")
    c[3].metric("Large exposures (≥ 10 % T1)", f"{int((le['% of Tier 1'] >= 0.10).sum())}", f"{int((le['% of Tier 1'] > 0.25).sum())} above 25 %", delta_color="off")
    alerts = []
    for g_, r_ in le[le["status"] != ""].iterrows():
        alerts.append(("Credit", f"Group {g_}: {r_['% of Tier 1']:.1%} of Tier 1 — {r_['status']}", RED if "breach" in r_["status"] else ORANGE))
    for d_, r_ in mk.iterrows():
        if r_["utilisation"] >= 0.8:
            alerts.append(("Market", f"{d_}: VaR at {r_['utilisation']:.0%} of limit", RED if r_["utilisation"] >= 1 else ORANGE))
    for e_, r_ in lq.iterrows():
        if r_["NSFR"] < 1 or r_["LCR"] < 1:
            alerts.append(("Liquidity", f"{e_}: LCR {r_['LCR']:.0%}, NSFR {r_['NSFR']:.0%} (minimum 100 %)", RED))
    for (e_, ccy), r_ in rl[rl["status"] == "break"].iterrows():
        alerts.append(("Data", f"Unexplained risk / finance break {e_} {ccy}: {r_['unexplained']:,.0f}", RED))
    for f_ in src.feeds[src.feeds["latest as-of"] < S.ASOF].itertuples():
        alerts.append(("Data", f"Feed {f_.source} ({f_.system}) as of {f_._4:%d %b} — stale", ORANGE))
    a, b = st.columns([2, 3])
    with a:
        st.subheader("Alerts")
        for cat, msg, col in alerts:
            st.markdown(f"<span style='color:{col}'>●</span> **{cat}** — {msg}", unsafe_allow_html=True)
    with b:
        g = A.credit(cr, ["entity", "business_line"]).reset_index()
        fig = px.bar(g, x="entity", y="EAD", color="business_line", title="Credit EAD by entity and business line (€)",
                     color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_layout(height=360, margin=dict(t=40, b=10), legend=dict(orientation="h", y=-0.2, title=""))
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- pipeline & lineage
with tabs[1]:
    rlog = model.run_log
    srcs = ["LOANS", "TRADES", "TREASURY", "OPLOSS"]
    labels = srcs + ["Data-quality controls", "Quarantine", "Risk data model", "Credit reports", "Market reports",
                     "Liquidity reports", "Operational reports", "Reconciliation (GL)"]
    ix = {l: i for i, l in enumerate(labels)}
    s_, t_, v_, cl = [], [], [], []
    for _, r_ in rlog.iterrows():
        s_.append(ix[r_["dataset"]]); t_.append(ix["Data-quality controls"]); v_.append(r_["ingested"]); cl.append("rgba(31,111,139,0.35)")
    s_.append(ix["Data-quality controls"]); t_.append(ix["Quarantine"]); v_.append(rlog["rejected (blocking)"].sum()); cl.append("rgba(184,67,58,0.5)")
    s_.append(ix["Data-quality controls"]); t_.append(ix["Risk data model"]); v_.append(rlog["loaded"].sum()); cl.append("rgba(59,140,90,0.4)")
    for rep, ds in [("Credit reports", "LOANS"), ("Market reports", "TRADES"), ("Liquidity reports", "TREASURY"), ("Operational reports", "OPLOSS")]:
        s_.append(ix["Risk data model"]); t_.append(ix[rep]); v_.append(int(rlog.set_index("dataset").loc[ds, "loaded"])); cl.append("rgba(154,165,177,0.4)")
    s_.append(ix["Risk data model"]); t_.append(ix["Reconciliation (GL)"]); v_.append(int(model.credit["on_balance"].sum())); cl.append("rgba(212,137,43,0.4)")
    fig = go.Figure(go.Sankey(node=dict(label=labels, pad=18, thickness=16,
                                        color=[BLUE] * 4 + [GREY, RED, GREEN, GREY, GREY, GREY, GREY, ORANGE]),
                              link=dict(source=s_, target=t_, value=v_, color=cl)))
    fig.update_layout(title="Records from source systems to reports (reference data REF, market data MKT and GL feed the enrichment and checks)",
                      height=420, margin=dict(t=50, b=10))
    st.plotly_chart(fig, width="stretch")
    a, b = st.columns([3, 2])
    a.markdown("**Run log**")
    a.dataframe(rlog.set_index("dataset"), width="stretch", height=h(rlog))
    b.markdown("**Source feeds**")
    b.dataframe(text(src.feeds.assign(status=np.where(src.feeds["latest as-of"] < S.ASOF, "stale", "on time")),
                     {"latest as-of": "{:%d %b %Y}"}), hide_index=True, width="stretch", height=h(src.feeds))
    st.subheader("Record-level lineage")
    c1, c2 = st.columns(2)
    kind = c1.radio("Record type", ["Loan", "Trade"], horizontal=True)
    if kind == "Loan":
        lid = c2.selectbox("Loan", model.credit["loan_id"].tolist()[:400])
        rec = model.credit.set_index("loan_id").loc[lid]
        raw = src.loans[src.loans["loan_id"] == lid].iloc[0]
        steps = rec["lineage"].split(" → ")
        st.markdown(" → ".join(f"`{s}`" for s in steps))
        a, b = st.columns(2)
        a.markdown("**Source record (LOANS)**")
        a.dataframe(raw.astype(str).rename("value").to_frame(), width="stretch", height=420)
        b.markdown("**Risk data model record**")
        cols = ["entity", "lei", "name", "group", "country", "sector", "rating", "currency", "fx_rate", "outstanding_eur", "ead_eur", "rw",
                "rwa_eur", "el_eur", "warnings"]
        b.dataframe(rec[cols].astype(str).rename("value").to_frame(), width="stretch", height=420)
        st.caption(f"Used in: credit dashboard (entity {rec['entity']}, sector {rec['sector']}, rating {rec['rating']}); large exposures "
                   f"(group {rec['group']}); RWA; risk / finance reconciliation ({rec['entity']} / {rec['currency']}).")
    else:
        tid = c2.selectbox("Trade", model.market["trade_id"].tolist()[:400])
        rec = model.market.set_index("trade_id").loc[tid]
        st.markdown(" → ".join(f"`{s}`" for s in rec["lineage"].split(" → ")))
        p = model.pnl[tid]
        fig = px.line(p, title=f"P&L vector of {tid} (€) — input of the VaR aggregation", color_discrete_sequence=[BLUE])
        fig.update_layout(height=280, margin=dict(t=40, b=10), showlegend=False, xaxis_title="")
        st.plotly_chart(fig, width="stretch")
    st.subheader("Quarantine")
    st.dataframe(model.quarantine, hide_index=True, width="stretch", height=300)

# ---------------------------------------------------------------- data quality
with tabs[2]:
    res = model.dq_results
    sc = dq.scores(res)
    a, b = st.columns([3, 2])
    fig = px.imshow(sc, text_auto=".1%", color_continuous_scale="RdYlGn", zmin=0.9, zmax=1.0, aspect="auto",
                    title="Data-quality score by dataset and dimension")
    fig.update_layout(height=360, margin=dict(t=40, b=10))
    a.plotly_chart(fig, width="stretch")
    with b:
        st.metric("Overall score (mean pass rate)", f"{res['pass rate'].mean():.2%}")
        st.metric("Rules failing (blocking / warning)", f"{int((res['status'] == 'fail').sum())} / {int((res['status'] == 'warning').sum())}")
        dd = src.defects
        if len(dd):
            st.metric("Injected defects detected", f"{len(dd)}", help="Ground truth of the synthetic sources: every injected defect type "
                                                                     "is caught by at least one rule.")
    st.dataframe(text(res, {"pass rate": "{:.2%}", "threshold": "{:.1%}"}).style.apply(
        lambda r: ["background-color: #f6dcd9" if r["status"] == "fail" else ("background-color: #fbecd6" if r["status"] == "warning" else "")] * len(r), axis=1),
        hide_index=True, width="stretch", height=h(res))
    rid = st.selectbox("Failing records of rule", res.loc[res["failed"] > 0, "rule"].tolist() or ["–"])
    if rid != "–":
        fr = dq.failing_records(src, model.dq_fails, rid)
        st.dataframe(fr.head(200), width="stretch", height=min(h(fr), 380))
    st.subheader("Issue log")
    owners = src.feeds.set_index("source")["data owner"]
    issues = res[res["status"] != "pass"].assign(owner=lambda d: d["dataset"].map(lambda x: owners.get(x.split(" ")[0], "Data office")),
                                                  due=(S.ASOF + pd.Timedelta(days=10)).date(), state="open")
    st.dataframe(issues[["rule", "dataset", "description", "failed", "severity", "owner", "due", "state"]], hide_index=True, width="stretch",
                 height=h(issues))

# ---------------------------------------------------------------- reconciliation
with tabs[3]:
    st.markdown("**Loans and advances** — general ledger gross carrying amount vs risk on-balance outstanding, by entity and "
                "currency (original currency). The difference is explained by accrued interest (accounted, not in the risk "
                "exposure) and by records quarantined by the data-quality controls; the residual must stay within the "
                f"tolerance (max of {R.TOL_REL:.1%} of the balance and € {R.TOL_ABS / M:.0f} m).")
    st.dataframe(text(rl.reset_index(), {c: "{:,.0f}" for c in ["GL", "risk outstanding", "accrued interest", "rejected by DQ", "difference",
                                                               "unexplained", "tolerance", "unexplained (EUR)"]}).style.apply(
        lambda r: ["background-color: #f6dcd9" if r["status"] == "break" else ""] * len(r), axis=1), hide_index=True, width="stretch", height=h(rl))
    br = rl[rl["status"] == "break"]
    pick = st.selectbox("Bridge for", [f"{e} / {c}" for e, c in rl.index], index=[f"{e} / {c}" for e, c in rl.index].index(f"{br.index[0][0]} / {br.index[0][1]}") if len(br) else 0)
    e, ccy = pick.split(" / ")
    row = rl.loc[(e, ccy)]
    fig = go.Figure(go.Waterfall(x=["Risk outstanding", "Accrued interest", "Rejected by DQ", "Unexplained", "GL balance"],
                                 y=[row["risk outstanding"], row["accrued interest"], row["rejected by DQ"], row["unexplained"], row["GL"]],
                                 measure=["absolute", "relative", "relative", "relative", "total"],
                                 increasing=dict(marker_color=ORANGE), decreasing=dict(marker_color=GREEN), totals=dict(marker_color=BLUE)))
    lo = min(row["risk outstanding"], row["GL"]) * 0.97
    fig.update_layout(title=f"Bridge from risk to finance — {e} {ccy}", height=380, margin=dict(t=40, b=10), yaxis_range=[lo, max(row["GL"], row["risk outstanding"]) * 1.02])
    st.plotly_chart(fig, width="stretch")
    st.markdown("**Derivatives** — GL net fair value vs sum of risk MtM (EUR)")
    rd = rd.copy()
    num = rd.select_dtypes("number").columns
    rd[num] = rd[num].round(0) + 0.0
    st.dataframe(text(rd.reset_index(), {c: "{:,.0f}" for c in ["GL (EUR)", "risk MtM (EUR)", "difference"]}), hide_index=True, width="stretch", height=h(rd))

# ---------------------------------------------------------------- credit
with tabs[4]:
    c = st.columns(5)
    c[0].metric("EAD", f"€{cr['ead_eur'].sum() / 1e9:,.2f} bn")
    c[1].metric("RWA (standardised)", f"€{cr['rwa_eur'].sum() / 1e9:,.2f} bn")
    c[2].metric("Expected loss", f"€{cr['el_eur'].sum() / M:,.1f} m")
    c[3].metric("HHI (groups)", f"{A.hhi(cr.groupby('group')['ead_eur'].sum()):.3f}")
    c[4].metric("Large exposures (≥ 10 % T1)", f"{int((le['% of Tier 1'] >= 0.10).sum())}")
    path = st.multiselect("Drill-down hierarchy", ["entity", "business_line", "country", "sector", "rating", "group", "product"],
                          default=["entity", "sector", "group"])
    if path:
        tm = cr.groupby(path, dropna=False)["ead_eur"].sum().reset_index()
        tm = tm[tm["ead_eur"] > 0]
        fig = px.treemap(tm, path=[px.Constant("Group")] + path, values="ead_eur", color_discrete_sequence=px.colors.qualitative.Safe,
                         title="EAD — click to drill down")
        fig.update_layout(height=480, margin=dict(t=40, b=10))
        st.plotly_chart(fig, width="stretch")
        ag = A.credit(cr, path)
        st.dataframe(text(ag.reset_index(), {"outstanding": "{:,.0f}", "EAD": "{:,.0f}", "RWA": "{:,.0f}", "EL": "{:,.0f}", "provisions": "{:,.0f}",
                                             "RW density": "{:.0%}", "share of EAD": "{:.1%}"}), hide_index=True, width="stretch", height=380)
    st.subheader("Large exposures (groups of connected clients)")
    st.dataframe(text(le.head(15).reset_index(), {"loans EAD": "{:,.0f}", "derivatives (positive MtM)": "{:,.0f}", "exposure": "{:,.0f}",
                                                  "% of Tier 1": "{:.1%}"}), hide_index=True, width="stretch", height=h(le.head(15)))
    grp = st.selectbox("Transactions of group", le.index[:15].tolist())
    st.dataframe(text(cr[cr["group"] == grp][["loan_id", "entity", "name", "lei", "product", "currency", "outstanding_eur", "ead_eur", "rating", "rwa_eur",
                                              "source", "warnings"]], {"outstanding_eur": "{:,.0f}", "ead_eur": "{:,.0f}", "rwa_eur": "{:,.0f}"}),
                 hide_index=True, width="stretch", height=300)

# ---------------------------------------------------------------- market
with tabs[5]:
    c = st.columns(4)
    c[0].metric("Group VaR 99 % 1d", f"€{mk.attrs['total VaR'] / M:,.2f} m")
    c[1].metric("Sum of desk VaRs", f"€{mk.attrs['sum of VaRs'] / M:,.2f} m")
    c[2].metric("Diversification benefit", f"{1 - mk.attrs['total VaR'] / mk.attrs['sum of VaRs']:.0%}")
    c[3].metric("Group ES 97.5 % 1d", f"€{mk.attrs['total ES'] / M:,.2f} m")
    a, b = st.columns(2)
    fig = go.Figure()
    fig.add_bar(x=mk.index, y=mk["VaR 99 % 1d"] / M, name="VaR", marker_color=[RED if u >= 1 else ORANGE if u >= 0.8 else BLUE for u in mk["utilisation"]])
    fig.add_scatter(x=mk.index, y=mk["limit"] / M, mode="markers", marker=dict(symbol="line-ew-open", size=40, color="#222"), name="limit")
    fig.update_layout(title="VaR 99 % 1d by desk vs limit (€ m)", height=380, margin=dict(t=40, b=10), legend=dict(orientation="h", y=-0.2))
    a.plotly_chart(fig, width="stretch")
    me = A.market(model, "entity")
    fig = go.Figure(go.Bar(x=me.index, y=me["VaR 99 % 1d"] / M, marker_color=BLUE))
    fig.add_hline(y=me.attrs["total VaR"] / M, line_dash="dot", annotation_text="group VaR (diversified)")
    fig.update_layout(title="VaR by legal entity (€ m)", height=380, margin=dict(t=40, b=10))
    b.plotly_chart(fig, width="stretch")
    st.dataframe(text(mk.reset_index(), {"MtM": "{:,.0f}", "VaR 99 % 1d": "{:,.0f}", "ES 97.5 % 1d": "{:,.0f}", "limit": "{:,.0f}", "utilisation": "{:.0%}"}),
                 hide_index=True, width="stretch", height=h(mk))
    sens = A.sensitivities(model)
    fig = px.imshow(sens / 1e3, text_auto=",.0f", color_continuous_scale="RdBu", color_continuous_midpoint=0, aspect="auto",
                    title="Net sensitivities by risk factor and desk (k€ per bp for rates and credit, per 100 % for FX and equity)")
    fig.update_layout(height=380, margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("VaRs are aggregated from P&L vectors (sum of the vectors, then quantile), never by adding VaRs: the gap between "
               "the sum of desk VaRs and the group VaR is the diversification benefit.")

# ---------------------------------------------------------------- liquidity
with tabs[6]:
    st.dataframe(text(lq.reset_index(), {"HQLA": "{:,.0f}", "outflows": "{:,.0f}", "inflows (capped)": "{:,.0f}", "LCR": "{:.0%}",
                                         "ASF": "{:,.0f}", "RSF": "{:,.0f}", "NSFR": "{:.0%}"}), hide_index=True, width="stretch", height=h(lq))
    a, b = st.columns(2)
    for col_, ratio in ((a, "LCR"), (b, "NSFR")):
        fig = go.Figure(go.Bar(x=lq.index, y=lq[ratio], marker_color=[RED if x < 1 else GREEN for x in lq[ratio]]))
        fig.add_hline(y=1.0, line_dash="dash", annotation_text="minimum 100 %")
        fig.update_layout(title=ratio, height=340, margin=dict(t=40, b=10), yaxis_tickformat=".0%")
        col_.plotly_chart(fig, width="stretch")
    st.caption("Simplified Basel III factors: LCR outflow rates (retail stable 5 %, less stable 10 %, operational 25 %, non-operational "
               "corporates 40 %, financials 100 %, committed facilities 10 %), inflows capped at 75 % of outflows, Level 2 caps; NSFR "
               "ASF and RSF factors. Transferability of liquidity between entities is not modelled.")

# ---------------------------------------------------------------- operational
with tabs[7]:
    rec_ = op["recent"]
    c = st.columns(4)
    c[0].metric("Net losses, 12 months", f"€{rec_['net_loss_eur'].sum() / M:,.1f} m")
    c[1].metric("Events, 12 months", f"{len(rec_)}")
    c[2].metric("Largest event", f"€{rec_['net_loss_eur'].max() / M:,.1f} m" if len(rec_) else "–")
    c[3].metric("Average annual loss (3 y)", f"€{op['annual']['sum'].mean() / M:,.1f} m")
    fig = px.imshow(op["matrix"] / 1e3, text_auto=",.0f", aspect="auto", color_continuous_scale="Oranges",
                    title="Net losses by Basel event type and business line, last 12 months (k€)")
    fig.update_layout(height=420, margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")
    a, b = st.columns(2)
    an = op["annual"].reset_index()
    a.plotly_chart(px.bar(an, x="year", y="sum", title="Net losses by year (€)", color_discrete_sequence=[ORANGE])
                   .update_layout(height=320, margin=dict(t=40, b=10)), width="stretch")
    b.markdown("**Largest events (12 months)**")
    b.dataframe(text(op["top"][["event_id", "entity", "business_line", "event_type", "event_date", "net_loss_eur"]],
                     {"net_loss_eur": "{:,.0f}", "event_date": "{:%d %b %Y}"}), hide_index=True, width="stretch", height=320)

# ---------------------------------------------------------------- ad hoc
with tabs[8]:
    st.markdown("Build a report on the fly from the risk data model (BCBS 239 principle 6, adaptability) — for instance a "
                "stress question: *what is our exposure to real estate in Germany and the UK, by entity, and how much "
                "would expected loss rise if PDs tripled?*")
    c = st.columns(3)
    f_ent = c[0].multiselect("Entity", sorted(cr["entity"].unique()))
    f_cty = c[1].multiselect("Country", sorted(cr["country"].dropna().unique()))
    f_sec = c[2].multiselect("Sector", sorted(cr["sector"].dropna().unique()))
    c = st.columns(3)
    f_rat = c[0].multiselect("Rating", S.RATINGS)
    f_bl = c[1].multiselect("Business line", sorted(cr["business_line"].unique()))
    f_pr = c[2].multiselect("Product", sorted(cr["product"].unique()))
    c = st.columns(3)
    by = c[0].multiselect("Group by", ["entity", "country", "sector", "rating", "business_line", "product", "group"], default=["entity"])
    pdm = c[1].slider("Stress: PD multiplier", 1.0, 5.0, 1.0, 0.5)
    t0 = time.time()
    sel = cr.copy()
    for col_, vals in (("entity", f_ent), ("country", f_cty), ("sector", f_sec), ("rating", f_rat), ("business_line", f_bl), ("product", f_pr)):
        if vals:
            sel = sel[sel[col_].isin(vals)]
    sel = sel.assign(el_stressed=sel["ead_eur"] * np.minimum(sel["pd"] * pdm, 1.0) * sel["lgd"])
    rep = (sel.groupby(by).agg(loans=("loan_id", "count"), EAD=("ead_eur", "sum"), RWA=("rwa_eur", "sum"), EL=("el_eur", "sum"),
                               EL_stressed=("el_stressed", "sum")) if by else sel[["ead_eur", "rwa_eur", "el_eur", "el_stressed"]].sum().to_frame("total").T)
    secs = time.time() - t0
    st.session_state["adhoc_seconds"] = secs
    c[2].metric("Produced in", f"{secs:.2f} s", f"{len(sel)} transactions", delta_color="off")
    st.dataframe(text(rep.reset_index(), {k: "{:,.0f}" for k in ["EAD", "RWA", "EL", "EL_stressed", "ead_eur", "rwa_eur", "el_eur", "el_stressed"]}),
                 hide_index=True, width="stretch", height=min(h(rep), 460))
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        rep.to_excel(w, sheet_name="report")
        sel.drop(columns=["lineage"]).to_excel(w, sheet_name="transactions", index=False)
        pd.DataFrame({"filter": ["entity", "country", "sector", "rating", "business line", "product", "PD multiplier", "reporting date"],
                      "value": [", ".join(f_ent) or "all", ", ".join(f_cty) or "all", ", ".join(f_sec) or "all", ", ".join(f_rat) or "all",
                                ", ".join(f_bl) or "all", ", ".join(f_pr) or "all", pdm, f"{S.ASOF:%Y-%m-%d}"]}).to_excel(w, sheet_name="parameters", index=False)
    c1, c2 = st.columns(2)
    c1.download_button("Download (Excel, with transactions and parameters)", buf.getvalue(), "adhoc_report.xlsx")
    c2.download_button("Download (CSV)", rep.to_csv().encode(), "adhoc_report.csv", "text/csv")

# ---------------------------------------------------------------- BCBS 239
with tabs[9]:
    ga = G.assess(model, rl, rd, src.feeds, S.ASOF, st.session_state.get("adhoc_seconds"))
    st.markdown("**Self-assessment of the 14 principles** — automatic ratings come from the current run; manual ratings can be "
                "edited. 4 = fully compliant, 3 = largely compliant, 2 = materially non-compliant, 1 = non-compliant.")
    ga = ga[["title", "rating", "basis", "evidence", "block"]]
    ed = st.data_editor(ga, width="stretch", height=h(ga), disabled=["title", "block", "evidence", "basis"], key="g_ed",
                        column_config={"rating": st.column_config.NumberColumn(min_value=1, max_value=4, step=1)})
    a, b = st.columns(2)
    sc2 = ed.dropna(subset=["rating"]).groupby("block")["rating"].mean()
    fig = go.Figure(go.Bar(x=sc2.values, y=sc2.index, orientation="h", marker_color=[GREEN if v >= 3.5 else ORANGE if v >= 2.5 else RED for v in sc2.values]))
    fig.update_layout(title="Average rating by block", height=300, margin=dict(t=40, b=10), xaxis_range=[0, 4])
    a.plotly_chart(fig, width="stretch")
    with b:
        st.markdown("**Remediation priorities**")
        for p_, r_ in ed.dropna(subset=["rating"]).sort_values("rating").head(4).iterrows():
            st.markdown(f"* **P{p_} {r_['title']}** (rating {int(r_['rating'])}) — {r_['evidence']}")
    st.subheader("Data dictionary — critical data elements")
    st.dataframe(G.DICTIONARY, hide_index=True, width="stretch", height=h(G.DICTIONARY))
    st.download_button("Download the self-assessment (CSV)", ed.to_csv().encode(), "bcbs239_self_assessment.csv", "text/csv")

# ---------------------------------------------------------------- import
with tabs[10]:
    st.markdown("Replace any source table with your own file (same columns as the template). The pipeline, the controls, the "
                "reconciliation and every report then run on it.")
    name = st.selectbox("Table", list(IO.TABLES))
    st.download_button(f"Template — {name}", IO.template(src, name), f"{name.replace(' ', '_').lower()}_template.csv", "text/csv")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None and st.button("Load into the pipeline", type="primary"):
        try:
            df = pd.read_csv(up)
            IO.replace_table(src, name, df)
            imp = st.session_state.get("imported", {}) if st.session_state.get("imp_base") == ("synthetic", seed, rate, late) else {}
            imp[name] = df
            st.session_state["imported"] = imp
            st.session_state["imp_base"] = ("synthetic", seed, rate, late)
            st.success(f"{len(df)} rows loaded into {name}.")
            st.rerun()
        except ValueError as e:
            st.error(str(e))
    if st.session_state.get("imported"):
        st.info("Imported tables: " + ", ".join(st.session_state["imported"]))
        if st.button("Back to the synthetic sources"):
            st.session_state.pop("imported")
            st.rerun()

# ---------------------------------------------------------------- methodology
with tabs[11]:
    st.markdown(r"""
### Architecture
1. **Sources.** Reference data (counterparties with LEI, group, sector, country, rating; local code mapping; books), loans,
   trades with sensitivities, market data, treasury, operational losses and the general ledger, each with its own
   conventions and with injected defects.
2. **Controls.** 23 data-quality rules across completeness, accuracy, consistency, integrity, timeliness and uniqueness
   (BCBS 239 principles 3–5). Blocking failures go to quarantine; warnings are loaded and flagged.
3. **Integration.** Enrichment from the reference data, conversion to EUR and derived measures: EAD = drawn + CCF ×
   undrawn, standardised RWA (CRE20), P&L vectors = sensitivities × historical moves.
4. **Lineage.** Every record keeps its source key and the list of transformations applied to it.
5. **Reconciliation.** Risk is reconciled to the general ledger, with an explained bridge (accrued interest, quarantined
   records) and a residual tested against a tolerance.
6. **Reporting.** Credit (drill-down, large exposures at 10 % / 25 % of Tier 1), market (VaR and ES aggregated from P&L
   vectors, limits), liquidity (LCR, NSFR) and operational risk (Basel event types × business lines). An ad hoc report
   builder with export completes it.
7. **Governance.** Self-assessment of the 14 principles from the run's evidence, plus a data dictionary of critical data
   elements.

### References
* Basel Committee (2013), *Principles for effective risk data aggregation and risk reporting* (BCBS 239).
* Basel Committee (2023), *Progress in adopting the Principles for effective risk data aggregation and risk reporting*.
* ECB (2024), *Guide on effective risk data aggregation and risk reporting*.
* CRR (Regulation (EU) No 575/2013) art. 392–395 (large exposures); Basel III LCR (2013) and NSFR (2014); CRE20 (standardised
  approach for credit risk).

### Limits
* Synthetic data and simplified regulatory calculations (no CRM, no netting, simplified LCR / NSFR).
* Single reporting date.
* No user management or workflow on the issue log.
""")
