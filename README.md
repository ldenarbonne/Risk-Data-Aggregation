# Risk Data Aggregator – BCBS 239 Risk Data Aggregation and Risk Reporting

An interactive Streamlit application that aggregates **credit, market, liquidity and operational risk** data from several source systems of a banking group into one risk data model. It covers:

* **pipeline and lineage**: ingestion → data-quality controls → quarantine → enrichment → risk data model → reports, with the source key and the transformations of every record;
* **data quality**: 23 rules across completeness, accuracy, consistency, integrity, timeliness and uniqueness, with scores, failing records and an issue log;
* **reconciliation** of risk data with the general ledger: an explained bridge (accrued interest, quarantined records) and a residual tested against a tolerance;
* **risk reporting with drill-down**:
  * credit (EAD, RWA, large exposures);
  * market (VaR aggregated from P&L vectors, limits, sensitivities);
  * liquidity (LCR, NSFR);
  * operational losses;
* an **ad hoc report builder** with a PD stress and Excel export;
* **BCBS 239 governance**: self-assessment of the 14 principles from the run's evidence, and a data dictionary of critical data elements;
* **CSV import** of any source table.

**Live app:** `https://<your-app-name>.streamlit.app` *(to be completed after deployment)*

![tests](https://github.com/ldenarbonne/risk-data-aggregator/actions/workflows/tests.yml/badge.svg)

---

## 1. Source systems (synthetic) and injected defects

| Source | Content | Injected defects |
|---|---|---|
| REF | 220 counterparties (LEI, group, country, sector, rating), local-code mapping, books / desks | – |
| LOANS | 1,500 loans in 4 legal entities, local codes and currencies | unmapped codes, missing currency, PD out of range, negative amounts, matured loans, duplicates |
| TRADES | 900 trades with MtM and sensitivities | missing LEI, unknown book, MtM unit error (× 1,000), late feed (stale as-of date) |
| MKT | 500 days of risk-factor moves | – |
| TREASURY | HQLA, deposits, inflows and funding by entity | – |
| OPLOSS | 260 operational loss events | future event dates, recovery above the gross loss |
| GL | General ledger balances | one unexplained risk / finance break |

The defect rate and the late feed are set in the sidebar. A test checks that **every injected defect is caught by its rule**.

## 2. Tabs

| Tab | Content |
|---|---|
| **Overview** | Key indicators by risk type, data quality and reconciliation; alerts (limit breaches, regulatory ratios, breaks, stale feeds) |
| **Pipeline & lineage** | Sankey diagram from the sources to the reports, run log, feed status. Record-level lineage (source record → transformations → model record → reports). Quarantine |
| **Data quality** | Scores by dataset × dimension, rule results, failing records per rule, issue log with data owners |
| **Reconciliation** | Loans GL vs risk by entity × currency, with the bridge (waterfall) for any cell; derivatives fair value vs MtM |
| **Credit** | Treemap drill-down along any hierarchy (entity, business line, country, sector, rating, group, product). Large exposures vs Tier 1 (10 % / 25 %). Transactions of a group |
| **Market** | Group and desk VaR, ES, diversification, limits, VaR by entity, sensitivities heatmap |
| **Liquidity** | LCR and NSFR by entity and for the group |
| **Operational** | Net losses by Basel event type × business line, annual losses, largest events |
| **Ad hoc report** | Filters, group-by, PD stress, timing, Excel export (report, transactions and parameters) and CSV export |
| **BCBS 239** | 14-principle self-assessment (automatic evidence and editable ratings), average rating by block, remediation priorities, data dictionary |
| **Import CSV** | Template and upload for any source table |

## 3. Illustrative results (default settings)

* **Data quality:**
  * 2,746 records ingested, 150 quarantined by 7 failing blocking rules;
  * overall score 98.5 %;
  * the stale trading feed brings the TRADES timeliness score down to 86 %.
* **Reconciliation:** 12 of 13 entity × currency cells reconcile after explaining accrued interest and quarantined records. The injected €12.5 m break (Luxembourg, EUR) is isolated.
* **Credit:**
  * EAD €13.0 bn, standardised RWA €10.1 bn;
  * one group at 28.1 % of Tier 1, above the 25 % large-exposure limit, and one at 13.1 %.
* **Market:**
  * group VaR (99 %, 1 day) €2.0 m, against €3.5 m for the sum of desk VaRs (42 % diversification);
  * the credit trading desk is at 106 % of its limit.
* **Liquidity:** group LCR 233 % and NSFR 113 %; the Luxembourg entity has an NSFR of 95 %.
* **BCBS 239 self-assessment:** the weakest principles are completeness (P4) and accuracy (P3), driven by the rejected records, and distribution (P11), which is not yet documented.

## 4. Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests (9 tests, a few seconds) cover:

* detection of every injected defect;
* clean sources;
* record counts and lineage;
* EAD and RWA identities;
* the reconciliation isolating exactly the injected break;
* VaR aggregation;
* large exposures and liquidity;
* governance and CSV import;
* the app.

```bash
pip install -r requirements-dev.txt
pytest -q
```

## 5. Repository structure

```
app.py               Streamlit application (12 tabs)
rda/sources.py       Synthetic source systems with injected defects
rda/dq.py            Data-quality rule catalogue and engine
rda/pipeline.py      Controls, quarantine, enrichment, FX, derived measures, lineage, run log
rda/recon.py         Risk / finance reconciliation with explained bridge
rda/aggregate.py     Credit, large exposures, market VaR / ES, sensitivities, LCR / NSFR, operational losses
rda/governance.py    BCBS 239 self-assessment and data dictionary
rda/io.py            CSV import and templates
tests/               pytest suite
```

## 6. References

* Basel Committee on Banking Supervision (2013), *Principles for effective risk data aggregation and risk reporting* (BCBS 239).
* Basel Committee (2023), *Progress in adopting the Principles for effective risk data aggregation and risk reporting*.
* European Central Bank (2024), [*Guide on effective risk data aggregation and risk reporting*](https://www.ssm.europa.eu/ecb/pub/pdf/ssm.supervisory_guides240503_riskreporting.en.pdf).
* Regulation (EU) No 575/2013 (CRR), art. 392–395 (large exposures); Basel III LCR (2013) and NSFR (2014); CRE20 (standardised approach).

## 7. Author

**Laurent Motais de Narbonne** – Risk management and quantitative finance · [ldenarbonne.github.io](https://ldenarbonne.github.io/)

Licence: MIT.
