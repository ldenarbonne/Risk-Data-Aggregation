"""CSV import of a source table (same columns as the synthetic one) and templates."""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

from . import sources as S

TABLES = {"LOANS": "loans", "TRADES": "trades", "TREASURY": "treasury", "OPLOSS": "oploss", "GL": "gl",
          "REF counterparties": "ref_cpty", "REF code mapping": "ref_map", "REF books": "ref_books"}
DATES = {"loans": ["maturity_date", "asof"], "trades": ["trade_date", "asof"], "treasury": ["asof"], "oploss": ["event_date"]}


def template(src: S.Sources, name: str, n: int = 50) -> bytes:
    return getattr(src, TABLES[name]).head(n).to_csv(index=False).encode()


def replace_table(src: S.Sources, name: str, df: pd.DataFrame) -> S.Sources:
    attr = TABLES[name]
    ref = getattr(src, attr)
    miss = [c for c in ref.columns if c not in df.columns]
    if miss:
        raise ValueError(f"{name}: missing columns {', '.join(miss)}")
    df = df[list(ref.columns)].copy()
    for c in DATES.get(attr, []):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ref.columns:
        if pd.api.types.is_numeric_dtype(ref[c]) and not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return replace(src, **{attr: df, "defects": src.defects[src.defects["dataset"] != name]})
