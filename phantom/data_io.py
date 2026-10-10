"""Load the plastic tables (features only; labels are loaded separately by evaluate.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "synthetic"


def load_reference() -> dict:
    return yaml.safe_load((Path(__file__).parent / "reference.yaml").read_text(encoding="utf-8"))


def _fy_fast(dates: pd.Series) -> pd.Series:
    d = pd.to_datetime(dates)
    y = (d.dt.year - (d.dt.month < 4).astype(int)).to_numpy()
    codes, uniq = pd.factorize(y)
    labels = np.array([f"FY{u}-{str(u + 1)[2:]}" for u in uniq])
    return pd.Series(labels[codes], index=dates.index)


def load_all(data_dir: Path | str = DEFAULT_DATA) -> dict[str, pd.DataFrame]:
    d = Path(data_dir)
    t: dict[str, pd.DataFrame] = {}
    ent = pd.read_parquet(d / "registry" / "entities.parquet")
    t["entities"] = ent[ent.stream == "plastic"].reset_index(drop=True)
    t["tariff"] = pd.read_parquet(d / "registry" / "tariff_reference.parquet")
    for name in ["collectors", "offtakers", "brands", "brand_obligations", "certificate_lots", "certificate_sales",
                 "feedstock_invoices", "output_invoices", "monthly_processing_reported", "electricity_bills"]:
        t[name] = pd.read_parquet(d / "plastic" / f"{name}.parquet")
    for k, cols in {
        "feedstock_invoices": ["invoice_date"], "output_invoices": ["invoice_date"],
        "certificate_lots": ["period_month", "generated_on"], "certificate_sales": ["sale_date"],
        "monthly_processing_reported": ["month"], "electricity_bills": ["billing_month"],
        "collectors": ["registration_date"], "offtakers": ["incorporation_date"],
        "brands": ["registration_date"],
    }.items():
        for c in cols:
            t[k][c] = pd.to_datetime(t[k][c])
    t["entities"]["incorporation_date"] = pd.to_datetime(t["entities"]["incorporation_date"])
    t["entities"]["epr_registration_date"] = pd.to_datetime(t["entities"]["epr_registration_date"])
    t["electricity_bills"]["fy"] = _fy_fast(t["electricity_bills"]["billing_month"])
    return t
