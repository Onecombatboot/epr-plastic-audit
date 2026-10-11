"""Validate observable tables and join stream physics without touching ground truth."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "synthetic"
DEFAULT_RUN = ROOT / "yieldbal" / "outputs" / "seed42"


def load_reference() -> dict:
    return yaml.safe_load(Path(__file__).with_name("reference.yaml").read_text())


def load_configs(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    processes, products, seen = [], [], set()
    for path in sorted(Path(directory).glob("*.yaml")):
        config = yaml.safe_load(path.read_text())
        if config.get("certificate_basis") != "input_tonnes_processed":
            raise ValueError(f"Unsupported certificate basis in {path.name}")
        stream = config["stream"]
        if stream in seen:
            raise ValueError(f"Duplicate stream config: {stream}")
        seen.add(stream)
        for process, spec in config["processes"].items():
            low, high = spec["kwh_per_t"]
            if not np.isfinite([low, high]).all() or not 0 < low <= high:
                raise ValueError(f"Invalid energy range: {stream}/{process}")
            saleable = []
            for product, values in spec["outputs"].items():
                expected, tolerance = float(values["expected"]), float(values["tolerance"])
                if not np.isfinite([expected, tolerance]).all() or not 0 <= expected <= 1 or tolerance < 0:
                    raise ValueError(f"Invalid product fraction: {product}")
                if values["saleable"]:
                    saleable.append((expected, max(expected - tolerance, 0)))
                    products.append(dict(stream=stream, process_type=process, product=product,
                                         expected=expected, tolerance=tolerance, lower=max(expected-tolerance, 0)))
            lower = sum(x[1] for x in saleable)
            processes.append(dict(stream=stream, process_type=process, kwh_min=float(low), kwh_max=float(high),
                                  saleable_expected=sum(x[0] for x in saleable), saleable_lower=lower))
    if not processes:
        raise ValueError("No stream configurations found")
    return pd.DataFrame(processes), pd.DataFrame(products)


def _unique(frame: pd.DataFrame, keys: list[str], name: str) -> None:
    if frame.duplicated(keys).any():
        raise ValueError(f"Duplicate {name} keys: {keys}")


def load(data_dir: Path = DEFAULT_DATA) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    tables = {name: pd.read_parquet(data_dir / "yield" / f"{name}.parquet")
              for name in ("operations_monthly", "outputs_monthly", "electricity_bills")}
    tables["entities"] = pd.read_parquet(data_dir / "registry" / "entities.parquet")
    tables["tariffs"] = pd.read_parquet(data_dir / "registry" / "tariff_reference.parquet")
    tables["processes"], tables["products"] = load_configs(data_dir / "yield" / "stream_configs")
    for name, col in (("operations_monthly", "month"), ("outputs_monthly", "month"), ("electricity_bills", "billing_month")):
        tables[name][col] = pd.to_datetime(tables[name][col]).dt.to_period("M").dt.to_timestamp()
    for name, keys in (("entities", ["entity_id"]), ("operations_monthly", ["entity_id", "month"]),
                       ("electricity_bills", ["entity_id", "billing_month"]), ("tariffs", ["state", "fy"])):
        _unique(tables[name], keys, name)
    _unique(tables["electricity_bills"], ["bill_id"], "bill ids")
    for name, columns in {"operations_monthly": ["input_t_reported", "certificates_t"],
                          "outputs_monthly": ["tonnes_sold", "value_inr"],
                          "electricity_bills": ["units_kwh", "energy_charge_inr"]}.items():
        values = tables[name][columns].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Non-finite or negative observed values in {name}")
    ops = tables["operations_monthly"]
    if ops.empty:
        raise ValueError("No operations to audit")
    entities = tables["entities"].set_index("entity_id")
    if not ops.entity_id.isin(entities.index).all():
        raise ValueError("Operations reference an unknown entity")
    observed_entities = entities.loc[ops.entity_id.unique()]
    for column in ("declared_capacity_tpa", "connected_load_kw"):
        values = observed_entities[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"Non-positive or unknown plant physics: {column}")
    solar = observed_entities.captive_solar_kw.dropna().to_numpy(dtype=float)
    if not np.isfinite(solar).all() or (solar < 0).any():
        raise ValueError("Invalid declared solar capacity")
    if tables["tariffs"].energy_rate_inr_per_kwh.le(0).any():
        raise ValueError("Tariff reference rates must be positive")
    for name in ("outputs_monthly", "electricity_bills"):
        if not tables[name].entity_id.isin(ops.entity_id).all():
            raise ValueError(f"Unknown audited entity in {name}")
    for col in ("stream", "process_type"):
        if not ops[col].eq(ops.entity_id.map(entities[col])).all():
            raise ValueError(f"Registry and operations disagree on {col}")
    keys = pd.MultiIndex.from_frame(tables["processes"][["stream", "process_type"]])
    if not pd.MultiIndex.from_frame(ops[["stream", "process_type"]]).isin(keys).all():
        raise ValueError("A stream/process has no configuration")
    observed_physics = ops[["stream", "process_type"]].drop_duplicates().merge(tables["processes"])
    if observed_physics.saleable_lower.le(0).any():
        raise ValueError("An observed process has no positive saleable output lower bound")
    return tables


def monthly_frame(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ops = tables["operations_monthly"]
    end = max(ops.month.max(), tables["outputs_monthly"].month.max(), tables["electricity_bills"].billing_month.max())
    months = pd.date_range(ops.month.min(), end, freq="MS")
    grid = pd.MultiIndex.from_product([sorted(ops.entity_id.unique()), months], names=["entity_id", "month"])
    frame = ops.set_index(["entity_id", "month"]).reindex(grid).reset_index()
    frame["operation_present"] = frame.certificates_t.notna()
    frame = frame.drop(columns=["stream", "process_type"])
    cols = ["entity_id", "stream", "process_type", "state", "legal_name", "declared_capacity_tpa",
            "connected_load_kw", "captive_solar_kw", "epr_registration_date"]
    frame = frame.merge(tables["entities"][cols], on="entity_id", validate="many_to_one")
    registered = pd.to_datetime(frame.epr_registration_date).dt.to_period("M").dt.to_timestamp()
    frame["active"] = frame.month >= registered
    frame[["input_t_reported", "certificates_t"]] = frame[["input_t_reported", "certificates_t"]].fillna(0)
    frame = frame.merge(tables["processes"], on=["stream", "process_type"], validate="many_to_one")
    bills = tables["electricity_bills"].rename(columns={"billing_month": "month"})
    frame = frame.merge(bills, on=["entity_id", "month"], how="left", validate="one_to_one")
    year = frame.month.dt.year - (frame.month.dt.month < 4).astype(int)
    frame["fy"] = "FY" + year.astype(str) + "-" + ((year+1) % 100).astype(str).str.zfill(2)
    frame = frame.merge(tables["tariffs"], on=["state", "fy"], how="left", validate="many_to_one")
    return frame.sort_values(["entity_id", "month"]).reset_index(drop=True)
