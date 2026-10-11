"""A new stream is supplied as twelve lines of YAML plus observable tables, with no detector edits."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd

from .pipeline import run

CONFIG = """stream: foam
certificate_basis: input_tonnes_processed
assumption: true
processes:
  recovery:
    kwh_per_t: [50, 200]
    outputs:
      recovered_foam:
        expected: 0.8
        tolerance: 0.1
        saleable: true
note: Made-up demonstration stream; not a physical norm.
"""


def make_world(directory: Path) -> Path:
    directory = Path(directory)
    for name in ("registry", "yield/stream_configs"):
        (directory/name).mkdir(parents=True, exist_ok=True)
    (directory/"yield/stream_configs/foam.yaml").write_text(CONFIG)
    ids = [f"FOAM-{i}" for i in range(5)]
    entities = pd.DataFrame([dict(entity_id=entity, stream="foam", process_type="recovery", state="Demo",
                                  legal_name=f"Foam recovery {i}", declared_capacity_tpa=2400., connected_load_kw=100.,
                                  captive_solar_kw=float("nan"), epr_registration_date="2023-04-01") for i, entity in enumerate(ids)])
    entities.to_parquet(directory/"registry/entities.parquet", index=False)
    pd.DataFrame([dict(state="Demo", fy="FY2023-24", energy_rate_inr_per_kwh=8.,
                       fixed_rate_inr_per_kw_month=50., duty_rate=0.05)]).to_parquet(directory/"registry/tariff_reference.parquet", index=False)
    operations, outputs, bills = [], [], []
    for i, entity in enumerate(ids):
        for month in pd.date_range("2023-04-01", periods=12, freq="MS"):
            claimed = 300. if i == 4 else 100.+i
            operations.append(dict(entity_id=entity, stream="foam", process_type="recovery", month=month,
                                   input_t_reported=claimed, certificates_t=claimed))
            outputs.append(dict(entity_id=entity, month=month, product="recovered_foam", tonnes_sold=80.+i,
                                value_inr=(80.+i)*10000))
            bills.append(dict(bill_id=f"BILL-{entity}-{month:%Y%m}", entity_id=entity, billing_month=month,
                              discom="Demo", consumer_no=entity, sanctioned_load_kw=100., units_kwh=10000.,
                              energy_charge_inr=80000., fixed_charge_inr=5000., duty_inr=4250., total_amount_inr=89250.))
    for name, rows in (("operations_monthly", operations), ("outputs_monthly", outputs), ("electricity_bills", bills)):
        pd.DataFrame(rows).to_parquet(directory/"yield"/f"{name}.parquet", index=False)
    return directory


def prove(directory: Path) -> dict:
    make_world(directory/"data")
    result = run(directory/"data", directory/"run")
    assert result["meta"]["streams"] == ["foam"]
    assert result["scores"].loc["FOAM-4", "screening"]
    assert len(CONFIG.splitlines()) == 12
    return {"pass": True, "stream": "foam", "yaml_lines": 12, "code_changes": 0,
            "entities": len(result["scores"]), "top_entity": result["scores"].index[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out:
        result = prove(args.out)
    else:
        with tempfile.TemporaryDirectory() as temporary:
            result = prove(Path(temporary))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
