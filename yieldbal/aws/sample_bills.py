"""Ten sample PDFs exercise the parser offline; this does not test cloud OCR accuracy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from yieldbal.prep import DEFAULT_DATA, load_configs, load_reference
from .textract_bill import check_bill, parse_analyze_expense, render_bill_pdf

LABELS = {"Sanctioned Load (kW)": "sanctioned_load_kw", "Units Consumed (kWh)": "units_kwh",
          "Fixed Charges (Rs)": "fixed_charge_inr", "Energy Charges (Rs)": "energy_charge_inr",
          "Electricity Duty (Rs)": "duty_inr", "Total Amount Payable (Rs)": "total_amount_inr"}


def offline_response_from_pdf(path: Path) -> dict:
    lines = PdfReader(path).pages[0].extract_text().splitlines()
    fields = []
    for i, label in enumerate(lines):
        if label in LABELS:
            fields.append({"LabelDetection": {"Text": label},
                           "ValueDetection": {"Text": lines[i+1], "Confidence": 100}})
    return {"ExpenseDocuments": [{"SummaryFields": fields}]}


def sample_roundtrip(data: Path, out: Path) -> dict:
    bills = pd.read_parquet(Path(data)/"yield/electricity_bills.parquet").head(10)
    entities = pd.read_parquet(Path(data)/"registry/entities.parquet").set_index("entity_id")
    tariffs = pd.read_parquet(Path(data)/"registry/tariff_reference.parquet").set_index(["state", "fy"])
    operations = pd.read_parquet(Path(data)/"yield/operations_monthly.parquet")
    operations["month"] = pd.to_datetime(operations.month)
    operations = operations.set_index(["entity_id", "month"])
    processes, _ = load_configs(Path(data)/"yield/stream_configs")
    processes = processes.set_index(["stream", "process_type"])
    reference = load_reference()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for i, bill in enumerate(bills.to_dict("records")):
        date = pd.Timestamp(bill["billing_month"])
        year = date.year-(date.month < 4)
        fy = f"FY{year}-{(year+1)%100:02d}"
        rate = float(tariffs.loc[(entities.at[bill["entity_id"], "state"], fy), "energy_rate_inr_per_kwh"])
        if i == 8:
            bill["units_kwh"] *= 2
        elif i == 9:
            bill["energy_charge_inr"] = bill["units_kwh"]*rate*0.88
        path = render_bill_pdf(bill, out/f"sample_{i+1:02d}.pdf")
        response = offline_response_from_pdf(path)
        (out/f"sample_{i+1:02d}.json").write_text(json.dumps(response, indent=2))
        extracted = parse_analyze_expense(response)
        for key in LABELS.values():
            assert abs(extracted[key]-bill[key]) <= 0.011
        entity = entities.loc[bill["entity_id"]]
        minimum = float(processes.loc[(entity.stream, entity.process_type), "kwh_min"])
        claim = float(operations.loc[(bill["entity_id"], date), "certificates_t"])
        solar_kw = entity.captive_solar_kw if pd.notna(entity.captive_solar_kw) else 0
        solar_credit = solar_kw*reference["solar_hours_per_month"]*reference["solar_conversion_efficiency"]
        result = check_bill(extracted, rate, claimed_t=claim, kwh_per_t_min=minimum, solar_credit_kwh=solar_credit)
        if i >= 8:
            assert result["bill_discrepancy"] > 0.03
        results.append({"sample": path.name, "deliberately_doctored": i >= 8, "claimed_t": claim,
                        "kwh_per_t_min": minimum, "base_load_note": "Zero base load: conservative single-bill energy ceiling", **result})
    report = {"samples": len(results), "doctored_samples": 2, "numeric_roundtrip_pass": True,
              "cloud_ocr_tested": False, "results": results}
    (out/"report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=Path("yieldbal/aws/samples"))
    args = parser.parse_args()
    result = sample_roundtrip(args.data, args.out)
    print(json.dumps({key: value for key, value in result.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
