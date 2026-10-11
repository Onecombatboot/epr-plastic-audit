"""Parse one AnalyzeExpense document and preserve unknown fields instead of inventing zeroes."""
from __future__ import annotations

import math
import re
from pathlib import Path


def render_bill_pdf(bill: dict, path: Path) -> Path:
    from reportlab.lib.colors import HexColor
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    document.setFillColor(HexColor("#173c35"))
    document.rect(0, height-125, width, 125, fill=1, stroke=0)
    document.setFillColor(HexColor("#ffffff"))
    document.setFont("Helvetica-Bold", 21)
    document.drawString(40, height-55, "Electricity bill")
    document.setFont("Helvetica", 11)
    document.drawString(40, height-80, str(bill.get("discom", "State DISCOM")))
    document.drawString(40, height-103, "SYNTHETIC SAMPLE - FOR AUDIT TESTING")
    document.setFillColor(HexColor("#21332f"))
    document.setFont("Helvetica", 11)
    fields = [("Bill ID", bill.get("bill_id", "sample")), ("Consumer No", bill.get("consumer_no", "sample")),
              ("Billing Month", f"{pd.Timestamp(bill['billing_month']):%b %Y}"),
              ("Sanctioned Load (kW)", f"{bill['sanctioned_load_kw']:.2f}"),
              ("Units Consumed (kWh)", f"{bill['units_kwh']:.2f}"),
              ("Fixed Charges (Rs)", f"{bill['fixed_charge_inr']:.2f}"),
              ("Energy Charges (Rs)", f"{bill['energy_charge_inr']:.2f}"),
              ("Electricity Duty (Rs)", f"{bill['duty_inr']:.2f}"),
              ("Total Amount Payable (Rs)", f"{bill['total_amount_inr']:.2f}")]
    y = height-165
    for label, value in fields:
        document.drawString(40, y, label)
        document.drawRightString(width-40, y, str(value))
        document.setStrokeColor(HexColor("#dde5e1"))
        document.line(40, y-12, width-40, y-12)
        y -= 38
    document.setFont("Helvetica", 9)
    document.drawString(40, 50, "This document represents synthetic observations. No real consumer appears here.")
    document.save()
    return path


def numeric(text: str) -> float | None:
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", text or "")
    return float(match.group().replace(",", "")) if match else None


def parse_fields(fields: dict[str, str]) -> dict:
    parsed = {}
    for label, value in fields.items():
        label = label.lower()
        key = None
        if "unit" in label and ("consum" in label or "kwh" in label or label.strip() == "units"):
            key = "units_kwh"
        elif "energy" in label and "charge" in label:
            key = "energy_charge_inr"
        elif "fixed" in label:
            key = "fixed_charge_inr"
        elif "sanction" in label or "load" in label:
            key = "sanctioned_load_kw"
        elif "duty" in label:
            key = "duty_inr"
        elif "total" in label or "payable" in label:
            key = "total_amount_inr"
        if key:
            number = numeric(value)
            if number is not None and (not math.isfinite(number) or number < 0):
                raise ValueError(f"Invalid extracted value for {key}")
            if key in parsed and parsed[key] != number:
                raise ValueError(f"Conflicting extracted values for {key}")
            parsed[key] = number
    return parsed


def parse_analyze_expense(response: dict, minimum_confidence: float = 70) -> dict:
    documents = response.get("ExpenseDocuments", [])
    if len(documents) != 1:
        raise ValueError("Expected exactly one expense document")
    fields = {}
    for field in documents[0].get("SummaryFields", []):
        value = field.get("ValueDetection", {})
        if value.get("Confidence", 100) < minimum_confidence:
            continue
        key = field.get("LabelDetection", {}).get("Text") or field.get("Type", {}).get("Text", "")
        text = value.get("Text", "")
        if key in fields and fields[key] != text:
            raise ValueError("Conflicting expense labels")
        fields[key] = text
    return parse_fields(fields)


def textract_extract(path: Path, region: str | None = None, client=None) -> dict:
    if client is None:
        import boto3
        client = boto3.client("textract", region_name=region)
    return parse_analyze_expense(client.analyze_expense(Document={"Bytes": Path(path).read_bytes()}))


def check_bill(extracted: dict, energy_rate_inr_per_kwh: float, claimed_t: float,
               kwh_per_t_min: float, solar_credit_kwh: float = 0, base_load_kwh: float = 0,
               tolerance: float = 0.03) -> dict:
    if energy_rate_inr_per_kwh <= 0 or kwh_per_t_min <= 0 or min(claimed_t, solar_credit_kwh, base_load_kwh) < 0:
        raise ValueError("Bill physics must have positive rates and non-negative quantities")
    units, charge = extracted.get("units_kwh"), extracted.get("energy_charge_inr")
    for number in (units, charge):
        if number is not None and (not math.isfinite(number) or number < 0):
            raise ValueError("Extracted quantities must be finite and non-negative")
    findings = []
    discrepancy = None
    if units is not None and charge is not None:
        expected = units*energy_rate_inr_per_kwh
        discrepancy = abs(charge-expected)/max(expected, 1)
        if discrepancy > tolerance:
            findings.append(f"Energy charge differs {discrepancy:.1%} from {units:,.0f} kWh × Rs {energy_rate_inr_per_kwh:.2f}/kWh.")
    effective = min(units, charge/energy_rate_inr_per_kwh) if units is not None and charge is not None else units
    supported = max(effective+solar_credit_kwh-base_load_kwh, 0)/kwh_per_t_min if effective is not None else None
    if supported is not None and supported < claimed_t:
        findings.append(f"Energy supports at most {supported:,.1f} t of the {claimed_t:,.1f} t claim.")
    unknown = [key for key in ("units_kwh", "energy_charge_inr") if extracted.get(key) is None]
    return {"consistent": not findings if not unknown else None, "findings": findings,
            "unknown_fields": unknown, "bill_discrepancy": discrepancy, "energy_supported_t": supported}
