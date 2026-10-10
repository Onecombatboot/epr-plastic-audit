"""Electricity-bill extraction with Amazon Textract (AnalyzeExpense) and a consistency check against the claimed tonnes.

`render_bill_pdf` makes a one-page bill from a row of electricity_bills.parquet so the demo can run the same path a
portal upload would take. `check_bill` works on the extracted fields and needs no AWS.
"""
from __future__ import annotations

import re
from pathlib import Path


def render_bill_pdf(bill: dict, path: str | Path, discom_name: str = "State DISCOM", consumer_name: str = "") -> Path:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    path = Path(path)
    c = canvas.Canvas(str(path), pagesize=A4)
    _w, h = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, h - 50, f"{discom_name} - Electricity Bill")
    c.setFont("Helvetica", 11)
    rows = [("Consumer No", bill["consumer_no"]), ("Consumer Name", consumer_name or "-"), ("Billing Month", f"{bill['billing_month']:%b %Y}"),
            ("Sanctioned Load (kW)", f"{bill['sanctioned_load_kw']:.0f}"), ("Units Consumed (kWh)", f"{bill['units_kwh']:.0f}"),
            ("Fixed Charges (Rs)", f"{bill['fixed_charge_inr']:.2f}"), ("Energy Charges (Rs)", f"{bill['energy_charge_inr']:.2f}"),
            ("Electricity Duty (Rs)", f"{bill['duty_inr']:.2f}"), ("Total Amount Payable (Rs)", f"{bill['total_amount_inr']:.2f}")]
    y = h - 100
    for k, v in rows:
        c.drawString(50, y, k)
        c.drawString(300, y, str(v))
        y -= 22
    c.save()
    return path


_NUM = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def _num(s: str) -> float | None:
    m = _NUM.search(s or "")
    return float(m.group().replace(",", "")) if m else None


def parse_fields(fields: dict[str, str]) -> dict:
    """Map Textract key/value text (lower-cased labels) to numeric bill fields."""
    out: dict = {}
    for k, v in fields.items():
        if "unit" in k:
            out["units_kwh"] = _num(v)
        elif "energy" in k and "charge" in k:
            out["energy_charge_inr"] = _num(v)
        elif "fixed" in k:
            out["fixed_charge_inr"] = _num(v)
        elif "sanction" in k or "load" in k:
            out["sanctioned_load_kw"] = _num(v)
        elif "total" in k or "payable" in k:
            out["total_amount_inr"] = _num(v)
    return out


def textract_extract(path: str | Path, region: str = "us-east-1") -> dict:
    """AnalyzeExpense on a single-page bill PDF/image. Returns normalised numeric fields."""
    import boto3
    tx = boto3.client("textract", region_name=region)
    resp = tx.analyze_expense(Document={"Bytes": Path(path).read_bytes()})
    fields: dict[str, str] = {}
    for doc in resp.get("ExpenseDocuments", []):
        for f in doc.get("SummaryFields", []):
            k = (f.get("LabelDetection") or {}).get("Text") or f.get("Type", {}).get("Text", "")
            v = (f.get("ValueDetection") or {}).get("Text", "")
            if k:
                fields[k.strip().lower()] = v
    return parse_fields(fields)


def check_bill(extracted: dict, energy_rate_inr_per_kwh: float, claimed_t: float, kwh_per_t_min: float, tolerance: float = 0.04) -> dict:
    """Two independent checks on one bill: charges equal units x tariff, and units are enough for the tonnes claimed."""
    findings = []
    u, e = extracted.get("units_kwh"), extracted.get("energy_charge_inr")
    if u and e:
        ratio = e / (u * energy_rate_inr_per_kwh)
        if abs(ratio - 1) > tolerance:
            findings.append(f"Energy charge is {ratio:.0%} of units x tariff; the bill does not reconcile")
    if u and claimed_t:
        need = claimed_t * kwh_per_t_min
        if u < need:
            findings.append(f"{u:,.0f} kWh cannot process {claimed_t:,.1f} t (needs at least {need:,.0f} kWh)")
    return {"consistent": not findings, "findings": findings}
