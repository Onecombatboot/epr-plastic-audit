"""Offline adapters are tested without credentials or service calls."""
import io
import json
from pathlib import Path

import pytest
import yaml

from yieldbal.aws import lambda_handler
from yieldbal.aws.agent import AuditTools, OfflineAgent, build_agent, make_tools
from yieldbal.aws.deploy import plan
from yieldbal.aws.sample_bills import offline_response_from_pdf, sample_roundtrip
from yieldbal.aws.textract_bill import check_bill, parse_analyze_expense, parse_fields, render_bill_pdf, textract_extract
from yieldbal.config_demo import make_world
from yieldbal.pipeline import run


@pytest.fixture(scope="module")
def audit_world(tmp_path_factory):
    path = tmp_path_factory.mktemp("aws_world")
    make_world(path/"data")
    run(path/"data", path/"run")
    return path


@pytest.mark.parametrize("units,charge,expected", [(100., 800., True), (100., 400., False), (0., 0., True),
                                                  (0., 800., False), (None, 800., None)])
def test_bill_arithmetic(units, charge, expected):
    result = check_bill({"units_kwh": units, "energy_charge_inr": charge}, 8, 0, 100)
    assert result["consistent"] is expected


def test_zero_energy_and_missing_energy_are_different():
    assert check_bill({"units_kwh": 0., "energy_charge_inr": 0.}, 8, 10, 100)["energy_supported_t"] == 0
    missing = check_bill({}, 8, 10, 100)
    assert missing["energy_supported_t"] is None
    assert missing["consistent"] is None


def test_parser_thousands_and_zeroes():
    values = parse_fields({"Units Consumed (kWh)": "12,345.67 kWh", "Energy Charges": "Rs 0.00"})
    assert values == {"units_kwh": 12345.67, "energy_charge_inr": 0.}


def test_negative_extraction_rejected():
    with pytest.raises(ValueError):
        parse_fields({"units": "-20"})


def test_low_confidence_becomes_unknown():
    response = {"ExpenseDocuments": [{"SummaryFields": [{"LabelDetection": {"Text": "units"},
                                                            "ValueDetection": {"Text": "100", "Confidence": 20}}]}]}
    assert parse_analyze_expense(response) == {}


def test_multiple_documents_not_combined():
    with pytest.raises(ValueError):
        parse_analyze_expense({"ExpenseDocuments": [{}, {}]})


def test_duplicate_fields_rejected():
    with pytest.raises(ValueError):
        parse_fields({"units": "100", "Units consumed": "200"})


def test_pdf_numeric_roundtrip(audit_world, tmp_path):
    result = sample_roundtrip(audit_world/"data", tmp_path)
    assert result["samples"] == 10
    assert result["doctored_samples"] == 2
    assert result["numeric_roundtrip_pass"]


def test_textract_api_shape_with_mock(audit_world, tmp_path):
    import pandas as pd
    row = pd.read_parquet(audit_world/"data/yield/electricity_bills.parquet").iloc[0].to_dict()
    pdf = render_bill_pdf(row, tmp_path/"bill.pdf")
    class Client:
        def analyze_expense(self, Document):
            assert Document["Bytes"].startswith(b"%PDF")
            return offline_response_from_pdf(pdf)
    values = textract_extract(pdf, client=Client())
    assert values["units_kwh"] == row["units_kwh"]


def test_tools_and_offline_summary(audit_world, monkeypatch):
    monkeypatch.delenv("YIELDBAL_BEDROCK_MODEL", raising=False)
    tools = AuditTools(audit_world/"run", audit_world/"data")
    assert tools.top_suspects(1)[0]["entity_id"] == "FOAM-4"
    assert tools.top_suspects(1, state="demo", stream="foam")
    assert not tools.top_suspects(1, state="elsewhere")
    assert tools.check_energy_ceiling("FOAM-4")["bill_months"] == 12
    assert tools.check_output_balance("FOAM-4")["lag_months"] == 2
    assert len(tools.explain_recycler("FOAM-4")["reasons"]) >= 2
    with pytest.raises(ValueError):
        tools.explain_recycler("../bad")
    with pytest.raises(ValueError):
        tools.top_suspects(0)
    agent = build_agent(audit_world/"run", audit_world/"data")
    assert isinstance(agent, OfflineAgent)
    assert json.loads(agent("Explain FOAM-4"))["entity_id"] == "FOAM-4"
    assert json.loads(agent("Top cases"))["mode"] == "offline_template"


def test_strands_tool_wrappers(audit_world):
    wrappers = make_tools(audit_world/"run", audit_world/"data")
    assert len(wrappers) == 4
    assert json.loads(wrappers[0](entity_id="FOAM-4"))["bill_months"] == 12
    assert json.loads(wrappers[1](entity_id="FOAM-4"))["testable_windows"] > 0
    assert json.loads(wrappers[2](entity_id="FOAM-4"))["reasons"]
    assert json.loads(wrappers[3](n=1))[0]["entity_id"] == "FOAM-4"


@pytest.fixture
def cloud_signals(monkeypatch):
    records = [{"entity_id": "E-1", "score": 0.7, "stream": "tyre", "state": "Demo", "reasons": ["1", "2"]},
               {"entity_id": "E-2", "score": 0.9, "stream": "battery", "state": "Elsewhere", "reasons": ["1", "2"]}]
    monkeypatch.setattr(lambda_handler, "_load_signals", lambda: records)
    return records


def test_lambda_ranking_filters_and_case(cloud_signals):
    result = lambda_handler.handler({"rawPath": "/top", "queryStringParameters": {"n": "1"}}, None)
    assert result["statusCode"] == 200
    assert json.loads(result["body"])["cases"][0]["entity_id"] == "E-2"
    result = lambda_handler.handler({"rawPath": "/top", "queryStringParameters": {"stream": "tyre"}}, None)
    assert len(json.loads(result["body"])["cases"]) == 1
    assert lambda_handler.handler({"rawPath": "/case/E-1"}, None)["statusCode"] == 200
    assert lambda_handler.handler({"rawPath": "/case/unknown"}, None)["statusCode"] == 404


@pytest.mark.parametrize("event", [{"rawPath": "/top", "queryStringParameters": {"n": "bad"}},
                                   {"rawPath": "/top", "queryStringParameters": {"n": "101"}},
                                   {"rawPath": "/case/../../etc"}])
def test_lambda_rejects_invalid_requests(event, cloud_signals):
    assert lambda_handler.handler(event, None)["statusCode"] == 400


def test_lambda_unknown_route_does_not_load(monkeypatch):
    def fail():
        raise AssertionError("Should not fetch results")
    monkeypatch.setattr(lambda_handler, "_load_signals", fail)
    assert lambda_handler.handler({"rawPath": "/other"}, None)["statusCode"] == 404


def test_lambda_cloud_failure_is_503(monkeypatch):
    def fail():
        raise RuntimeError("sensitive internal detail")
    monkeypatch.setattr(lambda_handler, "_load_signals", fail)
    result = lambda_handler.handler({"rawPath": "/top"}, None)
    assert result["statusCode"] == 503
    assert "sensitive" not in result["body"]


def test_s3_signal_loading_and_cache(monkeypatch):
    import boto3
    monkeypatch.setenv("RESULTS_BUCKET", "test-bucket")
    monkeypatch.setenv("RESULTS_KEY", "test.jsonl")
    calls = []
    class S3:
        def get_object(self, **kwargs):
            calls.append(kwargs)
            return {"Body": io.BytesIO(b'{"entity_id":"E-1","score":0.5}\n')}
    monkeypatch.setattr(boto3, "client", lambda name: S3())
    lambda_handler._CACHE.clear()
    assert lambda_handler._load_signals()[0]["entity_id"] == "E-1"
    lambda_handler._load_signals()
    assert len(calls) == 1
    lambda_handler._CACHE.clear()


def test_deployment_plan_and_sam_template(audit_world):
    commands = plan("demo-stack", "ap-south-1", audit_world/"run")
    assert commands[0][:2] == ["sam", "build"]
    assert commands[1][:2] == ["sam", "deploy"]
    class Loader(yaml.SafeLoader):
        pass
    Loader.add_multi_constructor("!", lambda loader, tag, node: loader.construct_scalar(node))
    template = yaml.load(Path("yieldbal/aws/template.yaml").read_text(), Loader=Loader)
    assert template["Resources"]["ApiFunction"]["Properties"]["Handler"] == "lambda_handler.handler"
    assert len(template["Resources"]["ApiFunction"]["Properties"]["Events"]) == 2
