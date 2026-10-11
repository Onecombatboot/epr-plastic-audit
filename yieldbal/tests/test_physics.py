"""Physics and contract tests include confounders and counterfactual edge cases."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yieldbal.checks import compute, product_balance, what_if
from yieldbal.config_demo import make_world, prove
from yieldbal.pipeline import run
from yieldbal.prep import load, load_configs, load_reference


@pytest.fixture
def tables(tmp_path):
    make_world(tmp_path)
    return load(tmp_path)


def test_config_only_stream(tmp_path):
    assert prove(tmp_path)["code_changes"] == 0


def test_ceiling_and_missing_bill(tables):
    bill = tables["electricity_bills"].iloc[0]
    tables["electricity_bills"] = tables["electricity_bills"].iloc[1:]
    monthly, features = compute(tables, load_reference())
    missing = monthly[(monthly.entity_id == bill.entity_id) & (monthly.month == bill.billing_month)].iloc[0]
    assert pd.isna(missing.energy_supported_t)
    assert pd.isna(missing.energy_deficit)
    assert 0 <= features.loc[bill.entity_id, "bill_coverage"] < 1
    assert monthly.suspected_hollow_t.ge(0).all()
    assert monthly.suspected_hollow_t.le(monthly.certificates_t).all()


def test_zero_bill_is_observed_evidence(tables):
    tables["electricity_bills"].loc[0, ["units_kwh", "energy_charge_inr"]] = 0
    monthly, _ = compute(tables, load_reference())
    row = monthly.iloc[0]
    assert row.energy_supported_t == 0
    assert row.energy_deficit == 1
    assert row.bill_error == 0


def test_doctored_bill_uses_rupee_ceiling(tables):
    tables["electricity_bills"].loc[0, "units_kwh"] = 100000
    monthly, _ = compute(tables, load_reference())
    row = monthly.iloc[0]
    assert row.bill_error > 0.8
    assert row.effective_kwh == 10000


def test_declared_solar_adds_support(tables):
    before, _ = compute(tables, load_reference())
    tables["entities"].loc[0, "captive_solar_kw"] = 100.
    after, _ = compute(tables, load_reference())
    entity = tables["entities"].iloc[0].entity_id
    a, b = after[after.entity_id.eq(entity)], before[before.entity_id.eq(entity)]
    assert a.energy_supported_t.sum() > b.energy_supported_t.sum()


def test_inventory_lag_preserves_campaign_outputs(tables):
    entity = "FOAM-0"
    op = tables["operations_monthly"].entity_id.eq(entity)
    values = np.array([150., 0.]*6)
    tables["operations_monthly"].loc[op, ["certificates_t", "input_t_reported"]] = np.column_stack([values, values])
    output = tables["outputs_monthly"].entity_id.eq(entity)
    tables["outputs_monthly"].loc[output, "tonnes_sold"] = np.r_[0., 0., values[:-2]*0.8]
    monthly, _ = compute(tables, load_reference())
    campaign = monthly[monthly.entity_id.eq(entity)]
    assert campaign.output_deficit.dropna().max() < 1e-10
    assert campaign.output_supported_t.tail(2).isna().all()


def test_absent_product_and_non_saleable_config(tables):
    tables["outputs_monthly"] = tables["outputs_monthly"][~tables["outputs_monthly"].entity_id.eq("FOAM-4")]
    monthly, _ = compute(tables, load_reference())
    testable = monthly[monthly.entity_id.eq("FOAM-4")].output_deficit.dropna()
    assert testable.eq(1).all()


def test_what_if_all_available_tonnage_checks(tables):
    monthly, _ = compute(tables, load_reference())
    suspect = monthly[monthly.entity_id.eq("FOAM-4")]
    original = what_if(suspect, 1)
    assert original["unexplained_t"] > 0
    assert what_if(suspect, original["safe_fraction"])["tonnage_checks_pass"]
    assert what_if(suspect, 0)["unexplained_t"] == 0
    with pytest.raises(ValueError):
        what_if(suspect, 1.1)


def test_counterfactual_does_not_repair_bill(tables):
    tables["electricity_bills"].loc[0, "energy_charge_inr"] = 40000
    monthly, _ = compute(tables, load_reference())
    assert not what_if(monthly, 0)["bill_checks_pass"]


def test_pipeline_needs_no_labels_and_is_deterministic(tmp_path):
    make_world(tmp_path/"data")
    a = run(tmp_path/"data", tmp_path/"a")
    b = run(tmp_path/"data", tmp_path/"b")
    pd.testing.assert_frame_equal(a["scores"], b["scores"])
    assert a["signals"] == b["signals"]
    assert a["scores"].score.between(0, 1).all()
    for signal in a["signals"]:
        assert 2 <= len(signal["reasons"]) <= 4
        assert signal["evidence_refs"]


@pytest.mark.parametrize("table,columns", [("operations_monthly", ["entity_id", "month"]),
                                         ("electricity_bills", ["entity_id", "billing_month"])])
def test_duplicate_inputs_fail(tmp_path, table, columns):
    make_world(tmp_path)
    path = tmp_path/"yield"/f"{table}.parquet"
    frame = pd.read_parquet(path)
    pd.concat([frame, frame.head(1)]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        load(tmp_path)


def test_unknown_stream_rejected(tmp_path):
    make_world(tmp_path)
    path = tmp_path/"yield/stream_configs/foam.yaml"
    path.write_text(path.read_text().replace("recovery:", "another_process:"))
    with pytest.raises(ValueError, match="configuration"):
        load(tmp_path)


def test_invalid_physics_rejected(tmp_path):
    make_world(tmp_path)
    path = tmp_path/"yield/stream_configs/foam.yaml"
    path.write_text(path.read_text().replace("[50, 200]", "[0, 200]"))
    with pytest.raises(ValueError, match="energy"):
        load_configs(path.parent)


def test_negative_inputs_rejected(tmp_path):
    make_world(tmp_path)
    path = tmp_path/"yield/operations_monthly.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "certificates_t"] = -1
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="negative"):
        load(tmp_path)


def test_non_saleable_share_not_counted(tmp_path):
    make_world(tmp_path)
    config = tmp_path/"yield/stream_configs/foam.yaml"
    config.write_text(config.read_text()+"\n")
    import yaml
    value = yaml.safe_load(config.read_text())
    value["processes"]["recovery"]["outputs"]["gas"] = dict(expected=0.2, tolerance=0.05, saleable=False)
    config.write_text(yaml.safe_dump(value))
    process, products = load_configs(config.parent)
    assert process.saleable_expected.iloc[0] == 0.8
    assert "gas" not in products["product"].tolist()


def test_product_balance_quantifies_missing_product(tables):
    monthly, _ = compute(tables, load_reference())
    sales = tables["outputs_monthly"][~tables["outputs_monthly"].entity_id.eq("FOAM-4")]
    products = product_balance(monthly, sales, tables["products"], load_reference())
    missing = products[products.entity_id.eq("FOAM-4") & products.sold_window_t.notna()]
    assert missing.shortfall_t.gt(0).all()
    assert missing.sold_window_t.eq(0).all()
    assert missing.expected_product_t.ge(missing.lower_product_t).all()


def test_missing_tariff_does_not_invent_bill_mismatch(tables):
    tables["tariffs"] = tables["tariffs"].iloc[:0]
    monthly, _ = compute(tables, load_reference())
    assert monthly.bill_error.isna().all()
    assert not monthly.bill_testable.any()
    assert monthly.energy_supported_t.notna().all()


@pytest.mark.parametrize("column,value", [("connected_load_kw", 0.), ("declared_capacity_tpa", -1.),
                                         ("captive_solar_kw", -1.)])
def test_invalid_plant_physics_rejected(tmp_path, column, value):
    make_world(tmp_path)
    path = tmp_path/"registry/entities.parquet"
    entities = pd.read_parquet(path)
    entities.loc[0, column] = value
    entities.to_parquet(path, index=False)
    with pytest.raises(ValueError):
        load(tmp_path)
