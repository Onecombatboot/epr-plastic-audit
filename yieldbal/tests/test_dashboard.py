"""Every tab renders on each run; filters, empty states, cases and claim controls rerun."""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

from yieldbal.config_demo import make_world
from yieldbal.pipeline import run


def test_dashboard_controls(tmp_path, monkeypatch):
    make_world(tmp_path/"data")
    run(tmp_path/"data", tmp_path/"run")
    monkeypatch.setattr(sys, "argv", ["app.py", "--run", str(tmp_path/"run"), "--data", str(tmp_path/"data")])
    app = AppTest.from_file(Path(__file__).resolve().parents[1]/"app.py", default_timeout=180).run()
    assert not app.exception
    assert len(app.tabs) == 4
    for key in ("stream", "state", "policy", "entity"):
        for index in range(len(app.selectbox(key=key).options)):
            app.selectbox(key=key).select_index(index).run()
            assert not app.exception
        if key != "entity":
            app.selectbox(key=key).select_index(0).run()
    for percent in (100, 50, 0):
        app.slider(key="claim_percent").set_value(percent).run()
        assert not app.exception
    assert app.success


def test_dashboard_missing_run_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["app.py", "--run", str(tmp_path/"missing")])
    app = AppTest.from_file(Path(__file__).resolve().parents[1]/"app.py", default_timeout=180).run()
    assert not app.exception
    assert any("yieldbal.run_eval" in element.value for element in app.info)


def test_dashboard_rebuilds_omitted_monthly_artifact(tmp_path, monkeypatch):
    make_world(tmp_path/"data")
    run(tmp_path/"data", tmp_path/"run")
    (tmp_path/"run/monthly_checks.parquet").unlink()
    monkeypatch.setattr(sys, "argv", ["app.py", "--run", str(tmp_path/"run"), "--data", str(tmp_path/"data")])
    app = AppTest.from_file(Path(__file__).resolve().parents[1]/"app.py", default_timeout=180).run()
    assert not app.exception
    assert app.selectbox(key="entity").options


def test_empty_filtered_queue_does_not_open_invalid_case(tmp_path, monkeypatch):
    import pandas as pd
    make_world(tmp_path/"data")
    entity_path = tmp_path/"data/registry/entities.parquet"
    entities = pd.read_parquet(entity_path)
    entities.loc[4, "state"] = "Elsewhere"
    entities.to_parquet(entity_path, index=False)
    tariff_path = tmp_path/"data/registry/tariff_reference.parquet"
    tariff = pd.read_parquet(tariff_path)
    extra = tariff.copy()
    extra["state"] = "Elsewhere"
    pd.concat([tariff, extra]).to_parquet(tariff_path, index=False)
    run(tmp_path/"data", tmp_path/"run")
    monkeypatch.setattr(sys, "argv", ["app.py", "--run", str(tmp_path/"run"), "--data", str(tmp_path/"data")])
    app = AppTest.from_file(Path(__file__).resolve().parents[1]/"app.py", default_timeout=180).run()
    app.selectbox(key="state").select("Demo").run()
    app.selectbox(key="policy").select("Priority").run()
    assert not app.exception
    assert len(app.selectbox) == 3
    assert any("No recyclers" in element.value for element in app.info)
