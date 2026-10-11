"""Exercise real stored runs through the shared router and standalone entrypoints."""
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from ui.auditors import REQUIRED, missing_inputs
from ui.overview import summarize
from ui.settings import ROOT, Settings, settings


@pytest.fixture(autouse=True)
def clean_argv(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["app.py"])


@pytest.fixture
def app():
    config = settings([])
    if any(missing_inputs(name, config) for name in REQUIRED):
        pytest.skip("Stored seed-42 datasets and runs are required for full UI regression")
    result = AppTest.from_file(ROOT/"app.py", default_timeout=180).run()
    assert not result.exception
    return result


def test_explicit_paths_are_independent(tmp_path):
    config = settings(["--data", str(tmp_path/"data"), "--plastic-run", str(tmp_path/"p"),
                       "--biomed-run", str(tmp_path/"b"), "--chemical-run", str(tmp_path/"c")])
    assert config == Settings(tmp_path/"data", tmp_path/"p", tmp_path/"b", tmp_path/"c")


def test_overview_counts_match_actual_runs(app):
    config = settings([])
    summary = summarize(config)
    expected = {
        "plastic": len(pd.read_parquet(config.plastic/"recycler_scores.parquet")),
        "biomedical": len(pd.read_parquet(config.biomedical/"facility_scores.parquet")),
        "chemical": len(pd.read_parquet(config.chemical/"scores.parquet")),
    }
    assert {key: value["count"] for key, value in summary.items()} == expected
    assert len(app.button) == 3
    assert all(value["ready"] for value in summary.values())


@pytest.mark.parametrize("name,title,tabs", [("plastic", "Phantom Plastic", 5),
                                                ("biomedical", "Biomedical Waste Gap", 5),
                                                ("chemical", "Mass-Balance & Yield", 4)])
def test_home_buttons_open_complete_dashboards(app, name, title, tabs):
    app.button(key=f"open_{name}").click().run()
    assert not app.exception
    assert app.title[0].value == title
    assert len(app.tabs) == tabs
    assert len(app.dataframe) > 0


@pytest.mark.parametrize("page,title", [("plastic_legacy", "Phantom Plastic"),
                                        ("biomedical_legacy", "Biomedical Waste Gap")])
def test_existing_deployment_urls_remain_available(app, page, title):
    app.switch_page(f"ui/pages/{page}.py").run()
    assert not app.exception
    assert app.title[0].value == title
    assert len(app.tabs) == 5


def test_navigation_and_controls_across_all_agents(app):
    for name in ("chemical", "plastic", "biomedical", "chemical"):
        app.switch_page(f"ui/pages/{name}.py").run()
        assert not app.exception
        for index, selector in enumerate(list(app.selectbox)):
            # Alternate case/loop/plant selections without dropping the shared navigation.
            if len(selector.options) > 1:
                app.selectbox[index].select_index(1).run()
                assert not app.exception
        if name == "chemical":
            app.selectbox(key="stream").select_index(0).run()
            app.selectbox(key="state").select_index(0).run()
            app.selectbox(key="policy").select_index(0).run()
            for percent in (0, 50, 100):
                app.slider(key="claim_percent").set_value(percent).run()
                assert not app.exception
        if name == "plastic":
            app.slider[0].set_value(20).run()
            assert not app.exception
        for index, selector in enumerate(list(app.multiselect)):
            app.multiselect[index].select(selector.options[0]).run()
            assert not app.exception
        app.switch_page("ui/pages/overview.py").run()
        assert not app.exception
        assert len(app.button) == 3


def test_missing_auditor_does_not_block_other_pages(tmp_path, monkeypatch, app):
    monkeypatch.setattr(sys, "argv", ["app.py", "--plastic-run", str(tmp_path/"missing")])
    app.switch_page("ui/pages/overview.py").run()
    app.button(key="open_plastic").click().run()
    assert not app.exception
    assert app.info
    assert "phantom.pipeline" in app.code[0].value
    for name in ("biomedical", "chemical"):
        app.switch_page(f"ui/pages/{name}.py").run()
        assert not app.exception
        assert app.tabs


def test_all_missing_data_keeps_navigation(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["app.py", "--data", str(tmp_path/"missing")])
    app = AppTest.from_file(ROOT/"app.py", default_timeout=30).run()
    assert not app.exception
    for name in REQUIRED:
        app.button(key=f"open_{name}").click().run()
        assert not app.exception
        assert app.info
        assert not app.tabs
        app.switch_page("ui/pages/overview.py").run()
        assert not app.exception


@pytest.mark.parametrize("name,tabs", [("phantom", 5), ("biomed", 5), ("yieldbal", 4)])
def test_standalone_dashboards_still_work(app, name, tabs):
    standalone = AppTest.from_file(ROOT/name/"app.py", default_timeout=180).run()
    assert not standalone.exception
    assert len(standalone.tabs) == tabs


@pytest.mark.parametrize("module,score_file", [("phantom", "recycler_scores.parquet"),
                                               ("biomed", "facility_scores.parquet")])
def test_cache_uses_run_paths(app, tmp_path, module, score_file):
    from importlib import import_module
    config = settings([])
    original = ROOT/module/"outputs/seed42"
    dashboard = import_module(f"{module}.app")
    table = pd.read_parquet(original/score_file)
    for count in (2, 3):
        new_run = tmp_path/f"run_{count}"
        shutil.copytree(original, new_run)
        table.head(count).to_parquet(new_run/score_file)
        loaded = dashboard.load(new_run, config.data)
        assert len(loaded[0]) == count
