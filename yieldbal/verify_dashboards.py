"""Headless verification covers all three existing dashboards without editing sibling agents."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from .prep import DEFAULT_DATA, DEFAULT_RUN, ROOT


def verify(path: Path, run: Path, data: Path, exhaustive_filters: bool = False) -> dict:
    sys.argv = [str(path), "--run", str(run), "--data", str(data)]
    st.cache_data.clear()
    app = AppTest.from_file(path, default_timeout=180).run()
    assert not app.exception, [(error.message, error.stack_trace) for error in app.exception]
    tabs, selectors, reruns = len(app.tabs), len(app.selectbox), 1
    for selector_index in range(selectors):
        selector = app.selectbox[selector_index]
        n = len(selector.options)
        indices = range(n) if exhaustive_filters and selector_index < 3 else sorted({0, n//2, n-1})
        for index in indices:
            if selector_index >= len(app.selectbox) or not app.selectbox[selector_index].options:
                continue
            app.selectbox[selector_index].select_index(min(index, len(app.selectbox[selector_index].options)-1)).run()
            reruns += 1
            assert not app.exception, [error.message for error in app.exception]
        if selector_index < len(app.selectbox) and app.selectbox[selector_index].options:
            app.selectbox[selector_index].select_index(0).run()
            reruns += 1
            assert not app.exception
    for index in range(len(app.slider)):
        slider = app.slider[index]
        for value in (slider.min, slider.max):
            app.slider[index].set_value(value).run()
            reruns += 1
            assert not app.exception
    for index in range(len(app.multiselect)):
        options = app.multiselect[index].options
        if options:
            app.multiselect[index].set_value([options[0]]).run()
            reruns += 1
            assert not app.exception
            app.multiselect[index].set_value([]).run()
    return {"dashboard": path.parent.name, "pass": True, "tabs": tabs, "selectors": selectors, "reruns": reruns}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--siblings", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    targets = [(ROOT/"yieldbal/app.py", args.run, True)]
    if args.siblings:
        targets.extend([(ROOT/"phantom/app.py", ROOT/"phantom/outputs/seed42", False),
                        (ROOT/"biomed/app.py", ROOT/"biomed/outputs/seed42", False)])
    result = [verify(path, run, args.data, exhaustive) for path, run, exhaustive in targets]
    report = {"dashboards": result, "all_pass": all(item["pass"] for item in result)}
    if args.out:
        args.out.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
