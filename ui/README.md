# Unified waste-audit workspace

Start all three use cases from one Streamlit server and one sidebar:

```bash
pip install -r requirements-app.txt
streamlit run app.py
```

The overview reads saved results for Plastic, Biomedical and Chemical (Mass-Balance & Yield). It shows real population and inspection counts, with each queue's policy kept separate. No cross-agent risk score is invented. `/plastic`, `/biomedical` and `/chemical` open the complete existing dashboards in the same app session; case files, brand exposure, trading loops, treatment maps and chemical claim controls remain available. The existing `/phantom_plastic` and `/biomedical_waste` deployment paths also work.

All datasets are synthetic. The interface does not change detector weights, thresholds, pipelines, data, labels or stored results. Chemical's known accuracy limitations remain in its evaluation report. If one auditor is missing its results or data, its page explains how to generate them while the other pages remain accessible. Reading prepared results does not run the full scoring pipelines. Chemical rebuilds monthly evidence from observable data if that optional artifact is absent.

Each auditor can use its own result directory, with a shared dataset directory:

```bash
streamlit run app.py -- --data data/synthetic \
  --plastic-run phantom/outputs/seed42 \
  --biomed-run biomed/outputs/seed42 \
  --chemical-run yieldbal/outputs/seed42
```

Use matching data and output worlds. A custom `--data` directory must contain the source tables needed by the auditors being opened. Missing files never receive an invented zero count. Cached dashboard loaders include their data and result paths, so different runs cannot share results accidentally. The original three `streamlit run <agent>/app.py` commands remain supported.

The shared design uses white and neutral surfaces, Helvetica typography, cobalt `#002FA7`, hairline rules and numbered sections. daisyUI 5 supplies the scoped hero, cards, stats, badges and disclosure components. Tailwind 4 compiles local CSS. No font service, CDN or image network request is needed by the interface. Biomedical's existing map still needs OpenStreetMap tiles.

The compiled `assets/dashboard.css` is committed; Node is only needed to rebuild it:

```bash
cd ui
npm ci
npm run build
```

`assets/source.css` omits Tailwind's global reset and prefixes daisyUI classes with `audit-`, keeping library styles away from existing Streamlit components. `package-lock.json` records the build dependencies. The repository's Streamlit theme supplies matching colors for native widgets.

Run the integration and existing chemical tests from the repository root:

```bash
pip install -r yieldbal/requirements.txt
python -m pytest ui/tests yieldbal/tests -q
python -m yieldbal.verify_dashboards --siblings
```

Full UI regression uses the stored seed-42 data and result files; those integration cases skip explicitly if the dataset is absent. Missing-data and path-configuration tests also run without it. The tests cover all overview buttons, navigation in one session, selectors, claim adjustments, standalone compatibility, legacy deployment paths, unavailable auditors, and cache separation across result directories. Browser verification checks the rendered pages, direct routes, responsive layout and CSV download. See `verification.json` for the final run record. The existing EC2 script still starts `app.py`; no cloud deployment was performed for this UI change.
