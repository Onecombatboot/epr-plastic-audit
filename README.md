# epr-plastic-audit

Three audit agents for India's waste-compliance rules, built for the Environmental Hacks hackathon (Waste and Energy track):

1. **Phantom Plastic** (`phantom/`): unsupervised detection of **hollow EPR certificates**, recyclers whose plastic certificates are genuine portal entries backed by recycling that did not physically happen. Output: a ranked inspection list, an evidence-backed case file per recycler, and each brand's exposure to risky certificates.
2. **Biomedical Waste Gap** (`biomed/`): audit of 20,000 healthcare facilities and 200 treatment plants (CBWTFs) from barcode pickups, GPS stops and treatment logs. Finds facilities that hand over less waste than they must have produced, pickups that cannot have happened, and plants reporting more burning than their incinerator can do.

3. **Chemical recovery / Mass-Balance & Yield** (`yieldbal/`): checks e-waste, batteries, tyres and used oil against saleable output, electricity and plant capacity. Includes bill checks, an evidence-backed case file and claim-reduction controls.

All three emit signals in one shared format (`{entity_id, agent, score, reasons[], evidence_refs[]}`) and share one integrated inspector workspace. Their standalone dashboards also remain available.

> All data in this repository is **synthetic**, generated with planted fraud patterns and calibrated to published aggregates. Detector design and the data generator share assumptions, so reported accuracy shows that the mechanisms are detectable, not how the systems would perform on real portal data. See `phantom/README.md`, `biomed/README.md` and `yieldbal/README.md` for results, held-out tests and known weaknesses. Chemical does not meet every statistical accuracy target; these failures are documented.

## Phantom Plastic: how it works

| Layer | Method |
|---|---|
| Rules | 30 physics / market / forensic rules with a plain-language reason each: electricity needed for the tonnes claimed, capacity, certificate price against the cost of recycling, closed trading loops, round-number invoices, new or over-capacity collectors |
| Isolation Forest + SHAP | 100+ recycler features, each mapped to a one-sided suspicion level |
| Graph autoencoder | DOMINANT-style GCN autoencoder on the recycler / collector / offtaker / brand network |
| Ensemble + brand exposure | rank average; risk propagated along certificate sales to brands |

## Biomedical Waste Gap: how it works

Each pickup is compared with the waste the facility should have produced since the previous collection (beds in service x occupancy x kg per bed-day, net of declared closures), with season and weekday effects learned from the population. Separate physical tests flag pickups that cannot have happened (during a declared closure, off the facility's collection rhythm, GPS more than 1 km away, under 3 minutes at the stop, copied weights). Plants are checked for ghost pickups, inflated weights, burning above incinerator rating or permit, and implausibly low diesel use. Whole 4.65M-pickup dataset: about 20 s.

No model in any agent is trained on fraud labels. Labels live in `data/*/labels/` and are read only by the evaluation code.

## Quick start

```bash
pip install -r requirements-app.txt

# One UI: Overview / Plastic / Biomedical / Chemical
streamlit run app.py
```

The shared sidebar opens all three full dashboards on the same server. The overview reads real saved-result counts; scoring pipelines and policies remain separate. See [UI setup and verification](ui/README.md) for custom run directories, CSS builds and tests. The original standalone entrypoints below are also supported.

```bash
# Additional dependencies for generating scores and running tests
pip install -r requirements.txt -r yieldbal/requirements.txt

# Phantom Plastic
python -m phantom.pipeline --data data/synthetic --out phantom/outputs/seed42   # about 8 min on CPU
streamlit run phantom/app.py -- --run phantom/outputs/seed42 --data data/synthetic

# Biomedical Waste Gap
python -m biomed.run_eval --data data/synthetic --out biomed/outputs/seed42     # pipeline + evaluation, about 30 s
streamlit run biomed/app.py -- --run biomed/outputs/seed42 --data data/synthetic

# Chemical recovery / Mass-Balance & Yield
python -m yieldbal.run_eval --data data/synthetic --out yieldbal/outputs/seed42
streamlit run yieldbal/app.py -- --run yieldbal/outputs/seed42 --data data/synthetic
```

Re-generate the data (seeded, reproducible):

```bash
python -m synthdata.generate_all --scale large --seed 42 --out data/synthetic
python -m synthdata.validate --data data/synthetic
```

## Layout

```
phantom/     Phantom Plastic: features, rules, models, graph, exposure, evidence, pipeline, evaluate, experiments, dashboard
phantom/aws/ Bedrock summaries, Strands agent tools, Textract bill extraction, SAM template (written, not deployed)
biomed/      Biomedical Waste Gap: prep, ghosts, facility, cbwtf, pipeline, evaluate, run_eval, dashboard
yieldbal/    Chemical recovery: stream physics, scoring, evidence, evaluation, dashboard and AWS adapters
ui/          Unified overview, navigation, local CSS and integration tests
app.py       Single Streamlit entrypoint for all three use cases
synthdata/   seeded synthetic data generators and validation
data/        DATA_CARD.md; synthetic/ (seed 42, all streams), synthetic_seed7/ (held-out world, plastic and biomedical) and synthetic_evasive/ (plastic, fraudsters hiding their tracks)
```

Data documentation: `data/DATA_CARD.md`. In the stored datasets `bmw/pickups.parquet` is a directory of part files (GitHub's 100 MB file limit); readers treat it like a single file.
