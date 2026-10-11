# epr-plastic-audit

Two audit agents for India's waste-compliance rules, built for the Environmental Hacks hackathon (Waste and Energy track):

1. **Phantom Plastic** (`phantom/`): unsupervised detection of **hollow EPR certificates**, recyclers whose plastic certificates are genuine portal entries backed by recycling that did not physically happen. Output: a ranked inspection list, an evidence-backed case file per recycler, and each brand's exposure to risky certificates.
2. **Biomedical Waste Gap** (`biomed/`): audit of 20,000 healthcare facilities and 200 treatment plants (CBWTFs) from barcode pickups, GPS stops and treatment logs. Finds facilities that hand over less waste than they must have produced, pickups that cannot have happened, and plants reporting more burning than their incinerator can do.

Both emit signals in one shared format (`{entity_id, agent, score, reasons[], evidence_refs[]}`) and each has an inspector dashboard.

> All data in this repository is **synthetic**, generated with planted fraud patterns and calibrated to published aggregates. Detector design and the data generator share assumptions, so reported accuracy shows that the mechanisms are detectable, not how the systems would perform on real portal data. See `phantom/README.md` and `biomed/README.md` for results, held-out tests and known weaknesses.

## Phantom Plastic: how it works

| Layer | Method |
|---|---|
| Rules | 30 physics / market / forensic rules with a plain-language reason each: electricity needed for the tonnes claimed, capacity, certificate price against the cost of recycling, closed trading loops, round-number invoices, new or over-capacity collectors |
| Isolation Forest + SHAP | 100+ recycler features, each mapped to a one-sided suspicion level |
| Graph autoencoder | DOMINANT-style GCN autoencoder on the recycler / collector / offtaker / brand network |
| Ensemble + brand exposure | rank average; risk propagated along certificate sales to brands |

## Biomedical Waste Gap: how it works

Each pickup is compared with the waste the facility should have produced since the previous collection (beds in service x occupancy x kg per bed-day, net of declared closures), with season and weekday effects learned from the population. Separate physical tests flag pickups that cannot have happened (during a declared closure, off the facility's collection rhythm, GPS more than 1 km away, under 3 minutes at the stop, copied weights). Plants are checked for ghost pickups, inflated weights, burning above incinerator rating or permit, and implausibly low diesel use. Whole 4.65M-pickup dataset: about 20 s.

No model in either agent is trained on fraud labels. Labels live in `data/*/labels/` and are read only by the evaluation code.

## Quick start

```bash
pip install -r requirements.txt

# Phantom Plastic
python -m phantom.pipeline --data data/synthetic --out phantom/outputs/seed42   # about 8 min on CPU
streamlit run phantom/app.py -- --run phantom/outputs/seed42 --data data/synthetic

# Biomedical Waste Gap
python -m biomed.run_eval --data data/synthetic --out biomed/outputs/seed42     # pipeline + evaluation, about 30 s
streamlit run biomed/app.py -- --run biomed/outputs/seed42 --data data/synthetic
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
synthdata/   seeded synthetic data generators and validation
data/        DATA_CARD.md; synthetic/ (seed 42, all streams), synthetic_seed7/ and synthetic_evasive/ (plastic held-out and evasive-fraud worlds)
```

Data documentation: `data/DATA_CARD.md`. The biomedical held-out world (seed 7) is regenerated with
`python -m synthdata.generate_all --scale large --seed 7 --only bmw --out data/synthetic_seed7`.
