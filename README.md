# epr-plastic-audit

Unsupervised detection of **hollow EPR certificates** in India's plastic packaging scheme: recyclers whose certificates are genuine portal entries but are backed by recycling that did not physically happen. The output is a ranked inspection list for State Pollution Control Board auditors, an evidence-backed case file per recycler, and each brand's exposure to risky certificates.

Built for the Environmental Hacks hackathon (Waste and Energy track).

> All data in this repository is **synthetic**, generated with planted fraud patterns and calibrated to published CPCB aggregates. Detector design and the data generator share assumptions, so reported accuracy shows that the mechanisms are detectable, not how the system would perform on real portal data. See `phantom/README.md` for results, held-out tests and known weaknesses.

## How it works

| Layer | Method |
|---|---|
| Rules | 30 physics / market / forensic rules with a plain-language reason each: electricity needed for the tonnes claimed, capacity, certificate price against the cost of recycling, closed trading loops, round-number invoices, new or over-capacity collectors |
| Isolation Forest + SHAP | 100+ recycler features, each mapped to a one-sided suspicion level |
| Graph autoencoder | DOMINANT-style GCN autoencoder on the recycler / collector / offtaker / brand network |
| Ensemble + brand exposure | rank average; risk propagated along certificate sales to brands |

No model is trained on fraud labels. Labels live in `data/*/labels/` and are read only by the evaluation code.

## Quick start

```bash
pip install -r requirements.txt
python -m phantom.pipeline --data data/synthetic --out phantom/outputs/seed42   # about 8 min on CPU
streamlit run phantom/app.py -- --run phantom/outputs/seed42 --data data/synthetic
```

Re-generate the data (seeded, reproducible):

```bash
python -m synthdata.generate_all --scale large --seed 42 --out data/synthetic
python -m synthdata.validate --data data/synthetic
```

## Layout

```
phantom/     features, rules, models, graph, exposure, evidence, pipeline, evaluate, experiments, dashboard (app.py)
phantom/aws/ Bedrock summaries, Strands agent tools, Textract bill extraction, SAM template (written, not deployed)
synthdata/   seeded synthetic data generators and validation
data/        DATA_CARD.md; synthetic/ (seed 42), synthetic_seed7/ (held-out), synthetic_evasive/ (fraudsters hiding their tracks)
```

Data documentation: `data/DATA_CARD.md`.
