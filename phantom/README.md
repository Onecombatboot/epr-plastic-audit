# Phantom Plastic

Finds plastic recyclers whose EPR certificates are probably **hollow**: genuine portal certificates backed by recycling that did not physically happen. Output is a ranked inspection list, an evidence-backed case file per recycler, and each brand's exposure to risky certificates.

> **Read this first.** Everything below was measured on **synthetic data** that we generated, with fraud patterns that we planted. The detectors were designed knowing those mechanisms, so the numbers show the mechanisms are detectable, **not** how accurate the system would be on real CPCB portal data. All yields, kWh/t, certificate cost floors and tariffs are assumptions (`synthdata/calibration.yaml`, `phantom/reference.yaml`).

## How it works

| Layer | What | Trained? |
|---|---|---|
| Features | 106 recycler features from invoices, bills, certificates, outputs, registry (`features.py`) | no |
| 1. Rules | 30 physics/market/forensic rules with a human-readable reason each, combined by weighted noisy-OR (`rules.py`) | no |
| 2. Isolation Forest + SHAP | on direction-aware suspicion levels (which direction is worse comes from domain knowledge, never from labels) (`models.py`) | unsupervised fit on the 2,500 recyclers |
| 3. Graph autoencoder | DOMINANT-style GCN autoencoder on a 44k-node recycler / collector / offtaker / brand graph, sparse negative-sampled structure loss (`graph.py`) | unsupervised, ~7 min on CPU |
| Ensemble | rank-average, weights rules 1 : forest 1 : graph 0.25 | no fitting |
| Brand exposure | risk propagated along certificate sales, volume-weighted (`exposure.py`) | no |

No labels are used for fitting anywhere. Labels in `data/synthetic/labels/` are read only by `evaluate.py`.

Why the graph model gets a low weight: on this data it is the weakest detector (AUC 0.88) and an equal-weight average made the ensemble worse than rules + forest alone (precision@225 0.83 vs 0.94 on seed 42). It is kept as a small tie-breaker and because it is the only layer that scores collectors and offtakers by network position. The weight was chosen on seed 42 and not changed for the other worlds.

## Run it

```
python -m phantom.pipeline --data data/synthetic --out phantom/outputs/seed42      # ~8 min
python -m phantom.experiments                                                       # stress tests, ~10 min
python -m streamlit run phantom/app.py -- --run phantom/outputs/seed42
```

Outputs per run (`phantom/outputs/<run>/`): `recycler_scores.parquet`, `signals.jsonl` (shared agent contract: `{entity_id, agent, score, reasons[], evidence_refs[]}`), `cases/*.json` (top 100), `brand_exposure.parquet`, `collector_scores.parquet`, `offtaker_scores.parquet`, `metrics.json`.

## Results

Three independent worlds, same code and thresholds (tuned only on seed 42). 2,500 recyclers, 225 planted fraudulent in the baseline; tiers obvious / moderate / subtle.

| World | Detector | AUC all | AUC subtle | precision@100 | precision@225 | hollow tonnes in top 100 | subtle recall@225 |
|---|---|---|---|---|---|---|---|
| Seed 42 (tuned on) | rules | 0.997 | 0.994 | 1.00 | 0.94 | 79% | 83% |
| | Isolation Forest | 0.996 | 0.991 | 1.00 | 0.93 | 72% | 83% |
| | graph autoencoder | 0.878 | 0.809 | 0.59 | 0.42 | 66% | 22% |
| | **ensemble** | **0.997** | **0.993** | **1.00** | **0.92** | **81%** | **81%** |
| Seed 7 (held out) | ensemble | 0.996 | 0.990 | 1.00 | 0.92 | 71% | 79% |
| Evasive fraudsters | ensemble | 0.997 | 0.990 | 1.00 | 0.91 | 81% | 75% |

*Evasive world:* fraudsters leave 60% fewer side traces (underpricing, bursts, photo reuse, ...) and hide behind fabricated outputs, doctored bills and ghost feedstock far more often.

Other layers (seed 42 / seed 7 / evasive): shell offtakers AUC 0.91 / 0.91 / 0.95; ghost collectors AUC 0.81 / 0.81 / 0.79; brand ranking Spearman vs true hollow tonnes 0.93 / 0.92 / 0.92. Top-100 brands by estimated hollow tonnes hold only 36% / 26% / 29% of true hollow tonnes (exposure is spread over many brands), and 89 / 83 / 83 of the true top 100 are found.

### Which evidence matters (rules + forest only, seed 42 / evasive)

Removing any one evidence source barely changes anything, because fraud leaves traces in several. Removing the electricity data hurts most when fraudsters are evasive:

| Scenario | AUC all | subtle recall@225 |
|---|---|---|
| all evidence | 0.998 / 0.998 | 0.86 / 0.81 |
| no electricity bills or consumption | 0.995 / 0.984 | 0.80 / 0.58 |
| no price data | 0.997 / 0.997 | 0.90 / 0.81 |
| no identity data (photos, shared bank/address/director) | 0.996 / 0.998 | 0.86 / 0.80 |
| transactions only (no bills, registry, capacity) | 0.979 / 0.956 | 0.63 / 0.36 |

A single group alone (seed 42): energy AUC 0.95, identity 0.90, price 0.89, collectors 0.86, output 0.86, cert timing 0.83, capacity 0.81, feedstock 0.77, bills 0.59, network loops 0.59, Benford 0.54. Full table in `outputs/experiments.json`.

## Known weaknesses (say these in the demo)

- Synthetic data, designed alongside the detectors (see top). The near-perfect ranking at the top says the mechanisms are detectable and redundant across sources; real portal data will be noisier and more missing.
- precision@225 of 0.92 is not the whole story: at k = number of fraudsters, the last 8% of slots go to legitimate plants that look odd (undeclared captive solar, fast-growing plants, efficient low-price sellers).
- Subtle fraud: 19-25% of subtle-tier fraudsters are missed at k = 225. Without electricity data and with evasive fraudsters, subtle recall falls to 58%.
- Brand exposure depends on an assumed prevalence of hollow recyclers (default 10%), which is a setting, not a measurement. The hollow-fraction estimate is a heuristic.
- Ghost-collector detection is weak (AUC ~0.8) compared with recycler detection.
- Photos are perceptual hashes, not images. A real CLIP / OpenSearch near-duplicate step needs real photos.
- **AWS pieces in `phantom/aws/` are written but not deployed or tested against AWS**: there were no AWS credentials on this machine. Tested offline: PDF bill rendering, field parsing, bill check, template summaries. Untested: Bedrock calls, Strands agent LLM loop, Textract call, SAM deploy. The Bedrock model id is an assumption.

## Files

```
phantom/
  features.py   rules.py   models.py   graph.py   exposure.py   evidence.py   pipeline.py   evaluate.py   experiments.py   app.py
  reference.yaml                      detector priors (kWh/t by route, cost floors, expected category mix)
  aws/  narrative.py agent.py textract_bill.py lambda_handler.py template.yaml deploy.py
  outputs/<run>/ ...
```
