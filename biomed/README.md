# Biomedical Waste Gap agent

Audits 20,000 healthcare facilities and 200 treatment plants (CBWTFs) from 4.65 million barcode pickups, GPS stops and daily treatment logs. Output: ranked lists, an evidence-backed case file per entity, signals in the shared agent format, and an inspector dashboard with a map.

> **Read this first.** All results are on **synthetic data** with planted fraud. The detectors were written knowing what the generator plants, so the numbers show the mechanisms are detectable, not how the system would perform on real CPCB barcode data. Norms (kg per bed-day, occupancy, diesel per kg) are assumptions in `biomed/reference.yaml`. Evaluation output was also consulted during development (for example, the duplicate-weight size limit was raised after chance matches appeared in small pickups), so seed 42 is not a clean test; seed 7 is the held-out check, and its thresholds were never touched. Seed-7 data is not stored in this repository; regenerate it with `python -m synthdata.generate_all --scale large --seed 7 --only bmw --out data/synthetic_seed7`.

## What it does

**Facilities.** Waste accumulates between collections, so each pickup is compared with what the site should have produced *since the previous one*: beds in service (declared expansions) x occupancy x kg per bed-day, or patients x kg per patient, net of declared closure days. Season (+16% Aug-Nov) and weekday (Sunday -9%) effects are learned from population medians, not assumed. Three views: level against peers and against the lowest handover any legitimate site can have; dips against its own normal level (intermittent dumping); CUSUM level shift with an onset date.

**Pickups that cannot have happened** (no model, just physical tests): recorded during a declared closure; off the facility's own collection rhythm (alternate-day parity, twice-weekly weekdays); GPS more than 1 km from the facility; under 3 minutes at the stop; a copy of another pickup's four weights (large pickups only) with no GPS fix. A missing GPS fix alone is never evidence (about 3% of stops lose GPS).

**Treatment plants.** Ghost-pickup counts; weight padding (facilities it serves pushed above anything legitimate, counted against a population-estimated base rate); burning above rated kg/hr x 24 h or above the permit; diesel per kg below the physical minimum; intake above permit (a low-weight compliance finding, not treated as fraud).

No model is trained on labels. Pipeline: `load 1.5 s -> ghost tests -> accumulation -> scores` = **about 20 s** for all 4.65M pickups.

## Run

```
python -m biomed.run_eval --data data/synthetic --out biomed/outputs/seed42     # pipeline + evaluation + acceptance checks (~30 s)
python -m streamlit run biomed/app.py -- --run biomed/outputs/seed42 --data data/synthetic
```

Signals: `biomed/outputs/<run>/signals.jsonl` with `{entity_id, entity_kind, agent, score, reasons[], evidence_refs[]}`.

## Results

| | Seed 42 | Seed 7 (held out) |
|---|---|---|
| Facility under-handover AUC (obvious / moderate / subtle) | 0.987 / 0.909 / 0.792 | 0.987 / 0.909 / 0.800 |
| precision@10 / @25 / @50 / @100 | 1.00 / 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 / 1.00 |
| Flagged facilities (precision) | 679 (1.00) | 653 (1.00) |
| Recall at flagging threshold, obvious / moderate / subtle | 0.78 / 0.50 / 0.41 | 0.77 / 0.49 / 0.41 |
| Recall if 5,000 of 20,000 are inspected (precision 0.20) | 0.99 / 0.81 / 0.64 | 0.99 / 0.83 / 0.66 |
| Intermittent under-handover recall (flagged) | 1.00 | 1.00 |
| False-positive rate on clean facilities (incl. 187 expansions, 280 closures) | 0.005% | 0.011% |
| Ghost pickups: recall / false-alarm rate on clean pickups | 0.959 / 0.0001% | 0.951 / 0.0000% |
| Plants: ghost-pickup recall | 0.95 | 0.94 |
| Plants: weight-padding recall | 0.43 | 0.36 |
| Plants: overcapacity recall (obvious / moderate / subtle) | 0.50 / 1.00 / 1.00 | 0.83 / 0.75 / 0.67 |
| Plants: emitted precision | 0.97 | 0.91 |

## Acceptance criteria (seed 42, `python -m biomed.run_eval`)

| Criterion | Result |
|---|---|
| Precision@25 facility under-handover >= 0.85 | **PASS** (1.00) |
| Recall >= 0.80 on obvious and moderate facility under-handover | **FAIL** (0.78 / 0.50 at the flagging threshold; 0.99 / 0.81 only when 25% of all facilities are inspected) |
| Ghost-pickup recall >= 0.90 with false alarms <= 0.05 on clean pickups | **PASS** (0.959 / 0.0001%) |
| Overcapacity: 100% obvious, >= 0.80 moderate | **FAIL on obvious** (1 of 2), **PASS on moderate** (4 of 4) |
| 100% valid signals | **PASS** (718 / 718) |
| 4.65M pickups processed in < 60 s | **PASS** (about 20 s) |
| No labels in detector inputs or tuning | Labels are read only by `evaluate.py`; see the disclosure at the top about how evaluation output was used |

## Why the two failures are real limits, not tuning gaps

- **Persistent under-handover overlaps legitimate variation.** A site's true generation depends on occupancy (0.55-0.92) and kg per bed-day (a 2x range), which are not observable. A site that hands over 60% of a high-generating site's waste looks like an ordinary low-generating site. Only handover below the lowest legitimate level is provable; the rest is probabilistic. Intermittent dumping (dips against its own level) is caught completely.
- **One obvious overcapacity plant** (10 kg/hr incinerator) reports burning within its rated capacity and permit every day, with normal diesel use. Nothing in its logs contradicts it, so it cannot be detected from them.

## Other weaknesses

- Weight padding is weak (recall 0.43 / 0.36): padding lifts only some of a plant's facilities, by 12-70%, and part of that lands inside the legitimate range.
- Ghost pickups missed (about 4-5%): copies placed on a skipped Sunday at a small daily facility, with no GPS and weights too small to prove a copy. They count toward the plant-level ghost tally only when other tests also fire.
- Off-cadence is evidence here because the synthetic collection rhythm is strict. In real data, rescheduled collections would make it weaker.
- Diesel per kg is noisy and carries a low weight. Intake above permit is a separate compliance finding: 9 clean plants serve more facilities than their permit allows.
- Map tiles load from OpenStreetMap and need an internet connection.
- Not deployed to AWS. There were no AWS credentials on this machine.

## Files

```
biomed/ reference.yaml  prep.py  ghosts.py  facility.py  cbwtf.py  pipeline.py  evaluate.py  run_eval.py  app.py
biomed/outputs/<run>/  facility_scores  cbwtf_scores  flagged_pickups  signals.jsonl  top_facility_pickups  metrics.json  run.json
```
