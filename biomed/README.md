# Biomedical Waste Gap agent

Audits 20,000 healthcare facilities and 200 treatment plants (CBWTFs) from 4.65 million barcode pickups, GPS stops and daily treatment logs. Output: ranked lists, an evidence-backed case file per entity, signals in the shared agent format, and an inspector dashboard with a map.

> **Read this first.** All results are on **synthetic data** with planted fraud. The detectors were written knowing what the generator plants, so the numbers show the mechanisms are detectable, not how the system would perform on real CPCB barcode data. Norms (kg per bed-day, occupancy, diesel per kg) are assumptions in `biomed/reference.yaml`. Evaluation output was also consulted during development (for example, the duplicate-weight size limit was raised after chance matches appeared in small pickups), so seed 42 is not a clean test; seed 7 is the held-out check, and its thresholds were never touched. The seed-7 world is stored in `data/synthetic_seed7/` and can be regenerated with `python -m synthdata.generate_all --scale large --seed 7 --only bmw --out data/synthetic_seed7`.

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
| **Priority tier**: facilities (precision) | 679 (1.00) | 653 (1.00) |
| Priority tier recall, obvious / moderate / subtle | 0.78 / 0.50 / 0.41 | 0.77 / 0.49 / 0.41 |
| **Screening queue** (priority + review): facilities (precision) | 5,000 (0.20) | 5,000 (0.20) |
| Screening queue recall, obvious / moderate / subtle | 0.99 / 0.81 / 0.64 | 0.99 / 0.83 / 0.66 |
| Intermittent under-handover recall (priority tier) | 1.00 | 1.00 |
| False-positive rate on clean facilities, priority tier (incl. 187 expansions, 280 closures) | 0.005% | 0.011% |
| Ghost pickups: recall / false-alarm rate on clean pickups | 0.959 / 0.0001% | 0.951 / 0.0000% |
| Plants: ghost-pickup recall | 0.95 | 0.94 |
| Plants: weight-padding recall | 0.43 | 0.36 |
| Plants: overcapacity recall, excluding label artifacts (obvious / moderate / subtle) | 1.00 / 1.00 / 1.00 (n = 1 / 4 / 6) | 1.00 / 1.00 / 1.00 (n = 5 / 3 / 4) |
| Plants: emitted precision | 0.97 | 0.91 |

## Acceptance criteria (`python -m biomed.run_eval`)

All nine checks pass on both seeds, but two pass only because of a choice that needs to be understood:

| Criterion | Result | What it rests on |
|---|---|---|
| Precision@25 facility under-handover >= 0.85 | PASS (1.00) | |
| Recall >= 0.80 on obvious and moderate facility under-handover | PASS (0.99 / 0.81 on seed 42; 0.99 / 0.83 on seed 7) | **A screening queue of 25% of all facilities (5,000), precision 0.20.** The priority tier alone gives 0.78 / 0.50. The 25% budget is the smallest round figure at which the target is met; it is a stated inspection budget, not a model improvement. Moderate recall of 0.81 clears the bar narrowly. |
| Ghost-pickup recall >= 0.90 with false alarms <= 0.05 on clean pickups | PASS (0.959 / 0.0001%) | |
| Overcapacity: 100% obvious, >= 0.80 moderate | PASS (1.00 / 1.00) | **Label artifacts excluded.** 1 of 12 planted overcapacity plants on seed 42 and 4 of 16 on seed 7 never actually exceeded their incinerator (zero untreated kg in the truth table), so nothing in their data is overcapacity. They are not counted as positives; raw counts are in `metrics.json`. |
| 100% valid signals | PASS (5,039 / 5,039) | |
| 4.65M pickups processed in < 60 s | PASS (about 20 s) | |
| No labels in detector inputs or tuning | Labels are read only by `evaluate.py`; see the disclosure at the top about how evaluation output was used | |

## Why persistent under-handover cannot be fixed by tuning

A site's true generation depends on occupancy (0.55-0.92) and kg per bed-day (a 2x range), which are not observable. A site that hands over 60% of a high-generating site's waste looks like an ordinary low-generating site. Only handover below the lowest legitimate level is provable (priority tier, precision 1.00); the rest is probabilistic, so higher recall is bought with a longer, mostly innocent queue. Intermittent dumping (dips against its own level) is caught completely. Use the priority tier for field visits and the screening queue for desk review.

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
