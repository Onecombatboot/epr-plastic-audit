# Data Card: Synthetic Waste-Compliance Datasets

Synthetic datasets for training and evaluating the multi-agent waste compliance auditor (Phantom Plastic + extension agents). **All data is synthetic.** Magnitudes are calibrated to public figures where available; every other constant is a labelled assumption in `synthdata/calibration.yaml`.

- **Regenerate:** `python -m synthdata.generate_all --scale large --seed 42 --out data/synthetic` (about 6 min; `small` about 20 s for development)
- **Validate:** `python -m synthdata.validate --data data/synthetic`, which writes `data/synthetic/validation_report.md`
- **Inventory:** `data/synthetic/manifest.json` (row counts, columns, bytes, timings)

---

## 1. Design principles

1. **Truth first, then fabrication.** Each generator simulates the *true* physical world (tonnes processed, kWh consumed, waste generated, effluent quality), then derives the *reported* world = truth + fabrication. Fraud signals (capacity overrun, energy deficit, ghost suppliers, below-cost pricing, ...) **emerge** from the simulation instead of being painted on. Clean entities are physically consistent by construction, and `validate` proves it.
2. **One shared registry.** Plastic, yield and shell-network data are built on the same `registry/entities`. A fraud recycler in the plastic graph is the *same entity* whose directors appear in a shell cluster and whose bills appear in the yield data, so **cross-agent corroboration is testable**.
3. **Three difficulty tiers.** Every fraud entity/episode is `obvious`, `moderate` or `subtle` (30/40/30). Validation reports naive single-signal AUC per tier: obvious is easy, subtle is hard.
4. **Benign confounders.** Realistic legitimate behaviour that *looks* suspicious (captive solar, shared consultants, legit business groups, efficient low-price recyclers, deadline-rush buying, campaign processing, stuck sensors, bed expansions, ...) is planted, so precision is meaningful.
5. **Labels are quarantined.** Everything under `labels/` is ground truth. **Never join `labels/` into model features.** Use it only for evaluation (or for training supervised baselines you explicitly report as such). IDs are random, so ID order leaks nothing.

---

## 2. Directory layout

```
data/synthetic/
├── registry/         shared entities across 5 EPR streams (+ people, consultants, photos, tariffs)
├── plastic/          Phantom Plastic graph: collectors → recyclers → certificate lots → brands
├── yield/            Mass-balance & yield agent: e-waste, battery, tyre, used-oil operations
├── ocems/            Effluent monitoring readings (chemical-waste agent)
├── bmw/              Biomedical waste pickups and treatment
├── labels/           GROUND TRUTH ONLY (registry/, plastic/, yield/, ocems/, bmw/)
├── manifest.json
└── validation_report.md
```

---

## 3. Registry (`registry/`), used by the Shell-Network agent and as entity master data

| Table | Grain | Key columns |
|---|---|---|
| `entities` | one recycler (5 streams) | entity_id, stream, legal_name, constitution, pan, gstin (valid checksum), state/district/estate, address, lat/lon, incorporation_date, epr_registration_date, process_type, declared_capacity_tpa, connected_load_kw, declared_staff, captive_solar_kw, contact_phone/email, filing_phone/email (consultant's if used), bank_ifsc/account, consultant_id, authorised_signatory_id |
| `people` | person | person_id, din, name, pan, phone, email, residential_city |
| `entity_directors` | entity × person | role, appointed_date |
| `consultants` | compliance consultancy | shared filing contacts across many clients (benign link) |
| `plant_photos` | uploaded photo | photo_type, **phash** (64-bit perceptual hash, hex), exif_present/lat/lon/datetime, upload_date |
| `tariff_reference` | state × FY | energy rate, fixed charge per kW, duty (used to check bill arithmetic) |

**Shell ground truth** (`labels/registry/`): `entity_labels` (is_fraud, operator_id, operator_size, difficulty, hollow_fraction, is_migration_shell, behaviour flags, confounders), `same_operator_pairs` (positive pairs for link prediction), `benign_link_pairs` (legit groups: hard negatives), `photo_origin` (original / reused_sibling / reused_external / benign duplicates), `entity_physics` (true capacity, kWh/t, utilisation: hidden physics).

**How shells are linked:** members of one operator share identifiers along a chain, so each cluster is connected in principle. Obvious = 3–5 shared identifier types (director, signatory, phone, email, bank, exact address, near-duplicate photo); moderate = 2; subtle = 1, possibly only a weak *relative* link (same surname + residential city). **Migration shells** are registered within 0–120 days after a late stream's EPR launch (used oil Apr 2024, e-waste Apr 2023).

---

## 4. Plastic (`plastic/`), the Phantom Plastic agent

| Table | Grain | Notes |
|---|---|---|
| `collectors` | waste collector/aggregator (incl. hidden ghosts) | type, state, gstin, phone, address, registration_date, declared_monthly_capacity_t, vehicle_count |
| `feedstock_invoices` | invoice (collector or recycler → recycler) | date, category (CAT_I–IV), tonnes, rate, value, GST, vehicle_reg, eway_bill_no, distance_km |
| `monthly_processing_reported` | recycler × month × category | what the recycler reports to the portal |
| `certificate_lots` | recycler × month × category | EPR certificates generated (tonnes, generated_on, fy, eol_type) |
| `certificate_sales` | lot-level sale to a brand | transaction_id, buyer (brand), sale_date, tonnes, price_inr_per_t |
| `brands` / `brand_obligations` | PIBO / PIBO × FY × category | sector drives category mix (snacks → multilayer heavy) |
| `offtakers` / `output_invoices` | buyers of recycled output | recycled granules/flakes/pyrolysis oil sales (mass-balance output side) |
| `electricity_bills` | recycler × month | DISCOM bill: units, fixed, energy, duty, total |

**Fraud behaviours** (from `labels/registry/entity_labels`; realised effects are in the data):

| Behaviour | What happens in the data |
|---|---|
| hollow claims (all fraud) | reported tonnes = true / (1 − hollow_fraction); paper shells claim a full plant while processing ~nothing |
| `capacity_inflated` | declared capacity 1.5–6× true, which hides capacity overrun but breaks tonnes-per-kW |
| `ghost_feedstock` | 2–8 new ghost collectors invoice the hollow tonnes; obvious ghosts: round tonnages, month-end bulk invoices, one reused truck, missing e-way bills, shared phone/address with the recycler |
| inflated real invoices | (moderate/subtle without ghosts) real collectors "supply" beyond their real volumes |
| no backing | (obvious without ghosts) hollow tonnes have no invoices at all |
| `round_trip` | rings of 3–5 recyclers invoicing "granules" to each other to justify inputs |
| `underprice` | certificates sold below the recycling cost floor (obvious 0.20–0.45×, moderate 0.45–0.75×, subtle 0.75–0.95×) |
| `cat3_inflation` | hollow tonnes skewed to multilayer (CAT_III), where genuine recycling capacity is scarce |
| `deadline_burst` | hollow generation concentrated in Feb–Mar; lots batch-generated just before 30 June |
| `output_fabrication` | sales to shell offtakers so output balance looks right (careful fraud) |
| `doctored_bill` | bill units inflated; amount left unedited (obvious/moderate) or edited with a stale tariff (subtle) |
| `photo_reuse` | registry photos copied from siblings or unrelated plants (near-duplicate phash, wrong/stripped EXIF) |

**Benign confounders:** large efficient legit recyclers pricing near the floor; genuine multilayer recyclers (high CAT_III, higher kWh/t); captive solar (declared or not) giving low grid bills; fast-growing new legit recyclers; consultants sharing filing contacts; legit business groups sharing directors/compound addresses; **every brand** rushes purchases into Apr–Jun before the deadline; monsoon feedstock dip.

**Market model:** brand obligations ≈ 1.02× genuine supply, so hollow certificates undercut and **crowd out** genuine ones (some legit certificates go unsold). Price-sensitive brands end up with higher hollow exposure.

**Labels** (`labels/plastic/`): `collector_labels` (is_ghost), `invoice_labels` (legit / ghost / inflated_real / round_trip), `lot_labels` (hollow_tonnes, hollow_share), `brand_labels` (**true_hollow_exposure**, compliance_ratio, procurement_style), `monthly_truth`, `offtaker_labels` (is_shell), `output_invoice_labels`, `round_trip_edges`, `recycler_extra_labels` (confounder flags, certificate markup).

---

## 5. Yield (`yield/`), the Mass-Balance & Yield agent (e-waste, battery, tyre, used oil)

| Table | Grain | Notes |
|---|---|---|
| `operations_monthly` | entity × month | input_t_reported, certificates_t, process_type |
| `outputs_monthly` | entity × month × product | tonnes_sold, value_inr (saleable products only) |
| `electricity_bills` | entity × month | same schema as plastic |
| `stream_configs/*.yaml` | stream | **agent config:** expected output fractions ± tolerance, kWh/t ranges, saleable flags. Adding a stream = adding a YAML. |

Plastic's equivalents for the yield agent: `plastic/monthly_processing_reported` + `plastic/output_invoices` + `plastic/electricity_bills` + `yield/stream_configs/plastic.yaml`.

**Fraud:** lazy fraud inflates inputs/certificates while outputs stay true (output-balance gap); careful fraud fabricates outputs too (only energy/capacity reveal it); doctored bills as in plastic. **Confounders:** campaign processors (idle/peak months, 1–2 month inventory lag), atypical-but-legit yields, captive solar, ramp-up, fast growers. **Labels:** `labels/yield/monthly_truth` (true_input_t, hollow_t), `entity_extra_labels` (campaign, atypical yield, true per-entity yields).

---

## 6. OCEMS (`ocems/`), the Effluent Tamper Detector (chemical waste)

| Table | Grain | Notes |
|---|---|---|
| `units` | industrial unit / CETP | industry (distillery, pulp & paper, textile, pharma, chemical, tannery, sugar, dairy, CETP), schedule, design flow, standards |
| `readings` | unit × 15-min | ph, cod_mgl, bod_mgl, tss_mgl, flow_m3h, status (OK / CAL / MAINT / NODATA) |

Signals: production schedule → ETP retention smoothing → AR(1) noise → genuine upsets (real exceedances) → sensor noise and quantisation.

**Tampering (8 types):** flatline, over-smoothing, replay (copy-paste of an earlier compliant window), dilution (flow up, concentration down), night bypass (23:00–05:00), clipping just under the standard, implausible BOD/COD, scaling. 60% of episodes sit on top of genuine upsets (motive). Obvious = exact constants and exact copies; subtle = small jitter, scaled BOD so ratios stay plausible.

**Benign faults:** communication gaps, power outages, calibration (70% flagged, 30% not), slow drift, flagged maintenance flatlines, **genuine unflagged stuck sensors** (the hardest confounder for flatline detection).

**Labels** (`labels/ocems/`): `row_labels` (tamper_type, fault_type, true_exceedance, **hidden_exceedance**, true COD/BOD/flow), `tamper_episodes`, `benign_fault_episodes`, `unit_labels`.

---

## 7. Biomedical waste (`bmw/`), the BMW Gap agent

> Storage note: in this repository `bmw/pickups.parquet` is a directory of part files (GitHub's 100 MB file limit); `pandas.read_parquet` and `pyarrow` read it exactly like the single file the generator writes.

| Table | Grain | Notes |
|---|---|---|
| `facilities` | healthcare facility | type, beds / avg daily patients, state, lat/lon, assigned CBWTF, pickup schedule |
| `facility_changes` | declared change | bed expansions, temporary closures (benign explanations) |
| `cbwtfs` | treatment facility | incinerator kg/hr, permitted kg/day, vehicles |
| `pickups` | barcode-recorded pickup | yellow/red/white/blue kg, bags, vehicle, GPS stop, arrival, dwell_min |
| `cbwtf_daily_treatment` | CBWTF × day | intake, incinerated/autoclaved reported, incinerator hours, diesel purchased |

**Fraud:** facility under-handover (persistent or intermittent; the rest is dumped); CBWTF ghost pickups (no GPS / GPS 2–10 km away / near but ~0 dwell; often during declared closures); CBWTF weight padding; CBWTF overcapacity (undersized incinerator, reported burned > permitted, diesel per kg too low). **Confounders:** bed expansions, closures, dengue-season peaks, weekday effects, 3% GPS device failures. **Labels** (`labels/bmw/`): `facility_labels`, `cbwtf_labels`, `pickup_labels` (true accumulated kg, handed kg, padding, is_ghost), `cbwtf_treatment_truth` (true incinerated, untreated kg).

---

## 8. How to use for training and evaluation

- **Unsupervised detectors** (Isolation Forest, DOMINANT, matrix profile): fit on feature tables only; evaluate with `labels/`.
- **Report per difficulty tier** (obvious / moderate / subtle) and per fraud behaviour, not just overall AUC.
- **Metrics:** precision@k (auditors inspect a top-k list), recall per behaviour, AUC; for brands, rank correlation between predicted and `true_hollow_exposure`; for OCEMS, episode-level recall and false alarms on benign faults; for shells, pairwise precision/recall against `same_operator_pairs` with `benign_link_pairs` as hard negatives.
- **Supervised baselines:** if you train on labels, split **by operator** (all entities of one operator in the same fold), or the model memorises clusters.
- **Seeds:** different `--seed` values give independent worlds; use one for tuning and another for the final reported numbers.

## 9. Known simplifications (state these in the demo)

- Certificates are based on input tonnes processed; categories are drawn per invoice from the recycler's mix (approximate).
- Yields, kWh/t, certificate cost floors, feedstock prices and tariffs are **assumptions** (see `calibration.yaml`); cite real figures before quoting numbers.
- Photos are represented by perceptual hashes + EXIF metadata, not images. A CLIP/real-image demo needs real photos.
- GSTINs/PANs are format-valid with correct checksums but random; they do not correspond to real taxpayers.
- Trade-mirror data is **not** synthetic: use real UN Comtrade data for that agent.
