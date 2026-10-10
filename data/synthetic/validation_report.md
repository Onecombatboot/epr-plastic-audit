# Validation report

Data: `data\synthetic`

| Section | Check | Pass | Detail |
|---|---|---|---|
| registry | entity_id unique | ✅ | 4550 |
| registry | labels align with entities | ✅ |  |
| registry | GSTIN checksum valid | ✅ | 1.0000 |
| registry | PAN format | ✅ |  |
| registry | incorporation <= EPR registration | ✅ |  |
| registry | directors reference people | ✅ |  |
| registry | fraud rate per stream in 5-12% | ✅ | {'battery': 0.071, 'ewaste': 0.07, 'plastic': 0.09, 'tyre': 0.08, 'used_oil': 0.1} |
| registry | shell clusters connected via shared identifiers | ✅ | 83/83 operators |
| registry | photo hashes are 16 hex chars | ✅ |  |
| plastic | invoice_id unique | ✅ | 2572616 |
| plastic | lot_id unique | ✅ | 289275 |
| plastic | sale_id unique | ✅ | 646106 |
| plastic | collector_id unique | ✅ | 25607 |
| plastic | brand_id unique | ✅ | 12000 |
| plastic | bill_id unique | ✅ | 73719 |
| plastic | invoice buyers are plastic recyclers | ✅ |  |
| plastic | invoice sellers exist | ✅ |  |
| plastic | sales reference lots & brands | ✅ |  |
| plastic | no lot oversold | ✅ | 0 |
| plastic | sale_date >= lot generation date | ✅ | 1.0000 |
| plastic | e-way bill missing rate (value>50k): legit low | ✅ | {'ghost': 0.567, 'inflated_real': 0.015, 'legit': 0.015, 'round_trip': 0.262} |
| plastic | GSTIN checksum valid (collectors) | ✅ |  |
| plastic | PHYSICS legit claims <= true capacity | ✅ | 1.0000 |
| plastic | PHYSICS legit invoice coverage 0.95-1.10 | ✅ | median 1.016 |
| plastic | PHYSICS legit bills arithmetically consistent | ✅ | max 0.0004 |
| plastic | DIFFICULTY naive AUC [neg_price] | ✅ | {'all': 0.892, 'obvious': 0.881, 'moderate': 0.915, 'subtle': 0.871} |
| plastic | DIFFICULTY naive AUC [neg_kwh] | ✅ | {'all': 0.767, 'obvious': 0.903, 'moderate': 0.795, 'subtle': 0.554} |
| plastic | DIFFICULTY naive AUC [load_ratio] | ✅ | {'all': 0.779, 'obvious': 0.867, 'moderate': 0.832, 'subtle': 0.585} |
| plastic | DIFFICULTY naive AUC [neg_inv_cov] | ✅ | {'all': 0.554, 'obvious': 0.756, 'moderate': 0.513, 'subtle': 0.368} |
| plastic | DIFFICULTY naive AUC [bill tariff gap, doctored-bill fraud only] | ✅ | {'all': 1.0, 'obvious': 1.0, 'moderate': 1.0, 'subtle': 1.0} |
| plastic | brand exposure higher for price-sensitive buyers | ✅ | {'balanced': 0.0499, 'price_sensitive': 0.1721, 'quality': 0.0327} |
| plastic | ghost collector share 0.2-5% | ✅ | 0.0237 |
| plastic | some certificates go unsold (crowding-out) | ✅ | 0.122 |
| yield | stream configs present | ✅ | ['battery', 'ewaste', 'plastic', 'tyre', 'used_oil'] |
| yield | PHYSICS legit |output balance gap| < 12% (median) | ✅ | 0.025 |
| yield | PHYSICS legit annual input <= true capacity | ✅ |  |
| yield | DIFFICULTY naive AUC [balance_gap] | ✅ | {'all': 0.832, 'obvious': 0.951, 'moderate': 0.834, 'subtle': 0.612} |
| yield | DIFFICULTY naive AUC [neg_kwh_ratio] | ✅ | {'all': 0.952, 'obvious': 0.979, 'moderate': 0.974, 'subtle': 0.844} |
| ocems | rows = units x timesteps | ✅ | 3456000 |
| ocems | row labels aligned | ✅ |  |
| ocems | tampered row share 1-15% | ✅ | 0.0308 |
| ocems | all 8 tamper types present | ✅ | {'scaling': 29, 'flatline': 26, 'night_bypass': 23, 'replay': 19, 'smoothing': 18, 'clipping': 16, 'dilution': 16, 'implausible_ratio': 15} |
| ocems | benign faults present | ✅ | {'': 2670667, 'sensor_drift': 645536, 'comm_gap': 62791, 'power_outage': 45464, 'maintenance': 12346, 'calibration_flagged': 10719, 'calibration_unflagged': 4898, 'stuck_sensor': 3579} |
| ocems | PHYSICS clean BOD < COD | ✅ | 1.00000 |
| ocems | hidden exceedances exist (motive) | ✅ | 51878 |
| ocems | DIFFICULTY naive AUC [low COD variability -> flatline/smoothing day] | ✅ | 0.759 |
| bmw | pickup_id unique | ✅ | 4652377 |
| bmw | pickups reference facilities | ✅ |  |
| bmw | pickup labels aligned | ✅ |  |
| bmw | PHYSICS legit kg/bed/day in 0.08-0.8 | ✅ | median 0.263 |
| bmw | DIFFICULTY naive AUC [low kg per bed -> under-handover] | ✅ | {'all': 0.815, 'obvious': 0.898, 'moderate': 0.842, 'subtle': 0.719} |
| bmw | ghost pickups present | ✅ | 8145 |
| bmw | DIFFICULTY naive AUC [no/short GPS dwell -> ghost pickup] | ✅ | 0.957 |
| bmw | PHYSICS legit CBWTFs never burn above permitted capacity | ✅ | 0.000 |
| bmw | DIFFICULTY naive AUC [days above permitted capacity -> overcapacity] | ✅ | {'all': 0.958, 'obvious': 0.75, 'moderate': 1.0, 'subtle': 1.0} |
| bmw | DIFFICULTY naive AUC [low diesel per kg -> overcapacity] | ✅ | 0.698 |

**56/56 checks passed.**