"""Cross-stream entity registry: recyclers in all 5 EPR streams, their people, identifiers,
plant photos, and the hidden ground truth (fraud operators, shell clusters, behaviours,
true physical parameters).

Everything downstream (plastic graph, yield operations, shell network) is built on this
registry, so one fraud recycler is the *same* entity across every agent's data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .bills import make_tariff
from .common import (CLUSTERS, addresses, bank_accounts, company_names, emails, gstins, jitter_latlon, make_ids,
                     pan_numbers, person_names, phones, rng_for, sample_clusters, write_table)

STREAMS = ["plastic", "ewaste", "battery", "tyre", "used_oil"]
LINK_TYPES = ["director", "signatory", "phone", "email", "bank", "address", "photo", "relative"]
STRONG = ["director", "signatory", "phone", "email", "bank", "address", "photo"]
PHOTO_TYPES = ["raw_material_storage", "production_area", "dispatch_area", "machinery", "weighbridge", "signboard"]


def _process_types(rng, cal, stream, n):
    if stream == "plastic":
        eol = cal["plastic"]["eol_types"]
        names = list(eol)
        p = np.array([eol[k]["share"] for k in names])
    else:
        procs = cal["streams"][stream]["processes"]
        names = list(procs)
        p = np.array([procs[k]["share"] for k in names])
    return rng.choice(names, n, p=p / p.sum())


def _kwh_range(cal, stream, proc):
    if stream == "plastic":
        return cal["plastic"]["eol_types"][proc]["kwh_per_t"]
    return cal["streams"][stream]["processes"][proc]["kwh_per_t"]


def _cap_spec(cal, stream):
    return cal["plastic"]["capacity_tpa_lognormal"] if stream == "plastic" else cal["streams"][stream]["capacity_tpa_lognormal"]


def _sample_operator_sizes(rng, cal):
    probs = cal["fraud"]["operator_size_probs"]
    sizes = np.array([int(k) for k in probs])
    p = np.array(list(probs.values()), float)
    return sizes, p / p.sum()


def generate(cal: dict, scale: str, seed: int, out: Path, manifest: dict) -> dict:
    rng = rng_for(seed, "registry")
    sc = cal["scales"][scale]
    counts = {s: int(sc[s]) for s in STREAMS}
    stream = np.concatenate([[s] * counts[s] for s in STREAMS]).astype(object)
    n = len(stream)
    ids = make_ids(rng, "ENT", n)
    df = pd.DataFrame({"entity_id": ids, "stream": stream})

    # ------------------------------------------------------------------ fraud slots & operators
    fraud_rate = cal["fraud"]["entity_fraud_rate"]
    is_fraud = np.zeros(n, bool)
    slots: dict[str, list[int]] = {}
    for s in STREAMS:
        idx = np.flatnonzero(stream == s)
        k = int(round(len(idx) * fraud_rate[s]))
        chosen = rng.choice(idx, k, replace=False)
        is_fraud[chosen] = True
        slots[s] = list(rng.permutation(chosen))

    sizes, size_p = _sample_operator_sizes(rng, cal)
    diff_names = list(cal["fraud"]["difficulty_mix"])
    diff_p = np.array(list(cal["fraud"]["difficulty_mix"].values()))
    operator_of = np.full(n, -1)
    difficulty = np.array([""] * n, dtype=object)
    migration = np.zeros(n, bool)
    op_streams = []
    op_id = 0
    late_streams = ["used_oil", "ewaste"]  # streams whose EPR launched last -> migration targets
    while any(slots[s] for s in STREAMS):
        remaining = np.array([len(slots[s]) for s in STREAMS], float)
        k = int(rng.choice(sizes, p=size_p))
        primary = STREAMS[rng.choice(len(STREAMS), p=remaining / remaining.sum())]
        members: list[int] = []
        member_streams = [primary]
        if k >= 2 and rng.random() < cal["fraud"]["cross_stream_operator_prob"]:
            others = [s for s in STREAMS if s != primary and slots[s]]
            if others:
                w = np.array([len(slots[s]) * (2.5 if s in late_streams else 1.0) for s in others])
                member_streams += list(rng.choice(others, min(len(others), rng.integers(1, 3)), replace=False, p=w / w.sum()))
        for j in range(k):
            s = member_streams[j % len(member_streams)] if j > 0 else primary
            if not slots[s]:
                s = next((x for x in member_streams if slots[x]), None)
                if s is None:
                    break
            members.append(slots[s].pop())
        d = rng.choice(diff_names, p=diff_p / diff_p.sum())
        for m in members:
            operator_of[m] = op_id
            difficulty[m] = d
        streams_here = sorted(set(stream[members]))
        if len(streams_here) > 1:
            for m in members:
                if stream[m] in late_streams and rng.random() < cal["fraud"]["late_stream_migration_prob"]:
                    migration[m] = True
        op_streams.append(streams_here)
        op_id += 1
    n_ops = op_id

    # ------------------------------------------------------------------ geography & names
    cl = sample_clusters(rng, n)
    # operators concentrate their shells in 1-2 home clusters (realistic + detectable)
    for o in range(n_ops):
        m = np.flatnonzero(operator_of == o)
        if len(m) > 1:
            homes = sample_clusters(rng, 2)
            cl[m] = np.where(rng.random(len(m)) < 0.75, homes[0], homes[1])
    lat0 = np.array([CLUSTERS[c].lat for c in cl])
    lon0 = np.array([CLUSTERS[c].lon for c in cl])
    lat, lon = jitter_latlon(rng, lat0, lon0, km=2.5)
    names = np.empty(n, dtype=object)
    cons = np.empty(n, dtype=object)
    for s in STREAMS:
        idx = np.flatnonzero(stream == s)
        nm, cs = company_names(rng, len(idx), s)
        names[idx], cons[idx] = nm, cs
    pans = pan_numbers(rng, n, cons, names)
    state_codes = [CLUSTERS[c].gst_code for c in cl]
    df["legal_name"] = names
    df["constitution"] = cons
    df["pan"] = pans
    df["gstin"] = gstins(state_codes, pans, rng.integers(1, 4, n))
    df["state"] = [CLUSTERS[c].state for c in cl]
    df["district"] = [CLUSTERS[c].district for c in cl]
    df["estate"] = [CLUSTERS[c].estate for c in cl]
    plots = [f"{rng.integers(1, 900)}{rng.choice(['', '', '', '/A', '/B', '-1', '-2'])}" for _ in range(n)]
    df["lat"], df["lon"] = lat.round(6), lon.round(6)

    # ------------------------------------------------------------------ dates
    launch = {s: pd.Timestamp(cal["epr_launch"][s]) for s in STREAMS}
    last_reg = pd.Timestamp("2026-02-28")
    reg = []
    for i in range(n):
        L = launch[stream[i]]
        span = (last_reg - L).days
        if migration[i]:
            d = L + pd.Timedelta(days=int(rng.integers(0, 120)))
        elif is_fraud[i]:
            d = L + pd.Timedelta(days=int(rng.uniform(0.15, 0.95) * span))
        else:
            # most legit recyclers register early; a tail of new entrants keeps arriving
            d = L + pd.Timedelta(days=int(min(span - 1, rng.exponential(span * 0.28))))
        reg.append(min(d, last_reg))
    reg = pd.to_datetime(pd.Series(reg))
    inc = []
    confound: list[list[str]] = [[] for _ in range(n)]
    for i in range(n):
        if is_fraud[i] and rng.random() < 0.6:
            days = int(rng.uniform(15, 540))          # freshly incorporated shell
        elif (not is_fraud[i]) and rng.random() < 0.06:
            days = int(rng.uniform(30, 700))          # genuine new company (confounder)
            confound[i].append("new_legit_company")
        else:
            days = int(rng.lognormal(np.log(9 * 365), 0.6))  # established firm (or bought dormant shell)
        inc.append(reg.iloc[i] - pd.Timedelta(days=days))
    df["incorporation_date"] = pd.to_datetime(pd.Series(inc)).dt.date
    df["epr_registration_date"] = reg.dt.date

    # ------------------------------------------------------------------ process, capacity, physics (true vs declared)
    proc = np.empty(n, dtype=object)
    true_cap = np.zeros(n)
    kwh = np.zeros(n)
    util = rng.uniform(*cal["plastic"]["utilization_base"], n)
    for s in STREAMS:
        idx = np.flatnonzero(stream == s)
        proc[idx] = _process_types(rng, cal, s, len(idx))
        spec = _cap_spec(cal, s)
        true_cap[idx] = np.clip(rng.lognormal(np.log(spec["median"]), spec["sigma"], len(idx)), spec["min"], spec["max"]).round(0)
        for i in idx:
            lo, hi = _kwh_range(cal, s, proc[i])
            kwh[i] = rng.uniform(lo, hi)
    behaviours = {b: np.zeros(n, bool) for b in
                  ["ghost_feedstock", "underprice", "cat3_inflation", "deadline_burst", "round_trip",
                   "output_fabrication", "doctored_bill", "capacity_inflated", "photo_reuse"]}
    hollow = np.zeros(n)
    hf = cal["fraud"]["hollow_fraction"]
    # Scenario knobs (defaults leave the baseline world unchanged): side_trace_scale shrinks the by-products of
    # fraud (underpricing, deadline bursts, ...), evasion_boost pushes the covering behaviours (fabricated outputs,
    # doctored bills, ghost feedstock) towards 1, i.e. fraudsters who know what auditors look at.
    sc = float(cal["fraud"].get("side_trace_scale", 1.0))
    eb = float(cal["fraud"].get("evasion_boost", 0.0))
    ev = lambda p: p + eb * (1.0 - p)
    for i in np.flatnonzero(is_fraud):
        d = difficulty[i]
        hollow[i] = rng.uniform(*hf[d])
        careful = {"obvious": 0.10, "moderate": 0.40, "subtle": 0.75}[d]
        behaviours["output_fabrication"][i] = rng.random() < ev(careful)
        behaviours["doctored_bill"][i] = rng.random() < ev({"obvious": 0.05, "moderate": 0.15, "subtle": 0.25}[d])
        behaviours["capacity_inflated"][i] = rng.random() < 0.5 * sc
        behaviours["photo_reuse"][i] = rng.random() < 0.45 * sc
        behaviours["underprice"][i] = rng.random() < 0.65 * sc
        behaviours["deadline_burst"][i] = rng.random() < 0.40 * sc
        if stream[i] == "plastic":
            behaviours["ghost_feedstock"][i] = rng.random() < ev({"obvious": 0.45, "moderate": 0.6, "subtle": 0.7}[d])
            behaviours["cat3_inflation"][i] = (proc[i] == "recycling") and rng.random() < 0.30 * sc
            behaviours["round_trip"][i] = rng.random() < 0.20 * sc
    declared_cap = true_cap.copy()
    ci = behaviours["capacity_inflated"]
    declared_cap[ci] = (true_cap[ci] * rng.uniform(1.5, 6.0, ci.sum())).round(-1)
    # legit declared capacity has small rounding noise only
    lg = ~ci
    declared_cap[lg] = (true_cap[lg] * rng.uniform(0.97, 1.05, lg.sum())).round(-1).clip(min=50)
    headroom = rng.uniform(1.2, 1.9, n)
    connected_kw = (true_cap * 0.85 * kwh / 6000.0 * headroom).clip(min=15).round(0)
    staff = (6 + declared_cap / rng.uniform(90, 160, n) * rng.uniform(0.8, 1.2, n)).round(0)

    solar_kw = np.full(n, np.nan)
    solar_frac = np.zeros(n)
    for i in range(n):
        if (not is_fraud[i]) and rng.random() < cal["benign"]["solar_adoption_prob"]:
            solar_frac[i] = rng.uniform(*cal["benign"]["solar_offset_range"])
            if rng.random() >= cal["benign"]["solar_undeclared_prob"]:
                solar_kw[i] = round(connected_kw[i] * solar_frac[i] * rng.uniform(0.9, 1.3), 0)
                confound[i].append("captive_solar_declared")
            else:
                confound[i].append("captive_solar_undeclared")
    fast = (~is_fraud) & (rng.random(n) < cal["benign"]["new_legit_fast_grower_prob"])
    for i in np.flatnonzero(fast):
        confound[i].append("fast_growing_legit")
    df["process_type"] = proc
    df["declared_capacity_tpa"] = declared_cap
    df["connected_load_kw"] = connected_kw
    df["declared_staff"] = staff.astype(int)
    df["captive_solar_kw"] = solar_kw

    # ------------------------------------------------------------------ people, consultants, groups
    n_people = int(n * 2.2)
    pid = make_ids(rng, "PER", n_people)
    pnames = person_names(rng, n_people)
    people = pd.DataFrame({
        "person_id": pid,
        "din": rng.choice(np.arange(1_000_000, 99_999_999), n_people, replace=False).astype(str),
        "name": pnames,
        "pan": pan_numbers(rng, n_people, ["person"] * n_people, pnames),
        "phone": phones(rng, n_people),
        "email": emails(rng, pnames, corporate_prob=0.0),
        "residential_city": [CLUSTERS[c].district for c in sample_clusters(rng, n_people)],
    })
    free_people = list(rng.permutation(n_people))

    def take_person():
        return free_people.pop()

    directors: list[list[int]] = []
    for i in range(n):
        k = 1 if cons[i] == "Proprietorship" else int(rng.choice([1, 2, 2, 3]))
        directors.append([take_person() for _ in range(k)])
    signatory = np.array([d[0] for d in directors])

    n_cons = cal["benign"]["consultant_count"]
    cons_names, _ = company_names(rng, n_cons, "plastic", np.array(["LLP"] * n_cons))
    cons_names = np.array([x.replace("Polymers", "EPR Consultants").replace("Plast", "Compliance").replace("Recyclers", "Advisory")
                           .replace("Polytech", "Enviro Consultants").replace("Granules", "Green Advisors")
                           .replace("Poly Industries", "Regulatory Services").replace("Plastics", "Compliance")
                           .replace("Enviro Recycling", "Enviro Consultancy") for x in cons_names], dtype=object)
    consultants = pd.DataFrame({
        "consultant_id": make_ids(rng, "CON", n_cons),
        "firm_name": cons_names,
        "phone": phones(rng, n_cons),
        "email": emails(rng, cons_names, corporate_prob=1.0),
        "state": [CLUSTERS[c].state for c in sample_clusters(rng, n_cons)],
    })
    cons_pop = rng.pareto(1.3, n_cons) + 1
    cons_state = consultants["state"].to_numpy()
    consultant_of = np.full(n, -1)
    for i in range(n):
        if rng.random() < cal["benign"]["consultant_use_prob"]:
            w = cons_pop * np.where(cons_state == df["state"].iat[i], 4.0, 1.0)
            consultant_of[i] = rng.choice(n_cons, p=w / w.sum())

    contact_phone = phones(rng, n)
    contact_email = emails(rng, names)
    ifsc, acct = bank_accounts(rng, n)
    plot_arr = np.array(plots, dtype=object)

    # legit business groups (benign shared directors / registered office)
    group_of = np.full(n, -1)
    legit_idx = list(rng.permutation(np.flatnonzero(~is_fraud)))
    n_group_members = int(n * cal["benign"]["business_group_share"])
    g = 0
    benign_pairs = []
    while n_group_members > 1 and len(legit_idx) > 4:
        k = int(rng.integers(2, 5))
        mem = [legit_idx.pop() for _ in range(k)]
        n_group_members -= k
        promoter = directors[mem[0]][0]
        for m in mem:
            group_of[m] = g
            confound[m].append("legit_business_group")
            if promoter not in directors[m]:
                directors[m].append(promoter)
        if rng.random() < 0.5:   # shared corporate email domain
            dom = contact_email[mem[0]].split("@")[1]
            for m in mem[1:]:
                contact_email[m] = contact_email[m].split("@")[0] + "@" + dom
        if rng.random() < 0.3:   # same compound, adjacent sheds (near-duplicate address)
            base = plot_arr[mem[0]].split("/")[0].split("-")[0]
            for j, m in enumerate(mem):
                cl[m] = cl[mem[0]]
                plot_arr[m] = f"{base}/{'ABCD'[j]}"
                confound[m].append("shared_compound_address")
        for a in range(len(mem)):
            for b in range(a + 1, len(mem)):
                benign_pairs.append((ids[mem[a]], ids[mem[b]], f"GRP{g}"))
        g += 1

    # ------------------------------------------------------------------ plant photos (64-bit perceptual hashes)
    photo_rows = []
    ent_photos: dict[int, list[int]] = {}
    for i in range(n):
        k = int(rng.integers(4, 9))
        types = list(rng.choice(PHOTO_TYPES, k, replace=k > len(PHOTO_TYPES)))
        ent_photos[i] = []
        for t in types:
            h = int(rng.integers(0, 2**63, dtype=np.int64)) ^ (int(rng.integers(0, 2)) << 63)
            exif = rng.random() > 0.15
            plat, plon = jitter_latlon(rng, df["lat"].iat[i], df["lon"].iat[i], km=0.15)
            when = pd.Timestamp(df["epr_registration_date"].iat[i]) - pd.Timedelta(days=int(rng.integers(1, 60)))
            photo_rows.append([i, t, h, exif, float(plat) if exif else np.nan, float(plon) if exif else np.nan,
                               when if exif else pd.NaT, "original"])
            ent_photos[i].append(len(photo_rows) - 1)

    def flip_bits(h, kbits):
        for b in rng.choice(64, kbits, replace=False):
            h ^= (1 << int(b))
        return h

    # ------------------------------------------------------------------ shell clusters: plant shared identifiers
    pair_rows = []
    for o in range(n_ops):
        mem = list(rng.permutation(np.flatnonzero(operator_of == o)))
        if len(mem) < 2:
            continue
        d = difficulty[mem[0]]
        n_types = {"obvious": int(rng.integers(3, 6)), "moderate": 2, "subtle": 1}[d]
        pool = LINK_TYPES if d == "subtle" else STRONG
        weights = np.array([{"director": 3, "signatory": 2, "phone": 2, "email": 2, "bank": 1, "address": 2,
                             "photo": 2, "relative": 3}[t] for t in pool], float)
        types = list(rng.choice(pool, min(n_types, len(pool)), replace=False, p=weights / weights.sum()))
        mastermind = directors[mem[0]][0]
        for a, b in zip(mem[:-1], mem[1:]):
            # every consecutive pair shares >=1 identifier, so the cluster is connected in principle
            k = 1 if d == "subtle" else int(rng.integers(1, len(types) + 1))
            for t in rng.choice(types, k, replace=False):
                if t == "director":
                    directors[b].append(mastermind if rng.random() < 0.6 else directors[a][0])
                elif t == "signatory":
                    signatory[b] = signatory[a]
                elif t == "phone":
                    contact_phone[b] = contact_phone[a]
                elif t == "email":
                    contact_email[b] = contact_email[a]
                elif t == "bank":
                    ifsc[b], acct[b] = ifsc[a], acct[a]
                elif t == "address":
                    cl[b] = cl[a]
                    plot_arr[b] = plot_arr[a]
                elif t == "photo":
                    src = ent_photos[a][int(rng.integers(len(ent_photos[a])))]
                    r = photo_rows[src]
                    photo_rows.append([b, r[1], flip_bits(r[2], int(rng.integers(0, 7))), r[3], r[4], r[5], r[6], "reused_sibling"])
                    ent_photos[b].append(len(photo_rows) - 1)
                elif t == "relative":
                    # a director sharing surname + residential city with a sibling's director (weak link)
                    rel = take_person()
                    src_p = directors[a][0]
                    sur = people.at[src_p, "name"].split()[-1]
                    people.at[rel, "name"] = people.at[rel, "name"].split()[0] + " " + sur
                    people.at[rel, "residential_city"] = people.at[src_p, "residential_city"]
                    directors[b].append(rel)
        for x in range(len(mem)):
            for y in range(x + 1, len(mem)):
                pair_rows.append((ids[mem[x]], ids[mem[y]], f"OP{o}", difficulty[mem[x]]))

    # photo reuse from unrelated legit plants (stock-photo style fraud)
    legit_all = np.flatnonzero(~is_fraud)
    for i in np.flatnonzero(behaviours["photo_reuse"]):
        donor = int(rng.choice(legit_all))
        for src in rng.choice(ent_photos[donor], min(len(ent_photos[donor]), int(rng.integers(1, 4))), replace=False):
            r = photo_rows[src]
            strip = rng.random() < 0.4
            photo_rows.append([i, r[1], flip_bits(r[2], int(rng.integers(0, 5))), not strip and r[3],
                               np.nan if strip else r[4], np.nan if strip else r[5], pd.NaT if strip else r[6], "reused_external"])
    # benign duplicates: same entity re-uploads at renewal; group siblings share a common-facility photo
    for i in rng.choice(n, int(n * 0.08), replace=False):
        src = ent_photos[i][0]
        r = photo_rows[src]
        photo_rows.append([i, r[1], flip_bits(r[2], int(rng.integers(0, 3))), r[3], r[4], r[5], r[6], "benign_self_reupload"])
    for gid in range(g):
        mem = np.flatnonzero(group_of == gid)
        if len(mem) > 1 and rng.random() < 0.35:
            r = photo_rows[ent_photos[mem[0]][0]]
            photo_rows.append([mem[1], r[1], flip_bits(r[2], int(rng.integers(0, 4))), r[3], r[4], r[5], r[6], "benign_group_shared"])

    # finalise addresses after cluster/plot edits
    df["state"] = [CLUSTERS[c].state for c in cl]
    df["district"] = [CLUSTERS[c].district for c in cl]
    df["estate"] = [CLUSTERS[c].estate for c in cl]
    df["address"] = addresses(rng, cl, plot_arr)
    same_addr_fix = df.groupby(df["address"].str.split(" - ").str[0])["address"].transform("first")
    df["address"] = same_addr_fix   # identical plot -> identical full address string (incl. pincode)
    df["contact_phone"] = contact_phone
    df["contact_email"] = contact_email
    df["bank_ifsc"] = ifsc
    df["bank_account"] = acct
    df["consultant_id"] = [consultants["consultant_id"].iat[c] if c >= 0 else None for c in consultant_of]
    df["filing_phone"] = [consultants["phone"].iat[c] if c >= 0 else p for c, p in zip(consultant_of, contact_phone)]
    df["filing_email"] = [consultants["email"].iat[c] if c >= 0 else e for c, e in zip(consultant_of, contact_email)]
    df["authorised_signatory_id"] = people["person_id"].to_numpy()[signatory]

    ed_rows = []
    for i in range(n):
        for j, p in enumerate(dict.fromkeys(directors[i])):
            role = "proprietor" if cons[i] == "Proprietorship" and j == 0 else ("partner" if cons[i] in ("LLP", "Partnership") else "director")
            appointed = pd.Timestamp(df["incorporation_date"].iat[i]) + pd.Timedelta(days=int(rng.integers(0, 400)) if j else 0)
            ed_rows.append((ids[i], people["person_id"].iat[p], role, appointed.date()))
    entity_directors = pd.DataFrame(ed_rows, columns=["entity_id", "person_id", "role", "appointed_date"])

    photos = pd.DataFrame(photo_rows, columns=["ent_idx", "photo_type", "phash_int", "exif_present", "exif_lat", "exif_lon",
                                               "exif_datetime", "_origin"])
    photos = photos.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
    photos.insert(0, "photo_id", make_ids(rng, "IMG", len(photos), 8))
    photos.insert(1, "entity_id", ids[photos["ent_idx"].to_numpy()])
    photos["phash"] = [format(int(h) & ((1 << 64) - 1), "016x") for h in photos["phash_int"]]
    photos["upload_date"] = pd.to_datetime(df.set_index("entity_id").loc[photos["entity_id"], "epr_registration_date"].to_numpy())
    photo_labels = photos[["photo_id", "entity_id", "_origin"]].rename(columns={"_origin": "origin"})
    photos = photos.drop(columns=["ent_idx", "phash_int", "_origin"])

    # ------------------------------------------------------------------ labels
    beh = pd.DataFrame(behaviours)
    labels = pd.DataFrame({
        "entity_id": ids, "stream": stream, "is_fraud": is_fraud,
        "operator_id": np.where(operator_of >= 0, np.char.add("OP", operator_of.astype(str)), None),
        "operator_size": pd.Series(operator_of).map(pd.Series(operator_of[operator_of >= 0]).value_counts()).fillna(0).astype(int).to_numpy(),
        "difficulty": np.where(is_fraud, difficulty, None),
        "hollow_fraction": hollow.round(4),
        "is_migration_shell": migration,
        "true_capacity_tpa": true_cap,
        "benign_group_id": np.where(group_of >= 0, np.char.add("GRP", group_of.astype(str)), None),
        "confounders": [";".join(c) for c in confound],
    })
    labels = pd.concat([labels, beh], axis=1)
    physics = pd.DataFrame({"entity_id": ids, "stream": stream, "process_type": proc, "true_capacity_tpa": true_cap,
                            "kwh_per_t": kwh.round(1), "utilization_base": util.round(4), "solar_fraction": solar_frac.round(4),
                            "hollow_fraction": hollow.round(4), "is_fraud": is_fraud, "difficulty": np.where(is_fraud, difficulty, None),
                            "fast_grower": fast})
    physics = pd.concat([physics, beh], axis=1)
    same_op = pd.DataFrame(pair_rows, columns=["entity_a", "entity_b", "operator_id", "difficulty"])
    benign = pd.DataFrame(benign_pairs, columns=["entity_a", "entity_b", "benign_group_id"])

    reg_dir = out / "registry"
    lab_dir = out / "labels" / "registry"
    write_table(df, reg_dir / "entities.parquet", manifest)
    write_table(people, reg_dir / "people.parquet", manifest)
    write_table(entity_directors, reg_dir / "entity_directors.parquet", manifest)
    write_table(consultants, reg_dir / "consultants.parquet", manifest)
    write_table(photos, reg_dir / "plant_photos.parquet", manifest)
    write_table(labels, lab_dir / "entity_labels.parquet", manifest)
    write_table(physics, lab_dir / "entity_physics.parquet", manifest)
    write_table(same_op, lab_dir / "same_operator_pairs.parquet", manifest)
    write_table(benign, lab_dir / "benign_link_pairs.parquet", manifest)
    write_table(photo_labels, lab_dir / "photo_origin.parquet", manifest)
    tariff = make_tariff(rng_for(seed, "tariff"), cal["plastic"], {c.state for c in CLUSTERS})
    write_table(tariff, reg_dir / "tariff_reference.parquet", manifest)
    return {"entities": df, "physics": physics, "labels": labels, "tariff": tariff}
