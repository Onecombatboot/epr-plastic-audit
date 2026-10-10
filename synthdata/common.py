"""Shared helpers: RNG, Indian identifiers (PAN/GSTIN/DIN/IFSC), geography, names, IO."""
from __future__ import annotations

import json
import string
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ALNUM = string.digits + string.ascii_uppercase


def load_calibration() -> dict:
    with open(HERE / "calibration.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def rng_for(seed: int, name: str) -> np.random.Generator:
    """Independent, reproducible stream per generator (adding one generator never shifts another)."""
    salt = int.from_bytes(name.encode(), "little") % (2**32)
    return np.random.default_rng([seed, salt])


# --------------------------------------------------------------------------- geography
# (state, gst_code, pin_prefix, weight, [(cluster/estate, district, lat, lon), ...])
GEO = [
    ("Gujarat", "24", "3", 0.13, [("Vatva GIDC", "Ahmedabad", 22.960, 72.630), ("Vapi GIDC", "Valsad", 20.370, 72.910),
                                 ("Ankleshwar GIDC", "Bharuch", 21.630, 73.010), ("Sachin GIDC", "Surat", 21.080, 72.880),
                                 ("Shapar Veraval", "Rajkot", 22.150, 70.780), ("Morbi Ceramic Zone", "Morbi", 22.820, 70.840)]),
    ("Maharashtra", "27", "4", 0.13, [("Bhiwandi", "Thane", 19.300, 73.060), ("Taloja MIDC", "Raigad", 19.060, 73.120),
                                     ("Chakan MIDC", "Pune", 18.760, 73.860), ("Butibori MIDC", "Nagpur", 20.930, 79.000),
                                     ("Waluj MIDC", "Aurangabad", 19.840, 75.240), ("Tarapur MIDC", "Palghar", 19.820, 72.700)]),
    ("Delhi", "07", "1", 0.06, [("Narela Industrial Area", "North Delhi", 28.850, 77.090), ("Bawana Industrial Area", "North West Delhi", 28.800, 77.030),
                               ("Mundka Industrial Area", "West Delhi", 28.680, 77.030)]),
    ("Haryana", "06", "1", 0.07, [("Panipat Industrial Area", "Panipat", 29.390, 76.970), ("Faridabad Sector 24", "Faridabad", 28.400, 77.310),
                                 ("IMT Manesar", "Gurugram", 28.360, 76.940), ("Kundli Industrial Area", "Sonipat", 28.880, 77.120)]),
    ("Uttar Pradesh", "09", "2", 0.10, [("Noida Phase II", "Gautam Buddh Nagar", 28.540, 77.390), ("Loni Industrial Area", "Ghaziabad", 28.750, 77.290),
                                       ("Panki Industrial Area", "Kanpur", 26.450, 80.330), ("Partapur", "Meerut", 28.980, 77.710),
                                       ("Sikandra", "Agra", 27.180, 78.010)]),
    ("Tamil Nadu", "33", "6", 0.09, [("Ambattur Industrial Estate", "Chennai", 13.100, 80.160), ("SIDCO Kurichi", "Coimbatore", 11.020, 76.960),
                                    ("Tiruppur Knitwear Cluster", "Tiruppur", 11.110, 77.340), ("SIPCOT Hosur", "Krishnagiri", 12.740, 77.830)]),
    ("Karnataka", "29", "5", 0.07, [("Peenya Industrial Area", "Bengaluru Urban", 13.030, 77.520), ("Hebbal Industrial Area", "Mysuru", 12.300, 76.640),
                                   ("Udyambag", "Belagavi", 15.850, 74.500)]),
    ("Telangana", "36", "5", 0.06, [("Patancheru IDA", "Sangareddy", 17.530, 78.260), ("Jeedimetla IDA", "Medchal", 17.510, 78.450)]),
    ("Andhra Pradesh", "37", "5", 0.04, [("Autonagar", "Visakhapatnam", 17.690, 83.220), ("Sri City", "Tirupati", 13.550, 79.990)]),
    ("West Bengal", "19", "7", 0.05, [("Howrah Foundry Park", "Howrah", 22.590, 88.310), ("Durgapur Industrial Area", "Paschim Bardhaman", 23.520, 87.310)]),
    ("Rajasthan", "08", "3", 0.05, [("VKI Area", "Jaipur", 26.980, 75.770), ("RIICO Bhiwadi", "Khairthal-Tijara", 28.210, 76.860),
                                   ("Basni Industrial Area", "Jodhpur", 26.240, 73.020)]),
    ("Punjab", "03", "1", 0.04, [("Focal Point", "Ludhiana", 30.900, 75.850), ("Industrial Area Phase 8", "Mohali", 30.700, 76.720)]),
    ("Madhya Pradesh", "23", "4", 0.04, [("Pithampur", "Dhar", 22.610, 75.680), ("Govindpura", "Bhopal", 23.250, 77.460)]),
    ("Uttarakhand", "05", "2", 0.02, [("SIDCUL Haridwar", "Haridwar", 29.940, 78.070), ("SIDCUL Pantnagar", "Udham Singh Nagar", 28.980, 79.400)]),
    ("Kerala", "32", "6", 0.02, [("Eloor Industrial Area", "Ernakulam", 10.070, 76.300)]),
    ("Odisha", "21", "7", 0.02, [("Jagatpur Industrial Estate", "Cuttack", 20.500, 85.880)]),
    ("Himachal Pradesh", "02", "1", 0.01, [("Baddi Industrial Area", "Solan", 30.960, 76.790)]),
]


@dataclass(frozen=True)
class Cluster:
    state: str
    gst_code: str
    pin_prefix: str
    estate: str
    district: str
    lat: float
    lon: float


CLUSTERS: list[Cluster] = [Cluster(s, g, p, e, d, la, lo) for s, g, p, _w, cl in GEO for e, d, la, lo in cl]
CLUSTER_WEIGHTS = np.array([w / len(cl) for _s, _g, _p, w, cl in GEO for _ in cl])
CLUSTER_WEIGHTS = CLUSTER_WEIGHTS / CLUSTER_WEIGHTS.sum()
STATE_OF_CLUSTER = np.array([c.state for c in CLUSTERS])


def sample_clusters(rng, n: int) -> np.ndarray:
    return rng.choice(len(CLUSTERS), size=n, p=CLUSTER_WEIGHTS)


def jitter_latlon(rng, lat, lon, km: float):
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    d = km / 111.0
    return (lat + rng.normal(0, d, lat.shape), lon + rng.normal(0, d, lon.shape) / np.cos(np.radians(lat)))


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def pincode(rng, prefix: str) -> str:
    return prefix + "".join(rng.choice(list(string.digits), 5))


# --------------------------------------------------------------------------- names
FIRST = ("Aarav Aditya Ajay Akash Amit Anand Anil Ankit Arjun Arun Ashok Deepak Dinesh Ganesh Gaurav Harish Hitesh Jayesh "
         "Kamal Karan Kiran Krishna Kunal Mahesh Manish Manoj Mohan Mukesh Naresh Nikhil Nitin Pankaj Paresh Pradeep Prakash "
         "Pramod Rahul Rajesh Rakesh Ramesh Ravi Rohit Sachin Sanjay Santosh Satish Shailesh Shyam Sunil Suresh Tushar Vijay "
         "Vikas Vinod Vishal Yogesh Aisha Anita Anjali Deepa Divya Geeta Kavita Lakshmi Meena Neha Pooja Priya Rekha Ritu "
         "Sangeeta Seema Shalini Sunita Swati Usha Farhan Imran Irfan Salim Zoya Gurpreet Harpreet Jaspreet Manpreet Senthil "
         "Murugan Karthik Venkatesh Srinivas Subramanian Lakshman Basavaraj Mallikarjun Debashis Sourav Partha Arijit").split()
SURNAMES = ("Shah Patel Mehta Desai Joshi Trivedi Parmar Chauhan Solanki Jain Agarwal Gupta Bansal Goyal Mittal Singhal "
            "Sharma Verma Mishra Tiwari Pandey Dubey Yadav Singh Thakur Rathore Shekhawat Kumar Rao Reddy Naidu Iyer Iyengar "
            "Nair Menon Pillai Kulkarni Deshpande Patil Pawar Jadhav Shinde More Gaikwad Khan Qureshi Ansari Sheikh Siddiqui "
            "Malhotra Kapoor Khanna Arora Sethi Bhatia Gill Sandhu Dhillon Grewal Banerjee Chatterjee Mukherjee Das Ghosh Bose "
            "Sen Mondal Biswas Murugan Subramaniam Krishnan Gowda Hegde Shetty").split()
PREFIX = ("Shree Shri Om Sai Jai Maa Balaji Ganesh Krishna Laxmi Durga Bharat Hind National Royal Supreme Global Green "
          "Eco Earth Prakriti Vasundhara Sanjeevani Navjeevan Amrit Pragati Unity Galaxy Sunrise Everest Ganga Narmada "
          "Kaveri Himalaya Sahyadri Deccan Western Eastern Northern Southern Pioneer Apex Prime Sterling Metro Star").split()
STREAM_WORDS = {
    "plastic": ["Polymers", "Plast", "Plastics", "Recyclers", "Polytech", "Poly Industries", "Enviro Recycling", "Granules"],
    "ewaste": ["E-Waste Solutions", "E-Recyclers", "Ecotech Recycling", "Electronics Recycling", "Enviro Solutions", "E-Cycle"],
    "battery": ["Metals", "Smelters", "Battery Recyclers", "Alloys", "Lead Industries", "Energy Materials"],
    "tyre": ["Tyre Recyclers", "Rubber Industries", "Pyro Oils", "Carbon & Fuels", "Rubber Reclaim", "Green Fuels"],
    "used_oil": ["Petrochem", "Lubricants", "Oil Refinery", "Re-Refiners", "Hydrocarbons", "Petro Products"],
    "offtaker": ["Packaging", "Moulders", "Containers", "Pipes", "Films", "Industries", "Products"],
    "collector": ["Waste Traders", "Scrap Suppliers", "Kabadi Centre", "Recyclable Traders", "Waste Management", "Scrap Co"],
    "brand": ["Foods", "Beverages", "Consumer Products", "FMCG", "Personal Care", "Retail", "Pharma", "Dairy", "Snacks", "Home Care"],
}
CONSTITUTION = ["Private Limited", "LLP", "Proprietorship", "Partnership", "Limited"]
CONSTITUTION_P = [0.48, 0.18, 0.18, 0.12, 0.04]
SUFFIX = {"Private Limited": "Pvt Ltd", "LLP": "LLP", "Proprietorship": "", "Partnership": "& Co", "Limited": "Ltd"}


def person_names(rng, n):
    return np.char.add(np.char.add(rng.choice(FIRST, n).astype(str), " "), rng.choice(SURNAMES, n).astype(str))


def company_names(rng, n, kind: str, constitutions=None):
    words = STREAM_WORDS[kind]
    if constitutions is None:
        constitutions = rng.choice(CONSTITUTION, n, p=CONSTITUTION_P)
    p = rng.choice(PREFIX, n)
    p2 = np.where(rng.random(n) < 0.35, np.char.add(" ", rng.choice(PREFIX, n).astype(str)), "")
    w = rng.choice(words, n)
    out = [f"{a}{b} {c} {SUFFIX[k]}".strip() for a, b, c, k in zip(p, p2, w, constitutions)]
    return np.array(out, dtype=object), constitutions


# --------------------------------------------------------------------------- identifiers
def _rand_str(rng, alphabet, n, k):
    idx = rng.integers(0, len(alphabet), size=(n, k))
    arr = np.array(list(alphabet))[idx]
    return np.array(["".join(r) for r in arr], dtype=object)


def make_ids(rng, prefix: str, n: int, width: int = 7) -> np.ndarray:
    """Random-looking unique IDs (not sequential, so ID order leaks nothing)."""
    out: set[str] = set()
    while len(out) < n:
        out.update(_rand_str(rng, ALNUM, n - len(out) + 16, width))
    arr = np.array(sorted(out)[:n], dtype=object)
    rng.shuffle(arr)
    return np.array([f"{prefix}-{x}" for x in arr], dtype=object)


PAN_TYPE = {"Private Limited": "C", "Limited": "C", "LLP": "F", "Partnership": "F", "Proprietorship": "P", "person": "P"}


def pan_numbers(rng, n, entity_kinds, names) -> np.ndarray:
    letters = string.ascii_uppercase
    a = _rand_str(rng, letters, n, 3)
    d = _rand_str(rng, string.digits, n, 4)
    e = _rand_str(rng, letters, n, 1)
    out = []
    for i in range(n):
        kind = PAN_TYPE.get(entity_kinds[i], "C")
        nm = str(names[i])
        # 5th char = first letter of surname (person) or entity name
        fifth = (nm.split()[-1][0] if kind == "P" else nm[0]).upper()
        if fifth not in letters:
            fifth = "X"
        out.append(f"{a[i]}{kind}{fifth}{d[i]}{e[i]}")
    return np.array(out, dtype=object)


def gstin_checksum(first14: str) -> str:
    total = 0
    for i, ch in enumerate(first14):
        v = ALNUM.index(ch) * (1 if i % 2 == 0 else 2)
        total += v // 36 + v % 36
    return ALNUM[(36 - total % 36) % 36]


def gstins(state_codes, pans, entity_nums=None) -> np.ndarray:
    out = []
    for i, (sc, pan) in enumerate(zip(state_codes, pans)):
        en = "1" if entity_nums is None else str(entity_nums[i])
        base = f"{sc}{pan}{en}Z"
        out.append(base + gstin_checksum(base))
    return np.array(out, dtype=object)


def phones(rng, n):
    first = rng.choice(list("6789"), n)
    rest = _rand_str(rng, string.digits, n, 9)
    return np.char.add(np.char.add("+91", first.astype(str)), rest.astype(str)).astype(object)


EMAIL_DOMAINS = ["gmail.com", "yahoo.co.in", "rediffmail.com", "outlook.com", "hotmail.com"]


def emails(rng, names, corporate_prob=0.45):
    out = []
    for nm in names:
        slug = "".join(ch for ch in str(nm).lower().replace(" ", ".") if ch.isalnum() or ch == ".")[:28].strip(".")
        if rng.random() < corporate_prob:
            dom = slug.split(".")[0] + rng.choice(["recycling", "enviro", "industries", "group", "india"]) + rng.choice([".com", ".in", ".co.in"])
            out.append(f"{rng.choice(['info', 'accounts', 'compliance', 'admin', 'epr'])}@{dom}")
        else:
            out.append(f"{slug}{rng.integers(1, 999)}@{rng.choice(EMAIL_DOMAINS)}")
    return np.array(out, dtype=object)


BANKS = ["SBIN", "HDFC", "ICIC", "UTIB", "PUNB", "BARB", "CNRB", "UBIN", "KKBK", "IDIB", "YESB", "IOBA", "BKID", "MAHB"]


def bank_accounts(rng, n):
    ifsc = np.array([f"{b}0{x}" for b, x in zip(rng.choice(BANKS, n), _rand_str(rng, ALNUM, n, 6))], dtype=object)
    lens = rng.integers(11, 17, n)
    acct = np.array([str(rng.integers(10 ** (k - 1), 10**k - 1)) for k in lens], dtype=object)
    return ifsc, acct


STATE_RTO = {"Gujarat": "GJ", "Maharashtra": "MH", "Delhi": "DL", "Haryana": "HR", "Uttar Pradesh": "UP", "Tamil Nadu": "TN",
             "Karnataka": "KA", "Telangana": "TS", "Andhra Pradesh": "AP", "West Bengal": "WB", "Rajasthan": "RJ", "Punjab": "PB",
             "Madhya Pradesh": "MP", "Uttarakhand": "UK", "Kerala": "KL", "Odisha": "OD", "Himachal Pradesh": "HP"}


def vehicle_regs(rng, states):
    states = np.asarray(states)
    n = len(states)
    rto = np.array([STATE_RTO.get(s, "DL") for s in states], dtype=object)
    dist = rng.integers(1, 60, n)
    ser = _rand_str(rng, string.ascii_uppercase, n, 2)
    num = rng.integers(1, 9999, n)
    return np.array([f"{a}{b:02d}{c}{d:04d}" for a, b, c, d in zip(rto, dist, ser, num)], dtype=object)


def addresses(rng, cluster_idx, plots=None):
    out = []
    for i, ci in enumerate(cluster_idx):
        c = CLUSTERS[ci]
        plot = plots[i] if plots is not None else f"{rng.integers(1, 900)}{rng.choice(['', '', '', '/A', '/B', '-1', '-2'])}"
        out.append(f"Plot No. {plot}, {c.estate}, {c.district}, {c.state} - {pincode(rng, c.pin_prefix)}")
    return np.array(out, dtype=object)


# --------------------------------------------------------------------------- time
def fy_months(cal) -> pd.DatetimeIndex:
    return pd.date_range(cal["period"]["fy_start"], cal["period"]["fy_end"], freq="MS")


def fy_label(ts) -> pd.Series:
    ts = pd.to_datetime(pd.Series(ts))
    y = ts.dt.year - (ts.dt.month < 4).astype(int)
    return "FY" + y.astype(str) + "-" + ((y + 1) % 100).astype(str).str.zfill(2)


def lognormal_capacity(rng, n, spec):
    v = rng.lognormal(np.log(spec["median"]), spec["sigma"], n)
    return np.clip(v, spec["min"], spec["max"]).round(0)


def dirichlet_around(rng, mean, conc, n):
    mean = np.asarray(mean, float)
    mean = np.clip(mean, 1e-3, None)
    mean = mean / mean.sum()
    return rng.dirichlet(mean * conc, n)


# --------------------------------------------------------------------------- IO
def write_table(df: pd.DataFrame, path: Path, manifest: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")
    if manifest is not None:
        manifest[str(path)] = {"rows": int(len(df)), "cols": list(map(str, df.columns))}


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
