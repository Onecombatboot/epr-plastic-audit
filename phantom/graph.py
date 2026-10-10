"""Layer 3: graph autoencoder (DOMINANT-style) on the recycler / collector / offtaker / brand graph.

DOMINANT (Ding et al., SDM 2019) trains a GCN autoencoder to reconstruct node attributes AND the adjacency
matrix; nodes that reconstruct badly are anomalous. We implement it directly in PyTorch + PyG because
PyGOD's mini-batch path needs pyg-lib / torch-sparse (no Windows wheels for Python 3.14) and its full-batch
path builds a dense N x N adjacency (N = 44,000 here, about 7.7 GB). Our structure decoder scores observed
edges and sampled non-edges instead of the dense matrix: same objective, linear memory.

Anomaly score of a node = alpha * attribute error + (1 - alpha) * structure error, each z-scored within the
node's own type so collectors, offtakers, recyclers and brands are compared with their peers.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch_geometric.nn import GCNConv

from .features import collector_table, offtaker_table

TYPES = ["recycler", "collector", "offtaker", "brand"]


def _log_std(df: pd.DataFrame) -> pd.DataFrame:
    x = np.log1p(df.clip(lower=0))
    return (x - x.mean()) / x.std().replace(0, 1)


LOG_COLS = ["in_deg", "in_w", "in_n", "out_deg", "out_w", "out_n", "mean_rate_in", "mean_rate_out"]
LIN_COLS = ["in_hhi", "in_top", "out_hhi", "out_top"]


def _type_block(St: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in St.columns if St[c].abs().sum() > 0]          # drop structure that does not exist for this node type
    L = _log_std(St[[c for c in LOG_COLS if c in keep]])
    Z = St[[c for c in LIN_COLS if c in keep]]
    Z = (Z - Z.mean()) / Z.std().replace(0, 1)
    return pd.concat([L, Z], axis=1)


def build_graph(t: dict[str, pd.DataFrame], F: pd.DataFrame, U: pd.DataFrame | None = None):
    """Return (x, edge_index, nodes) where nodes is a DataFrame(node_id, type, idx).

    U (optional) is the recycler suspicion matrix; if given it is appended to recycler attributes.
    """
    fi = t["feedstock_invoices"].rename(columns={"buyer_id": "dst", "seller_id": "src"})
    fi["invoice_date"] = pd.to_datetime(fi["invoice_date"])
    fi["month"] = fi["invoice_date"].dt.to_period("M").dt.to_timestamp()
    o = t["output_invoices"].rename(columns={"seller_id": "entity_id"})
    o["invoice_date"] = pd.to_datetime(o["invoice_date"])
    s = t["certificate_sales"]
    col = t["collectors"].set_index("collector_id")
    ofk = t["offtakers"].set_index("offtaker_id")
    brands = t["brands"].set_index("brand_id")
    ent = t["entities"].set_index("entity_id")

    e1 = fi.groupby(["src", "dst"], observed=True).agg(w=("tonnes", "sum"), n=("tonnes", "size"), rate=("rate_inr_per_t", "mean")).reset_index()
    e2 = o.groupby(["entity_id", "buyer_id"], observed=True).agg(w=("tonnes", "sum"), n=("tonnes", "size"), rate=("rate_inr_per_t", "mean")).reset_index().rename(columns={"entity_id": "src", "buyer_id": "dst"})
    e3 = s.groupby(["seller_id", "buyer_id"], observed=True).agg(w=("tonnes", "sum"), n=("tonnes", "size"), rate=("price_inr_per_t", "mean")).reset_index().rename(columns={"seller_id": "src", "buyer_id": "dst"})
    E = pd.concat([e1, e2, e3], ignore_index=True)

    ids = list(ent.index) + list(col.index) + list(ofk.index) + list(brands.index)
    typ = ["recycler"] * len(ent) + ["collector"] * len(col) + ["offtaker"] * len(ofk) + ["brand"] * len(brands)
    nodes = pd.DataFrame({"node_id": ids, "type": typ})
    nodes["idx"] = np.arange(len(nodes))
    pos = pd.Series(nodes["idx"].to_numpy(), index=nodes["node_id"].to_numpy())
    E["si"] = E["src"].map(pos)
    E["di"] = E["dst"].map(pos)
    E = E.dropna(subset=["si", "di"])
    si = E["si"].astype(int).to_numpy()
    di = E["di"].astype(int).to_numpy()
    N = len(nodes)

    # structural attributes
    def agg(idx, w, name):
        d = pd.DataFrame({"i": idx, "w": w})
        g = d.groupby("i")["w"]
        tot = g.sum()
        out = pd.DataFrame({f"{name}_deg": g.size(), f"{name}_w": tot})
        d["s2"] = (d["w"] / d["i"].map(tot)) ** 2
        out[f"{name}_hhi"] = d.groupby("i")["s2"].sum()
        out[f"{name}_top"] = g.max() / tot
        return out.reindex(range(N)).fillna(0.0)
    S = pd.concat([agg(di, E["w"].to_numpy(), "in"), agg(si, E["w"].to_numpy(), "out")], axis=1)
    S["in_n"] = pd.Series(E["n"].to_numpy(), index=di).groupby(level=0).sum().reindex(range(N)).fillna(0)
    S["out_n"] = pd.Series(E["n"].to_numpy(), index=si).groupby(level=0).sum().reindex(range(N)).fillna(0)
    S["mean_rate_in"] = pd.Series((E["rate"] * E["w"]).to_numpy(), index=di).groupby(level=0).sum().reindex(range(N)).fillna(0) / S["in_w"].replace(0, np.nan)
    S["mean_rate_out"] = pd.Series((E["rate"] * E["w"]).to_numpy(), index=si).groupby(level=0).sum().reindex(range(N)).fillna(0) / S["out_w"].replace(0, np.nan)
    S = S.fillna(0.0)

    # per-type attribute blocks (z-scored within type)
    blocks = []
    for ty in TYPES:
        m = (nodes["type"] == ty).to_numpy()
        base = _type_block(S[m])
        extra = pd.DataFrame(index=base.index)
        ids_ty = nodes.loc[m, "node_id"].to_numpy()
        if ty == "collector":
            cst = collector_table(fi.rename(columns={"src": "seller_id", "dst": "entity_id"}), col)
            c = col.reindex(ids_ty)
            ex = pd.DataFrame({"cap": c["declared_monthly_capacity_t"].to_numpy(), "veh": c["vehicle_count"].to_numpy(),
                               "ratio": cst["c_ratio_max"].reindex(ids_ty).fillna(0).to_numpy(),
                               "nbuy": cst["c_n_buyers"].reindex(ids_ty).fillna(0).to_numpy()})
            ex["age"] = cst["c_age_at_first_inv_days"].reindex(ids_ty).fillna(0).clip(lower=0).to_numpy()
            extra = _log_std(ex)
        elif ty == "offtaker":
            ost = offtaker_table(o, ofk)
            ex = pd.DataFrame({"nsell": ost["o_n_sellers"].reindex(ids_ty).fillna(0).to_numpy(),
                               "age": ost["o_age_days"].reindex(ids_ty).fillna(0).clip(lower=0).to_numpy(),
                               "t": ost["o_t"].reindex(ids_ty).fillna(0).to_numpy()})
            extra = _log_std(ex)
        elif ty == "recycler" and U is not None:
            extra = U.reindex(ids_ty).fillna(0.0)
        blocks.append((ty, m, base.reset_index(drop=True), extra.reset_index(drop=True)))

    d_base = max(b.shape[1] for _, _, b, _ in blocks)
    d_extra = sum(e.shape[1] for _, _, _, e in blocks)
    x = np.zeros((N, 4 + d_base + d_extra), dtype=np.float32)
    off = 4 + d_base
    for k, (ty, m, b, e) in enumerate(blocks):
        rows = np.where(m)[0]
        x[rows, k] = 1.0
        x[rows, 4:4 + b.shape[1]] = b.to_numpy(dtype=np.float32)
        if e.shape[1]:
            x[rows, off:off + e.shape[1]] = e.to_numpy(dtype=np.float32)
        off += e.shape[1]
    x = np.nan_to_num(x)
    ei = np.stack([np.concatenate([si, di]), np.concatenate([di, si])])     # undirected for message passing
    return torch.tensor(x), torch.tensor(ei, dtype=torch.long), nodes, (si, di)


class Dominant(nn.Module):
    def __init__(self, d_in: int, hid: int = 64):
        super().__init__()
        self.enc = nn.ModuleList([GCNConv(d_in, hid), GCNConv(hid, hid), GCNConv(hid, hid)])
        self.attr = nn.ModuleList([GCNConv(hid, hid), GCNConv(hid, d_in)])
        self.struct = GCNConv(hid, hid)

    def forward(self, x, ei):
        h = x
        for c in self.enc:
            h = Fn.relu(c(h, ei))
        a = Fn.relu(self.attr[0](h, ei))
        xa = self.attr[1](a, ei)
        z = self.struct(h, ei)
        return xa, z


def fit_dominant(x, ei, pos_edges, epochs: int = 120, hid: int = 64, alpha: float = 0.8, lr: float = 5e-3,
                 neg_per_pos: int = 1, seed: int = 0, verbose: bool = True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    N = x.shape[0]
    model = Dominant(x.shape[1], hid)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    si, di = [torch.tensor(a, dtype=torch.long) for a in pos_edges]
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        xa, z = model(x, ei)
        attr_err = ((xa - x) ** 2).sum(dim=1)
        ps = torch.sigmoid((z[si] * z[di]).sum(dim=1))
        ni = torch.randint(0, N, (len(si) * neg_per_pos,))
        nsrc = si.repeat(neg_per_pos)
        ns = torch.sigmoid((z[nsrc] * z[ni]).sum(dim=1))
        pos_err = torch.zeros(N).index_add_(0, si, (1 - ps) ** 2) / torch.bincount(si, minlength=N).clamp(min=1)
        neg_err = torch.zeros(N).index_add_(0, nsrc, ns ** 2) / torch.bincount(nsrc, minlength=N).clamp(min=1)
        struct_err = pos_err + neg_err
        loss = (alpha * attr_err + (1 - alpha) * struct_err).mean()
        loss.backward()
        opt.step()
        if verbose and (ep % 20 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    with torch.no_grad():
        xa, z = model(x, ei)
        attr_err = ((xa - x) ** 2).sum(dim=1)
        # average structure error over several negative draws for a stable score
        se = torch.zeros(N)
        for r in range(5):
            ps = torch.sigmoid((z[si] * z[di]).sum(dim=1))
            ni = torch.randint(0, N, (len(si),))
            ns = torch.sigmoid((z[si] * z[ni]).sum(dim=1))
            pe = torch.zeros(N).index_add_(0, si, (1 - ps) ** 2) / torch.bincount(si, minlength=N).clamp(min=1)
            ne = torch.zeros(N).index_add_(0, si, ns ** 2) / torch.bincount(si, minlength=N).clamp(min=1)
            se += (pe + ne) / 5
    return attr_err.numpy(), se.numpy()


def dominant_scores(t, F, U=None, epochs=120, alpha=0.8, seed=0, verbose=True) -> pd.DataFrame:
    x, ei, nodes, pos = build_graph(t, F, U)
    if verbose:
        print(f"graph: {len(nodes):,} nodes, {len(pos[0]):,} edges, {x.shape[1]} attrs", flush=True)
    a, s = fit_dominant(x, ei, pos, epochs=epochs, alpha=alpha, seed=seed, verbose=verbose)
    df = nodes.assign(attr_err=a, struct_err=s)
    for c in ["attr_err", "struct_err"]:
        g = df.groupby("type")[c]
        df[c + "_z"] = (df[c] - g.transform("median")) / (1.4826 * g.transform(lambda v: np.median(np.abs(v - np.median(v)))) + 1e-9)
    df["score"] = alpha * df["attr_err_z"] + (1 - alpha) * df["struct_err_z"]
    return df.set_index("node_id")
