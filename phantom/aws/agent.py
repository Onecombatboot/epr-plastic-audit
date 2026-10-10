"""Phantom Plastic as a Strands agent: tools the orchestrator (or an inspector chatting) can call.

    from phantom.aws.agent import build_agent
    agent = build_agent("phantom/outputs/seed42")          # needs AWS credentials + Bedrock model access
    agent("Which recyclers should we inspect first in Gujarat, and why?")

The tool functions work without AWS (plain Python); only the LLM planner needs Bedrock.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .narrative import summarise

_STATE: dict = {}


def _load(run: str, data: str) -> dict:
    if _STATE.get("run") != run:
        r, d = Path(run), Path(data)
        _STATE.update(run=run, S=pd.read_parquet(r / "recycler_scores.parquet"), BX=pd.read_parquet(r / "brand_exposure.parquet"),
                      ent=pd.read_parquet(d / "registry" / "entities.parquet").set_index("entity_id"),
                      brands=pd.read_parquet(d / "plastic" / "brands.parquet").set_index("brand_id"), cases=r / "cases")
    return _STATE


def make_tools(run: str, data: str = "data/synthetic"):
    from strands import tool

    @tool
    def top_suspects(n: int = 10, state: str = "") -> str:
        """List the plastic recyclers with the highest hollow-certificate risk. Optionally filter by Indian state name."""
        st = _load(run, data)
        df = st["S"].join(st["ent"][["legal_name", "state"]])
        if state:
            df = df[df["state"].str.lower() == state.lower()]
        df = df.sort_values("ens_score", ascending=False).head(int(n))
        return json.dumps([{"entity_id": i, "name": r.legal_name, "state": r.state, "risk_percentile": round(float(r.ens_pct), 3), "rank": int(r["rank"])}
                           for i, r in df.iterrows()])

    @tool
    def explain_recycler(entity_id: str) -> str:
        """Return the evidence-backed case summary for one recycler: reasons, evidence to pull, inspection checklist."""
        st = _load(run, data)
        f = st["cases"] / f"{entity_id}.json"
        if not f.exists():
            return json.dumps({"error": "no case file; only the top 100 recyclers have one"})
        case = json.loads(f.read_text(encoding="utf-8"))
        out = summarise(case)
        return json.dumps({"summary": out["text"], "reasons": case["reasons"], "evidence": case["evidence"], "checklist": case["inspection_checklist"]})

    @tool
    def brand_exposure(brand_name_contains: str = "", n: int = 10) -> str:
        """Show brands whose purchased certificates trace most to high-risk recyclers. Optionally filter by part of the brand name."""
        st = _load(run, data)
        bx = st["BX"].join(st["brands"][["name", "sector"]]).sort_values("hollow_t_est", ascending=False)
        if brand_name_contains:
            bx = bx[bx["name"].str.contains(brand_name_contains, case=False)]
        return bx.head(int(n))[["name", "sector", "bought_t", "exposure", "hollow_t_est", "top_sources"]].round(3).to_json(orient="records")

    return [top_suspects, explain_recycler, brand_exposure]


def build_agent(run: str, data: str = "data/synthetic", model_id: str | None = None, region: str = "us-east-1"):
    from strands import Agent
    from strands.models import BedrockModel

    from .narrative import MODEL_ID
    if not (model_id or MODEL_ID):
        raise ValueError("Set PHANTOM_BEDROCK_MODEL (or pass model_id) to a Bedrock model enabled in your account")
    model = BedrockModel(model_id=model_id or MODEL_ID, region_name=region)
    return Agent(model=model, tools=make_tools(run, data),
                 system_prompt=("You assist State Pollution Control Board inspectors. Answer only from tool results; never invent figures. "
                                "Say plainly that the data is synthetic when asked about accuracy. Offer the inspection checklist when explaining a case."))
