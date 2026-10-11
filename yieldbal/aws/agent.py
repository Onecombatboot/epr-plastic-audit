"""Inspector tools share the local evidence; Bedrock is optional and configured explicitly."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from yieldbal.checks import compute, product_balance
from yieldbal.prep import DEFAULT_DATA, DEFAULT_RUN, load, load_configs, load_reference
from yieldbal.scoring import reasons


class AuditTools:
    def __init__(self, run: Path = DEFAULT_RUN, data: Path = DEFAULT_DATA):
        self.data = Path(data)
        self.scores = pd.read_parquet(Path(run)/"scores.parquet")
        path = Path(run)/"monthly_checks.parquet"
        self.monthly = pd.read_parquet(path) if path.exists() else compute(load(Path(data)), load_reference())[0]
        self.signals = {signal["entity_id"]: signal for signal in
                        (json.loads(line) for line in (Path(run)/"signals.jsonl").read_text().splitlines())}

    def _case(self, entity_id: str) -> pd.DataFrame:
        if entity_id not in self.scores.index:
            raise ValueError("Unknown recycler")
        return self.monthly[self.monthly.entity_id.eq(entity_id)]

    def top_suspects(self, n: int = 10, state: str = "", stream: str = "") -> list[dict]:
        if not 1 <= n <= 100:
            raise ValueError("n must be between 1 and 100")
        scores = self.scores
        if state:
            scores = scores[scores.state.str.casefold().eq(state.casefold())]
        if stream:
            scores = scores[scores.stream.eq(stream)]
        return json.loads(scores.head(n).rename_axis("entity_id").reset_index()[
            ["entity_id", "legal_name", "stream", "state", "score", "rank", "tier", "suspected_hollow_t"]].to_json(orient="records"))

    def explain_recycler(self, entity_id: str) -> dict:
        case = self._case(entity_id)
        row = self.scores.loc[entity_id]
        signal = self.signals.get(entity_id)
        return {"entity_id": entity_id, "synthetic": True, "score": float(row.score), "tier": row.tier,
                "reasons": signal["reasons"] if signal else reasons(row, case),
                "evidence_refs": signal["evidence_refs"] if signal else [f"operations_monthly:{entity_id}:{case.iloc[0].month:%Y-%m}"],
                "summary": " ".join(signal["reasons"] if signal else reasons(row, case))}

    def check_energy_ceiling(self, entity_id: str) -> dict:
        case = self._case(entity_id)
        return {"entity_id": entity_id, "months": len(case), "bill_months": int(case.bill_present.sum()),
                "missing_bill_months": int((~case.bill_present).sum()),
                "energy_supported_t_observed_months": float(case.energy_supported_t.sum()),
                "claimed_t_observed_bill_months": float(case.loc[case.bill_present, "certificates_t"].sum()),
                "declared_solar_credit_kwh": float(case.solar_credit_kwh.sum())}

    def check_output_balance(self, entity_id: str) -> dict:
        case = self._case(entity_id)
        _, products = load_configs(self.data/"yield/stream_configs")
        sales = pd.read_parquet(self.data/"yield/outputs_monthly.parquet", filters=[("entity_id", "==", entity_id)])
        sales["month"] = pd.to_datetime(sales.month)
        balance = product_balance(case, sales, products, load_reference())
        latest = balance[balance.sold_window_t.notna()].groupby("product", sort=False).tail(1)
        return {"entity_id": entity_id, "claimed_t": float(case.certificates_t.sum()),
                "sold_t": float(case.sold_t.sum()), "lag_months": load_reference()["sales_lag_months"],
                "expected_saleable_share": float(case.saleable_expected.iloc[0]),
                "minimum_saleable_share": float(case.saleable_lower.iloc[0]),
                "testable_windows": int(case.output_deficit.notna().sum()),
                "average_6m_shortfall": float(case.output_deficit.mean()) if case.output_deficit.notna().any() else None,
                "products_latest_complete_window": json.loads(latest.to_json(orient="records", date_format="iso"))}


def make_tools(run: Path = DEFAULT_RUN, data: Path = DEFAULT_DATA):
    from strands import tool
    audit = AuditTools(run, data)

    @tool
    def top_suspects(n: int = 10, state: str = "", stream: str = "") -> str:
        """List up to 100 ranked recyclers, optionally filtered by state and waste stream."""
        return json.dumps(audit.top_suspects(n, state, stream), allow_nan=False)

    @tool
    def explain_recycler(entity_id: str) -> str:
        """Explain a recycler's risk with numeric reasons and observable evidence references."""
        return json.dumps(audit.explain_recycler(entity_id), allow_nan=False)

    @tool
    def check_energy_ceiling(entity_id: str) -> str:
        """Compare certificates to the energy ceiling, preserving months with missing bills."""
        return json.dumps(audit.check_energy_ceiling(entity_id), allow_nan=False)

    @tool
    def check_output_balance(entity_id: str) -> str:
        """Compare saleable product sales with lag-adjusted reported input using stream physics."""
        return json.dumps(audit.check_output_balance(entity_id), allow_nan=False)

    return [check_energy_ceiling, check_output_balance, explain_recycler, top_suspects]


class OfflineAgent:
    def __init__(self, audit: AuditTools):
        self.audit = audit

    def __call__(self, prompt: str) -> str:
        entities = [entity for entity in self.audit.scores.index if entity in prompt]
        if entities:
            return json.dumps(self.audit.explain_recycler(entities[0]), allow_nan=False)
        return json.dumps({"mode": "offline_template", "synthetic": True,
                           "message": "Use a recycler ID for an evidence summary; otherwise the top 10 cases are shown.",
                           "top_suspects": self.audit.top_suspects()}, allow_nan=False)


def build_agent(run: Path = DEFAULT_RUN, data: Path = DEFAULT_DATA, model_id: str | None = None,
                region: str | None = None):
    model_id = model_id or os.getenv("YIELDBAL_BEDROCK_MODEL")
    if not model_id:
        return OfflineAgent(AuditTools(run, data))
    from strands import Agent
    from strands.models import BedrockModel
    model = BedrockModel(model_id=model_id, region_name=region or os.getenv("AWS_REGION", "ap-south-1"))
    return Agent(model=model, tools=make_tools(run, data),
                 system_prompt="Assist pollution-control inspectors using only tool evidence. Give quantitative reasons and evidence references. State that all data is synthetic. Do not present scores as fraud probabilities.")
