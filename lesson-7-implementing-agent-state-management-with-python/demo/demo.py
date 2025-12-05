#!/usr/bin/env python3
"""
Pharmacovigilance Agent Demo: Drug-Event Signal Loop
Demonstrates state management with transitions using OpenAI structured outputs,
FAERS API integration, and Pydantic validation.
"""
import traceback
from typing import List, Dict, Optional, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI
import requests
import math
import os
import traceback

# Load environment variables
load_dotenv(".env")

# Setup OpenAI client
client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1"),
)
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

NO_LLM = False
try:
    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY) if "vocareum" in OPENAI_BASE_URL else OpenAI(api_key=OPENAI_API_KEY)
except:
    traceback.print_exc()
    client = None
    NO_LLM = True

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ============================================================================
# 2) FAERS Helper & Signal Math
# ============================================================================

def _query_faers_api(search_query: str) -> int:
    """
    Helper function to query the openFDA API and return the total count of matching records.
    Returns 0 if the query fails or finds no results.
    """
    base_url = "https://api.fda.gov/drug/event.json"
    params = {"search": search_query, "limit": 1}
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("meta", {}).get("results", {}).get("total", 0)
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return 0

def get_faers_contingency_counts(drug_name: str, adverse_event: str) -> Dict[str, int]:
    """
    Build the 2x2 table for ROR:
    a: drug & event, b: drug & !event, c: !drug & event, d: !drug & !event
    """
    print(f"🛠️  Fetching FAERS data for {drug_name} and {adverse_event}...")
    drug_query = f'patient.drug.medicinalproduct:"{drug_name}"'
    event_query = f'patient.reaction.reactionmeddrapt:"{adverse_event}"'

    a = _query_faers_api(f"({drug_query}) AND ({event_query})")
    total_drug_reports = _query_faers_api(drug_query)
    b = max(total_drug_reports - a, 0)

    total_event_reports = _query_faers_api(event_query)
    c = max(total_event_reports - a, 0)

    total_reports = _query_faers_api("")  # empty query → total
    d = max(total_reports - a - b - c, 0)
    counts = {"a": a, "b": b, "c": c, "d": d}
    print(f"📊 Contingency Table: {counts}")
    return counts

def compute_ror_and_ci(a: int, b: int, c: int, d: int, alpha: float = 0.05) -> Dict[str, float]:
    """
    Reporting Odds Ratio (ROR) with Wald 95% CI on log scale.
    Adds 0.5 continuity correction if any cell is zero.
    """
    cc = 0.5 if 0 in (a, b, c, d) else 0.0
    a_, b_, c_, d_ = a + cc, b + cc, c + cc, d + cc

    ror = (a_ / b_) / (c_ / d_)
    se_log = math.sqrt(1/a_ + 1/b_ + 1/c_ + 1/d_)
    z = 1.96  # approx for 95%
    lower = math.exp(math.log(ror) - z * se_log)
    upper = math.exp(math.log(ror) + z * se_log)
    return {"ror": ror, "ci_lower": lower, "ci_upper": upper}

# ============================================================================
# 3) Pydantic Models
# ============================================================================

class CandidateEvent(BaseModel):
    meddra_pt: Annotated[str, Field(description="MedDRA Preferred Term to query in FAERS")]

class CandidateEvents(BaseModel):
    drug: Annotated[str, Field(description="Drug of interest")]
    candidates: Annotated[List[CandidateEvent], Field(description="Ranked list of candidate adverse events")]

class SignalStats(BaseModel):
    a: int
    b: int
    c: int
    d: int
    ror: float
    ci_lower: float
    ci_upper: float

class SignalDecision(BaseModel):
    drug: str
    meddra_pt: str
    threshold_rule: str  # e.g., "lower_CI>1"
    is_signal: bool
    stats: SignalStats

class FinalReport(BaseModel):
    drug: str
    evaluated_events: List[SignalDecision]
    selected_signal: Optional[SignalDecision] = None
    notes: Optional[str] = ""

# ============================================================================
# 4) Agentic Loop
# ============================================================================

SYSTEM = (
    "You are a pharmacovigilance assistant. "
    "Given a drug name, propose likely adverse events as MedDRA PTs. "
    "Focus on events with known or plausible associations, diverse organ systems, "
    "and practical signal detection value. Return only structured JSON."
)

def propose_candidates(drug: str, k: int = 6) -> CandidateEvents:
    if NO_LLM:
        # Mock data for demo purposes when API is not available
        mock_candidates = {
            "ibuprofen": [
                "Gastrointestinal haemorrhage",
                "Myocardial infarction",
                "Renal failure",
                "Anaphylactic reaction",
                "Hepatic failure",
                "Asthma"
            ],
            "aspirin": [
                "Gastrointestinal haemorrhage",
                "Reye's syndrome",
                "Cerebral haemorrhage",
                "Tinnitus",
                "Urticaria",
                "Asthma"
            ]
        }
        drug_lower = drug.lower()
        if drug_lower in mock_candidates:
            candidates = [CandidateEvent(meddra_pt=pt) for pt in mock_candidates[drug_lower][:k]]
        else:
            # Generic mock data for unknown drugs
            candidates = [
                CandidateEvent(meddra_pt="Adverse event"),
                CandidateEvent(meddra_pt="Drug ineffective"),
                CandidateEvent(meddra_pt="Nausea")
            ]
        return CandidateEvents(drug=drug, candidates=candidates)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Drug: {drug}\nReturn {k} candidate MedDRA PTs most worth checking in FAERS."}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "candidate_events",
                "schema": {
                    "type": "object",
                    "properties": {
                        "drug": {"type": "string"},
                        "candidates": {
                            "type": "array",
                            "items": {"type": "object", "properties": {"meddra_pt": {"type": "string"}}, "required": ["meddra_pt"]},
                            "minItems": 1
                        }
                    },
                    "required": ["drug", "candidates"],
                    "additionalProperties": False
                }
            }
        },
        temperature=0.2,
    )
    raw = response.choices[0].message.content
    return CandidateEvents.model_validate_json(raw)

def evaluate_drug(drug: str, ci_rule: str = "lower_CI>1") -> FinalReport:
    # Get candidate events
    print(f"\n🔍 Proposing candidate adverse events for: {drug}")
    candidates = propose_candidates(drug).candidates
    print(f"📋 Candidates: {[c.meddra_pt for c in candidates]}\n")

    evaluated: List[SignalDecision] = []
    selected: Optional[SignalDecision] = None

    for ce in candidates:
        counts = get_faers_contingency_counts(drug, ce.meddra_pt)
        stats = compute_ror_and_ci(counts["a"], counts["b"], counts["c"], counts["d"])
        decision = SignalDecision(
            drug=drug,
            meddra_pt=ce.meddra_pt,
            threshold_rule=ci_rule,
            is_signal=(stats["ci_lower"] > 1.0),
            stats=SignalStats(a=counts["a"], b=counts["b"], c=counts["c"], d=counts["d"],
                              ror=stats["ror"], ci_lower=stats["ci_lower"], ci_upper=stats["ci_upper"])
        )
        evaluated.append(decision)
        print(f"{'✅' if decision.is_signal else '❌'} {ce.meddra_pt}: ROR={stats['ror']:.2f} (95% CI {stats['ci_lower']:.2f}–{stats['ci_upper']:.2f})")

        # Stop at first signal for demo (or continue to rank signals)
        if decision.is_signal and selected is None:
            selected = decision
            print(f"🎯 Signal detected! Stopping search.\n")
            break

    return FinalReport(drug=drug, evaluated_events=evaluated, selected_signal=selected,
                       notes="Demo threshold: lower 95% CI of ROR > 1 indicates a signal.")

# ============================================================================
# 5) Main Execution
# ============================================================================

def main():
    # Example usage (you can change the drug via env var or default)
    drug = os.getenv("DEMO_DRUG", "ibuprofen")

    print("=" * 80)
    print("Pharmacovigilance Agent: Drug-Event Signal Detection")
    print("=" * 80)

    try:
        report = evaluate_drug(drug)
        print("\n" + "=" * 80)
        print("✅ Final Report (dict):")
        print("=" * 80)
        print(report.model_dump())

        # 5) Access Parsed Data
        print("\n" + "=" * 80)
        print("📊 Summary:")
        print("=" * 80)

        if report.selected_signal:
            print(f"\n📌 First detected signal for {report.drug}: {report.selected_signal.meddra_pt}")
            s = report.selected_signal.stats
            print(f"ROR={s.ror:.2f} (95% CI {s.ci_lower:.2f}–{s.ci_upper:.2f}); a={s.a}, b={s.b}, c={s.c}, d={s.d}")
        else:
            print(f"\nℹ️ No signal detected for {report.drug} under the demo rule.")
            print("Top evaluated event and stats (if any):")
            if report.evaluated_events:
                top = report.evaluated_events[0]
                ts = top.stats
                print(f"{top.meddra_pt}: ROR={ts.ror:.2f} (95% CI {ts.ci_lower:.2f}–{ts.ci_upper:.2f})")

        print("\n" + "=" * 80)
        print("✅ Demo completed successfully!")
        print("=" * 80)

    except ValidationError as ve:
        print("❌ Validation error:", ve)
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
