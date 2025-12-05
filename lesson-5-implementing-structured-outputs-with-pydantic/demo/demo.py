"""
Demo: LLM output structured with Pydantic + OpenAI SDK

This demonstration shows how to parse free-text variant notes into a well-typed
ClinVarEntry using the OpenAI SDK, function-calling "tools," and Pydantic validation.
"""

import os
import json
from typing import List, Optional, Literal, Annotated
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================================
# Stub functions for ls_action_space helpers (if imports fail)
# ============================================================================
def query_clinvar(query: str) -> dict:
    """Stub for ClinVar lookup (read-only)."""
    return {
        "title": f"Mock ClinVar entry for {query}",
        "clinical_significance": "Pathogenic",
        "gene": "BRCA1"
    }


def query_pubmed(query: str, max_results: int = 10) -> List[dict]:
    """Stub for PubMed lookup (read-only)."""
    pmids = query.split(" OR ")
    return [
        {
            "pmid": pmid.strip(),
            "title": f"Mock article for PMID {pmid.strip()}",
            "year": "2020",
            "journal": "Mock Journal"
        }
        for pmid in pmids[:max_results]
    ]


def query_clinicaltrials(query: str, max_results: int = 5) -> dict:
    """Stub for ClinicalTrials.gov lookup (read-only)."""
    return {
        "count": 2,
        "studies": [
            {"NCTId": "NCT00000001", "BriefTitle": f"Mock trial for {query} #1"},
            {"NCTId": "NCT00000002", "BriefTitle": f"Mock trial for {query} #2"}
        ][:max_results]
    }


# Try importing real helpers if available
try:
    from ls_action_space.action_space import (
        query_clinvar as _qc,
        query_pubmed as _qp,
        query_clinicaltrials as _qct,
    )
    query_clinvar = _qc
    query_pubmed = _qp
    query_clinicaltrials = _qct
    print("[INFO] Using real ls_action_space helpers")
except ImportError:
    print("[INFO] Using stub functions for ls_action_space helpers")


# ============================================================================
# Load environment and initialize OpenAI client
# ============================================================================

# Load environment variables
load_dotenv(".env")

# Setup OpenAI client
client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1"),
)
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ============================================================================
# Define the Domain Model (Pydantic)
# ============================================================================
class ClinVarEntry(BaseModel):
    """Structured snapshot of a ClinVar variant extraction from free text."""
    variant_label: Annotated[
        str,
        Field(description="User-provided label or alias for the variant (e.g., 'BRCA1 variant in proband')")
    ]
    hgvs: Annotated[
        List[str],
        Field(description="All HGVS expressions found (c., g., p., r.). Deduplicate; prefer canonical if known.")
    ]
    acmg_class: Annotated[
        Optional[Literal["Pathogenic", "Likely pathogenic", "VUS", "Likely benign", "Benign"]],
        Field(description="ACMG/AMP classification if present.")
    ] = None
    gene: Annotated[
        Optional[str],
        Field(description="Gene symbol if present (e.g., BRCA1).")
    ] = None
    clinvar_accessions: Annotated[
        List[str],
        Field(description="Any VCV/RCV accessions mentioned or inferred.")
    ] = []
    pubmed_pmids: Annotated[
        List[str],
        Field(description="All PubMed IDs mentioned or linked.")
    ] = []
    notes: Annotated[
        Optional[str],
        Field(description="Short free-text rationale or comments.")
    ] = None


# ============================================================================
# Sample free text
# ============================================================================
free_text = """
Patient note: BRCA1 variant observed. HGVS: NM_007294.3(BRCA1):c.5266dup (p.Gln1756ProfsTer74).
Classified as Pathogenic per ACMG criteria and ClinVar (VCV000009123). See PMIDs 20301390, 24728327.
"""


# ============================================================================
# Step 3: Basic String Output Parsing
# ============================================================================
def step3_basic_string_parsing():
    """Extract HGVS strings using simple prompt and line splitting."""
    print("\n" + "="*70)
    print("STEP 3: Basic String Output Parsing")
    print("="*70)

    # Simulate what LLM would extract (fallback for budget/API issues)
    try:
        prompt = f"""Extract all HGVS expressions from the text below; one per line only.
Text:
{free_text}
"""

        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        raw_text = resp.choices[0].message.content or ""
        hgvs_list = [line.strip() for line in raw_text.splitlines() if line.strip()]
    except Exception as e:
        print(f"[DEMO MODE] API call failed ({e}), using simulated extraction")
        hgvs_list = [
            "NM_007294.3(BRCA1):c.5266dup",
            "p.Gln1756ProfsTer74"
        ]

    print("HGVS (string parse):", hgvs_list)
    return hgvs_list


# ============================================================================
# Step 4: Enforce Structure with an LLM "Tool"
# ============================================================================
def step4_structured_tool_output():
    """Use function calling to get structured JSON output."""
    print("\n" + "="*70)
    print("STEP 4: Enforce Structure with LLM Tool")
    print("="*70)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "clinvar_snapshot",
                "description": "Identify HGVS, ACMG class, PubMed PMIDs, and any ClinVar accessions from free text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "variant_label": {
                            "type": "string",
                            "description": "A short alias for this variant entry."
                        },
                        "hgvs": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "acmg_class": {
                            "type": "string",
                            "enum": ["Pathogenic", "Likely pathogenic", "VUS", "Likely benign", "Benign"]
                        },
                        "gene": {"type": "string"},
                        "clinvar_accessions": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "pubmed_pmids": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "notes": {"type": "string"}
                    },
                    "required": ["variant_label", "hgvs", "clinvar_accessions", "pubmed_pmids"]
                }
            }
        }
    ]

    messages = [
        {
            "role": "system",
            "content": "You are a meticulous variant curation assistant for clinical genomics."
        },
        {
            "role": "user",
            "content": f"Parse this note into a ClinVar snapshot:\n{free_text}"
        }
    ]

    # Try API call, fallback to simulated response
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        choice = resp.choices[0].message
        if choice.tool_calls:
            tool_args = choice.tool_calls[0].function.arguments
        else:
            raise RuntimeError("Model did not call the expected tool.")
    except Exception as e:
        print(f"[DEMO MODE] API call failed ({e}), using simulated tool response")
        tool_args = json.dumps({
            "variant_label": "BRCA1 variant in proband",
            "hgvs": [
                "NM_007294.3(BRCA1):c.5266dup",
                "p.Gln1756ProfsTer74"
            ],
            "acmg_class": "Pathogenic",
            "gene": "BRCA1",
            "clinvar_accessions": ["VCV000009123"],
            "pubmed_pmids": ["20301390", "24728327"],
            "notes": "Frameshift variant classified as Pathogenic per ACMG criteria"
        })

    print("Tool args (JSON):", tool_args)
    return tool_args


# ============================================================================
# Step 5: Validate & Cast into ClinVarEntry (Pydantic)
# ============================================================================
def step5_validate_with_pydantic(tool_args: str) -> ClinVarEntry:
    """Parse JSON and validate with Pydantic model."""
    print("\n" + "="*70)
    print("STEP 5: Validate & Cast into ClinVarEntry (Pydantic)")
    print("="*70)

    try:
        data = json.loads(tool_args)
        entry = ClinVarEntry(**data)
        print("Validated ClinVarEntry:", json.dumps(entry.model_dump(), indent=2))
        return entry
    except (json.JSONDecodeError, ValidationError) as e:
        print("Validation error:", e)
        raise


# ============================================================================
# Step 6: Access the Parsed Data
# ============================================================================
def step6_access_parsed_data(entry: ClinVarEntry):
    """Demonstrate accessing fields from the validated model."""
    print("\n" + "="*70)
    print("STEP 6: Access the Parsed Data")
    print("="*70)

    print("PMIDs:", entry.pubmed_pmids)
    print("ACMG class:", entry.acmg_class)
    print("HGVS expressions:", entry.hgvs)
    print("Gene:", entry.gene)
    print("ClinVar accessions:", entry.clinvar_accessions)


# ============================================================================
# Step 7: Enrich/Verify with Domain Helpers
# ============================================================================
def step7_enrich_with_helpers(entry: ClinVarEntry):
    """Use domain-specific helpers to enrich/verify the data."""
    print("\n" + "="*70)
    print("STEP 7: Enrich/Verify with Domain Helpers")
    print("="*70)

    # A. Verify ClinVar accessions / HGVS (best-effort)
    print("\n[A] ClinVar lookups:")
    for tag in (entry.clinvar_accessions + entry.hgvs):
        try:
            cv = query_clinvar(tag)
            print(f"  [ClinVar:{tag}] title={cv.get('title')!r} "
                  f"significance={cv.get('clinical_significance')!r} "
                  f"gene={cv.get('gene')!r}")
        except Exception as ex:
            print(f"  [ClinVar:{tag}] lookup failed: {ex}")

    # B. Pull PubMed metadata for cited PMIDs
    print("\n[B] PubMed lookups:")
    if entry.pubmed_pmids:
        try:
            pmq = " OR ".join(entry.pubmed_pmids)
            articles = query_pubmed(pmq, max_results=len(entry.pubmed_pmids))
            for a in articles:
                print(f"  PMID {a['pmid']}: {a['title']} ({a.get('year')}) in {a.get('journal')}")
        except Exception as ex:
            print(f"  [PubMed] lookup failed: {ex}")

    # C. Search trials related to the gene/condition (if present)
    print("\n[C] ClinicalTrials.gov lookups:")
    if entry.gene:
        try:
            trials = query_clinicaltrials(entry.gene, max_results=5)
            print(f"  Trials mentioning {entry.gene}: {trials.get('count')} total; showing up to 5")
            for s in trials.get("studies", []):
                print(f"    - {s.get('NCTId')} — {s.get('BriefTitle')}")
        except Exception as ex:
            print(f"  [ClinicalTrials] lookup failed: {ex}")


# ============================================================================
# Main execution
# ============================================================================
def main():
    """Run all demonstration steps."""
    print("="*70)
    print("LLM Structured Outputs with Pydantic + OpenAI SDK Demo")
    print("="*70)

    # Step 3: Basic string parsing
    step3_basic_string_parsing()

    # Step 4: Structured tool output
    tool_args = step4_structured_tool_output()

    # Step 5: Validate with Pydantic
    entry = step5_validate_with_pydantic(tool_args)

    # Step 6: Access parsed data
    step6_access_parsed_data(entry)

    # Step 7: Enrich with domain helpers
    step7_enrich_with_helpers(entry)

    print("\n" + "="*70)
    print("Demo completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
