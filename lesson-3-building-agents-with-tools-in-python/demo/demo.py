#!/usr/bin/env python3
"""
Demo: Function Calling with OpenAI SDK for ClinVar Entry Extraction
"""
import json
import os
from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables
load_dotenv(".env")

# Setup OpenAI client
client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1"),
)
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# Define the ClinVar entry extraction tool
clinvar_entry_tool = {
    "type": "function",
    "function": {
        "name": "extract_clinvar_entry",
        "description": "Parse free-text variant notes into a structured ClinVarEntry.",
        "parameters": {
            "type": "object",
            "properties": {
                "hgvs": {
                    "type": "string",
                    "description": "HGVS expression (e.g., NM_000059.3:c.5946del)."
                },
                "acmg_class": {
                    "type": "string",
                    "description": "ACMG/AMP classification.",
                    "enum": [
                        "Pathogenic",
                        "Likely_pathogenic",
                        "Uncertain_significance",
                        "Likely_benign",
                        "Benign"
                    ]
                },
                "pubmed_pmids": {
                    "type": "array",
                    "description": "List of PubMed IDs supporting the assertion.",
                    "items": {"type": "string", "pattern": "^[0-9]+$"},
                    "default": []
                },
                "notes": {
                    "type": "string",
                    "description": "Optional free text that was parsed."
                }
            },
            "required": ["hgvs", "acmg_class"]
        }
    }
}


def extract_clinvar_entry(hgvs: str, acmg_class: str, pubmed_pmids=None, notes: str = None):
    """
    Local implementation of the ClinVar entry extraction tool.
    Normalizes and validates the structured data.
    """
    pmids = []
    for p in (pubmed_pmids or []):
        # keep digits only; drop empties; deduplicate while preserving order
        d = "".join(ch for ch in p if ch.isdigit())
        if d and d not in pmids:
            pmids.append(d)

    return {
        "hgvs": hgvs.strip(),
        "acmg_class": acmg_class,
        "pubmed_pmids": pmids,
        "notes": notes
    }


# Stub functions for optional ClinVar/PubMed validation
# These would normally come from ls_action_space.action_space
def query_clinvar(hgvs: str):
    """Stub for ClinVar query - returns mock data"""
    print(f"[STUB] query_clinvar called with: {hgvs}")
    return {
        "hgvs": hgvs,
        "pubmed_pmids": ["20301390", "11157737"],
        "significance": "Pathogenic"
    }


def query_pubmed(query: str, max_results: int = 10):
    """Stub for PubMed query - returns mock data"""
    print(f"[STUB] query_pubmed called with: {query}, max_results={max_results}")
    return {
        "results": [
            {"pmid": "20301390", "title": "BRCA2 mutations study", "year": "2010"},
            {"pmid": "11157737", "title": "Genetic analysis of BRCA2", "year": "2001"}
        ]
    }


def demo_basic_interaction():
    """Step 1: Basic interaction with the language model"""
    print("\n" + "="*60)
    print("STEP 1: Basic Interaction with Language Models")
    print("="*60)

    try:
        # Single-turn query
        print("\n--- Single-Turn Query ---")
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "What is ClinVar?"}]
        )
        print(resp.choices[0].message.content)

        # Multi-turn conversation
        print("\n--- Multi-Turn Conversation ---")
        messages = [
            {"role": "system", "content": "You are a life sciences AI assistant."},
            {"role": "user", "content": "Explain HGVS briefly in one sentence."}
        ]
        resp = client.chat.completions.create(model=MODEL, messages=messages)
        print(resp.choices[0].message.content)
    except Exception as e:
        print(f"[DEMO MODE] API call skipped: {e}")
        print("ClinVar is a public database that aggregates information about genetic variants and their relationships to human health.")
        print("\n--- Multi-Turn Conversation ---")
        print("HGVS (Human Genome Variation Society nomenclature) is a standardized system for describing genetic sequence variants.")


def demo_function_calling():
    """Step 2 & 3: Function calling with ClinVar entry extraction"""
    print("\n" + "="*60)
    print("STEP 2 & 3: Function Calling with ClinVar Entry Extraction")
    print("="*60)

    try:
        # Set up the conversation
        messages = [
            {
                "role": "system",
                "content": (
                    "You extract ClinVar snapshots. If the user provides variant text, "
                    "call the tool to return a structured ClinVarEntry. "
                    "If HGVS is missing, infer it from context if safe."
                )
            },
            {
                "role": "user",
                "content": (
                    "Note: BRCA2 frameshift reported in patient; c.5946del (aka 6174delT). "
                    "Clinically known pathogenetic per ACMG; refs: PMID:20301390, PMID:11157737."
                )
            }
        ]

        # Ask the model; allow tool usage
        print("\n--- Sending request with tool support ---")
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=[clinvar_entry_tool],
            tool_choice="auto"
        )

        ai_msg = resp.choices[0].message
        tool_calls = ai_msg.tool_calls or []

        if not tool_calls:
            print("No tool calls made by the model")
            return None

        # Extract tool call information
        call = tool_calls[0]
        tool_name = call.function.name
        args = json.loads(call.function.arguments)
        print(f"\n--- Tool Call ---")
        print(f"Function: {tool_name}")
        print(f"Arguments: {json.dumps(args, indent=2)}")

        # Execute the tool locally
        result = extract_clinvar_entry(**args)
        tool_response_content = json.dumps(result)
        print(f"\n--- Tool Result ---")
        print(json.dumps(result, indent=2))

        # Create a tool response message and get the final AI answer
        messages.append({
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": call.function.arguments
                    }
                }
            ]
        })
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "name": tool_name,
            "content": tool_response_content
        })

        final = client.chat.completions.create(model=MODEL, messages=messages)
        print(f"\n--- Final AI Response ---")
        print(final.choices[0].message.content)

        return result
    except Exception as e:
        print(f"[DEMO MODE] API call skipped: {e}")
        print("\n--- Simulating Tool Call ---")
        args = {
            "hgvs": "NM_000059.3:c.5946del",
            "acmg_class": "Pathogenic",
            "pubmed_pmids": ["20301390", "11157737"],
            "notes": "BRCA2 frameshift (6174delT)"
        }
        print(f"Function: extract_clinvar_entry")
        print(f"Arguments: {json.dumps(args, indent=2)}")

        result = extract_clinvar_entry(**args)
        print(f"\n--- Tool Result ---")
        print(json.dumps(result, indent=2))

        print(f"\n--- Final AI Response ---")
        print("Successfully extracted ClinVar entry for BRCA2 c.5946del variant classified as Pathogenic with supporting PMIDs 20301390 and 11157737.")

        return result


def demo_validation_enrichment(entry):
    """Step 4: Validate/enrich via ClinVar & PubMed"""
    print("\n" + "="*60)
    print("STEP 4: Validate/Enrich via ClinVar & PubMed")
    print("="*60)

    if not entry:
        print("No entry to validate")
        return

    # Validate/enrich using ClinVar
    if entry["hgvs"]:
        print(f"\n--- Querying ClinVar for: {entry['hgvs']} ---")
        cv = query_clinvar(entry["hgvs"])

        # Add any additional PMIDs found in ClinVar
        maybe_pmids = set(entry["pubmed_pmids"])
        for p in cv.get("pubmed_pmids", []):
            if p.isdigit():
                maybe_pmids.add(p)
        entry["pubmed_pmids"] = sorted(maybe_pmids)
        print(f"Enriched PMIDs: {entry['pubmed_pmids']}")

    # Optionally fetch PubMed metadata
    if entry["pubmed_pmids"]:
        print(f"\n--- Querying PubMed ---")
        q = " OR ".join(entry["pubmed_pmids"])
        pm = query_pubmed(q, max_results=min(10, len(entry["pubmed_pmids"])))

        print("\nPubMed Results:")
        for r in pm.get("results", []):
            print(f"  - PMID:{r['pmid']} ({r['year']}): {r['title']}")

    print(f"\n--- Final Enriched Entry ---")
    print(json.dumps(entry, indent=2))


def main():
    """Run all demonstration steps"""
    print("\n" + "="*60)
    print("OpenAI Function Calling Demo - ClinVar Entry Extraction")
    print("="*60)

    # Step 1: Basic interaction
    demo_basic_interaction()

    # Steps 2 & 3: Function calling
    entry = demo_function_calling()

    # Step 4: Validation and enrichment
    demo_validation_enrichment(entry)

    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
