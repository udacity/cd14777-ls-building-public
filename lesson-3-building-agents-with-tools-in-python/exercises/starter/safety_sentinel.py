"""
GOAL:
Your task is to fill in the missing code in the `run_safety_analysis` function.
You will need to:
1.  Define the tools (the two provided Python functions) in the JSON schema
    required by the OpenAI API.
2.  Implement the agent's conversation loop to handle the tool calls.

This exercise focuses on the core logic of making an LLM use custom functions.
The helper functions for API calls and calculations are already provided for you.
"""

import os
import json
import math
import requests
from openai import OpenAI
from typing import Dict

# --- Setup: OpenAI Client ---
# Ensure your OPENAI_API_KEY is set as an environment variable
# --- Configuration & API Key ---
from dotenv import load_dotenv
load_dotenv(".env")

try:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
except KeyError:
    print("🔴 ERROR: OPENAI_API_KEY environment variable not set.")
    exit()


# --------------------------------------------------------------------------
# --- PRE-BUILT TOOLS (No changes needed in this section) ---
# --------------------------------------------------------------------------

def _query_faers_api(search_query: str) -> int:
    """Helper to query the openFDA API and return the total record count."""
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
    TOOL 1: Fetches data to build a 2x2 contingency table for ROR calculation.
    (a: drug+event, b: drug-only, c: event-only, d: neither)
    """
    print(f"🛠️  Calling Tool 1: get_faers_contingency_counts for '{drug_name}' and '{adverse_event}'...")
    drug_query = f'patient.drug.medicinalproduct:"{drug_name}"'
    event_query = f'patient.reaction.reactionmeddrapt:"{adverse_event}"'
    a = _query_faers_api(f"({drug_query}) AND ({event_query})")
    total_drug = _query_faers_api(drug_query)
    b = total_drug - a
    total_event = _query_faers_api(event_query)
    c = total_event - a
    total_reports = _query_faers_api("")
    d = total_reports - a - b - c
    counts = {"a": a, "b": b, "c": c, "d": d}
    print(f"📊 Contingency Table: {counts}")
    return counts


def calculate_ror(a: int, b: int, c: int, d: int) -> Dict[str, float]:
    """
    TOOL 2: Calculates the Reporting Odds Ratio (ROR) and its 95% CI.
    """
    print("🛠️  Calling Tool 2: calculate_ror...")
    if a * b * c * d == 0: a, b, c, d = [x + 0.5 for x in (a, b, c, d)]
    ror = (a * d) / (b * c)
    log_ror = math.log(ror)
    se_log_ror = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    ci_lower = math.exp(log_ror - 1.96 * se_log_ror)
    ci_upper = math.exp(log_ror + 1.96 * se_log_ror)
    result = {"ror": round(ror, 2), "ci_lower": round(ci_lower, 2), "ci_upper": round(ci_upper, 2)}
    print(f"📈 ROR Result: {result}")
    return result


# --------------------------------------------------------------------------
# --- AGENT IMPLEMENTATION (This is where you'll write your code) ---
# --------------------------------------------------------------------------

def run_safety_analysis(user_prompt: str):
    """
    Runs the full agent workflow using the tools provided.
    """
    print(f"\n💬 User Prompt: '{user_prompt}'")

    messages = [
        {"role": "system",
         "content": "You are a helpful pharmacovigilance assistant. Your goal is to assess drug safety signals. First, get data counts with `get_faers_contingency_counts`. Second, use those counts to call `calculate_ror`. Finally, interpret the result in plain English."},
        {"role": "user", "content": user_prompt}
    ]

    available_functions = {
        "get_faers_contingency_counts": get_faers_contingency_counts,
        "calculate_ror": calculate_ror,
    }

    # ----------------------------------------------------------------------
    # TODO 1: DEFINE THE TOOLS FOR THE AI AGENT
    # Describe the two functions above in the JSON schema required by the API.
    # Pay close attention to the `description`, `parameters`, `properties`,
    # `type`, and `required` fields.
    # ----------------------------------------------------------------------
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_faers_contingency_counts",
                "description": "Fetches data from FAERS to build a 2x2 contingency table for ROR calculation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        # Add properties for 'drug_name' and 'adverse_event' here
                    },
                    "required": [],  # Add the required properties here
                },
            },
        },
        # Add the full definition for the 'calculate_ror' function here
    ]

    # ----------------------------------------------------------------------
    # TODO 2: IMPLEMENT THE AGENT'S CONVERSATION LOOP
    # This loop should continue until the model gives a final text answer
    # instead of calling a tool.
    # ----------------------------------------------------------------------
    while True:
        # Step 1: Call the model with the current messages and tools.
        # response = client.chat.completions.create(...)

        # Step 2: Check if the response from the model contains tool calls.
        # If it does, execute the functions and append the results to `messages`.
        # HINT: The response message will have a `tool_calls` attribute.

        # Step 3: If the response does NOT contain tool calls, it's the
        # final answer. Print the content and break the loop.

        print("🚧 Loop not implemented yet. Exiting.")  # Remove this line
        break  # Remove this line


# --- Main Execution ---
if __name__ == "__main__":
    run_safety_analysis("Check the link between Imatinib and periorbital edema.")