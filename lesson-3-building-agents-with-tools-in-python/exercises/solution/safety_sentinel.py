import os
import json
import math
import requests
from openai import OpenAI
from typing import Dict

# --- Configuration & API Key ---
from dotenv import load_dotenv
load_dotenv("../../../../.env")
client = OpenAI(base_url="https://openai.vocareum.com/v1")

# --- Tool 1: FAERS Data Fetching ---

def _query_faers_api(search_query: str) -> int:
    """
    Helper function to query the openFDA API and return the total count of matching records.
    Returns 0 if the query fails or finds no results.
    """
    base_url = "https://api.fda.gov/drug/event.json"
    params = {"search": search_query, "limit": 1}
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()
        return data.get("meta", {}).get("results", {}).get("total", 0)
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return 0

def get_faers_contingency_counts(drug_name: str, adverse_event: str) -> Dict[str, int]:
    """
    Fetches data from the FAERS database to build a 2x2 contingency table
    for calculating the Reporting Odds Ratio (ROR).

    This table consists of four groups:
    a: Reports containing the specific drug AND the specific adverse event.
    b: Reports containing the drug BUT NOT the adverse event.
    c: Reports NOT containing the drug BUT containing the adverse event.
    d: All other reports in the database.

    Args:
        drug_name (str): The name of the drug to investigate (e.g., "ibuprofen").
        adverse_event (str): The adverse event to investigate (e.g., "headache").

    Returns:
        Dict[str, int]: A dictionary containing the counts for a, b, c, and d.
    """
    print(f"🛠️  Fetching FAERS data for {drug_name} and {adverse_event}...")

    # Define the search terms for the API queries
    drug_query = f'patient.drug.medicinalproduct:"{drug_name}"'
    event_query = f'patient.reaction.reactionmeddrapt:"{adverse_event}"'

    # 1. Get count for 'a' (drug AND event)
    a = _query_faers_api(f"({drug_query}) AND ({event_query})")

    # 2. Get total reports for the drug (a+b)
    total_drug_reports = _query_faers_api(drug_query)
    b = total_drug_reports - a

    # 3. Get total reports for the event (a+c)
    total_event_reports = _query_faers_api(event_query)
    c = total_event_reports - a

    # 4. Get total reports in the entire database (a+b+c+d)
    total_reports = _query_faers_api("") # An empty query gets the total count
    d = total_reports - a - b - c

    counts = {"a": a, "b": b, "c": c, "d": d}
    print(f"📊 Contingency Table: {counts}")
    return counts


# --- Tool 2: Statistical Calculation ---

def calculate_ror(a: int, b: int, c: int, d: int) -> Dict[str, float]:
    """
    Calculates the Reporting Odds Ratio (ROR) and its 95% confidence interval.

    ROR is a measure of disproportionality. It compares the odds of an
    adverse event occurring with a specific drug to the odds of it
    occurring with all other drugs.

    Formula: ROR = (a/c) / (b/d) = (a*d) / (b*c)

    Args:
        a (int): Count of (drug=YES, event=YES).
        b (int): Count of (drug=YES, event=NO).
        c (int): Count of (drug=NO, event=YES).
        d (int): Count of (drug=NO, event=NO).

    Returns:
        A dictionary with the calculated ROR and the lower/upper bounds
        of the 95% confidence interval. Returns None if calculation is impossible.
    """
    print("🛠️  Calculating ROR...")

    # Apply continuity correction to avoid division by zero if any cell is 0
    if a == 0 or b == 0 or c == 0 or d == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5

    # Calculate ROR
    ror = (a * d) / (b * c)

    # Calculate 95% Confidence Interval
    log_ror = math.log(ror)
    se_log_ror = math.sqrt(1/a + 1/b + 1/c + 1/d)
    
    log_ci_lower = log_ror - 1.96 * se_log_ror
    log_ci_upper = log_ror + 1.96 * se_log_ror

    ci_lower = math.exp(log_ci_lower)
    ci_upper = math.exp(log_ci_upper)

    result = {
        "ror": round(ror, 2),
        "ci_lower": round(ci_lower, 2),
        "ci_upper": round(ci_upper, 2)
    }
    print(f"📈 ROR Result: {result}")
    return result


# --- Agent Orchestration ---

def run_safety_analysis(user_prompt: str):
    """
    Runs the full agent workflow:
    1. Sends the user prompt and tool definitions to the LLM.
    2. The LLM decides which tool to call and with what arguments.
    3. The script executes the tool and sends the result back to the LLM.
    4. This continues until the LLM provides a final, natural-language answer.
    """
    print(f"\n💬 User Prompt: '{user_prompt}'")
    
    messages = [
        {
            "role": "system",
            "content": """You are a helpful pharmacovigilance assistant. Your goal is to help users assess potential drug safety signals using a two-step process:
            1. First, call the `get_faers_contingency_counts` tool to get the necessary data from the adverse event database.
            2. Second, use the data returned from the first tool to call the `calculate_ror` tool.
            3. Finally, interpret the numerical result from `calculate_ror` into a clear, plain-English summary for the user. Explain what the ROR and confidence interval mean. State if it indicates a potential signal (typically if the lower CI bound is > 1).
            Do not ask the user for clarification; proceed directly with the tool calls."""
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_faers_contingency_counts",
                "description": "Fetches data from FAERS to build a 2x2 contingency table for ROR calculation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "drug_name": {"type": "string", "description": "The name of the drug, e.g., 'ibuprofen'"},
                        "adverse_event": {"type": "string", "description": "The adverse event, e.g., 'headache'"},
                    },
                    "required": ["drug_name", "adverse_event"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_ror",
                "description": "Calculates the Reporting Odds Ratio (ROR) and its 95% confidence interval.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer", "description": "Count of (drug=YES, event=YES)"},
                        "b": {"type": "integer", "description": "Count of (drug=YES, event=NO)"},
                        "c": {"type": "integer", "description": "Count of (drug=NO, event=YES)"},
                        "d": {"type": "integer", "description": "Count of (drug=NO, event=NO)"},
                    },
                    "required": ["a", "b", "c", "d"],
                },
            }
        }
    ]
    
    # Available functions that the agent can call
    available_functions = {
        "get_faers_contingency_counts": get_faers_contingency_counts,
        "calculate_ror": calculate_ror,
    }

    # Loop until the model provides a text response instead of a tool call
    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # If the model wants to call a tool
        if tool_calls:
            messages.append(response_message)  # Append the assistant's turn
            
            # Execute each tool call
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                
                function_response = function_to_call(**function_args)
                
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response),
                })
        else:
            # If the model provides a final answer
            final_answer = response_message.content
            print(f"\n✅ AI Assistant's Final Analysis:\n{final_answer}")
            break


# --- Main Execution ---

if __name__ == "__main__":
    # Example: A common drug and a common symptom
    run_safety_analysis("Is there a signal for headache with ibuprofen?")

    print("Done")