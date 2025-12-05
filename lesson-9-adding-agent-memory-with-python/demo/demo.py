#!/usr/bin/env python3
"""
Demo: Building Short-Term Memory for Patient Cohort Tracking

This demo implements short-term memory in AI-driven clinical support using LangGraph.
It tracks patient-specific context across conversation turns.
"""

import os
import re
from typing import Annotated, TypedDict, List, Dict, Any
import operator
from dotenv import load_dotenv

# Try to import LangGraph and OpenAI
try:
    from openai import OpenAI
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("Warning: LangGraph/OpenAI dependencies not available. Running in stub mode.")


# Load environment variables
load_dotenv(".env")


class AgentState(TypedDict, total=False):
    """
    Agent state tracking:
    - messages: conversation history
    - patient_info: dictionary with patient ID and variant counts
    - output: agent's response
    """
    messages: Annotated[List, operator.add]
    patient_info: Dict[str, Any]
    output: str


def update_patient_info(state: AgentState) -> Dict[str, Any]:
    """
    Extract patient IDs and update variant counts from user input.
    """
    patient_info = state.get("patient_info", {}).copy()

    # Get last human message
    last_human = ""
    for m in reversed(state.get("messages", [])):
        if LANGGRAPH_AVAILABLE and isinstance(m, HumanMessage):
            last_human = m.content
            break
        elif isinstance(m, dict) and m.get("type") == "human":
            last_human = m.get("content", "")
            break

    # Extract patient ID (e.g., "patient 123")
    pid_match = re.search(r"patient\s*(\d+)", last_human.lower())
    if pid_match:
        patient_info["id"] = pid_match.group(1)

    # Count variants mentioned
    if "variant" in last_human.lower():
        patient_info["variant_count"] = patient_info.get("variant_count", 0) + 1

    return {"patient_info": patient_info}


def summarize_patient(state: AgentState) -> Dict[str, Any]:
    """
    Generate contextual responses about patient review progress.
    """
    pinfo = state.get("patient_info", {})

    if "id" in pinfo and "variant_count" in pinfo:
        summary = f"You've reviewed {pinfo['variant_count']} variants for patient {pinfo['id']}."
    else:
        summary = "No patient or variant information yet."

    if LANGGRAPH_AVAILABLE:
        return {"messages": [AIMessage(content=summary)], "output": summary}
    else:
        return {"messages": [{"type": "ai", "content": summary}], "output": summary}


def call_model(state: AgentState) -> Dict[str, Any]:
    """
    Invoke the LLM with system prompt and current patient state.
    """
    pinfo = state.get("patient_info", {})
    system_prompt = "You are a clinical assistant. Track patients and variants, and remind the user of context."

    context = f"Current patient info: {pinfo}"

    if LANGGRAPH_AVAILABLE:
        # Real implementation with OpenAI
        client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1"),
            api_key=os.getenv("OPENAI_API_KEY", "dummy-key")
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context}
        ]

        try:
            ai_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.5,
            )
            output_text = ai_response.choices[0].message.content
        except Exception as e:
            output_text = f"[Simulated AI response for: {context}]"

        return {"messages": [AIMessage(content=output_text)], "output": output_text}
    else:
        # Stub implementation
        output_text = f"[Simulated AI response] I understand you're tracking {context}"
        return {"messages": [{"type": "ai", "content": output_text}], "output": output_text}


def build_graph():
    """
    Build the LangGraph workflow with memory checkpoints.
    """
    if not LANGGRAPH_AVAILABLE:
        return None

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("update_info", update_patient_info)
    workflow.add_node("summarize", summarize_patient)
    workflow.add_node("agent", call_model)

    # Define flow
    workflow.set_entry_point("update_info")
    workflow.add_edge("update_info", "summarize")
    workflow.add_edge("summarize", "agent")
    workflow.add_edge("agent", END)

    # Add memory checkpointer
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    return app


def run_stub_demo():
    """
    Run a stub demonstration when LangGraph is not available.
    """
    print("\n" + "="*60)
    print("STUB MODE: Running simplified demo without LangGraph")
    print("="*60 + "\n")

    # Manual state tracking
    state = {
        "messages": [],
        "patient_info": {},
        "output": ""
    }

    turns = [
        "Review variant BRCA1 p.Gly123 for patient 123",
        "Now check variant TP53 p.Arg175 for patient 123",
        "Remind me what we've reviewed so far."
    ]

    for i, turn_text in enumerate(turns, 1):
        print(f"\nTurn {i}: {turn_text}")

        # Add human message
        state["messages"].append({"type": "human", "content": turn_text})

        # Update patient info
        update_result = update_patient_info(state)
        state["patient_info"].update(update_result.get("patient_info", {}))

        # Summarize
        summary_result = summarize_patient(state)
        state["messages"].extend(summary_result.get("messages", []))

        # Call model
        model_result = call_model(state)
        state["output"] = model_result["output"]

        print(f"AI: {state['output']}")
        print(f"Memory: Patient {state['patient_info'].get('id', 'N/A')}, "
              f"{state['patient_info'].get('variant_count', 0)} variants")

    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)


def run_langgraph_demo():
    """
    Run the full LangGraph demonstration.
    """
    print("\n" + "="*60)
    print("Running LangGraph Agent Memory Demo")
    print("="*60 + "\n")

    app = build_graph()
    session_id = "cohort_demo_001"
    config = {"configurable": {"thread_id": session_id}}

    # Turn 1: First variant review
    print("\nTurn 1: Review variant BRCA1 p.Gly123 for patient 123")
    out = app.invoke(
        {"messages": [HumanMessage(content="Review variant BRCA1 p.Gly123 for patient 123")]},
        config=config
    )
    print(f"AI: {out['output']}")

    # Turn 2: Second variant review
    print("\nTurn 2: Now check variant TP53 p.Arg175 for patient 123")
    out = app.invoke(
        {"messages": [HumanMessage(content="Now check variant TP53 p.Arg175 for patient 123")]},
        config=config
    )
    print(f"AI: {out['output']}")

    # Turn 3: Memory recall
    print("\nTurn 3: Remind me what we've reviewed so far.")
    out = app.invoke(
        {"messages": [HumanMessage(content="Remind me what we've reviewed so far.")]},
        config=config
    )
    print(f"AI: {out['output']}")

    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)


def main():
    """
    Main entry point for the demo.
    """
    print("Lesson 9: Adding Agent Memory with Python")
    print("Demo: Building Short-Term Memory for Patient Cohort Tracking")

    if LANGGRAPH_AVAILABLE:
        run_langgraph_demo()
    else:
        run_stub_demo()


if __name__ == "__main__":
    main()
