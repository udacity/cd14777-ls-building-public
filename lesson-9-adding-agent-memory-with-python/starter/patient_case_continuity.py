import re
import operator
from typing import Any, List, Dict, Annotated, TypedDict

import torch
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# -----------------------------------------------------------------------------
# SECTION 1: LOCAL LLM SETUP (PRE-CONFIGURED FOR YOU)
# -----------------------------------------------------------------------------
# This section handles loading a local, privacy-preserving LLM.
# You don't need to modify this part.
# -----------------------------------------------------------------------------
print("Setting up local LLM (this might take a moment)...")
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/Qwen2-0.5B-Instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)
tokenizer = get_chat_template(tokenizer, chat_template="qwen2")

class CustomUnslothChatModel(BaseChatModel):
    """A LangChain-compatible wrapper for the local Unsloth model."""
    model: Any
    tokenizer: Any

    def _generate(self, messages: List[BaseMessage], **kwargs: Any) -> ChatResult:
        prompt = self.tokenizer.apply_chat_template([{"role": m.type, "content": m.content} for m in messages], tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=150, use_cache=True, temperature=0.6)
        decoded = self.tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        return ChatResult(generations=[AIMessage(content=decoded)])

    @property
    def _llm_type(self) -> str:
        return "custom_unsloth_chat_model"

chat_model = CustomUnslothChatModel(model=model, tokenizer=tokenizer)
print("✅ LLM is ready.")

# -----------------------------------------------------------------------------
# SECTION 2: AGENT STATE DEFINITION
# -----------------------------------------------------------------------------
# The 'AgentState' is the memory of our agent. It's a dictionary that
# holds information that persists and evolves across conversation turns.
# -----------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    """
    Represents the agent's memory.

    Attributes:
        messages: The history of the conversation. `operator.add` appends new messages.
        patient_info: A dictionary to store structured data like age and symptoms.
        output: The final message to show the user.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    patient_info: Dict[str, Any]
    output: str


# -----------------------------------------------------------------------------
# SECTION 3: GRAPH NODES (YOUR PRIMARY TASK 📝)
# -----------------------------------------------------------------------------
# Nodes are the building blocks of our agent. Each node is a function that
# performs an action, like updating memory or calling the LLM.
# -----------------------------------------------------------------------------

def update_patient_info(state: AgentState) -> Dict[str, Any]:
    """
    Node 1: Extracts structured data from the user's message to update the agent's memory.
    """
    # Always start with the memory from the previous turn.
    patient_info = state.get("patient_info", {}).copy()
    last_human_message = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")

    # TODO: Using the `last_human_message`, find and add the patient's age and symptoms
    # to the `patient_info` dictionary.
    # Hint: Use `re.search()` or simple string checks like `'symptoms:' in message.lower()`.
    # For example, to find age:
    # age_match = re.search(r"(\d{1,3})\s*year", last_human_message, re.IGNORECASE)
    # if age_match:
    #     patient_info["age"] = int(age_match.group(1))

    # YOUR CODE GOES HERE 👇


    # Return the updated dictionary to be saved in the agent's state.
    return {"patient_info": patient_info}


def call_model(state: AgentState) -> Dict[str, Any]:
    """
    Node 2: Calls the LLM with the latest patient information to generate a clinical summary.
    """
    # Get the patient info that was built in the previous node.
    pinfo = state.get("patient_info", {})
    print(f"DEBUG: patient_info before LLM call: {pinfo}") # for checking your work

    # TODO: Create a `prompt` for the LLM that includes the patient information
    # from the `pinfo` dictionary. Then, invoke the LLM.
    #
    # Example prompt:
    # prompt = f"Patient Summary:\nAge: {pinfo.get('age', 'N/A')}\nSymptoms: {pinfo.get('symptoms', 'N/A')}\n\nProvide a brief assessment."
    # ai_response = chat_model.invoke([HumanMessage(content=prompt)])

    # YOUR CODE GOES HERE 👇
    # Replace this placeholder with your LLM call
    ai_response = AIMessage(content="[LLM response will go here]")


    # Return the LLM's response so it can be added to the message history.
    return {"messages": [ai_response], "output": ai_response.content}


# -----------------------------------------------------------------------------
# SECTION 4: GRAPH DEFINITION (YOUR SECONDARY TASK 🧠)
# -----------------------------------------------------------------------------
# Here, you will define the agent's workflow by connecting the nodes you built.
# The agent will move from one node to the next in the order you specify.
# -----------------------------------------------------------------------------
workflow = StateGraph(AgentState)

# TODO: Add the two nodes you defined above to the graph.
# Hint: workflow.add_node("unique_node_name", function_name)


# TODO: Define the sequence of execution.
# The entry point should be 'update_patient_info', which then leads to 'call_model'.
# Hint: workflow.set_entry_point(...)
#       workflow.add_edge(start_node, end_node)
workflow.add_edge("agent", END) # The graph ends after the agent speaks.


# -----------------------------------------------------------------------------
# SECTION 5: COMPILE AND RUN (PRE-CONFIGURED FOR YOU)
# -----------------------------------------------------------------------------
# This section compiles your graph and sets up the memory checkpointer.
# The `if __name__ == "__main__"` block runs a sample conversation.
# -----------------------------------------------------------------------------
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    session_id = "patient_case_abc_123"
    config = {"configurable": {"thread_id": session_id}}

    print("\n--- Patient Consultation (Enter 'quit' to exit) ---")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        # This is how you run the agent. The 'app.invoke' call processes the
        # input and automatically handles loading/saving memory for the session.
        out = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
        print("AI:", out.get("output", ""))

    print("\n--- Final Persisted Patient Info ---")
    final_state = checkpointer.get(config)
    print(final_state['values'].get('patient_info'))