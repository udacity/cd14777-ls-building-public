# dynamic_variant_triage_starter.py

import os
from typing import List, Dict, Any

# --- Prerequisites (Provided for the student) ---

# 1. SET YOUR OPENAI API KEY
# Best practice is to use an environment variable.
# Create a .env file in your project folder with the line:
# OPENAI_API_KEY='your_key_here'
from dotenv import load_dotenv
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("🔴 ERROR: OPENAI_API_KEY not found. Please set it in your .env file.")
    exit()

# LangChain components are pre-imported for you.
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from ls_action_space.action_space import query_pubmed, query_clinvar

# 3. AGENT AND PROMPT (Provided for the student)
# The reasoning engine for the agent is already defined.
# The student does not need to modify this.

def create_triage_agent(tools: list):
    """Creates and configures the LangChain agent."""
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a clinical bioinformatics assistant generating a report.
        1. Use the `clinvar_variant_lookup` tool to get the variant's details.
        2. If the variant is 'Pathogenic', it is High-Risk.
        3. For High-Risk variants, use the `pubmed_literature_search` tool.
           - Create a simple search query with the gene and the first condition.
           - Example Query: "BRAF Melanoma"
        4. Synthesize the results into a concise, well-formatted report.
        """),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


# --- EXERCISE: Student starts here ---

# ✏️ STEP 1: DEFINE THE TOOLS
# The Learning Objective: Make the external Python functions (`query_clinvar`,
# `query_pubmed`) available to the LangChain agent.
#
# INSTRUCTIONS:
#   - Use the `@tool` decorator from LangChain to wrap each mock API function.
#   - Write a clear docstring for each tool. This is what the AI agent will read
#     to understand what the tool does and when to use it. The docstring is critical!

# TODO: Create the `clinvar_variant_lookup` tool here (invoking query_clinvar imported above).
#       Its docstring should tell the agent that it's for looking up a variant's
#       clinical significance in ClinVar.

# TODO: Create the `pubmed_literature_search` tool here (invoking query_pubmed imported above).
#       Its docstring should explain that it searches PubMed for articles
#       based on a search query string.


# --- Main Execution Block (Provided for the student) ---

if __name__ == "__main__":
    print("--- Dynamic Variant Triage Agent ---")

    # ✏️ STEP 2: ASSEMBLE AND RUN THE AGENT
    # The Learning Objective: Use the tools you defined to build and run an
    # agent that incorporates live (mocked) data into its workflow.
    #
    # INSTRUCTIONS:
    #   1. Create a list called `agent_tools` that contains the two tools
    #      you defined above.
    #   2. Create the agent by calling the `create_triage_agent` function
    #      and passing your `agent_tools` list to it.
    #   3. Define a variant to analyze
    #      Examples:
    #      - pathogenic_variant = "rs113488022" # A well-known pathogenic variant of BRAF; should trigger PubMed search
    #      - benign_variant = "rs429358" # A benign variant (in APOE gene)
    #   4. Invoke the agent with the variant and print the final output.

    # TODO: Create a list containing your two tools.
    # agent_tools = [...]

    # TODO: Create the agent executor.
    # triage_agent = create_triage_agent(...)

    # TODO: Define a pathogenic variant to test the full workflow.
    # variant_to_analyze = "..."

    # TODO: Invoke the agent with the variant. The input must be a dictionary
    #       with the key "input".
    # result = triage_agent.invoke({"input": ...})
    # print("\n--- FINAL REPORT ---")
    # print(result['output'])

    print("\n--- Exercise Complete ---")