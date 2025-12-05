import os
from typing import List, Dict, Any

# Import the pre-defined tools from the Life Sciences Action Space
# These functions interact with the NCBI ClinVar and PubMed APIs.
from ls_action_space.action_space import query_clinvar, query_pubmed

# LangChain components for building the agent
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

# Ensure you have your OpenAI API key set in your environment variables
from dotenv import load_dotenv
load_dotenv("../../../.env")


# --- Tool Definition ---
# The agent needs to know what tools it can use. We wrap the imported
# functions with the @tool decorator to make them accessible to the agent.

@tool
def clinvar_variant_lookup(variant: str) -> Dict[str, Any]:
    """
    Looks up a genetic variant (e.g., an rsID like "rs7412" or HGVS
    like "BRAF V600E") in the ClinVar database. Returns its clinical
    significance, associated conditions, review status, and supporting
    publication PMIDs.
    """
    return query_clinvar(variant)

@tool
def pubmed_literature_search(search_query: str) -> List[Dict[str, Any]]:
    """
    Searches PubMed for articles matching a given query string.
    For example, 'BRAF V600E melanoma treatment'. Returns a list of the top
    3 most relevant articles with their title, journal, year, and PMID.
    """
    # We limit the results to 3 to keep the report concise.
    return query_pubmed(query=search_query, max_results=3)

# --- Agent Initialization ---

def create_triage_agent():
    """
    Creates and configures the LangChain agent for dynamic variant triage.
    """
    # 1. Define the tools the agent can use
    tools = [clinvar_variant_lookup, pubmed_literature_search]

    # 2. Define the LLM (Large Language Model)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url="https://openai.vocareum.com/v1")

    # 3. Create the Agent's Prompt
    prompt_template = """
    You are a specialized clinical bioinformatics assistant. Your task is to analyze
    a genetic variant and generate a summary report for a Molecular Tumor Board.

    **Workflow:**
    1.  Receive a variant query.
    2.  Use the `clinvar_variant_lookup` tool to get its clinical details.
    3.  **Analyze the `clinical_significance` field.** If it is 'Pathogenic' or 'Likely pathogenic',
        classify the variant as **High-Risk** and proceed to the literature search. Otherwise, do not search for literature.
    4.  For **High-Risk** variants, perform a three-step literature search:
        a. **Step 1: Simplify long condition names.**. Focus on the core disease keywords. 
           - Example: shorten 'Mendelian susceptibility to mycobacterial diseases due to complete ISG15 deficiency' to just 'mycobacterial disease'.
        b. **Step 2: Specific Search.** First, construct a query combining the `gene`, the simplified `protein change` (e.g., 'V600E'), and the primary `condition`.
           - Example: "BRAF V600E melanoma".
           - Use the `pubmed_literature_search` tool with this specific query.
        c. **Step 3: Fallback Search.** If and ONLY IF the specific search returns **zero results**, you must perform a second, broader search.
           - For this search, construct a query using only the `gene` and the `condition`.
           - Example: "BRAF melanoma".
           - Use the `pubmed_literature_search` tool again with this broader query.
    5.  Synthesize all gathered information into a final report. Use the results from whichever literature search was successful. If both searches fail, state that no relevant literature was found.

    **Critical Rule for PubMed Queries:**
    - **NEVER** use the full, complex HGVS notation (e.g., `NM_004333.6:c.1799T>G`) in a PubMed search. It will always fail. Stick to the simplified format described above.

    **Output Format (use Markdown):**

    # Variant Triage Report: {input}

    ## 1. Summary
    - **Gene:** [Gene Symbol]
    - **Risk Assessment:** [High-Risk / Low-Risk / Uncertain Significance]
    - **Justification:** [Briefly state why, e.g., "ClinVar classification is 'Pathogenic'."]

    ## 2. Clinical Significance (from ClinVar)
    - **Classification:** [e.g., Pathogenic]
    - **Review Status:** [e.g., criteria provided, multiple submitters, no conflicts]
    - **Associated Conditions:** [List conditions, separated by commas]

    ## 3. Supporting Literature (for High-Risk Variants)
    * **PMID:** [PMID_1] | **Title:** [Title of Article 1] | **Journal, Year:** [Journal, YYYY]
    * **PMID:** [PMID_2] | **Title:** [Title of Article 2] | **Journal, Year:** [Journal, YYYY]
    * **PMID:** [PMID_3] | **Title:** [Title of Article 3] | **Journal, Year:** [Journal, YYYY]

    ## 4. Actionable Recommendation
    - [e.g., "Variant flagged for immediate review by the Molecular Tumor Board due to its pathogenic status."]
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )


    # 4. Create the Agent
    agent = create_openai_tools_agent(llm, tools, prompt)

    # 5. Create the Agent Executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor

def generate_variant_report(agent_executor: AgentExecutor, variant: str):
    """
    Invokes the agent to generate and print a report for a given variant.
    """
    print(f"\n▶️  Generating report for variant: {variant}...")
    print("-" * 50)
    try:
        response = agent_executor.invoke({"input": variant})
        print(response['output'])
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your OPENAI_API_KEY is set correctly and the required packages are installed.")
    print("-" * 50)

# --- Main Execution Block ---

if __name__ == "__main__":
    # Check if the API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("🔴 ERROR: The OPENAI_API_KEY environment variable is not set.")
        print("Please set it before running the script.")
    else:
        # Create the agent executor
        triage_agent = create_triage_agent()

        # --- Example 1: A well-known pathogenic variant (BRAF V600E) ---
        # This variant is high-risk and should trigger a PubMed search.
        pathogenic_variant = "rs113488022"
        generate_variant_report(triage_agent, pathogenic_variant)

        # --- Example 2: A benign variant (in APOE gene) ---
        # This variant is not high-risk, so the agent should not search PubMed.
        benign_variant = "rs429358"
        generate_variant_report(triage_agent, benign_variant)