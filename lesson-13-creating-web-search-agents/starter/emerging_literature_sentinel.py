import os
import re
from typing import List, Dict, Any

# --- Provided Dependencies (No changes needed here) ---
import requests
import arxiv as arxiv_py
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


from ls_action_space.action_space import query_arxiv, query_scholar

from dotenv import load_dotenv
# Load environment variables from a .env file (for OPENAI_API_KEY)
load_dotenv(".env")

def format_search_results(results_dict: Dict[str, List[Dict]]) -> str:
    """Formats the raw results into a single string for the LLM."""
    output = []
    for source, results in [("arXiv Pre-prints", results_dict.get("arxiv")),
                            ("Google Scholar", results_dict.get("google_scholar"))]:
        if not results: continue
        output.append(f"--- {source} ---")
        for item in results:
            if "error" in item:
                output.append(f"  - Error: {item['error']}")
            else:
                summary = item.get('summary') or item.get('snippet', 'No summary available.')
                output.append(f"Title: {item.get('title', 'N/A')}\nSummary: {summary}\n")
    return "\n".join(output) if output else "No new literature was found."


# --- Configuration ---
TOPIC = "SARS-CoV-2 spike protein mutations and vaccine efficacy"

# ==============================================================================
# --- BUILD THE AGENT CHAIN (This is where you'll write your code) ---
# ==============================================================================

# 1. Set up the parallel data fetching using RunnableParallel.
#    This should be a dictionary where keys are names for the data sources
#    and values are the tool functions to call (e.g., query_arxiv).
parallel_fetcher = ...

# 2. Define the prompt for the LLM.
#    It should instruct the LLM to analyze, cluster, and summarize the documents.
#    Remember to include placeholders for the '{topic}' and '{documents}'.
prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert life sciences research assistant. Your goal is to create a concise daily digest for a busy scientist."),
    ("human", """
    **Topic:** {topic}

    **Instructions:** Please analyze the following documents. Cluster them into 2-3 key subtopics, and write a brief summary for each cluster. Conclude with a high-level 'Director's Summary'.

    **Today's Documents:**
    {documents}
    """)  # TODO: Feel free to improve this prompt!
])

# 3. Initialize the Language Model.
#    Use the "gpt-4o-mini" model for this exercise.
llm = ...

# 4. Combine everything into a single chain using the `|` operator.
#    The chain should:
#    a. Take the 'topic' as input and run the `parallel_fetcher`.
#    b. Format the fetched results using `format_search_results`.
#    c. Pass the formatted results and topic to the `prompt_template`.
#    d. Send the formatted prompt to the `llm`.
#    e. Parse the output to a clean string.
output_parser = StrOutputParser()

sentinel_chain = (
        {"topic": RunnablePassthrough(), "documents": parallel_fetcher}
        | ...  # TODO: Complete the rest of the chain here
)

# ==============================================================================
# --- Main Execution Block (No changes needed here) ---
# ==============================================================================

if __name__ == "__main__":
    print(f"🔬 Emerging Literature Sentinel")
    print(f"----------------------------------")
    print(f"Monitoring topic: '{TOPIC}'")
    print("Fetching and analyzing... (this may take a moment)\n")

    try:
        # Check if the chain is defined
        if sentinel_chain is ...:
            raise NotImplementedError("The 'sentinel_chain' has not been implemented yet!")

        # Invoke the chain with the topic
        daily_digest = sentinel_chain.invoke(TOPIC)

        print("--- Your Daily Digest ---")
        print(daily_digest)
        print("-------------------------")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")