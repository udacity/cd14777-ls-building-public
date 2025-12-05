# Emerging_Literature_Sentinel.py

import os
import sys
import re
from typing import List, Dict, Any
from datetime import datetime

# LangChain imports for building the agentic chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from ls_action_space.action_space import query_arxiv, query_scholar

from dotenv import load_dotenv
# Load environment variables from a .env file (for OPENAI_API_KEY)
load_dotenv("../../../.env")


def format_search_results(results_dict: Dict[str, List[Dict]]) -> str:
    """
    Formats the raw results from parallel fetching into a single string for the LLM.
    Handles and reports errors for each data source.
    """
    formatted_string = ""
    sources = {"arXiv Pre-prints": "arxiv", "Google Scholar Publications": "google_scholar"}

    for display_name, key in sources.items():
        results = results_dict.get(key, [])
        if results:
            formatted_string += f"--- {display_name} ---\n"
            for item in results:
                if "error" in item:
                    formatted_string += f"  - Error: {item['error']}\n"
                    continue
                # Use .get() for safe dictionary access
                title = item.get('title', 'N/A')
                summary = item.get('summary') or item.get('snippet', 'No summary available.')
                formatted_string += f"Title: {title}\n"
                formatted_string += f"Summary: {summary}\n\n"

    if not formatted_string.strip():
        return "No new literature was found for the specified topic."

    return formatted_string

# --- LLM and Agentic Chain Configuration ---

# 1. Define the topic to be monitored, or read it from command line parameter
TOPIC = sys.argv[1] if len(sys.argv) > 1 else "SARS-CoV-2 spike mutations and their impact on vaccine efficacy"

# 2. Create the prompt template for the LLM to structure its analysis.
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an expert research assistant in the life sciences. Your purpose is to analyze recent scientific literature and generate a concise intelligence briefing for a busy researcher."),
        ("human",
         """
         Please analyze the following list of recent publications and pre-prints on the topic of '{topic}'.
         Your task is to identify key subtopics, cluster the documents accordingly, and write a digest summarizing the most novel findings.

         **Instructions:**
         1.  Carefully read all provided titles and summaries.
         2.  Identify 2-3 distinct subtopics or research themes that emerge from the documents.
         3.  For each subtopic, create a section with a clear Markdown heading (e.g., `### Subtopic: ...`).
         4.  Under each subtopic heading, list the titles of the papers that fall into that cluster.
         5.  After the list of titles, write a 2-3 sentence synthesis of the key findings or arguments presented in that cluster.
         6.  Conclude with a "**Daily Executive Summary**" (1-2 paragraphs) that provides a high-level overview of the most significant developments across all topics.
         7.  If no literature is found, state that clearly.

         **Today's Literature Findings:**
         {documents}
         """)
    ]
)

# 3. Initialize the LLM model. Using 'gpt-4o-mini' for a balance of cost and capability.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# 4. Set up the parallel fetching mechanism using `RunnableParallel`.
# This runnable will execute both query functions concurrently with the same input query.
parallel_fetcher = RunnableParallel(
    arxiv=lambda x: query_arxiv(x, max_results=5),
    google_scholar=lambda x: query_scholar(x, max_results=5)
)

# 5. Construct the full processing chain using LangChain Expression Language (LCEL).
sentinel_chain = (
    # The input to the chain is the topic string.
    # Pass the topic through and simultaneously trigger the parallel fetcher.
    {"topic": RunnablePassthrough(), "documents": parallel_fetcher}
    # Re-assign the 'documents' key with the output of our formatting function.
    | RunnablePassthrough.assign(documents=lambda x: format_search_results(x["documents"]))
    # Pipe the resulting dictionary into the prompt template.
    | prompt_template
    # Pipe the formatted prompt into the LLM.
    | llm
    # Parse the LLM's chat message output into a simple string.
    | StrOutputParser()
)

# --- Main Execution Block ---

if __name__ == "__main__":
    print(f"🔬 **Emerging Literature Sentinel**")
    print(f"----------------------------------")
    print(f"**Monitoring Topic:** '{TOPIC}'")
    print(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
    print("\nFetching and analyzing recent publications... (this may take a moment)\n")

    try:
        # Invoke the chain with the defined topic
        daily_digest = sentinel_chain.invoke(TOPIC)

        # Print the final, formatted digest from the LLM
        print("--- Your Daily Digest ---")
        print(daily_digest)
        print("-------------------------")

    except Exception as e:
        print(f"\n❌ An error occurred during the process: {e}")
        print("Please check your OPENAI_API_KEY, internet connection, and installed packages.")