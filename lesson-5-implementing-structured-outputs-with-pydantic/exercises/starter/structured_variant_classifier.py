# main_starter.py

import os
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from dotenv import load_dotenv
# Load environment variables from a .env file (for OPENAI_API_KEY)
load_dotenv(".env")

# --- Pre-configured Setup (No changes needed here) ---
# Assumes OPENAI_API_KEY is set in your environment
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url="https://openai.vocareum.com/v1")

# Example free-text report to be parsed
free_text_report = """
Variant analysis for c.4538G>A in the BRCA2 gene. This variant, also known as 
rs80359550, results in a p.Arg1513Gln missense mutation. Multiple functional 
studies and co-segregation data have led to its classification. The expert panel has classified this variant 
as Pathogenic. This conclusion is supported by evidence from PMID: 18684768 and 
PMID: 20104584, which detail its impact on protein function.
"""
# --- End of Setup ---


# TODO 1: Define the data structure for the output using Pydantic.
# What are the key pieces of information you need to extract?
# Give them appropriate names and Python types (e.g., str, List[str]).
class VariantReport(BaseModel):
    """Structured representation of a genetic variant report."""
    # Add fields here. For example: gene_name: str = Field(description="...")
    pass


# TODO 2: Create the Output Parser.
# The parser needs to know which Pydantic model to use for structuring and validation.
parser = PydanticOutputParser(pydantic_object=...) # Link the parser to your VariantReport model


# TODO 3: Write the Prompt Template.
# Your prompt must include:
#   1. Clear instructions for the LLM.
#   2. The placeholder for the input text (`{variant_report_text}`).
#   3. The placeholder for the formatting instructions (`{format_instructions}`).
prompt = PromptTemplate(
    template="""
    ... Your instructions to the LLM go here ...
    """,
    input_variables=["variant_report_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


# TODO 4: Assemble the processing chain.
# The chain should pipe the components together in the correct order:
# prompt -> llm -> parser
chain = ... # Assemble the chain here using the pipe | operator


# --- Run and Verify (No changes needed here) ---
# This section invokes your chain and prints the result.
# If your code is correct, it will print a validated, structured object.
try:
    structured_report = chain.invoke({"variant_report_text": free_text_report})
    print("--- ✅ Successfully Parsed Variant Report ---")
    print(structured_report.model_dump_json(indent=2))
except Exception as e:
    print(f"--- ❌ An error occurred ---")
    print(e)