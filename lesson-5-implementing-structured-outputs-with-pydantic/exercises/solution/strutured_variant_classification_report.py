import os
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Ensure you have your OpenAI API key set in your environment variables
from dotenv import load_dotenv
load_dotenv("../../../../.env")

# --- 1. Define the Desired Data Structure with Pydantic ---
# This class defines the schema for our structured output.
# Pydantic will use it to validate the LLM's response.
class VariantReport(BaseModel):
    """Structured representation of a genetic variant report."""
    gene_name: str = Field(description="The official symbol of the gene associated with the variant.")
    variant_identifier: str = Field(description="The standardized identifier for the variant (e.g., dbSNP rsID or HGVS nomenclature).")
    clinical_classification: str = Field(description="The clinical significance of the variant (e.g., 'Pathogenic', 'Benign', 'Uncertain Significance').")
    evidence_citations: List[str] = Field(description="A list of citations or PubMed IDs (PMIDs) supporting the classification.")

# --- 2. Set up the Model and Output Parser ---
# Initialize the language model we'll use for extraction.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url="https://openai.vocareum.com/v1")

# Create a PydanticOutputParser instance linked to our data model.
# This parser will automatically generate formatting instructions and
# parse the LLM's output into a VariantReport object.
parser = PydanticOutputParser(pydantic_object=VariantReport)

# --- 3. Create a Prompt Template ---
# The prompt instructs the LLM on its task, provides the input text,
# and includes the formatting instructions from the parser.
prompt_template = """
You are an expert bioinformatics agent. Your task is to extract key information 
from a free-text genetic variant report and structure it as a JSON object.

CONTEXT:
Here is the genetic variant report:
"{variant_report_text}"

INSTRUCTIONS:
Please extract the gene name, variant identifier, clinical classification, and all evidence citations.
{format_instructions}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["variant_report_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# --- 4. Define the Input Data and Run the Extraction Chain ---
# Example free-text report similar to a ClinVar entry.
free_text_report = """
Variant analysis for c.4538G>A in the BRCA2 gene. This variant, also known as 
rs80359550, results in a p.Arg1513Gln missense mutation. Multiple functional 
studies and co-segregation data from families with hereditary breast and ovarian 
cancer have led to its classification. The expert panel has classified this variant 
as Pathogenic. This conclusion is supported by evidence from PMID: 18684768 and 
PMID: 20104584, which detail its impact on protein function.
"""

# Create the processing chain by piping together the prompt, LLM, and parser.
chain = prompt | llm | parser

# Invoke the chain with the input report.
# The result is a validated Pydantic object, not just a string.
structured_report: VariantReport = chain.invoke({"variant_report_text": free_text_report})

# --- 5. Print and Verify the Structured Output ---
print("--- Successfully Parsed Variant Report ---")
print(f"Gene: {structured_report.gene_name}")
print(f"Variant ID: {structured_report.variant_identifier}")
print(f"Classification: {structured_report.clinical_classification}")
print(f"Citations: {structured_report.evidence_citations}")

# Pydantic models can be easily converted to dictionaries or JSON for downstream use.
print("\n--- Output as JSON ---")
print(structured_report.model_dump_json(indent=2))