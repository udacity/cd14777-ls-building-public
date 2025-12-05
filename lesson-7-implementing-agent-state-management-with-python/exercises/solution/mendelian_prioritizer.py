#
# Udacity Life Sciences Agentic AI Nanodegree
# Exercise: Mendelian Candidate-Gene Prioritizer (LLM & Live ClinVar Version)
#
# This script uses an LLM (gpt-4o-mini) and a live NCBI ClinVar query
# to build a stateful agent that prioritizes candidate genes.
#

import os
import re
import operator
import time
from typing import TypedDict, Annotated, List, Dict, Any

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from ls_action_space.action_space import query_clinvar


# ==============================================================================
# 1. SET UP THE LLM
# ==============================================================================
# Set your OpenAI API key
from dotenv import load_dotenv
load_dotenv("../../../../.env")

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, base_url="https://openai.vocareum.com/v1")


# ==============================================================================
# 2. DEFINE THE STATE OF THE AGENT
# ==============================================================================

class GenePrioritizationState(TypedDict):
    """Updated state to handle LLM outputs and live ClinVar data."""
    phenotypes: List[str]
    raw_variants: List[str]
    llm_candidate_genes: List[str]
    variants_to_process: List[str]
    processed_variant_details: Annotated[list, operator.add]
    ranked_results: dict


# ==============================================================================
# 3. DEFINE THE WORKFLOW NODES
# ==============================================================================

def map_phenotypes_to_genes_llm(state: GenePrioritizationState) -> dict:
    """
    Node: Uses an LLM to identify candidate genes based on phenotypes.
    """
    print("## Node: map_phenotypes_to_genes_llm ##")
    phenotypes = state["phenotypes"]
    print(f"-> Input Phenotypes: {phenotypes}")

    prompt = (
        "You are a medical geneticist. Based on the following clinical phenotypes, "
        "provide a comma-separated list of the most relevant associated human gene symbols. "
        "Return ONLY the gene symbols and nothing else.\n\n"
        f"Phenotypes: {', '.join(phenotypes)}"
    )

    response = llm.invoke(prompt)
    gene_string = response.content
    candidate_genes = [gene.strip() for gene in gene_string.split(',') if gene.strip()]

    print(f"-> LLM Candidate Genes: {candidate_genes}")
    return {"llm_candidate_genes": candidate_genes}


def lookup_variant_in_clinvar(state: GenePrioritizationState) -> dict:
    """
    Node: Takes one variant from the queue and queries the live ClinVar DB.
    """
    print("\n## Node: lookup_variant_in_clinvar ##")
    variants_to_process = state["variants_to_process"]
    variant = variants_to_process.pop(0)
    print(f"-> Querying ClinVar for variant: {variant}")

    # Use the provided live query function
    variant_details = query_clinvar(variant)

    if "error" in variant_details:
        print(f"-> ClinVar Error: {variant_details['error']}")
        return {"variants_to_process": variants_to_process}  # Continue without this variant

    print(f"-> Found Gene: {variant_details.get('gene')}, Significance: {variant_details.get('clinical_significance')}")

    return {
        "variants_to_process": variants_to_process,
        "processed_variant_details": [variant_details],
    }


def rank_candidates(state: GenePrioritizationState) -> dict:
    """
    Node: Scores and ranks genes by intersecting LLM suggestions with ClinVar findings.
    """
    print("\n## Node: rank_candidates ##")
    llm_genes = set(state["llm_candidate_genes"])
    variant_details = state["processed_variant_details"]

    scores = {}
    pathogenic_keywords = {"pathogenic", "likely pathogenic"}

    for details in variant_details:
        gene = details.get("gene")
        significance = (details.get("clinical_significance") or "").lower()

        if not gene:
            continue

        # Scoring Logic:
        # +2 points if the gene was suggested by the LLM
        # +5 points if the variant is pathogenic or likely pathogenic
        score = 0
        if gene in llm_genes:
            score += 2
            print(f"-> Evidence match: Gene {gene} was suggested by LLM.")

        if any(keyword in significance for keyword in pathogenic_keywords):
            score += 5
            print(f"-> Evidence match: Variant in {gene} is pathogenic.")

        if score > 0:
            scores[gene] = scores.get(gene, 0) + score

    ranked_genes = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    print("\n-> Final Ranked Genes:")
    if not ranked_genes:
        print("  - No significant overlapping candidates found.")
    else:
        for gene, score in ranked_genes:
            print(f"  - {gene}: Score {score}")

    return {"ranked_results": dict(ranked_genes)}


# ==============================================================================
# 4. DEFINE CONDITIONAL LOGIC (EDGES)
# ==============================================================================

def should_continue_variant_lookup(state: GenePrioritizationState) -> str:
    """
    Conditional Edge: Decides whether to continue looking up variants.
    """
    print("\n-- Conditional Edge: more variants to check? --")
    if len(state["variants_to_process"]) > 0:
        print("--> YES, continuing variant lookup loop.")
        return "continue"
    else:
        print("--> NO, all variants processed. Proceeding to ranking.")
        return "end"


# ==============================================================================
# 5. BUILD AND COMPILE THE GRAPH
# ==============================================================================

workflow = StateGraph(GenePrioritizationState)

workflow.add_node("llm_phenotype_mapper", map_phenotypes_to_genes_llm)
workflow.add_node("clinvar_variant_lookup", lookup_variant_in_clinvar)
workflow.add_node("final_ranker", rank_candidates)

workflow.set_entry_point("llm_phenotype_mapper")

workflow.add_edge("llm_phenotype_mapper", "clinvar_variant_lookup")
workflow.add_edge("final_ranker", END)

workflow.add_conditional_edges(
    "clinvar_variant_lookup",
    should_continue_variant_lookup,
    {"continue": "clinvar_variant_lookup", "end": "final_ranker"},
)

app = workflow.compile()


# ==============================================================================
# 6. GENERATE A HUMAN-READABLE REPORT
# ==============================================================================

def generate_final_report(final_state: GenePrioritizationState):
    """
    Formats and prints the final results in a clear, human-readable report.
    """
    ranked_results = final_state.get("ranked_results", {})

    print("\n\n" + "=" * 60)
    print("🔬       Mendelian Candidate-Gene Prioritization Report       🔬")
    print("=" * 60)

    # --- Section 1: Top Conclusion ---
    print("\n## 1. Top Conclusion\n")
    if not ranked_results:
        print("No high-confidence candidate genes were identified based on the provided data.")
    else:
        top_gene = next(iter(ranked_results))
        print(
            f"The analysis points to **{top_gene}** as the most likely candidate gene driving the patient's phenotype.")
        print("This conclusion is based on converging evidence from phenotypic analysis and variant pathogenicity.")

    # --- Section 2: Patient Summary ---
    print("\n" + "-" * 60)
    print("\n## 2. Patient Input Summary\n")
    print(f"**Clinical Phenotypes:** {', '.join(final_state['phenotypes'])}")
    print(f"**Genomic Variants Analyzed:** {', '.join(final_state['raw_variants'])}")

    # --- Section 3: Evidence Breakdown ---
    print("\n" + "-" * 60)
    print("\n## 3. Evidence Breakdown\n")

    # LLM-generated hypotheses
    print(f"🧬 **Phenotype-Gene Hypotheses (from LLM):**")
    print(
        f"Based on the phenotypes, the AI suggested the following candidate genes: {', '.join(final_state['llm_candidate_genes'])}")

    # Detailed findings for each ranked gene
    print("\n🔍 **Variant-Gene Evidence (from ClinVar):**")
    if not ranked_results:
        print("No pathogenic variants were found within the list of LLM-suggested genes.")
    else:
        for gene, score in ranked_results.items():
            print(f"\n  ▶ **Gene: {gene} (Score: {score})**")
            # Find the variant details corresponding to this gene
            for detail in final_state['processed_variant_details']:
                if detail.get("gene") == gene:
                    significance = detail.get('clinical_significance', 'N/A')
                    pathogenic = "Pathogenic" in significance or "Likely pathogenic" in significance

                    print(f"    - **Variant Found:** {detail['query']}")
                    print(f"    - **Clinical Significance:** {significance} {'✅' if pathogenic else ''}")
                    print(
                        f"    - **Reasoning:** This gene was prioritized because it was identified by the AI based on phenotype AND a patient variant within this gene was confirmed as pathogenic by ClinVar.")

    # --- Section 4: Disclaimer ---
    print("\n\n" + "=" * 60)
    print("\n**Disclaimer:** This is an automated analysis for educational purposes only.")
    print("=" * 60)


# ==============================================================================
# 7. RUN THE WORKFLOW AND GENERATE THE REPORT
# ==============================================================================

# Patient data including phenotypes and raw variant calls
patient_input = {
    "phenotypes": ["Hypertrophic cardiomyopathy", "Atrial fibrillation"],
    "raw_variants": ["rs397516459", "rs121913629", "rs3756073"],
    "variants_to_process": ["rs397516459", "rs121913629", "rs3756073"],
    "processed_variant_details": [],
}

print("==============================================")
print("🚀 Starting LLM-Powered Gene Analysis 🚀")
print("==============================================")

if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY environment variable not set.")
else:
    # Invoke the graph to get the final state
    final_state = app.invoke(patient_input)

    print("\n==============================================")
    print("✅ Workflow Complete. Generating Final Report... ✅")
    print("==============================================")

    # Generate and print the nice report
    generate_final_report(final_state)