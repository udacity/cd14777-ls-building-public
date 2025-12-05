import os
import operator
import re
from typing import TypedDict, Annotated, List, Dict, Any

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# ==============================================================================
# SECTION 1: SETUP & PROVIDED TOOLS (Minimal edits for reliability)
# ==============================================================================

# --- 1a. Configure APIs ---
# Set your OpenAI API key and your email for NCBI (if needed)
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"
# Entrez.email = "your.email@example.com"

# Initialize the LLM
from dotenv import load_dotenv
load_dotenv(".env")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, base_url="https://openai.vocareum.com/v1")

# --- 1b. ClinVar Query Function (with safe fallback mock) ---
USING_MOCK_CLINVAR = False
try:
    from ls_action_space.action_space import query_clinvar  # real implementation
except Exception as e:
    USING_MOCK_CLINVAR = True
    print(f"[Notice] Using mock ClinVar (import failed: {e})")
    # Simple, deterministic mock to keep the exercise unblocked
    def query_clinvar(variant: str) -> dict:
        MOCK = {
            "rs397516459": {"query": "rs397516459", "gene": "MYH7", "clinical_significance": "Pathogenic"},
            "rs121913629": {"query": "rs121913629", "gene": "MYBPC3", "clinical_significance": "Likely pathogenic"},
        }
        return MOCK.get(
            variant,
            {"query": variant, "gene": None, "clinical_significance": "Uncertain significance"},
        )

# Quick preflight to fail fast on missing API key (students see the message up front)
if not os.getenv("OPENAI_API_KEY"):
    print("[Warning] OPENAI_API_KEY is not set. LLM calls will fail until you set it.")


# ==============================================================================
# SECTION 2: DEFINE THE AGENT'S STATE (No changes needed here)
# ==============================================================================

class GenePrioritizationState(TypedDict):
    phenotypes: List[str]
    raw_variants: List[str]
    llm_candidate_genes: List[str]
    variants_to_process: List[str]
    processed_variant_details: Annotated[list, operator.add]
    ranked_results: dict


# ==============================================================================
# SECTION 3: IMPLEMENT THE AGENT'S TOOLS (Nodes)
# ==============================================================================

## TODO: Implement the three nodes and the conditional logic ##

def map_phenotypes_to_genes_llm(state: GenePrioritizationState) -> dict:
    """
    Node 1: Use the LLM to get candidate genes from phenotypes.

    Instructions:
    1. Read `phenotypes` from `state`.
    2. Build a prompt that asks for a **comma-separated list of gene symbols ONLY**.
    3. Call `llm.invoke(prompt)`; get the text via `response.content`.
    4. Parse into a list: split on ',', strip whitespace, drop empties.
    5. Return: {"llm_candidate_genes": <list_of_gene_symbols>}

    Return shape (must match state keys):
        {"llm_candidate_genes": [...]}
    """
    print("## Node: map_phenotypes_to_genes_llm ##")
    # --- YOUR CODE HERE ---
    # phenotypes = state["phenotypes"]
    # prompt = f"... use {', '.join(phenotypes)} ..."
    # response = llm.invoke(prompt)
    # gene_string = response.content
    # genes = [g.strip() for g in gene_string.split(",") if g.strip()]
    # return {"llm_candidate_genes": genes}
    pass


def lookup_variant_in_clinvar(state: GenePrioritizationState) -> dict:
    """
    Node 2: Look up a single variant in ClinVar. This node will loop.

    Instructions:
    1. Get `variants_to_process` from state and pop the next variant with `.pop(0)`.
    2. Call `query_clinvar(variant)` (provided; a mock is available if import failed).
    3. Print a brief status line showing variant, gene, significance.
    4. Return updates for:
       - "variants_to_process": the shortened list
       - "processed_variant_details": [detail_dict]  (single-element list; LangGraph will append)

    Return shape (must match state keys):
        {
          "variants_to_process": [...],
          "processed_variant_details": [<one_detail_dict>]
        }
    """
    print("\n## Node: lookup_variant_in_clinvar ##")
    # --- YOUR CODE HERE ---
    # queue = state["variants_to_process"]
    # variant = queue.pop(0)
    # detail = query_clinvar(variant)
    # print(f"Queried {variant} -> gene={detail.get('gene')}, significance={detail.get('clinical_significance')}")
    # return {"variants_to_process": queue, "processed_variant_details": [detail]}
    pass


def rank_candidates(state: GenePrioritizationState) -> dict:
    """
    Node 3: Score and rank the genes based on all collected evidence.

    Instructions:
    1. Read `llm_candidate_genes` and `processed_variant_details` from state.
    2. Create a `scores = {}` dict keyed by gene.
    3. For each detail:
       a. If `gene` missing, skip.
       b. If gene is in the LLM list -> add small bonus (e.g., +2).
       c. If `clinical_significance` contains 'pathogenic' (case-insensitive) -> add larger bonus (e.g., +5).
    4. Sort by descending score. Convert to a dict preserving order (or a list of tuples—either is fine for the report).
    5. Return: {"ranked_results": <ranked_mapping>}

    Hint: use `.lower()` on significance for a case-insensitive check.
    """
    print("\n## Node: rank_candidates ##")
    # --- YOUR CODE HERE ---
    # llm_genes = set(state["llm_candidate_genes"])
    # details = state["processed_variant_details"]
    # scores: Dict[str, int] = {}
    # for d in details:
    #     gene = d.get("gene")
    #     if not gene:
    #         continue
    #     sig = (d.get("clinical_significance") or "").lower()
    #     score = 0
    #     if gene in llm_genes:
    #         score += 2
    #     if "pathogenic" in sig:
    #         score += 5
    #     if score:
    #         scores[gene] = scores.get(gene, 0) + score
    # ranked = dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))
    # return {"ranked_results": ranked}
    pass


def should_continue_variant_lookup(state: GenePrioritizationState) -> str:
    """
    Conditional Edge: Decide whether to loop or exit.

    Instructions:
    1. If `variants_to_process` still has items, return "continue".
    2. Otherwise, return "end".
    """
    print("\n-- Conditional Edge: more variants to check? --")
    # --- YOUR CODE HERE ---
    # return "continue" if state["variants_to_process"] else "end"
    pass


# ==============================================================================
# SECTION 4: BUILD THE GRAPH (Pre-wired to save time)
# ==============================================================================

workflow = StateGraph(GenePrioritizationState)

# 1) Add nodes
workflow.add_node("llm_phenotype_mapper", map_phenotypes_to_genes_llm)
workflow.add_node("clinvar_variant_lookup", lookup_variant_in_clinvar)
# TODO: Add the node to rank candidates

# 2) Set the entry point
workflow.set_entry_point("llm_phenotype_mapper")

# 3) Standard edges
# TODO: Add the edge from phenotype mapping to lookup

workflow.add_edge("final_ranker", END)

# 4) Conditional loop (continue → lookup again; end → rank)
workflow.add_conditional_edges(
    "clinvar_variant_lookup",
    should_continue_variant_lookup,
    {"continue": "clinvar_variant_lookup", "end": "final_ranker"},
)

# 5) Compile the graph
app = workflow.compile()


# ==============================================================================
# SECTION 5: RUN THE WORKFLOW (No changes needed here)
# ==============================================================================

def generate_final_report(final_state: GenePrioritizationState):
    """Provided function to generate a nice report."""
    print("\n\n" + "="*60 + "\n🔬 Mendelian Candidate-Gene Prioritization Report 🔬\n" + "="*60)
    ranked_results = final_state.get("ranked_results", {})
    if not ranked_results:
        print("\nNo high-confidence candidate genes were identified.")
    else:
        top_gene = next(iter(ranked_results))
        print(f"\n## Top Conclusion: **{top_gene}** is the most likely candidate gene.")
    print("\n" + "-"*60 + "\n## Evidence Breakdown\n")
    print(f"Phenotype-Gene Hypotheses (LLM): {final_state.get('llm_candidate_genes', [])}")
    for gene, score in ranked_results.items():
        for detail in final_state.get("processed_variant_details", []):
            if detail.get("gene") == gene:
                print(f"\n  ▶ Gene: {gene} (Score: {score})")
                print(f"    - Variant Found: {detail['query']}")
                print(f"    - Significance: {detail.get('clinical_significance', 'N/A')}")
    print("\n" + "="*60)


# Small, fixed input to keep the loop quick & deterministic
patient_input = {
    "phenotypes": ["Hypertrophic cardiomyopathy", "Atrial fibrillation"],
    "raw_variants": ["rs397516459", "rs121913629"],
    "variants_to_process": ["rs397516459", "rs121913629"],
    "processed_variant_details": [],
}

print("🚀 Starting Gene Analysis...")
if os.getenv("OPENAI_API_KEY") and query_clinvar:
    # NOTE: This will only succeed once you've implemented the TODOs above.
    final_state = app.invoke(patient_input)
    print("\n✅ Workflow finished. Generating report...")
    generate_final_report(final_state)
else:
    print("\nERROR: Please set your OPENAI_API_KEY. "
          "If ClinVar import failed, a mock is already enabled so you can proceed.")
