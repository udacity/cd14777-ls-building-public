# simplified_starter_code.py

import os
import json
import re
from typing import List, TypedDict, Dict, Any

import chromadb
from openai import OpenAI
from langgraph.graph import StateGraph, END

# Assume ls_action_space/action_space.py is in the PYTHONPATH
from ls_action_space.action_space import query_pubmed, query_clinvar

# --- 1. Configuration & Setup ---
from dotenv import load_dotenv
# Load environment variables from a .env file (for OPENAI_API_KEY)
load_dotenv(".env")

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("🔴 OPENAI_API_KEY environment variable not set.")

client = OpenAI(base_url="https://openai.vocareum.com/v1")
db_client = chromadb.PersistentClient(path="./gene_disease_db_simple")
collection = db_client.get_or_create_collection(name="gene_disease_literature_simple")
MAX_ITERATIONS = 3


# --- 2. State Definition (Provided) ---
class GraphState(TypedDict):
    """Represents the state of our agent's thought process."""
    question: str
    key_terms: List[str]
    documents: List[str]
    answer: str
    missing_terms: List[str]
    iterations: int


# --- 3. Helper Functions (Provided) ---
def _process_api_results(pubmed_docs: List[Dict], clinvar_doc: Dict) -> List[tuple[str, str]]:
    """Helper to format API results into (id, text) tuples."""
    processed = []
    for doc in pubmed_docs:
        processed.append((f"pmid_{doc['pmid']}", f"Title: {doc['title']}\nAbstract: {doc['abstract']}"))
    if clinvar_doc and 'error' not in clinvar_doc:
        vcv = clinvar_doc['accessions'].get('VCV', [None])[0]
        doc_id = f"vcv_{vcv}" if vcv else f"gene_{clinvar_doc.get('gene', 'unknown')}"
        content = f"ClinVar: {clinvar_doc['title']}. Significance: {clinvar_doc['clinical_significance']}."
        processed.append((doc_id, content))
    return processed


# --- 4. Agent Nodes (Student to Complete the TODOs) ---

def extract_key_terms(state: GraphState) -> Dict[str, Any]:
    """Identifies core concepts from the question."""
    print("--- 🔬 EXTRACTING KEY TERMS ---")
    question = state['question']

    # Prompt is provided for you
    prompt = f"""
    From the question "{question}", identify the primary gene, disease, and 1-2 critical biological keywords.
    Return a JSON object with keys: "gene", "disease", "keywords" (a list of strings).
    """

    # TODO: Call the OpenAI API to get the key terms.
    # 1. Make a call to `client.chat.completions.create` with model 'gpt-4o-mini' and the prompt above.
    #    - Remember to enable JSON mode: `response_format={"type": "json_object"}`
    # 2. Parse the JSON string from the response message.
    # 3. Combine the gene, disease, and keywords into a single list.
    response_data = {}  # Replace {} with your parsed JSON data
    key_terms = []  # Replace [] with your combined list

    print(f"🔑 Key Terms Identified: {key_terms}")
    return {"key_terms": key_terms, "iterations": 0}


def retrieve_and_embed(state: GraphState) -> Dict[str, Any]:
    """Fetches, embeds, and retrieves relevant documents."""
    print("\n--- 📚 RETRIEVING & EMBEDDING DOCUMENTS ---")

    # Data fetching and processing logic is provided
    search_query = ' AND '.join(state['key_terms'])
    pubmed_docs = query_pubmed(search_query, max_results=3)
    clinvar_doc = query_clinvar(state['key_terms'][0])  # Query ClinVar by gene
    all_docs = _process_api_results(pubmed_docs, clinvar_doc)

    # Logic to find which documents are new is provided
    doc_ids = [doc[0] for doc in all_docs]
    existing_ids = set(collection.get(ids=doc_ids)['ids'])
    new_docs = [doc for doc in all_docs if doc[0] not in existing_ids]

    if new_docs:
        print(f"Found {len(new_docs)} new documents to add to the database.")
        new_doc_ids = [doc[0] for doc in new_docs]
        new_doc_texts = [doc[1] for doc in new_docs]

        # TODO: Embed and add the new documents to ChromaDB.
        # 1. Create embeddings for `new_doc_texts` using OpenAI's "text-embedding-3-small" model.
        # 2. Add the documents, their embeddings, and their IDs to the Chroma `collection`.
        pass  # Remove this line after you add your code

    # TODO: Query ChromaDB for relevant documents.
    # 1. Create an embedding for the original user `state['question']`.
    # 2. Query the `collection` using this embedding to get the top 5 most relevant documents.
    # 3. Extract just the document texts from the query results.
    retrieved_documents = []  # Replace [] with your extracted document texts

    print(f"Retrieved {len(retrieved_documents)} documents from DB for synthesis.")
    return {"documents": retrieved_documents}


def generate_answer(state: GraphState) -> Dict[str, str]:
    """Synthesizes an answer using the retrieved context."""
    print("\n--- ✍️ GENERATING ANSWER ---")
    context = "\n\n---\n\n".join(state['documents'])

    # RAG prompt is provided for you
    prompt = f"""
    Based ONLY on the following context, provide a clear and concise answer to the user's question.
    If the context is insufficient, state that you cannot answer with the information provided.

    CONTEXT:
    {context}

    QUESTION:
    {state['question']}
    """

    # TODO: Call the OpenAI API to generate the final answer.
    # 1. Make a call to `client.chat.completions.create` with the prompt above.
    # 2. Extract the answer text from the response.
    answer = "..."  # Replace "..." with the answer from the LLM

    print("Generated Answer Snippet:", answer[:200] + "...")
    return {"answer": answer}


# --- 5. Critique & Decision Nodes (Provided) ---
# The logic for the agentic loop is fully provided so you can focus on the core RAG steps.
def critique_answer(state: GraphState) -> Dict[str, Any]:
    """Checks if the generated answer contains all required key terms."""
    print("\n--- 🤔 CRITIQUING ANSWER ---")
    missing_terms = [
        term for term in state['key_terms']
        if not re.search(r'\b' + re.escape(term) + r'\b', state['answer'], re.IGNORECASE)
    ]
    if not missing_terms:
        print("✅ All key terms are covered. Finalizing.")
    else:
        print(f"❌ Missing key terms: {missing_terms}. Will re-try.")
    return {"missing_terms": missing_terms, "iterations": state['iterations'] + 1}


def decide_to_finish(state: GraphState) -> str:
    """Determines whether to end the process or loop for refinement."""
    if not state['missing_terms'] or state['iterations'] >= MAX_ITERATIONS:
        return "finish"
    else:
        return "refine"


# --- 6. Graph Definition (Provided) ---
workflow = StateGraph(GraphState)
workflow.add_node("extract_key_terms", extract_key_terms)
workflow.add_node("retrieve_and_embed", retrieve_and_embed)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("critique_answer", critique_answer)
workflow.set_entry_point("extract_key_terms")
workflow.add_edge("extract_key_terms", "retrieve_and_embed")
workflow.add_edge("retrieve_and_embed", "generate_answer")
workflow.add_edge("generate_answer", "critique_answer")
workflow.add_conditional_edges("critique_answer", decide_to_finish, {"refine": "retrieve_and_embed", "finish": END})
app = workflow.compile()

# --- 7. Main Execution Block (Provided) ---
if __name__ == "__main__":
    question = "What is the mechanism linking the BRCA1 gene to ovarian cancer, specifically mentioning homologous recombination?"
    initial_state = {"question": question}
    print(f"🚀 Starting Agent for Question: \"{question}\"\n")
    final_state = app.invoke(initial_state)
    print("\n" + "=" * 50 + "\n✨ AGENT RUN COMPLETE ✨")
    print(f"\n📝 Final Answer:\n   {final_state.get('answer', 'No answer generated.')}")
    print("=" * 50)