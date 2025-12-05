"""
Traditional RAG Demo: Gene-Disease Association Assistant

A Retrieval-Augmented Generation system for life sciences that synthesizes
gene-disease associations from PubMed abstracts stored in a vector database.
"""

import os
from typing import Dict, Any, List, TypedDict
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env")

# Check if we can import OpenAI and ChromaDB
try:
    from openai import OpenAI
    import chromadb
    from chromadb.utils import embedding_functions
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("Warning: OpenAI or ChromaDB not installed. Running in stub mode.")


# Type definitions
class State(TypedDict, total=False):
    question: str
    documents: List[str]
    messages: List[Dict[str, str]]
    answer: str


class Resource(TypedDict, total=False):
    vars: Dict[str, Any]


# Configuration
PERSIST_DIR = os.getenv("PERSIST_DIR", "data/gene_disease_pubmed_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "gene_disease_pubmed")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def setup_openai_client():
    """Initialize OpenAI client with Vocareum base URL."""
    if not HAS_DEPS:
        return None

    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY not set in .env file")
        return None

    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


def call_llm(client, messages, model=None, temperature=0.2):
    """Helper function to call the LLM."""
    if not client:
        # Stub response for demo purposes
        return type('obj', (object,), {
            'content': """- CGRP — GWAS; multiple genome-wide studies identify this pathway
- KCNK18 — familial; rare variants segregate in migraine families
- SCN1A — functional; ion channel mutations affect neuronal excitability
- CACNA1A — familial; hemiplegic migraine mutations well-characterized
- MTHFR — GWAS; polymorphism shows modest association in meta-analyses

Caveats: Effect sizes are generally small, replication varies across populations, and migraine heterogeneity complicates gene-phenotype mapping."""
        })()

    model = model or MODEL_NAME
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages
    )
    return response.choices[0].message


def load_vector_store():
    """Load the prepared ChromaDB collection."""
    if not HAS_DEPS:
        print("Stub mode: Simulating vector store load")
        return None

    if not os.path.exists(PERSIST_DIR):
        print(f"Warning: Vector store directory '{PERSIST_DIR}' not found")
        print("The demo will run in stub mode with simulated data")
        return None

    # Create embedding function matching the one used to build the store
    embeddings_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL,
        api_base=OPENAI_BASE_URL,
    )

    # Load the persistent collection
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embeddings_fn
    )

    return collection


def retrieve(state: State, resource: Resource) -> Dict:
    """
    Retrieval step: Query the vector store for relevant documents.

    Args:
        state: Current state containing the question
        resource: Resource containing the collection

    Returns:
        Dictionary with retrieved documents
    """
    question = state["question"]
    collection = resource.get("vars", {}).get("collection")

    if not collection:
        # Stub mode: return simulated documents
        stub_docs = [
            "CGRP pathway genes show strong association with migraine in multiple GWAS studies.",
            "KCNK18 variants have been identified in familial hemiplegic migraine cases.",
            "SCN1A mutations affect neuronal excitability and are linked to migraine susceptibility.",
            "CACNA1A is well-characterized in hemiplegic migraine with functional evidence.",
            "MTHFR polymorphisms show modest association with migraine in meta-analyses."
        ]
        return {"documents": stub_docs}

    # Query the collection
    results = collection.query(
        query_texts=[question],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )

    return {"documents": results["documents"][0]}


def augment(state: State, resource: Resource = None) -> Dict:
    """
    Augmentation step: Frame the prompt with context and guardrails.

    Args:
        state: Current state containing question and documents
        resource: Optional resource (not used in this step)

    Returns:
        Dictionary with messages for the LLM
    """
    question = state["question"]
    docs = state.get("documents", [])
    context = "\n\n".join(docs)

    system = (
        "You are a biomedical assistant. Use HGNC gene symbols. "
        "Cite the strongest **type of evidence** per gene (e.g., GWAS, familial, functional). "
        "Keep it concise, do not give medical advice, and state uncertainty if applicable."
    )

    user = f"""Use the retrieved context to answer.

# Task
Return up to **5 genes** most consistently associated with the condition or phenotype in the question.

# Question
{question}

# Context
{context}

# Output format
- GENE — evidence type; 1 short clause of rationale
Caveats: one short sentence about evidence limits (e.g., heterogeneity, small samples, mixed replication)."""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    return {"messages": messages}


def generate(state: State, resource: Resource) -> Dict:
    """
    Generation step: Call the LLM to produce the final answer.

    Args:
        state: Current state containing messages
        resource: Resource containing the LLM client

    Returns:
        Dictionary with the answer and updated messages
    """
    client = resource.get("vars", {}).get("client")
    ai = call_llm(client, state["messages"], model=MODEL_NAME, temperature=0.2)

    return {
        "answer": ai.content,
        "messages": state["messages"] + [{"role": "assistant", "content": ai.content}]
    }


def run_workflow(initial_state: State, resource: Resource) -> State:
    """
    Execute the full RAG workflow: Retrieve → Augment → Generate.

    Args:
        initial_state: Starting state with the question
        resource: Shared resources (client, collection)

    Returns:
        Final state with the answer
    """
    state = {**initial_state}

    # Execute pipeline
    state.update(retrieve(state, resource))
    state.update(augment(state, resource))
    state.update(generate(state, resource))

    return state


def main():
    """Main execution function."""
    print("=" * 70)
    print("Traditional RAG Demo: Gene-Disease Association Assistant")
    print("=" * 70)
    print()

    # Setup
    print("Setting up OpenAI client...")
    client = setup_openai_client()

    print("Loading vector store...")
    collection = load_vector_store()

    # Create resource bundle
    resource: Resource = {
        "vars": {
            "client": client,
            "collection": collection,
            "llm": MODEL_NAME
        }
    }

    print()
    print("-" * 70)
    print("DEMO QUERY")
    print("-" * 70)

    # Example query
    initial_state: State = {
        "question": "Which genes show replicated association with migraine?"
    }

    print(f"Question: {initial_state['question']}")
    print()

    # Run the workflow
    print("Executing RAG workflow (Retrieve → Augment → Generate)...")
    print()
    final_state = run_workflow(initial_state, resource)

    # Display results
    print("-" * 70)
    print("ANSWER")
    print("-" * 70)
    print(final_state["answer"])
    print()

    # Additional examples
    print("=" * 70)
    print("Try these example queries:")
    print("=" * 70)
    examples = [
        "Top genes implicated in familial hypercholesterolemia (FH).",
        "Ion channel genes most associated with epilepsy.",
        "Genes frequently reported in inflammatory bowel disease."
    ]
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")
    print()

    print("To run with different queries, modify the 'question' in main.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
