#!/usr/bin/env python3
"""
Demo: Agent with Long‑Term Memory for miRNA Literature Notes (Self‑Contained)
"""

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
from dotenv import load_dotenv

# ----------------------------- Utility: deterministic RNG -----------------------------
try:
    from secrets import randbelow  # stdlib, fine
except Exception:  # pragma: no cover
    randbelow = None  # not critical

# ----------------------------- Optional LLM wiring -----------------------------
USE_LLM_DEFAULT = True

load_dotenv(".env")

class LLMSummarizer:
    """Thin wrapper; falls back to heuristic if no key or flag."""
    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm and bool(os.getenv("OPENAI_API_KEY"))
        self.client = None
        if self.use_llm:
            try:
                from openai import OpenAI
                base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
                self.client = OpenAI(base_url=base_url)
            except Exception:
                self.use_llm = False

    def summarize(self, text: str) -> str:
        if not text:
            return "• (no abstract text available)"
        if self.use_llm and self.client is not None:
            prompt = (
                "Summarize in exactly 3 bullets focusing on: (1) mechanism of miRNA–target regulation, "
                "(2) experimental evidence type, (3) biological/clinical consequence. Be concise.\n\n" + text
            )
            try:
                res = self.client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                out = res.choices[0].message.content.strip()
                return out
            except Exception:
                # fall back to local heuristics
                pass
        return heuristic_summarize(text)

# ----------------------------- Heuristic summarizer -----------------------------
MECH_HINTS = [
    "bind", "targets", "target", "3'UTR", "regulat", "suppress", "downregul", "upregul",
    "inhibit", "repress", "knockdown", "overexpress"
]
EVID_HINTS = ["luciferase", "reporter", "western", "immunoblot", "rt-pcr", "qpcr", "chip", "flow", "assay"]
CONS_HINTS = ["prolifer", "apoptosis", "invasion", "metast", "tumor", "oncogen", "chemo", "prognos"]

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def pick_sentence(text: str, hints: List[str]) -> Optional[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    scored: List[Tuple[int, str]] = []
    for s in sentences:
        s_l = s.lower()
        score = sum(s_l.count(h) for h in hints)
        if score:
            scored.append((score, s.strip()))
    if not scored and sentences:
        # fallback: pick a mid sentence to avoid boilerplate openings
        idx = min(1, len(sentences) - 1)
        return sentences[idx].strip()
    return max(scored, key=lambda t: t[0])[1] if scored else None


def heuristic_summarize(text: str) -> str:
    """Return three bullet points (mechanism, evidence, consequence) from the abstract."""
    mech = pick_sentence(text, MECH_HINTS)
    evid = pick_sentence(text, EVID_HINTS)
    cons = pick_sentence(text, CONS_HINTS)

    bullets = [
        f"• Mechanism: {mech}" if mech else "• Mechanism: (not detected)",
        f"• Evidence: {evid}" if evid else "• Evidence: (not detected)",
        f"• Consequence: {cons}" if cons else "• Consequence: (not detected)",
    ]
    return "\n".join(bullets)

# ----------------------------- Minimal TF‑IDF vector store -----------------------------

def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]


class VectorStore:
    def __init__(self, name: str):
        self.name = name
        self.docs: List[Dict[str, Any]] = []  # {id, content, metadata}
        self.df: Counter = Counter()
        self.vocab: Dict[str, int] = {}  # token -> df
        self._dirty = False

    # persistence (optional)
    def dump_jsonl(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for d in self.docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def load_jsonl(self, path: str) -> None:
        if not os.path.exists(path):
            return
        self.docs.clear(); self.df.clear(); self.vocab.clear()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                self.docs.append(d)
        # rebuild df
        for d in self.docs:
            terms = set(tokenize(d.get("content", "")))
            for t in terms:
                self.df[t] += 1
        self.vocab = dict(self.df)

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None, ids: Optional[List[str]] = None) -> None:
        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [f"doc_{len(self.docs)+i}" for i in range(len(texts))]
        for text, md, id_ in zip(texts, metadatas, ids):
            # upsert by id
            self.docs = [d for d in self.docs if d["id"] != id_]
            doc = {"id": id_, "content": text, "metadata": md}
            self.docs.append(doc)
            for t in set(tokenize(text)):
                self.df[t] += 1
            self._dirty = True
        self.vocab = dict(self.df)

    def _tfidf(self, text: str) -> Dict[str, float]:
        tokens = tokenize(text)
        tf = Counter(tokens)
        N = max(len(self.docs), 1)
        vec: Dict[str, float] = {}
        for t, c in tf.items():
            df_t = self.df.get(t, 0)
            idf = math.log((N + 1) / (df_t + 1)) + 1.0
            vec[t] = (c / len(tokens)) * idf
        return vec

    @staticmethod
    def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        num = sum(a[t] * b[t] for t in common)
        denom = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
        return num / denom if denom else 0.0

    def similarity_search(self, query: str, k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List["MockDocument"]:
        qv = self._tfidf(query)
        results: List[Tuple[float, Dict[str, Any]]] = []
        for d in self.docs:
            if filter and not self._passes_filter(d.get("metadata", {}), filter):
                continue
            dv = self._tfidf(d.get("content", ""))
            score = self._cosine(qv, dv)
            results.append((score, d))
        results.sort(key=lambda x: x[0], reverse=True)
        top = [MockDocument(dd["content"], dd.get("metadata", {})) for _, dd in results[:k]]
        return top

    @staticmethod
    def _passes_filter(md: Dict[str, Any], f: Dict[str, Any]) -> bool:
        for key, val in f.items():
            if key == "year" and isinstance(val, dict) and "$gte" in val:
                y = md.get("year")
                if y is None or int(y) < int(val["$gte"]):
                    return False
            else:
                if md.get(key) != val:
                    return False
        return True


class VectorStoreManager:
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self._stores: Dict[str, VectorStore] = {}

    def get_or_create_store(self, name: str) -> VectorStore:
        if name not in self._stores:
            self._stores[name] = VectorStore(name)
        return self._stores[name]


class MockDocument:
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.page_content = content
        self.metadata = metadata

# ----------------------------- Data structures -----------------------------

@dataclass
class LiteratureNote:
    content: str
    owner: str
    mirna: str
    target_gene: str
    pmid: str
    year: Optional[int] = None
    namespace: str = "mirna_lit"
    timestamp: int = field(default_factory=lambda: int(datetime.now().timestamp()))


@dataclass
class NoteSearchResult:
    notes: List[LiteratureNote]
    metadata: Dict[str, Any]


# ----------------------------- Memory wrapper -----------------------------

class LiteratureMemory:
    def __init__(self, db: VectorStoreManager, store_name: str = "mirna_literature_notes", persist_path: Optional[str] = None):
        self.vector_store = db.get_or_create_store(store_name)
        self.persist_path = persist_path
        if self.persist_path and os.path.exists(self.persist_path):
            self.vector_store.load_jsonl(self.persist_path)

    def register(self, note: LiteratureNote, extra: Optional[Dict[str, Any]] = None) -> None:
        md = {
            "owner": note.owner,
            "mirna": note.mirna,
            "target_gene": note.target_gene,
            "pmid": note.pmid,
            "year": note.year,
            "namespace": note.namespace,
            "timestamp": note.timestamp,
        }
        if extra:
            md.update(extra)
        self.vector_store.add_texts([note.content], metadatas=[md], ids=[note.pmid])
        if self.persist_path:
            self.vector_store.dump_jsonl(self.persist_path)

    def search(
        self,
        query_text: str,
        owner: str,
        limit: int = 5,
        namespace: str = "mirna_lit",
        year_min: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> NoteSearchResult:
        f = {"owner": owner, "namespace": namespace}
        if filters:
            f.update(filters)
        if year_min is not None:
            f["year"] = {"$gte": year_min}
        hits = self.vector_store.similarity_search(query_text, k=limit, filter=f)
        as_notes: List[LiteratureNote] = []
        for h in hits:
            m = h.metadata or {}
            as_notes.append(
                LiteratureNote(
                    content=h.page_content,
                    owner=m.get("owner", ""),
                    mirna=m.get("mirna", ""),
                    target_gene=m.get("target_gene", ""),
                    pmid=m.get("pmid", ""),
                    year=m.get("year"),
                    namespace=m.get("namespace", "mirna_lit"),
                    timestamp=m.get("timestamp", int(datetime.now().timestamp())),
                )
            )
        return NoteSearchResult(notes=as_notes, metadata={"count": len(as_notes)})


# ----------------------------- Parsing -----------------------------

SimpleRecord = Dict[str, Any]


def parse_simple_tsv_line(line: str) -> Optional[SimpleRecord]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 3:
        return None
    # Accept 4+ columns: miRNA, TargetGene, PMID, Abstract, [Year]
    if len(parts) >= 4:
        mirna, gene, pmid, abstract = parts[:4]
        year = int(parts[4]) if len(parts) >= 5 and parts[4].isdigit() else None
        return {"mirna": mirna, "gene": gene, "pmid": pmid, "abstract": abstract, "year": year}
    return None


def parse_blocked_pubmed_file(lines: Iterable[str]) -> List[SimpleRecord]:
    """Parse files where each PMID appears on multiple lines, with fields like
    "PMID\tTitle\t...", "PMID\tAbstract\t..." and entity tags T1/T2/Tn for miRNA/Target_gene.
    Expands to one record per (miRNA, target_gene) pair found for a PMID. If a PMID has miRNA but
    no target, we still emit a record with target_gene set to "(unknown)".
    """
    by_pmid: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"abstract": "", "mirnas": set(), "targets": set(), "year": None})
    title_cache: Dict[str, str] = {}

    for raw in lines:
        raw = raw.rstrip("\n")
        if not raw:
            continue
        parts = raw.split("\t")
        if len(parts) < 2:
            continue
        pmid = parts[0]
        label = parts[1].lower()
        if label == "title" and len(parts) >= 3:
            title_cache[pmid] = parts[2]
        elif label == "abstract" and len(parts) >= 3:
            by_pmid[pmid]["abstract"] = parts[2]
        elif parts[1].startswith("T") and len(parts) >= 4:
            # e.g., 19937137\tT2\tTarget_gene\tLATS2
            ent_type = parts[2].lower()
            ent_val = parts[3]
            if ent_type == "mirna":
                by_pmid[pmid]["mirnas"].add(ent_val)
            elif ent_type in ("target_gene", "target", "gene"):
                by_pmid[pmid]["targets"].add(ent_val)
        # Optionally infer year from title/abstract if present (not in sample). Skipped here.

    records: List[SimpleRecord] = []
    for pmid, blob in by_pmid.items():
        abstract = blob.get("abstract", "")
        mirnas = blob.get("mirnas") or set()
        targets = blob.get("targets") or set()
        title = title_cache.get(pmid, "")
        if not mirnas:
            mirnas = {"(unknown miRNA)"}
        if not targets:
            targets = {"(unknown target)"}
        for mi in sorted(mirnas):
            for tg in sorted(targets):
                records.append({
                    "mirna": mi,
                    "gene": tg,
                    "pmid": pmid,
                    "abstract": abstract,
                    "year": None,
                    "title": title,
                })
    return records


def parse_file(src: str) -> List[SimpleRecord]:
    records: List[SimpleRecord] = []
    with open(src, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # Heuristic: if most lines start with digits + TAB and include labels like Title/Abstract/T1,
    # treat as blocked file; otherwise try simple TSV rows.
    sample = lines[:20]
    if any("\tT1\t" in ln or "\tAbstract\t" in ln for ln in sample):
        return parse_blocked_pubmed_file(lines)
    # else: treat as simple TSV, skipping header if present
    for ln in lines:
        rec = parse_simple_tsv_line(ln)
        if rec:
            records.append(rec)
    return records


# ----------------------------- PubMed enrichment (stub) -----------------------------

def query_pubmed_stub(query: str, max_results: int = 1) -> List[Dict[str, Any]]:
    # Self‑contained demo: return empty; replace with real E-utilities client if needed.
    return []


# ----------------------------- Tools and Agent -----------------------------

class Agent:
    def __init__(self, tools: Dict[str, Any]):
        self.tools = tools

    def invoke(self, query: str) -> str:
        # This is intentionally simple; in a real demo you'd parse the query and route to tools.
        if query.lower().startswith("search:"):
            q = query.split(":", 1)[1].strip()
            res = self.tools["search"](query_text=q)
            return json.dumps(res, indent=2)
        return "(Agent stub) Provide 'search: your query' to hit the search tool."


def build_note_registration_tool(lm: "LiteratureMemory", owner: str, namespace: str, summarizer: LLMSummarizer):
    def _register(abstract: str, mirna: str, target_gene: str, pmid: str, year: Optional[int] = None):
        summary = summarizer.summarize(abstract)
        note = LiteratureNote(
            content=f"miRNA: {mirna} | Target: {target_gene} | PMID: {pmid}\nSummary:\n{summary}",
            owner=owner,
            mirna=mirna,
            target_gene=target_gene,
            pmid=pmid,
            year=year,
            namespace=namespace,
        )
        lm.register(note)
        return {"status": "ok", "pmid": pmid}
    return _register


def build_note_search_tool(lm: "LiteratureMemory", owner: str, namespace: str):
    def _search(
        query_text: str,
        limit: int = 5,
        year_min: Optional[int] = None,
        mirna: Optional[str] = None,
        target_gene: Optional[str] = None,
    ):
        filters: Dict[str, Any] = {}
        if mirna:
            filters["mirna"] = mirna
        if target_gene:
            filters["target_gene"] = target_gene
        res = lm.search(query_text, owner, limit, namespace, year_min, filters)
        return {
            "count": len(res.notes),
            "hits": [
                {
                    "pmid": n.pmid,
                    "mirna": n.mirna,
                    "target_gene": n.target_gene,
                    "year": n.year,
                    "summary": n.content,
                }
                for n in res.notes
            ],
        }
    return _search


# ----------------------------- Demo runner -----------------------------

def run_demo(src: Optional[str], owner: str, limit: int, use_llm: bool) -> None:
    # Defaults: try common filenames plus attached sample
    candidate_paths = [
        src,
        os.path.join(os.getcwd(), "miRTarBase_microRNA_target_interaction_pubmed_abtract.txt"),
        os.path.join(os.getcwd(), "miRTarBase_microRNA_target_interaction_pubmed_abtract_sample.txt"),
        "/mnt/data/miRTarBase_microRNA_target_interaction_pubmed_abtract_sample.txt",
    ]
    src_path = next((p for p in candidate_paths if p and os.path.exists(p)), None)
    if not src_path:
        raise SystemExit("Could not find a source data file. Provide --src path.")

    print(f"[1] Parsing source file: {src_path}")
    records = parse_file(src_path)
    print(f"    Parsed {len(records)} raw (miRNA,target,PMID) records")

    print("[2] Initializing vector store and memory…")
    db = VectorStoreManager(api_key=os.getenv("OPENAI_API_KEY", ""))
    persist_path = os.path.join(os.getcwd(), "mirna_literature_notes.jsonl")
    lm = LiteratureMemory(db, "mirna_literature_notes", persist_path=persist_path)
    summarizer = LLMSummarizer(use_llm=use_llm)

    print("[3] Summarizing and registering notes…")
    registered = 0
    for rec in records[: max(0, limit) or len(records)]:
        abstract = rec.get("abstract", "")
        pmid = str(rec.get("pmid", ""))
        if not pmid:
            continue
        summary = summarizer.summarize(abstract)
        note = LiteratureNote(
            content=f"miRNA: {rec.get('mirna','?')} | Target: {rec.get('gene','?')} | PMID: {pmid}\nSummary:\n{summary}",
            owner=owner,
            mirna=str(rec.get("mirna", "")),
            target_gene=str(rec.get("gene", "")),
            pmid=pmid,
            year=rec.get("year"),
        )
        lm.register(note)
        registered += 1
    print(f"    ✓ Registered {registered} note(s). Index persisted to {persist_path}")

    print("[4] Example queries…")
    # Example 1: miR-21 → PTEN since 2015
    res1 = lm.search(
        query_text="oncogenic effects via PTEN suppression",
        owner=owner,
        limit=3,
        year_min=2015,
        filters={"mirna": "miR-21", "target_gene": "PTEN"},
    )
    print(f"    Q1 hits: {len(res1.notes)}")
    for n in res1.notes:
        print(f"      [PMID {n.pmid}] {n.mirna} → {n.target_gene} (year={n.year})")

    # Example 2: inflammation theme
    res2 = lm.search(
        query_text="inflammation inflammatory macrophage immune",
        owner=owner,
        limit=2,
    )
    print(f"    Q2 hits: {len(res2.notes)}")
    for n in res2.notes:
        print(f"      [PMID {n.pmid}] {n.mirna} → {n.target_gene}")

    # Tools + Agent demo
    register_tool = build_note_registration_tool(lm, owner=owner, namespace="mirna_lit", summarizer=summarizer)
    search_tool = build_note_search_tool(lm, owner=owner, namespace="mirna_lit")
    agent = Agent(tools={"register": register_tool, "search": search_tool})

    print("[5] Agent tool call: search for gastric cancer growth inhibition signals…")
    agent_out = agent.invoke("search: gastric cancer growth inhibition miRNA luciferase")
    print(agent_out)


# ----------------------------- CLI -----------------------------

def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Self‑contained agentic AI demo for miRNA literature notes")
    p.add_argument("--src", type=str, default="miRTarBase_microRNA_target_interaction_pubmed_abtract_SYNTHETIC.txt", help="miRTarBase_microRNA_target_interaction_pubmed_abtract_SYNTHETIC.txt")
    p.add_argument("--owner", type=str, default="researcher_001", help="Owner identifier")
    p.add_argument("--limit", type=int, default=200, help="Max records to index (0 = no limit)")
    p.add_argument("--use-llm", action="store_false", help="Use OpenAI LLM for summaries (requires OPENAI_API_KEY)")
    return p


if __name__ == "__main__":
    args = make_argparser().parse_args()
    run_demo(args.src, args.owner, args.limit, args.use_llm)
