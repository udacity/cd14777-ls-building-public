#!/usr/bin/env python3
"""
Demo: Playing with external APIs for biomedical research
Integrates PubMed, ClinicalTrials.gov, ClinVar, and OpenAI for evidence-based answers.
"""

import os
import textwrap
import json
from typing import List, Dict, Any, Optional
from urllib.request import urlopen, Request
from urllib.parse import urlencode, quote
import xml.etree.ElementTree as ET

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(".env")
except ImportError:
    pass  # dotenv optional

# Try to import ls_action_space, fall back to local stubs if unavailable
try:
    from ls_action_space.action_space import (
        query_pubmed,
        query_clinicaltrials,
        query_clinvar
    )
    USING_STUBS = False
except ImportError:
    USING_STUBS = True

    # Local stub implementations using stdlib only
    def query_pubmed(query: str, max_results: int = 10, include_mesh: bool = False,
                     include_citations: bool = False) -> List[Dict[str, Any]]:
        """Stub implementation of PubMed query using NCBI E-utilities"""
        try:
            # Step 1: Search for PMIDs
            search_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json"
            }
            search_url = f"{search_base}?{urlencode(search_params)}"
            with urlopen(search_url, timeout=10) as response:
                search_data = json.loads(response.read().decode())

            pmids = search_data.get("esearchresult", {}).get("idlist", [])
            if not pmids:
                return []

            # Step 2: Fetch article details
            fetch_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml"
            }
            fetch_url = f"{fetch_base}?{urlencode(fetch_params)}"
            with urlopen(fetch_url, timeout=15) as response:
                xml_data = response.read().decode()

            # Parse XML
            root = ET.fromstring(xml_data)
            articles = []

            for article_elem in root.findall(".//PubmedArticle"):
                pmid_elem = article_elem.find(".//PMID")
                pmid = pmid_elem.text if pmid_elem is not None else "Unknown"

                title_elem = article_elem.find(".//ArticleTitle")
                title = title_elem.text if title_elem is not None else "No title"

                abstract_texts = article_elem.findall(".//AbstractText")
                abstract = " ".join([at.text for at in abstract_texts if at.text]) if abstract_texts else ""

                journal_elem = article_elem.find(".//Journal/Title")
                journal = journal_elem.text if journal_elem is not None else "Unknown"

                year_elem = article_elem.find(".//PubDate/Year")
                year = int(year_elem.text) if year_elem is not None and year_elem.text else None

                # Authors
                author_elems = article_elem.findall(".//Author")
                authors = []
                for auth in author_elems:
                    lastname = auth.find("LastName")
                    forename = auth.find("ForeName")
                    if lastname is not None:
                        name = lastname.text or ""
                        if forename is not None and forename.text:
                            name = f"{forename.text} {name}"
                        authors.append(name)

                # DOI
                doi_elem = article_elem.find(".//ArticleId[@IdType='doi']")
                doi = doi_elem.text if doi_elem is not None else None

                # PMCID
                pmc_elem = article_elem.find(".//ArticleId[@IdType='pmc']")
                pmcid = pmc_elem.text if pmc_elem is not None else None

                # MeSH terms
                mesh_terms = []
                if include_mesh:
                    mesh_elems = article_elem.findall(".//MeshHeading/DescriptorName")
                    mesh_terms = [m.text for m in mesh_elems if m.text]

                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "journal": journal,
                    "year": year,
                    "authors": authors,
                    "doi": doi,
                    "pmcid": pmcid,
                    "mesh_terms": mesh_terms
                })

            return articles
        except Exception as e:
            print(f"Warning: PubMed query failed: {e}")
            return []

    def query_clinicaltrials(expr: str, max_results: int = 25,
                            fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Stub implementation of ClinicalTrials.gov query"""
        try:
            # Using ClinicalTrials.gov API v2
            base_url = "https://clinicaltrials.gov/api/v2/studies"
            params = {
                "query.term": expr,
                "pageSize": min(max_results, 1000),
                "format": "json"
            }

            url = f"{base_url}?{urlencode(params)}"
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())

            studies = []
            for study in data.get("studies", []):
                protocol = study.get("protocolSection", {})
                id_module = protocol.get("identificationModule", {})
                status_module = protocol.get("statusModule", {})
                design_module = protocol.get("designModule", {})
                arms_module = protocol.get("armsInterventionsModule", {})
                outcomes_module = protocol.get("outcomesModule", {})
                eligibility_module = protocol.get("eligibilityModule", {})
                contacts_module = protocol.get("contactsLocationsModule", {})
                sponsor_module = protocol.get("sponsorCollaboratorsModule", {})

                interventions = arms_module.get("interventions", [])
                intervention_names = [i.get("name", "") for i in interventions]

                primary_outcomes = outcomes_module.get("primaryOutcomes", [])
                primary_outcome = primary_outcomes[0].get("measure", "") if primary_outcomes else ""

                locations = contacts_module.get("locations", [])
                countries = list(set([loc.get("country", "") for loc in locations if loc.get("country")]))

                studies.append({
                    "NCTId": id_module.get("nctId", ""),
                    "BriefTitle": id_module.get("briefTitle", ""),
                    "OverallStatus": status_module.get("overallStatus", ""),
                    "Phase": design_module.get("phases", ["N/A"])[0] if design_module.get("phases") else "N/A",
                    "InterventionName": ", ".join(intervention_names),
                    "PrimaryOutcomeMeasure": primary_outcome,
                    "EnrollmentCount": status_module.get("enrollmentInfo", {}).get("count", 0),
                    "StartDate": status_module.get("startDateStruct", {}).get("date", ""),
                    "CompletionDate": status_module.get("completionDateStruct", {}).get("date", ""),
                    "LocationCountry": ", ".join(countries),
                    "LeadSponsorName": sponsor_module.get("leadSponsor", {}).get("name", "")
                })

            return {
                "count": data.get("totalCount", 0),
                "studies": studies
            }
        except Exception as e:
            print(f"Warning: ClinicalTrials query failed: {e}")
            return {"count": 0, "studies": []}

    def query_clinvar(variant_id: str) -> Dict[str, Any]:
        """Stub implementation of ClinVar query"""
        try:
            # Using NCBI E-utilities for ClinVar
            search_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            search_params = {
                "db": "clinvar",
                "term": variant_id,
                "retmode": "json"
            }
            search_url = f"{search_base}?{urlencode(search_params)}"
            with urlopen(search_url, timeout=10) as response:
                search_data = json.loads(response.read().decode())

            ids = search_data.get("esearchresult", {}).get("idlist", [])
            if not ids:
                return {"error": "Variant not found"}

            # For simplicity, return basic info
            return {
                "query": variant_id,
                "title": f"ClinVar entry for {variant_id}",
                "gene": "Unknown (stub)",
                "clinical_significance": "See ClinVar for details",
                "review_status": "Not available in stub",
                "conditions": ["Condition data not available in stub"],
                "accessions": {"VCV": [], "RCV": []},
                "hgvs": [],
                "pubmed_pmids": [],
                "allele_frequencies": "Not available in stub"
            }
        except Exception as e:
            print(f"Warning: ClinVar query failed: {e}")
            return {"error": str(e)}


# Initialize OpenAI client
try:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")

    if api_key:
        client = OpenAI(api_key=api_key, base_url=base_url)
        OPENAI_AVAILABLE = True
    else:
        OPENAI_AVAILABLE = False
        print("⚠️  OPENAI_API_KEY not set. Skipping LLM summarization.")
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI library not installed. Skipping LLM summarization.")


def pack_citations(arts: List[Dict], limit: int = 10) -> str:
    """Format articles for LLM context"""
    lines = []
    for a in arts[:limit]:
        title = a["title"]
        year = a.get("year")
        pmid = a["pmid"]
        journal = a.get("journal")
        doi = a.get("doi")
        abstract = (a.get("abstract") or "").strip().replace("\n", " ")
        abstract_short = abstract[:1200] + ("..." if len(abstract) > 1200 else "")
        lines.append(textwrap.dedent(f"""
        - PMID {pmid} ({year}, {journal}): {title}
          DOI: {doi if doi else "N/A"}
          Abstract: {abstract_short}
        """).strip())
    return "\n".join(lines)


def main():
    print("=" * 80)
    print("Demo: Playing with External APIs - Biomedical Research Assistant")
    print("=" * 80)
    print()

    if USING_STUBS:
        print("ℹ️  Using built-in API stubs (ls_action_space not found)")

    print("⚠️  Educational use only. Do not treat any output as medical advice.")
    print()

    # Step 1: Query PubMed
    QUESTION = "Do GLP-1 receptor agonists improve NASH histology in adults?"
    PUBMED_QUERY = '("GLP-1 receptor agonist" OR semaglutide OR liraglutide) AND (NASH OR "nonalcoholic steatohepatitis") AND randomized'

    print(f"Question: {QUESTION}")
    print(f"PubMed Query: {PUBMED_QUERY}")
    print()
    print("Retrieving PubMed articles...")

    articles = query_pubmed(
        PUBMED_QUERY,
        max_results=15,
        include_mesh=True,
        include_citations=False
    )

    print(f"✓ Retrieved {len(articles)} PubMed records")

    if articles:
        example = articles[0]
        print("\nExample article:")
        for k in ("pmid", "title", "journal", "year", "doi"):
            print(f"  {k}: {example.get(k)}")
    print()

    # Step 2: Query ClinicalTrials.gov
    print("Retrieving ClinicalTrials.gov data...")
    trials = query_clinicaltrials(
        expr="(NASH OR nonalcoholic steatohepatitis) AND (semaglutide OR liraglutide)",
        max_results=25,
        fields=None
    )

    print(f"✓ Matching trials (approx): {trials.get('count')}")
    if trials.get("studies"):
        print(f"  First trial: {trials['studies'][0].get('BriefTitle', '—')}")
    print()

    # Step 3: Query ClinVar (variant example)
    print("Retrieving ClinVar data for rs429358 (APOE ε4)...")
    variant_info = query_clinvar("rs429358")

    if "error" not in variant_info:
        print(f"✓ Variant: {variant_info.get('gene')} - {variant_info.get('clinical_significance')}")
        pmids = variant_info.get('pubmed_pmids', [])
        if pmids:
            print(f"  Linked PMIDs: {pmids[:5]}")
    else:
        print(f"  Note: {variant_info['error']}")
    print()

    # Step 4: Generate LLM summary
    if OPENAI_AVAILABLE and articles:
        print("Generating evidence-based summary with gpt-4o-mini...")
        print("-" * 80)

        evidence_block = pack_citations(articles, limit=10)

        prompt = f"""
You are a biomedical research assistant. Answer the user's question with a concise,
evidence-based summary. Cite studies inline as [PMID:12345678]. Highlight populations,
interventions, comparators, outcomes, and major limitations. If evidence is mixed,
note the balance. Then list a short "Key Takeaways" section.

Question:
{QUESTION}

Evidence:
{evidence_block}
"""

        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )

            summary = completion.choices[0].message.content
            print(summary)
            print("-" * 80)
            print()
        except Exception as e:
            print(f"⚠️  LLM summarization failed: {e}")
            print()
    elif not OPENAI_AVAILABLE:
        print("Skipped LLM summarization (OpenAI not configured)")
        print()

    # Step 5: Variant-centric summary example
    if OPENAI_AVAILABLE and "error" not in variant_info:
        print("Generating variant interpretation summary...")
        print("-" * 80)

        clinvar_block = textwrap.dedent(f"""
        Variant query: {variant_info['query']}
        Title: {variant_info.get('title')}
        Gene: {variant_info.get('gene')}
        Clinical significance: {variant_info.get('clinical_significance')}
        Review status: {variant_info.get('review_status')}
        Conditions: {', '.join(variant_info.get('conditions', []))}
        """).strip()

        variant_prompt = f"""
Summarize the clinical interpretation of this variant and reference key literature [PMID:...].
Note conflicting assertions and the review status. Keep to ~150 words.

{clinvar_block}
"""

        try:
            completion3 = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": variant_prompt}],
                temperature=0.2,
            )
            print(completion3.choices[0].message.content)
            print("-" * 80)
            print()
        except Exception as e:
            print(f"⚠️  Variant summarization failed: {e}")
            print()

    print("=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
