QUERY_CLASSIFICATION_PROMPT = """You are an expert in EU digital legislation. Classify the user query along three axes to guide retrieval.

=== AXES ===

1. intent:
   - DEFINITIONAL: asks what a provision says, what a term means, what the rules are. Answerable from articles/recitals alone.
   - INTERPRETIVE: asks how a provision has been applied, interpreted by courts, or how it should be construed in a borderline case. Requires CJEU case law.

2. acts: a list of available acts will be provided, if you're not sure of what act the query is about, return an empty list.
If the question include the act EXPLICITLY include it.

=== AVAILABLE ACTS ===
{acts}

=== FEW-SHOT EXAMPLES ===

Query: "What entities fall under the personal scope of Chapter II?"
{{"intent": "DEFINITIONAL", "acts": ["32022R0868"]}}

Query: "What does communication 'in a clear and comprehensible manner' entail?"
{{"intent": "DEFINITIONAL", "acts": ["32023R2854"]}}

=== QUERY ===
{query}
"""

TOPIC_SELECTION_PROMPT = """Act as an expert analyst in EU legal
documents (GDPR, AI Act, Data Act, Data Governance Act), specialising 
in topic classification and legal concept mapping.

=== TASK ===
Given a user query, select the most relevant topics from the 
provided list to guide retrieval of relevant legal paragraphs.

=== SELECTION RULES ===
1. Only select topics whose legal scope directly overlaps with the 
   query — adjacent or loosely related topics should be excluded.
2. Select between 1 and 7 topics depending on query complexity. 
   Narrow queries warrant fewer topics; broad or multi-concept 
   queries warrant more.
3. Topics must appear exactly as listed. Do not invent, merge, 
   or rephrase topic names.
4. If no topics are sufficiently relevant, return an empty list 
   and provide a brief explanation.

=== OUTPUT FORMAT ===
<topic>Topic1, Topic2, TopicN..</topic>

=== AVAILABLE TOPICS ===
{topics}

=== USER QUERY ===
{query}
"""

ANSWER_SYNTHESIS_PROMPT = """You are an EU data law expert specialising in the GDPR, 
AI Act, Data Act, and Data Governance Act.

=== RETRIEVED CONTENT ===

{context}

=== QUESTION ===

{question}

=== INSTRUCTIONS ===

Answer strictly from the retrieved content above. Follow these rules:

1. PRIORITISE THE MOST SPECIFIC SOURCE. If a recital directly answers the question, 
   lead with it — recitals are often more actionable than articles for "how" and 
   "what in practice" questions. Do not default to article-level citations when a 
   recital provides the concrete answer.

2. CITE PRECISELY. Reference the exact provision (e.g. "Recital 24 Data Act", 
   "Article 25(2) DGA"). Never invent or infer article numbers not present in the 
   retrieved content.

3. IF THE RETRIEVED CONTENT IS INSUFFICIENT, say so explicitly. Identify which 
   regulation and provision would answer the question and recommend the user refine 
   the search toward those sources. Do not fill gaps with general legal reasoning 
   or principles from other regulations.

4. DO NOT IMPORT OBLIGATIONS FROM OTHER REGULATIONS unless the retrieved content 
   explicitly cross-references them. A question about the Data Act must not default 
   to GDPR principles unless the retrieved chunks support that connection.

5. FOR "HOW" AND "WHAT IN PRACTICE" QUESTIONS, extract the concrete mechanism 
   described in the source — a specific tool, format, process, or technical 
   arrangement — rather than paraphrasing it into generic compliance advice.

=== RESPONSE FORMAT ===

**Legal basis**: The specific provision(s) from the retrieved content that govern 
this question, with regulation name and article or recital number.

**Answer**: What the law requires or permits, grounded directly in the retrieved 
content. For practical questions, describe the concrete mechanism as specified 
in the source — not a generalisation of it.

**Related obligations**: Only if the retrieved content explicitly supports a 
cross-reference to another provision or regulation. Omit this section entirely 
if no such link appears in the retrieved content.
"""

# Prompt to extract case law document hierarchy rules from structural elements

CASE_LAW_DOCUMENT_PARSING_SYSTEM_PROMPT = """You are an expert document analyst.
Given structural elements extracted from a PDF, infer the document's hierarchical rules.
Consider the domain (legal, academic, technical, corporate, etc.) and its conventions
when assigning depth levels.
"""

CASE_LAW_DOCUMENT_PARSING_USER_PROMPT = """Analyze these structural elements from a PDF document:

{sample}

Infer the hierarchy rules. Output ONLY a valid JSON object with this structure:
{{
  "domain": "brief description of domain and document type",
  "rules": [
    {{"pattern": "...", "type": "prefix|regex", "depth": 0}},
    ...
  ],
  "notes": "any important observations"
}}

Rules for the "type" field — use ONLY these two values, nothing else:
- "prefix"  → case-insensitive prefix match (e.g. pattern "chapter" matches "Chapter 3")
- "regex"   → full Python regex, re.IGNORECASE applied (e.g. "^\\d+(\\d+)*\\s")

Rules for the "depth" field:
- depth 0 = top-level heading (e.g. part, chapter, top-level section)
- depth increases by 1 for each nesting level
- for numeric dot notation derive depth from the number of dots: "1" → 0, "1.1" → 1, "1.1.1" → 2
- every rule must have the correct depth; a subsection must never have the same depth as its parent

Rules for coverage and noise:
- include the document title (if present) as depth 0
- include every structural heading level observed in the sample
- after writing your rules, mentally check: does every element with label=section_header
  in the sample match at least one rule? if not, add the missing rules
- EXCLUDE elements that are not content headings: page headers/footers, author lines,
  conference/journal metadata, citation reference headers, copyright notices, boilerplate

Ordering:
- order rules from most specific to least specific (exact patterns before general regex)

Domain conventions to follow when applicable:
    EU legislation  → CHAPTER > Section > Article > paragraph
    EU case law (Judgment) → depth 0: top sections (Judgment, Legal context, The dispute in the main proceedings,
                               Consideration of the questions referred, Costs, Signatures)
                      depth 1: subsections of Legal context — any "[Country/adjective] law" heading
                               (e.g. "European Union law", "Spanish law", "French law", "National law")
                               AND question sub-sections (match with regex "^(Question|The .+ [Qq]uestion)")
                      depth 2: numbered paragraphs (^\\d+\\.?\\s)
    EU case law (Advocate General Opinion, CELEX type CC) → depth 0: Roman-numeral sections
                               (e.g. "I. The facts…", "II. My assessment", "III. Conclusion")
                      depth 1: lettered sub-sections nested under their Roman-numeral parent
                               (e.g. "A. The first and second questions", "B. …", "C. …")
                      depth 2: numbered paragraphs (^\\d+\\.?\\s)
                      Rules to generate for this document type:
                        {{"pattern": "^[IVX]+\\.\\s", "type": "regex", "depth": 0}}
                        {{"pattern": "^[A-Z]\\.\\s", "type": "regex", "depth": 1}}
                        {{"pattern": "^\\d+\\.?\\s", "type": "regex", "depth": 2}}
    Academic        → numeric dot notation (1 > 1.1 > 1.1.1)
    Technical doc   → Part > Chapter > Section > Subsection
- also use docling_level as a signal when it varies meaningfully across elements

CRITICAL for EU case law — these rules are MANDATORY and must always appear in your output:
  {{"pattern": "European Union law", "type": "prefix", "depth": 1}}
  {{"pattern": "^[A-Z][a-z]+(\\s+[A-Za-z]+)*\\s+law$", "type": "regex", "depth": 1}}
  {{"pattern": "^(Question|The.+[Qq]uestions?)$", "type": "regex", "depth": 1}}
  {{"pattern": "^\\d+\\.?\\s", "type": "regex", "depth": 2}}
  Reason: docling assigns legal-context subsections docling_level=1 (same as their parent),
  so without explicit depth-1 rules they end up at depth 0 instead of nested inside Legal context.
  The "[Country] law" regex covers all national-law variants: "Spanish law", "French law", etc.
  The question regex uses $ to avoid matching headings like "The dispute in the main proceedings
  and the question referred for a preliminary ruling" which contain "question" mid-string.

NEVER include EU case law running page headers as rules. These are repeated headers that appear on
every page in the format "JUDGMENT OF [date] — CASE [identifier] [party names]".
They are NOT content sections — exclude them completely even if they appear as section_header in the sample.
Example of what to NEVER add as a rule: "JUDGMENT OF 13. 5. 2014 — CASE C-131/12 GOOGLE SPAIN AND GOOGLE"
"""

CASE_LAW_ENTITY_SUMMARY_SYSTEM_PROMPT = """
You are an AI assistant that helps a human analyst to perform general information discovery. Information
discovery is the process of identifying and assessing relevant information associated with certain
entities (e.g., organizations and individuals) within a network.
"""

CASE_LAW_ENTITY_SUMMARY_USER_PROMPT = """
You are analyzing a single section of a structured EU legal document.

Section heading: {heading}
Section depth: {depth}
Section body:
{body}

Produce a concise summary (3–6 sentences) of this section capturing:
1. The main legal question or topic addressed
2. The key parties, authorities, or legal instruments mentioned
3. The core finding, ruling, or argument made
4. Any notable references to other sections or legal precedents (if present)

Return ONLY a JSON object with the following fields:
- "heading": the section heading (copied exactly as given)
- "summary": your concise summary
"""

CASE_LAW_ENTIRE_DOC_SUMMARY_SYSTEM_PROMPT = """You are an expert legal document summarizer."""

CASE_LAW_ENTIRE_DOC_SUMMARY_USER_PROMPT = """
Summarize the following CJEU judgment. Focus on:
- The case number (e.g., C-XXX/YY)
- The parties
- The specific articles or regulations interpreted
- The core legal question

The summary must be concise, maximum {char_length} 
characters long, and optimized for providing context 
to smaller text chunks. Output only the summary text.

Document: {document_content}
"""