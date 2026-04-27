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

ANSWER_SYNTHESIS_PROMPT = """Act as an EU legal expert specialising in GDPR, AI Act, Data Act, and Data Governance Act.

=== GROUND RULES ===

1. Ground every claim in the retrieved content below. Cite the specific article and regulation (e.g. "Article 32 GDPR"). Do not invent article numbers.
2. If the retrieved content is insufficient or off-topic, say so explicitly and point the user to the correct provisions.
3. When the question asks "how" or "what in practice", go beyond restating the law — explain what an organisation must concretely do, what documentation to maintain, and what technical or organisational measures to adopt.
4. Note cross-references to related articles or regulations when the retrieved content supports them.

=== RESPONSE FORMAT ===

**Legal Basis**: The applicable provision(s) and what they require.

**Practical Measures**: Specific, actionable steps — technical controls, process design, contractual clauses, access policies — grounded in the retrieved content.

**Documentation & Accountability**: Records, policies, or assessments the organisation should maintain.

**Related Obligations**: Connected provisions from the same or other regulations, if supported by the context.

=== RETRIEVED CONTENT ===

{context}

=== QUESTION ===

{question}
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
    EU case law     → depth 0: top sections (Judgment, Legal context, The dispute in the main proceedings,
                               Consideration of the questions referred, Costs, Signatures)
                      depth 1: named subsections of Legal context (European Union law, National law)
                               AND numbered sub-questions (The first question, The second question, The N-th question)
                      depth 2: numbered paragraphs (^\\d+\\s)
    Academic        → numeric dot notation (1 > 1.1 > 1.1.1)
    Technical doc   → Part > Chapter > Section > Subsection
- also use docling_level as a signal when it varies meaningfully across elements

CRITICAL for EU case law — these rules are MANDATORY if those headers appear in the sample:
  {{"pattern": "European Union law", "type": "prefix", "depth": 1}}
  {{"pattern": "National law", "type": "prefix", "depth": 1}}
  Reason: docling assigns them docling_level=1 (same as their parent "Legal context"),
  so without an explicit depth-1 rule they will be placed at depth 0 instead of nested inside Legal context.
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