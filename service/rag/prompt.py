QUERY_CLASSIFICATION_PROMPT = """You are an expert in EU digital legislation. Classify the user query along four axes to guide retrieval.

=== AXES ===

1. intent:
   - DEFINITIONAL: asks what a provision says, what a term means, what the rules are. Answerable from articles/recitals alone.
   - INTERPRETIVE: asks how a provision has been applied, interpreted by courts, or how it should be construed in a borderline case. Requires CJEU case law.

2. query_type: pick the retrieval strategy that best fits the structural shape of the question.
   - SCOPE_OF_ACT: asks about the overall subject matter, purpose, or scope of applicability of an entire act.
   - SCOPE_OF_CHAPTER: asks which entities, services, or data fall under (or are excluded from) a specific Chapter.
   - DEFINITION_LOOKUP: asks for the meaning of a specific defined term, body, or instrument.
   - ENUMERATION: asks for a list of conditions, requirements, obligations, rights, or procedures.
   - SPECIFIC_QUESTION: yes/no or single specific rule — asks whether something is required/permitted, or what a single provision says.

3. acts: identify which act(s) the query is about using the topic descriptions below to disambiguate.
   - If the act is mentioned explicitly, always include it.
   - If uncertain, pick the most likely act based on the topic match — do not return an empty list unless the query is completely unrelated to any available act.

4. chapter_number: Arabic integer when query_type = SCOPE_OF_CHAPTER, null otherwise.
   DISAMBIGUATION: when the query mentions "Chapter II", "Chapter III", or "Chapter IV" without naming an act,
   default to the Data Governance Act (32022R0868) — its chapters are the most frequently queried by number.

=== AVAILABLE ACTS ===
{acts}

=== KEY TOPICS PER ACT (use for disambiguation) ===
- Data Governance Act (32022R0868): public-sector data re-use and the conditions/fees/exclusions governing it (Chapter II covers public sector bodies, fees, protected data, re-use conditions); data intermediation services, notification procedure, one-stop-shop (Chapter III); data altruism organisations and their registration (Chapter IV); European Data Innovation Board (Chapter V)
- Data Act (32023R2854): connected products; user rights to access product and service data (Art. 3–4); third-party data sharing by the data holder (Art. 5); processing obligations of the third party (Art. 6); B2B data sharing and fair/unfair contract terms (Art. 8–13, including Art. 8(1) on data holder obligations, Art. 8(4)–8(5) on unfair terms, Art. 13(3) on the definition of unfairness); public sector exceptional need (Art. 14–22); switching between data processing services (Art. 23–31); smart contracts (Art. 36)
- AI Act (32024R1689): classification of AI systems by risk level; requirements for high-risk AI systems; general-purpose AI (GPAI) models; conformity assessment; market surveillance; prohibited AI practices
- GDPR (32016R0679): lawful basis for personal data processing; data subject rights (access, erasure, portability, rectification); controller and processor obligations; DPIAs (Art. 35–36); personal data breaches (Art. 33–34); international transfers (Art. 45–49)

=== FEW-SHOT EXAMPLES ===

Query: "What entities fall under the personal scope of Chapter II?"
{{"intent": "DEFINITIONAL", "query_type": "SCOPE_OF_CHAPTER", "acts": ["32022R0868"], "chapter_number": 2}}

Query: "Does Chapter II create an obligation to allow the re-use of data?"
{{"intent": "DEFINITIONAL", "query_type": "SCOPE_OF_CHAPTER", "acts": ["32022R0868"], "chapter_number": 2}}

Query: "Does Chapter II impose specific requirements regarding the nature of the data they are making available for re-use?"
{{"intent": "DEFINITIONAL", "query_type": "SCOPE_OF_CHAPTER", "acts": ["32022R0868"], "chapter_number": 2}}

Query: "What services fall under the material scope of Chapter IV of the Data Governance Act?"
{{"intent": "DEFINITIONAL", "query_type": "SCOPE_OF_CHAPTER", "acts": ["32022R0868"], "chapter_number": 4}}

Query: "What is the subject matter and objectives of the Data Governance Act?"
{{"intent": "DEFINITIONAL", "query_type": "SCOPE_OF_ACT", "acts": ["32022R0868"], "chapter_number": null}}

Query: "What is the scope of the GDPR — which processing activities does it cover?"
{{"intent": "DEFINITIONAL", "query_type": "SCOPE_OF_ACT", "acts": ["32016R0679"], "chapter_number": null}}

Query: "What does 'data intermediation service' mean under the Data Governance Act?"
{{"intent": "DEFINITIONAL", "query_type": "DEFINITION_LOOKUP", "acts": ["32022R0868"], "chapter_number": null}}

Query: "What does communication 'in a clear and comprehensible manner' entail?"
{{"intent": "DEFINITIONAL", "query_type": "DEFINITION_LOOKUP", "acts": ["32023R2854"], "chapter_number": null}}

Query: "What is a 'high-risk AI system' under the AI Act?"
{{"intent": "DEFINITIONAL", "query_type": "DEFINITION_LOOKUP", "acts": ["32024R1689"], "chapter_number": null}}

Query: "What is the definition of 'personal data' under the GDPR?"
{{"intent": "DEFINITIONAL", "query_type": "DEFINITION_LOOKUP", "acts": ["32016R0679"], "chapter_number": null}}

Query: "How should connected products and related services be designed and manufactured/provided?"
{{"intent": "DEFINITIONAL", "query_type": "SPECIFIC_QUESTION", "acts": ["32023R2854"], "chapter_number": null}}

Query: "Can a public sector body charge fees for allowing re-use of its data?"
{{"intent": "DEFINITIONAL", "query_type": "SPECIFIC_QUESTION", "acts": ["32022R0868"], "chapter_number": null}}

Query: "Which terms are considered and which are presumed to be unfair for the purposes of Article 8(1)?"
{{"intent": "DEFINITIONAL", "query_type": "ENUMERATION", "acts": ["32023R2854"], "chapter_number": null}}

Query: "In which cases is a data holder obliged to make data available to a public sector body, the Commission, the European Central Bank or a Union body?"
{{"intent": "DEFINITIONAL", "query_type": "ENUMERATION", "acts": ["32023R2854"], "chapter_number": null}}

Query: "What obligations does the data holder have towards the data recipient?"
{{"intent": "DEFINITIONAL", "query_type": "ENUMERATION", "acts": ["32023R2854"], "chapter_number": null}}

Query: "What are the conditions for lawful processing of personal data under the GDPR?"
{{"intent": "DEFINITIONAL", "query_type": "ENUMERATION", "acts": ["32016R0679"], "chapter_number": null}}

Query: "What requirements must a high-risk AI system meet before being placed on the market?"
{{"intent": "DEFINITIONAL", "query_type": "ENUMERATION", "acts": ["32024R1689"], "chapter_number": null}}

Query: "How has the CJEU interpreted the right to erasure in the context of search engines?"
{{"intent": "INTERPRETIVE", "query_type": "SPECIFIC_QUESTION", "acts": ["32016R0679"], "chapter_number": null}}

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

Each retrieved passage is prefixed with its source in the format:
[Regulation, Article N — Title]

=== ANSWER GUIDANCE ===
{guidance}

=== RETRIEVED CONTENT ===

{context}

=== QUESTION ===

{question}

=== STRICT RULES ===

1. Base your entire answer exclusively on the retrieved content above.
3. When citing a legal basis, copy the article or recital reference EXACTLY from the source prefix of the passage that supports the claim. Never infer, guess, or recall an article number from memory.
4. If the retrieved content does not contain enough information to answer the question fully, state this explicitly rather than filling gaps from memory.
5. Do not repeat or paraphrase any article reference that appears in the user question anywhere in your answer — not in the Legal basis, not in the Answer body, not anywhere. Cite only from the exact [Regulation, Article N — Title] source prefix headers of the retrieved passages.
6. If the primary question has a yes/no answer, state it first using the passage that most directly answers it. Add qualifications and exceptions only after the direct answer, and only if the retrieved content explicitly supports them as conditions on the primary rule — do not elevate a narrow exception into the main answer.
7. If the question asks about two distinct categories (e.g., "which are considered X" and "which are presumed Y"), address each category separately and explicitly in your answer, drawing from all relevant retrieved passages — do not merge them into a single list.
8. Address ONLY the specific question asked. Do not enumerate adjacent provisions, related obligations, procedural details, downstream consequences, or definitions that the question does not require. If the retrieved content covers multiple paragraphs of the same article but only some are responsive to the question, cite only the responsive ones.
9. Be exhaustive within scope: cover ALL rules in the retrieved content that bear on the question, but only those. Do not cap the number of claims, but do not invent or pad.
10. Do not pad the answer with definitions, scope clauses, or background already implicit in the question.

=== RESPONSE FORMAT ===

**Legal basis**: The article or recital reference(s) copied from the source prefix(es) of the passage(s) that directly support this answer. Cite only what is present in the retrieved content.

**Answer**: What the law requires or permits, described in the concrete terms used in the retrieved content — not a generalisation.

**Related obligations**: Only if the retrieved content explicitly states a cross-reference to another provision. Omit "Related obligations" if no such link appears.
"""

SYNTHESIS_GUIDANCE_BY_TYPE = {
    "SCOPE_OF_ACT": (
        "DISAMBIGUATE first:\n"
        "- 'subject matter': enumerate ONLY Art. 1(1) items (a),(b),(c),(d)\n"
        "- 'scope of applicability': enumerate ONLY personal-scope article items\n"
        "**FORBIDDEN**: do NOT mention Chapter II-VII when question is subject matter/applicability\n"
        "Cite exact article/paragraph. Aim 5-10 claims."
    ),
    "SCOPE_OF_CHAPTER": (
        "Exhaustive enumeration of ALL INCLUDED and EXCLUDED categories. "
        "Cite exact article/paragraph. Aim 6-10 items. "
        "IMPORTANT: cite Art.2 definitions ONLY when they define terms used by queried Chapter's "
        "substantive articles. Do NOT pull in definitions from OTHER chapters."
    ),
    "DEFINITION_LOOKUP": (
        "Definition first, then enumerate ALL key attributes/components. "
        "Cite defining article. Aim 4-8 attribute claims."
    ),
    "ENUMERATION": (
        "Enumerate ALL items — do NOT collapse. Aim 6-12 items. "
        "Cite specific article/paragraph for each. "
        "If governed by single article, state total count and enumerate each."
    ),
    "SPECIFIC_QUESTION": (
        "Answer directly first, then ground in cited provision.\n"
        "- Short self-contained provision: 2-4 claims.\n"
        "- Provision lists multiple conditions/modalities: enumerate ALL, aim 5-10.\n"
        "- Relevant recital present: extract practical details as additional claims.\n"
        "Cite exact governing provision for each claim."
    ),
}

DEFAULT_SYNTHESIS_GUIDANCE = SYNTHESIS_GUIDANCE_BY_TYPE["ENUMERATION"]

ANSWER_FILTER_PROMPT = """You are filtering a draft legal answer to keep only the sentences that directly answer the user's question.

=== QUESTION ===
{question}

=== DRAFT ANSWER ===
{draft_answer}

=== FILTER RULES ===

1. Keep ONLY sentences that directly answer the question. A sentence directly answers the question if removing it would leave the question unanswered or incomplete.
2. Remove sentences about:
   - Adjacent provisions, related obligations, or downstream consequences not asked for
   - Procedural details, exceptions, or timelines the question does not explicitly ask about
   - Background, definitions, or scope clauses already implicit in the question
   - Article references that merely list provisions without stating what they require
3. Apply this test to every sentence: "would this sentence still be true if the question were slightly different?" If yes, the sentence is contextual filler — remove it.
4. Preserve the original sentence wording — do not rephrase, paraphrase, or add new content.
5. Preserve the **Legal basis** / **Answer** / **Related obligations** structure, but:
   - If **Related obligations** content does not directly answer the question, omit that section entirely.
   - If **Legal basis** lists more provisions than needed to answer, keep only the ones that ground claims kept in **Answer**.
6. If a sentence partially answers the question, keep only the responsive clause and drop the rest.

=== FILTERED ANSWER ===
"""

ARTICLE_SUMMARY_SYSTEM_PROMPT = """You are an expert in EU digital legislation (GDPR, AI Act, Data Act, Data Governance Act).
Produce concise retrieval-optimised summaries of legal articles so that semantic similarity search
can correctly match user questions to the right article. Use plain, query-friendly language.
Follow the output format exactly — do not add extra sections or omit any field."""

ARTICLE_SUMMARY_USER_PROMPT = """Summarise the following legal article using the template below.

=== OUTPUT TEMPLATE ===
Act: {{regulation_name}} ({{celex}})
Article — {{title}}
Applies to: {{entities subject to this article}}
Core rule: {{main obligation / right / prohibition in plain language}}
Key conditions or exceptions: {{when it applies / when it does not}}
Procedure: {{numbered steps if the article is procedural, otherwise "N/A"}}
Key terms: {{defined or heavily-used legal terms, comma-separated}}
Cross-references: {{articles explicitly mentioned in the text, or "none"}}

=== FEW-SHOT EXAMPLES ===

--- Example 1 ---
INPUT
Act: General Data Protection Regulation (32016R0679) | Article — Right to erasure ('right to be forgotten')
Body:
1. The data subject shall have the right to obtain from the controller the erasure of personal data
concerning him or her without undue delay [...] where one of the following grounds applies:
(a) the personal data are no longer necessary in relation to the purposes for which they were collected;
(b) the data subject withdraws consent [...]; (c) the data subject objects [...]; (d) the personal data
have been unlawfully processed; (e) erasure required by Union or Member State law; (f) data collected
in relation to the offer of information society services referred to in Article 8(1).
2. Where the controller has made the personal data public [...] shall take reasonable steps to inform
controllers which are processing the personal data that the data subject has requested the erasure.
3. Paragraphs 1 and 2 shall not apply to the extent that processing is necessary: (a) for exercising
the right of freedom of expression and information; (b) for compliance with a legal obligation [...]

OUTPUT
Act: General Data Protection Regulation (32016R0679)
Article — Right to erasure ('right to be forgotten')
Applies to: Data subjects (natural persons); controllers (entities determining purpose and means of processing)
Core rule: Data subjects have the right to obtain erasure of their personal data without undue delay; controllers must comply when any of six grounds applies.
Key conditions or exceptions: Grounds: data no longer necessary for original purpose; consent withdrawn and no other legal ground; data subject objects with no overriding legitimate grounds; data unlawfully processed; erasure required by law; data collected from a child online. Exceptions: freedom of expression, legal obligation, public interest in health, archiving/research/statistics, establishment or defence of legal claims.
Procedure: Data subject submits erasure request → controller erases without undue delay → if data was made public, controller must notify other controllers processing it.
Key terms: erasure, right to be forgotten, consent, controller, lawful processing, public interest
Cross-references: Article 6(1)(a), Article 8(1), Article 9(2)(a), Article 21

--- Example 2 ---
INPUT
Act: Data Governance Act (32022R0868) | Article — Notification by data intermediation services providers
Body:
1. Any data intermediation services provider who intends to provide the data intermediation services
referred to in Article 10 shall submit a notification to the competent authority for data intermediation
services. 4. After having submitted a notification [...] the data intermediation services provider may
start the activity [...]. 5. The notification [...] shall entitle the data intermediation services provider
to provide data intermediation services in all Member States. 6. The notification [...] shall include:
(a) the name of the data intermediation services provider; (b) legal status, form, ownership structure;
(c) address of main establishment; (d) a public website [...]; (e) contact details; (f) a description of
the data intermediation service [...]; (g) the estimated date for starting the activity.
8. [...] the competent authority [...] shall, within one week [...], issue a standardised declaration.
11. The competent authority [...] may charge fees [...] proportionate [...]; in the case of SMEs and
start-ups, the competent authority may charge a discounted fee or waive the fee.
12. Data intermediation services providers shall notify [...] any changes [...] within 14 days.

OUTPUT
Act: Data Governance Act (32022R0868)
Article — Notification by data intermediation services providers
Applies to: Data intermediation services providers intending to offer services under Article 10; competent national authorities; the Commission
Core rule: Any provider intending to offer data intermediation services must notify the competent national authority before starting. A single notification grants the right to operate across all EU Member States.
Key conditions or exceptions: Providers not established in the EU must designate a legal representative. Fees must be proportionate and may be reduced or waived for SMEs and start-ups. Changes to notified information must be reported within 14 days.
Procedure: Provider submits notification → may start activity immediately after submission → competent authority issues standardised declaration within 1 week on request → authority notifies Commission → Commission updates public register.
Key terms: notification, data intermediation service, competent authority, standardised declaration, public register, legal representative, SME
Cross-references: Article 10 (types of intermediation services), Article 12 (conditions for provision), Article 13 (competent authorities), Article 33(2)

--- Example 3 ---
INPUT
Act: Data Act (32023R2854) | Article — Obligation to make product data and related service data accessible to the user
Body:
1. Connected products shall be designed and manufactured, and related services shall be designed and
provided, in such a manner that product data and related service data, including the relevant metadata
necessary to interpret and use those data, are, by default, easily, securely, free of charge, in a
comprehensive, structured, commonly used and machine-readable format, and, where relevant and technically
feasible, directly accessible to the user. 2. Before concluding a contract [...] the seller [...] shall
provide at least the following information to the user, in a clear and comprehensible manner:
(a) the type, format and estimated volume of product data; (b) whether the connected product is capable
of generating data continuously and in real-time; (c) whether the connected product is capable of
storing data on-device or on a remote server; (d) how the user may access, retrieve or erase the data.

OUTPUT
Act: Data Act (32023R2854)
Article — Obligation to make product data and related service data accessible to the user
Applies to: Manufacturers of connected products; providers of related services; sellers, rentors, and lessors
Core rule: Connected products must be designed so that product data and metadata are easily, securely, and free of charge accessible to the user by default, in a comprehensive, structured, commonly used, and machine-readable format.
Key conditions or exceptions: Direct accessibility required where relevant and technically feasible. Pre-contractual disclosure of data type, volume, format, storage and access methods is mandatory. Applies to connected products placed on the market after 12 September 2026.
Procedure: Before contract conclusion → seller discloses data type/format/volume/storage/access → user gains access directly from product or by request to data holder.
Key terms: connected product, related service, product data, metadata, machine-readable format, data holder, user, readily available data
Cross-references: Article 2 (definitions), Article 4 (rights and obligations of users and data holders), Article 8(1) GDPR

=== ARTICLE TO SUMMARISE ===
Act: {act_title} ({celex})
Article — {article_title}
Body:
{body}

OUTPUT
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