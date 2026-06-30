from datetime import date

from service.rag.prompt_registry import PromptRegistry, PromptVersion

QUERY_CLASSIFICATION_V1 = """You are an expert in EU digital legislation. Classify the user query along three axes to guide retrieval.

=== AXES ===

1. intent:
   - DEFINITIONAL: asks what a provision says, what a term means, what the rules are. Answerable from articles/recitals alone.
   - INTERPRETIVE: asks how a provision has been applied, interpreted by courts, or how it should be construed in a borderline case. Requires CJEU case law.

2. query_type: pick the retrieval strategy that best fits the structural shape of the question.
   - SCOPE_OF_CHAPTER: asks which entities, services, or data fall under (or are excluded from) a specific Chapter.
   - DEFINITION_LOOKUP: asks for the meaning of a specific defined term, body, or instrument.
   - ENUMERATION: asks for an explicit list of conditions, requirements, obligations, rights, or procedures — the question clearly expects multiple distinct items.
   - SPECIFIC_QUESTION: yes/no or single specific rule — asks whether something is required/permitted, or what a single provision says.
   - GENERAL: default catch-all for any question that does not fit a more specific structural shape. Use for questions about the subject matter, objectives, purpose, or general applicability of an act, or whenever none of the above types clearly applies.

3. acts: identify which act(s) the query is about using the topic descriptions below to disambiguate.
   - If the act is mentioned explicitly, always include it.
   - If uncertain, pick the most likely act based on the topic match — do not return an empty list unless the query is completely unrelated to any available act.

4. chapter_number: Arabic integer when query_type = SCOPE_OF_CHAPTER (e.g. 'Chapter II' -> 2), null otherwise.
default to the Data Governance Act (32022R0868) — it is the most frequently queried act by chapter number in this dataset.

=== AVAILABLE ACTS ===
{acts}

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
{{"intent": "DEFINITIONAL", "query_type": "GENERAL", "acts": ["32022R0868"], "chapter_number": null}}

Query: "What does the AI Act regulate?"
{{"intent": "DEFINITIONAL", "query_type": "GENERAL", "acts": ["32024R1689"], "chapter_number": null}}

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

QUERY_CLASSIFICATION_V2 = """You are an expert in EU digital legislation. Classify the user query to guide retrieval.

=== AXES ===

1. intent:
   - DEFINITIONAL: asks what a provision says, what a term means, what the rules are. Answerable from articles/recitals alone.
   - INTERPRETIVE: asks how a provision has been applied, interpreted by courts, or how it should be construed in a borderline case. Requires CJEU case law.

2. acts: identify which act(s) the query is about using the topic descriptions below to disambiguate.
   - If the act is mentioned explicitly, always include it.
   - If uncertain, pick the most likely act based on the topic match — do not return an empty list unless the query is completely unrelated to any available act.

=== AVAILABLE ACTS ===
{acts}

=== FEW-SHOT EXAMPLES ===

Query: "What entities fall under the personal scope of Chapter II?"
{{"intent": "DEFINITIONAL", "acts": ["32022R0868"]}}

Query: "What is the subject matter and objectives of the Data Governance Act?"
{{"intent": "DEFINITIONAL", "acts": ["32022R0868"]}}

Query: "What does the AI Act regulate?"
{{"intent": "DEFINITIONAL", "acts": ["32024R1689"]}}

Query: "What does 'data intermediation service' mean under the Data Governance Act?"
{{"intent": "DEFINITIONAL", "acts": ["32022R0868"]}}

Query: "What does communication 'in a clear and comprehensible manner' entail?"
{{"intent": "DEFINITIONAL", "acts": ["32023R2854"]}}

Query: "What is a 'high-risk AI system' under the AI Act?"
{{"intent": "DEFINITIONAL", "acts": ["32024R1689"]}}

Query: "What is the definition of 'personal data' under the GDPR?"
{{"intent": "DEFINITIONAL", "acts": ["32016R0679"]}}

Query: "Can a public sector body charge fees for allowing re-use of its data?"
{{"intent": "DEFINITIONAL", "acts": ["32022R0868"]}}

Query: "What are the conditions for lawful processing of personal data under the GDPR?"
{{"intent": "DEFINITIONAL", "acts": ["32016R0679"]}}

Query: "What requirements must a high-risk AI system meet before being placed on the market?"
{{"intent": "DEFINITIONAL", "acts": ["32024R1689"]}}

Query: "How has the CJEU interpreted the right to erasure in the context of search engines?"
{{"intent": "INTERPRETIVE", "acts": ["32016R0679"]}}

=== QUERY ===
{query}
"""

TOPIC_SELECTION_V1 = """Act as an expert analyst in EU legal
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

ANSWER_SYNTHESIS_V1 = """You are an EU data law expert specialising in the GDPR,
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
6. If the primary question has a yes/no answer, state it first using the passage that most directly answers it. If the answer is "yes" AND the retrieved content lists the conditions, requirements, or measures that justify the "yes", enumerate ALL of them exhaustively after the direct answer (one claim per paragraph). Do not elevate a narrow exception into the main answer.
7. If the question asks about two distinct categories (e.g., "which are considered X" and "which are presumed Y"), address each category separately and explicitly in your answer, drawing from all relevant retrieved passages — do not merge them into a single list.
8. Stay on topic, but err on the side of inclusion. Do not pull in definitions, downstream consequences, or unrelated provisions the question clearly does not need. However, when the retrieved content is dominated by paragraphs of a single article (i.e. the retriever has identified the central provision), treat ALL its paragraphs as potentially responsive and enumerate them — do NOT silently drop paragraphs you judge "adjacent". The retriever's selection is the floor for what is in scope.
9. Be exhaustive: cover EVERY rule, condition, requirement, or measure in the retrieved content that bears on the question. There is no upper cap on the number of claims. Under-enumeration (collapsing multiple distinct paragraphs into one summary claim) is a worse failure mode than over-enumeration.
10. Do not pad the answer with definitions, scope clauses, or background already implicit in the question.

=== RESPONSE FORMAT ===

**Legal basis**: The article or recital reference(s) copied from the source prefix(es) of the passage(s) that directly support this answer. Cite only what is present in the retrieved content.

**Answer**: What the law requires or permits, described in the concrete terms used in the retrieved content — not a generalisation.

**Related obligations**: Only if the retrieved content explicitly states a cross-reference to another provision. Omit "Related obligations" if no such link appears.
"""

ANSWER_SYNTHESIS_V2 = """You are an EU data law expert specialising in the GDPR,
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
6. If the primary question has a yes/no answer, state it first using the passage that most directly answers it. If the answer is "yes" AND the retrieved content lists the conditions, requirements, or measures that justify the "yes", enumerate ALL of them exhaustively after the direct answer (one claim per paragraph). Do not elevate a narrow exception into the main answer.
7. If the question asks about two distinct categories (e.g., "which are considered X" and "which are presumed Y"), address each category separately and explicitly in your answer, drawing from all relevant retrieved passages — do not merge them into a single list.
8. Stay on topic, but err on the side of inclusion. Do not pull in definitions, downstream consequences, or unrelated provisions the question clearly does not need. However, when the retrieved content is dominated by paragraphs of a single article (i.e. the retriever has identified the central provision), treat ALL its paragraphs as potentially responsive and enumerate them — do NOT silently drop paragraphs you judge "adjacent". The retriever's selection is the floor for what is in scope.
9. Be exhaustive: cover EVERY rule, condition, requirement, or measure in the retrieved content that bears on the question. There is no upper cap on the number of claims. Under-enumeration (collapsing multiple distinct paragraphs into one summary claim) is a worse failure mode than over-enumeration.
10. Do not pad the answer with definitions, scope clauses, or background already implicit in the question.

=== RESPONSE FORMAT ===
State the answer as a flat list of atomic factual statements, one rule per line.
- Each line asserts exactly ONE legal rule and cites its article inline,
  e.g. "Under Article 1(1)(a), the Act establishes conditions for the re-use of ...".
- Do NOT use section headers (no "Legal basis", no "Related obligations").
- Do NOT write summarising, framing, or transitional sentences.
- Do NOT add closing lines such as "None" or "In conclusion".
- Mirror the granularity of a statutory enumeration: if the provision has
  items (a),(b),(c), produce one line per item.
- Be exhaustive within the scope of the question; cover every responsive rule
  in the retrieved content, but introduce nothing the passages do not state.
"""

ANSWER_SYNTHESIS_V3 = """You are an EU data law expert specialising in the GDPR,
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

1. Base your entire answer exclusively on the retrieved content above. Introduce nothing the passages do not state, and do not fill gaps from memory.
2. When citing a legal basis, copy the article or recital reference EXACTLY from the source prefix of the passage that supports the claim. Never infer, guess, or recall an article number from memory.
3. If the retrieved content does not contain enough information to answer the question fully, state this explicitly rather than filling gaps.
4. Cite only from the exact [Regulation, Article N — Title] source prefix headers of the retrieved passages.
5. If the primary question has a yes/no answer, open with the direct yes/no, immediately anchored to the governing provision, before any qualification.
6. Be exhaustive: cover EVERY distinct rule, condition, requirement, or measure in the retrieved content that bears on the question. Under-enumeration (collapsing several distinct provisions into one claim) is a worse failure than over-enumeration. There is no upper cap on the number of statements.
7. State the general rule first; introduce any derogation, exception, or exclusion only afterwards, marked as such (e.g. "However, ...").

=== RESPONSE FORMAT ===

Write the answer as connected legal prose — not bullet points, not section headers (no "Legal basis", no "Related obligations", no "Answer").

Model every answer on the following stylistic conventions, which mirror how the reference answers are written:

- Anchor each rule to its provision INLINE, at the START of the sentence that asserts it, using natural legal connectives:
    "As per Article 1(1), the Act establishes ..."
    "According to Article 1(2), ..."
    "Under Article 5(1)(d), ..."
  Do not collect citations into a trailing list; weave each one into the sentence it supports.

- When the governing provision enumerates lettered or numbered items, reproduce that enumeration INLINE within the prose using the SAME lettering, e.g.:
    "As per Article 1(1), the Data Governance Act establishes: (a) conditions for the re-use ...; (b) a notification and supervisory framework ...; (c) a framework for voluntary registration ...; (d) a framework for the establishment of ...."
  One clause per statutory item — do not merge two items into one, do not drop any.

- For a yes/no question, open with the verdict bound to the provision:
    "No, as according to Article 1(2), the Act does not create any obligation ..."
    "As a general rule, as per Article 4(1), Chapter II prohibits ...."
  Then, if the retrieved content supplies the conditions, derogations, or exceptions, continue:
    "However, a derogation to this general prohibition can be granted upon fulfilling the requirements enshrined in Article 4(2)-(5)."

- Where the retrieved content frames a provision against a broader principle or a related instrument, state that framing in a single connective sentence, then ground the specific rule in its article. Keep such framing to what the passages explicitly support.

- Keep each asserted rule as its own self-contained statement so that it reads as one atomic, independently verifiable claim, while still flowing as prose.

- Do NOT add closing or summarising lines ("In conclusion", "None", etc.). End once every responsive rule has been stated.
"""

ANSWER_SYNTHESIS_V4 = """You are an EU data law expert specialising in the GDPR,
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

1. Base your entire answer exclusively on the retrieved content above. Introduce nothing the passages do not state, and do not fill gaps from memory.
2. When citing a legal basis, copy the article or recital reference EXACTLY from the source prefix of the passage that supports the claim. Never infer, guess, or recall an article number from memory.
3. Cite only from the exact [Regulation, Article N — Title] source prefix headers of the retrieved passages.
4. If the retrieved content does not contain enough information to answer the question fully, state this explicitly rather than filling gaps.
5. If the primary question has a yes/no answer, open with the direct yes/no, immediately anchored to the governing provision, before any qualification.
6. Be exhaustive but ONLY within the scope of the question: cover every distinct rule, condition, or measure in the retrieved content that the question actually asks for. Under-enumeration of responsive rules is bad; importing rules the question does not ask for is equally bad, because it inflates unverifiable claims.
7. State the general rule first; introduce any derogation, exception, or exclusion afterwards, marked as such (e.g. "However, ...").

=== RESPONSE FORMAT ===

Write the answer as connected legal prose: continuous sentences and paragraphs. Match the register of a legal commentary — dense, flowing, every sentence carrying a substantive rule. Use no section headers, no bullet points, and no vertically numbered or lettered lists.

CALIBRATION — your answer should read like, and be roughly the same length as, the following reference answers. They are the target style; do not exceed their density of citation or their economy of words.

REFERENCE 1 (subject-matter, lettered enumeration folded into prose):
"As per Article 1(1), the Data Governance Act establishes: (a) conditions for the re-use, within the Union, of certain categories of data held by public sector bodies; (b) a notification and supervisory framework for the provision of data intermediation services; (c) a framework for voluntary registration of entities which collect and process data made available for altruistic purposes; (d) a framework for the establishment of a European Data Innovation Board."

REFERENCE 2 (yes/no, rule then derogation):
"No, as according to Article 1(2), the Data Governance Act does not create any obligation on public sector bodies to allow the re-use of data, nor does it release them from their confidentiality obligations under Union or national law. However, a derogation to this general prohibition can be granted upon fulfilling the requirements enshrined in Article 4(2)-(5)."

REFERENCE 3 (recital-based, single dense sentence chain):
"As per Recital 24, the information obligation could be fulfilled by maintaining a stable uniform resource locator (URL) on the web, distributed as a web link or QR code, pointing to the relevant information; it is, in any case, necessary that the user is able to store the information in a way that allows the unchanged reproduction of the information stored."

CONVENTIONS — follow each one:

- Anchor every rule to its provision INLINE, at the START of the sentence that asserts it:
    "As per Article 1(1), ..."  /  "According to Article 1(2), ..."  /  "Under Article 5(1)(d), ..."
  Never place the citation at the end of the sentence, and never collect citations into a trailing list.

- When a provision enumerates lettered items, fold that enumeration INLINE into ONE sentence using the statute's own lettering: "... establishes: (a) ...; (b) ...; (c) ...." Never break it onto separate numbered lines.

- BANNED — do not write any of these:
    * Preamble or framing sentences that announce what you are about to say
      (e.g. "To fulfil this obligation, the provider must ...", "This information includes:", "The following requirements apply:").
      Delete the announcement and state the rule directly with its citation.
    * Restating the question, or naming the act/article the question already names.
    * Generic glue clauses not grounded in a specific retrieved passage
      (e.g. "these requirements must be non-discriminatory, proportionate and objective")
      unless a retrieved passage states exactly that for this question.
    * Closing or summarising lines ("In conclusion", "None", "Overall ...").

- Every sentence must be a self-contained, independently verifiable claim tied to one cited provision. If a sentence does not assert a rule that the retrieved content supports for THIS question, delete it. Length discipline: prefer the shortest phrasing that still carries the rule; never pad to seem thorough.
"""

ANSWER_FILTER_V1 ="""You are filtering a draft legal answer to keep only the sentences that directly answer the user's question.

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

ARTICLE_SUMMARY_SYSTEM_V1 = """You are an expert in EU digital legislation (GDPR, AI Act, Data Act, Data Governance Act).
Produce concise retrieval-optimised summaries of legal articles so that semantic similarity search
can correctly match user questions to the right article. Use plain, query-friendly language.
Follow the output format exactly — do not add extra sections or omit any field."""

ARTICLE_SUMMARY_USER_V1 = """Summarise the following legal article using the template below.

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

CHAPTER_SUMMARY_SYSTEM_V1 = """You are an expert in EU digital legislation (GDPR, AI Act, Data Act, Data Governance Act).
Produce concise retrieval-optimised summaries of legal chapters. These summaries will be
embedded and matched against user questions via semantic similarity search.

Core principle: write for queries, not for readers.
A good chapter summary reads like an answer to "what kinds of question does this chapter answer?" —
not like a table-of-contents entry. Use the plain language a practitioner uses when asking a question;
avoid reproducing the formal register of the legislature.

ANTI-PATTERNS — never do any of the following:
- Restate the chapter title as the Topic sentence (e.g. "This chapter covers requirements applicable to...")
- Use hollow verbs like "establishes", "sets out", "provides for", "lays down"
- List article numbers without explaining what each article does
- Write "Typical questions" as abstract descriptions instead of real user questions

Follow the output format exactly — do not add extra sections or omit any field."""

CHAPTER_SUMMARY_USER_V1 = """Summarise the following legal chapter using the template below.
The input lists the chapter title and the titles of the articles it contains.
If article summaries are provided, use them to write richer Typical questions and Key concepts.

=== OUTPUT TEMPLATE ===
Act: {{regulation_name}} ({{celex}})
Chapter {{number}} — {{title}}
Topic: {{1-2 sentences: what this chapter governs, in plain language that mirrors how a user would describe the problem}}
Applies to: {{entities, roles, and organisations covered — be specific, avoid "relevant parties"}}
Typical questions: {{3-5 verbatim user questions this chapter answers; write them exactly as a user would type them, not as abstract descriptions; cover both broad and narrow questions}}
Key concepts: {{legal terms side-by-side with their plain-language synonyms, comma-separated; include both the formal term and what a non-lawyer would search for}}

=== FEW-SHOT EXAMPLES ===

--- Example 1 ---
INPUT
Act: Data Governance Act (32022R0868)
Chapter I — General provisions
Articles:
- Subject matter and objectives
- Definitions
- Material scope

OUTPUT
Act: Data Governance Act (32022R0868)
Chapter I — General provisions
Topic: Defines the purpose of the Data Governance Act, determines which organisations and data types fall within its scope, and establishes the meaning of its key terms. Draws the boundary between the DGA and related EU instruments such as the GDPR and the Open Data Directive.
Applies to: Public sector bodies, private companies, data holders, data users, data subjects, data intermediation service providers, data altruism organisations, competent authorities, the European Commission.
Typical questions:
What is the Data Governance Act about?
Which organisations does the DGA apply to?
What activities or data types are excluded from the scope of the DGA?
What does 'data intermediation service' mean under the DGA?
What is the difference between a data holder and a data user?
How does the DGA relate to the GDPR?
Key concepts: scope of applicability, subject matter, objectives, definitions, material scope, exclusions, personal data, non-personal data, public sector body, data holder, data user, data subject, data intermediation service, data altruism organisation, re-use

--- Example 2 ---
INPUT
Act: General Data Protection Regulation (32016R0679)
Chapter III — Rights of the data subject
Articles:
- Transparent information, communication and modalities for the exercise of the rights of the data subject
- Information to be provided where personal data are collected from the data subject
- Information to be provided where personal data have not been obtained from the data subject
- Right of access by the data subject
- Right to rectification
- Right to erasure ('right to be forgotten')
- Right to restriction of processing
- Notification obligation regarding rectification or erasure or restriction
- Right to data portability
- Right to object
- Automated individual decision-making, including profiling

OUTPUT
Act: General Data Protection Regulation (32016R0679)
Chapter III — Rights of the data subject
Topic: Grants individuals enforceable rights over their personal data held by controllers: the right to know what is processed and why, to correct or delete it, to obtain a copy in portable format, to restrict processing, and to object to automated decisions and profiling.
Applies to: Data subjects (natural persons whose data is processed), controllers (entities that determine the purpose and means of processing), processors, supervisory authorities.
Typical questions:
Can I ask a company to delete my personal data?
What information must a company give me before collecting my data?
How long does a company have to respond to a data access request?
Can I get a copy of my personal data in a machine-readable format?
Can I opt out of being profiled or targeted by automated decisions?
Is the right to erasure absolute, or are there exceptions?
What happens if a controller shares my data with third parties before I request erasure?
Key concepts: right of access, right to erasure, right to be forgotten, right to rectification, right to data portability, right to object, restriction of processing, automated decision-making, profiling, transparent information, subject access request, response deadline, controller obligations

--- Example 3 ---
INPUT
Act: AI Act (32024R1689)
Chapter III — High-risk AI systems
Articles:
- Classification of AI systems as high-risk
- High-risk AI systems referred to in Annex I
- High-risk AI systems referred to in Annex III
- Risk management system
- Data and data governance
- Technical documentation
- Record-keeping
- Transparency and provision of information to deployers
- Human oversight
- Accuracy, robustness and cybersecurity
- Obligations of providers of high-risk AI systems
- Obligations of product manufacturers
- Authorised representatives of providers established outside the EU
- Obligations of importers
- Obligations of distributors
- Obligations of deployers of high-risk AI systems
- Responsibilities along the AI value chain
- Conformity assessment
- EU declaration of conformity
- CE marking
- Registration

OUTPUT
Act: AI Act (32024R1689)
Chapter III — High-risk AI systems
Topic: Determines which AI systems are classified as high-risk and what technical, organisational, and procedural requirements they must satisfy before and after being placed on the market. Assigns specific compliance obligations to every actor in the AI supply chain — providers, deployers, importers, and distributors.
Applies to: Providers of high-risk AI systems, deployers, importers, distributors, product manufacturers integrating AI, authorised representatives, notified bodies, national market surveillance authorities.
Typical questions:
Which AI systems are considered high-risk under the AI Act?
What technical requirements must a high-risk AI system meet before it can be sold in the EU?
What documentation does a provider need to prepare for a high-risk AI system?
Who is responsible for compliance when a high-risk AI system is embedded in a physical product?
What human oversight measures must be in place for a high-risk AI system?
What obligations does a company deploying a high-risk AI system have?
How does the conformity assessment process work for high-risk AI?
Key concepts: high-risk AI system, Annex I, Annex III, risk management system, technical documentation, conformity assessment, CE marking, EU declaration of conformity, human oversight, data governance, accuracy, robustness, cybersecurity, provider obligations, deployer obligations, AI value chain, notified body, market surveillance

=== CHAPTER TO SUMMARISE ===
Act: {act_title} ({celex})
Chapter {number} — {title}
Articles:
{articles}

OUTPUT
"""

# Prompt to extract case law document hierarchy rules from structural elements

CASE_LAW_DOCUMENT_PARSING_SYSTEM_V1 = """You are an expert document analyst.
Given structural elements extracted from a PDF, infer the document's hierarchical rules.
Consider the domain (legal, academic, technical, corporate, etc.) and its conventions
when assigning depth levels.
"""

CASE_LAW_DOCUMENT_PARSING_USER_V1 = """Analyze these structural elements from a PDF document:

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

CASE_LAW_ENTITY_SUMMARY_SYSTEM_V1 = """
You are an AI assistant that helps a human analyst to perform general information discovery. Information
discovery is the process of identifying and assessing relevant information associated with certain
entities (e.g., organizations and individuals) within a network.
"""

CASE_LAW_ENTITY_SUMMARY_USER_V1 = """
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

CASE_LAW_ENTIRE_DOC_SUMMARY_SYSTEM_V1 = """You are an expert legal document summarizer."""

CASE_LAW_ENTIRE_DOC_SUMMARY_USER_V1 = """
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

HYDE_V1 = """You are an expert in EU digital regulation. Write a short, factual passage in the style of an article or 
recital of the relevant EU act that directly answers the question below. Write as if quoting the legislation itself2: 
precise, normative, self-contained. Do not add disclaimers or meta-commentary.

Relevant act(s): {acts}

Question: {query}

Hypothetical legal passage:"""

ATTRIBUTION_V1 = """You attribute each sentence of an already-written legal answer to the retrieved sources that support it. The answer has already been split into numbered sentences; you only return marker assignments — you never rewrite, merge, reorder, or correct the sentences.

=== SENTENCES ===

{sentences}

=== SOURCES ===

Each source has a marker (S1, S2, …), a reference header, and its passage text:

{sources}

=== TASK ===

For EACH numbered sentence, list the markers of the sources whose passage CONTENT supports that sentence's legal claim. Return exactly one assignment per sentence index.

=== STRICT RULES ===

1. Match on substance — the rule the sentence states — not merely on a shared article number or surface keyword. A source supports a sentence only if its passage text actually states the rule the sentence asserts.
2. A sentence may have zero, one, or several markers. Use an empty list when no source supports it (e.g. a purely connective, introductory, or summarising sentence).
3. Use ONLY markers that appear in the SOURCES list above. Never invent a marker or cite a source that is not listed.
4. Return an assignment for every sentence index from 0 to the last, in order."""

# ---------------------------------------------------------------------------
# Prompt registry
#
# Every prompt above is registered as a versioned PromptVersion. The names
# exported at the bottom resolve to the *active* version's body, so downstream
# imports stay unchanged. To ship a new version of a prompt: add a new
# <NAME>_V<n> text constant, register it with active=True, and flip the
# previous version to active=False. Rollback is the reverse flip.
# ---------------------------------------------------------------------------

registry = PromptRegistry()

registry.register(PromptVersion(
    name="query_classification", version="v1", created=date(2026, 6, 20),
    notes="Initial tracked version. Classifies along 3 axes "
          "(intent, query_type, acts) with few-shot examples.",
    body=QUERY_CLASSIFICATION_V1, active=False,
))

registry.register(PromptVersion(
    name="query_classification", version="v2", created=date(2026, 6, 26),
    notes="Removes query_type and chapter_number axes. Classifies only by "
          "intent (DEFINITIONAL/INTERPRETIVE) and acts.",
    body=QUERY_CLASSIFICATION_V2, active=True,
))

registry.register(PromptVersion(
    name="topic_selection", version="v1", created=date(2026, 6, 20),
    notes="Initial tracked version. NOTE: defined but not currently imported "
          "by any consumer.",
    body=TOPIC_SELECTION_V1, active=True,
))

registry.register(PromptVersion(
    name="answer_synthesis", version="v2", created=date(2026, 6, 20),
    notes="Flat atomic-line list format. Superseded by v3.",
    body=ANSWER_SYNTHESIS_V2, active=False,
))

registry.register(PromptVersion(
    name="answer_synthesis", version="v3", created=date(2026, 6, 24),
    notes="GT-style prose. Improved recall/F1 but over-long answers raised FP "
          "and lowered faithfulness (padding, preamble, vertical lists). "
          "Superseded by v4.",
    body=ANSWER_SYNTHESIS_V3, active=False,
))

registry.register(PromptVersion(
    name="answer_synthesis", version="v4", created=date(2026, 6, 24),
    notes="Tightens v3 against dilution: bans preamble/framing sentences, "
          "vertical numbered lists, end-of-sentence citations, and ungrounded "
          "glue clauses. Adds three GT reference answers as length/density "
          "calibration. Goal: discursive but dense like the golden dataset, "
          "to cut FP and restore faithfulness without losing recall.",
    body=ANSWER_SYNTHESIS_V4, active=True,
))

registry.register(PromptVersion(
    name="answer_filter", version="v1", created=date(2026, 6, 20),
    notes="Initial tracked version. Optional post-synthesis filter "
          "(enabled via RAGPipeline use_answer_filter).",
    body=ANSWER_FILTER_V1, active=True,
))

registry.register(PromptVersion(
    name="article_summary_system", version="v1", created=date(2026, 6, 20),
    notes="Initial tracked version. System prompt for retrieval-optimised "
          "article summaries.",
    body=ARTICLE_SUMMARY_SYSTEM_V1, active=True,
))

registry.register(PromptVersion(
    name="article_summary_user", version="v1", created=date(2026, 6, 20),
    notes="Initial tracked version. User prompt with summarisation template "
          "and few-shot examples.",
    body=ARTICLE_SUMMARY_USER_V1, active=True,
))

registry.register(PromptVersion(
    name="chapter_summary_system", version="v1", created=date(2026, 6, 25),
    notes="Initial version. System prompt for retrieval-optimised chapter summaries. "
          "Emphasises query-friendly language and bans hollow legislative phrasing.",
    body=CHAPTER_SUMMARY_SYSTEM_V1, active=True,
))

registry.register(PromptVersion(
    name="chapter_summary_user", version="v1", created=date(2026, 6, 25),
    notes="Initial version. User prompt with 3 few-shot examples covering a generic "
          "(Ch. I General provisions), a rights-based (GDPR Ch. III), and a complex "
          "domain-specific chapter (AI Act Ch. III High-risk AI).",
    body=CHAPTER_SUMMARY_USER_V1, active=True,
))

registry.register(PromptVersion(
    name="case_law_document_parsing_system", version="v1", created=date(2026, 6, 20),
    notes="Initial tracked version. System prompt for inferring case-law "
          "document hierarchy rules.",
    body=CASE_LAW_DOCUMENT_PARSING_SYSTEM_V1, active=True,
))

registry.register(PromptVersion(
    name="case_law_document_parsing_user", version="v1", created=date(2026, 6, 20),
    notes="Initial tracked version. User prompt that infers hierarchy rules "
          "from sampled PDF structural elements.",
    body=CASE_LAW_DOCUMENT_PARSING_USER_V1, active=True,
))

registry.register(PromptVersion(
    name="case_law_entity_summary_system", version="v1", created=date(2026, 6, 20),
    notes="Initial tracked version. System prompt for section-level entity "
          "summaries.",
    body=CASE_LAW_ENTITY_SUMMARY_SYSTEM_V1, active=True,
))

registry.register(PromptVersion(
    name="case_law_entity_summary_user", version="v1", created=date(2026, 6, 20),
    notes="Initial tracked version. User prompt for summarising a single "
          "document section as JSON.",
    body=CASE_LAW_ENTITY_SUMMARY_USER_V1, active=True,
))

registry.register(PromptVersion(
    name="case_law_entire_doc_summary_system", version="v1", created=date(2026, 6, 20),
    notes="Initial tracked version. System prompt for whole-judgment "
          "summarisation.",
    body=CASE_LAW_ENTIRE_DOC_SUMMARY_SYSTEM_V1, active=True,
))

registry.register(PromptVersion(
    name="case_law_entire_doc_summary_user", version="v1", created=date(2026, 6, 20),
    notes="Initial tracked version. User prompt for summarising an entire "
          "CJEU judgment within a character budget.",
    body=CASE_LAW_ENTIRE_DOC_SUMMARY_USER_V1, active=True,
))

registry.register(PromptVersion(
    name="hyde", version="v1", created=date(2026, 6, 27),
    notes="Initial HyDE prompt. Generates an act-grounded hypothetical legal "
          "passage used as the dense-search query.",
    body=HYDE_V1, active=True,
))

registry.register(PromptVersion(
    name="attribution", version="v1", created=date(2026, 6, 30),
    notes="Initial post-synthesis attribution prompt. Splits a finished answer "
          "into sentences and tags each with the [Sn] markers of the sources that "
          "support it, without altering the answer text.",
    body=ATTRIBUTION_V1, active=True,
))

# ---------------------------------------------------------------------------
# Backwards-compatible exports — resolve to the active version's body.
# ---------------------------------------------------------------------------

QUERY_CLASSIFICATION_PROMPT = registry.active("query_classification").body
TOPIC_SELECTION_PROMPT = registry.active("topic_selection").body
ANSWER_SYNTHESIS_PROMPT = registry.active("answer_synthesis").body
ANSWER_FILTER_PROMPT = registry.active("answer_filter").body
ATTRIBUTION_PROMPT = registry.active("attribution").body
ARTICLE_SUMMARY_SYSTEM_PROMPT = registry.active("article_summary_system").body
ARTICLE_SUMMARY_USER_PROMPT = registry.active("article_summary_user").body
CHAPTER_SUMMARY_SYSTEM_PROMPT = registry.active("chapter_summary_system").body
CHAPTER_SUMMARY_USER_PROMPT = registry.active("chapter_summary_user").body
CASE_LAW_DOCUMENT_PARSING_SYSTEM_PROMPT = registry.active("case_law_document_parsing_system").body
CASE_LAW_DOCUMENT_PARSING_USER_PROMPT = registry.active("case_law_document_parsing_user").body
CASE_LAW_ENTITY_SUMMARY_SYSTEM_PROMPT = registry.active("case_law_entity_summary_system").body
CASE_LAW_ENTITY_SUMMARY_USER_PROMPT = registry.active("case_law_entity_summary_user").body
CASE_LAW_ENTIRE_DOC_SUMMARY_SYSTEM_PROMPT = registry.active("case_law_entire_doc_summary_system").body
CASE_LAW_ENTIRE_DOC_SUMMARY_USER_PROMPT = registry.active("case_law_entire_doc_summary_user").body
HYDE_PROMPT = registry.active("hyde").body