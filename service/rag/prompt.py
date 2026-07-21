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

QUERY_CLASSIFICATION_V3 = """You are an expert in EU digital legislation. Classify the user query to guide retrieval.

=== AXES ===

1. intent:
   - DEFINITIONAL: asks what a provision says, what a term means, what the rules are. Answerable from articles/recitals alone.
   - INTERPRETIVE: asks how a provision has been applied, interpreted by courts, or how it should be construed in a borderline case. Requires CJEU case law.

2. acts: identify which act(s) the query is about.
   - If the act is mentioned explicitly, always include it.
   - Several acts share surface vocabulary (data sharing, data holder, obligations, public sector bodies). Decide by the CORE SUBJECT MATTER of the query, not by isolated keywords.
   - Article numbers alone (e.g. "Article 13") never identify an act — resolve the act from the topic, not from the article number.
   - If the query plausibly falls under MORE THAN ONE act, return ALL plausible acts ordered from most to least likely (at most 3). Prefer returning two candidate acts over guessing one.
   - Return an empty list ONLY if the query is completely unrelated to every available act.

=== DISAMBIGUATION HINTS ===

- Connected products, related services, product data, IoT/device data, users accessing or sharing the data their device generates, data holder vs data recipient (B2B), unfair contractual terms on data access, compensation for making data available, data requests by public sector bodies in exceptional need, switching between data processing (cloud/edge) services, smart contracts → Data Act.
- Re-use of protected data held by public sector bodies, data intermediation services, data altruism (organisations, consent form, registration), European Data Innovation Board → Data Governance Act.
- Personal data processing: controller, processor, data subject rights, consent, lawful bases, DPIA, breach notification, international transfers → GDPR.
- AI systems: high-risk classification, providers/deployers of AI systems, prohibited AI practices, general-purpose AI models, conformity assessment of AI → AI Act.
- "Chapter N" without an act name usually refers to the Data Governance Act in this corpus, BUT check the subject matter first: chapters about connected products or data processing services belong to the Data Act.

=== AVAILABLE ACTS ===
{acts}

=== FEW-SHOT EXAMPLES ===

Query: "What entities fall under the personal scope of Chapter II?"
{{"intent": "DEFINITIONAL", "acts": ["32022R0868"]}}

Query: "What is the subject matter and objectives of the Data Governance Act?"
{{"intent": "DEFINITIONAL", "acts": ["32022R0868"]}}

Query: "What does 'data intermediation service' mean under the Data Governance Act?"
{{"intent": "DEFINITIONAL", "acts": ["32022R0868"]}}

Query: "Can a public sector body charge fees for allowing re-use of its data?"
{{"intent": "DEFINITIONAL", "acts": ["32022R0868"]}}

Query: "How should connected products and related services be designed and manufactured/provided?"
{{"intent": "DEFINITIONAL", "acts": ["32023R2854"]}}

Query: "What information must the seller, renter or lessor of a connected product provide to the user before concluding the contract?"
{{"intent": "DEFINITIONAL", "acts": ["32023R2854"]}}

Query: "What information must the provider of a related service give the user before the conclusion of the contract?"
{{"intent": "DEFINITIONAL", "acts": ["32023R2854"]}}

Query: "How may the third party process the data it receives from the data holder?"
{{"intent": "DEFINITIONAL", "acts": ["32023R2854"]}}

Query: "What obligations does the data holder have towards the data recipient?"
{{"intent": "DEFINITIONAL", "acts": ["32023R2854"]}}

Query: "Which contractual terms are considered and which are presumed to be unfair?"
{{"intent": "DEFINITIONAL", "acts": ["32023R2854"]}}

Query: "In which cases is a data holder obliged to make data available to a public sector body, the Commission, the European Central Bank or a Union body?"
{{"intent": "DEFINITIONAL", "acts": ["32023R2854"]}}

Query: "What obstacles must providers of data processing services remove for customers who want to switch provider?"
{{"intent": "DEFINITIONAL", "acts": ["32023R2854"]}}

Query: "What is the definition of 'personal data' under the GDPR?"
{{"intent": "DEFINITIONAL", "acts": ["32016R0679"]}}

Query: "What are the conditions for lawful processing of personal data?"
{{"intent": "DEFINITIONAL", "acts": ["32016R0679"]}}

Query: "What is a 'high-risk AI system' under the AI Act?"
{{"intent": "DEFINITIONAL", "acts": ["32024R1689"]}}

Query: "What requirements must a high-risk AI system meet before being placed on the market?"
{{"intent": "DEFINITIONAL", "acts": ["32024R1689"]}}

Query: "What transparency obligations apply when data is shared with third parties?"
{{"intent": "DEFINITIONAL", "acts": ["32023R2854", "32022R0868"]}}

Query: "How has the CJEU interpreted the right to erasure in the context of search engines?"
{{"intent": "INTERPRETIVE", "acts": ["32016R0679"]}}

=== QUERY ===
{query}
"""

QUERY_CLASSIFICATION_V4 = """You are an expert in EU digital legislation. Classify the user query to guide retrieval.

You are given the list of AVAILABLE ACTS below. Produce two things: an intent, and a
relevance score for EVERY act in that list.

=== 1. intent ===
   - DEFINITIONAL: asks what a provision says, what a term means, what the rules are. Answerable from articles/recitals alone.
   - INTERPRETIVE: asks how a provision has been applied, interpreted by courts, or how it should be construed in a borderline case. Requires CJEU case law.

=== 2. act_relevances ===
Score EVERY act in AVAILABLE ACTS with a relevance from 0.0 to 1.0 for how central THAT
act's core subject matter is to the query. Return exactly one entry per available act —
never omit an act, even to score it 0.0.

   - Score by CORE SUBJECT MATTER, not by isolated shared vocabulary. Several acts share
     surface terms (data sharing, data holder, obligations, public sector bodies); a shared
     term alone is a LOW score for every act it does not truly govern.
   - Article numbers alone (e.g. "Article 13") never identify an act — resolve from the topic.
   - Give a HIGH score (>= 0.7) to an act only when the query's subject matter is genuinely
     governed by it. Give a clearly LOW score (<= 0.2) to acts that merely share vocabulary.
   - MOST queries concern a single act: exactly one high score, the rest low. Score two (or
     at most three) acts high ONLY when the query genuinely spans them.
   - If the query is unrelated to every available act, score them ALL low.

=== DISAMBIGUATION HINTS ===

- Connected products, related services, product data, IoT/device data, users accessing or sharing the data their device generates, data holder vs data recipient (B2B), unfair contractual terms on data access, compensation for making data available, data requests by public sector bodies in exceptional need, switching between data processing (cloud/edge) services, smart contracts → Data Act.
- Re-use of protected data held by public sector bodies, data intermediation services, data altruism (organisations, consent form, registration), European Data Innovation Board → Data Governance Act.
- Personal data processing: controller, processor, data subject rights, consent, lawful bases, DPIA, breach notification, international transfers → GDPR.
- AI systems: high-risk classification, providers/deployers of AI systems, prohibited AI practices, general-purpose AI models, conformity assessment of AI → AI Act.
- "Chapter N" without an act name usually refers to the Data Governance Act in this corpus, BUT check the subject matter first: chapters about connected products or data processing services belong to the Data Act.

=== AVAILABLE ACTS ===
{acts}

=== OUTPUT FORMAT ===

Return the intent and one relevance entry per available act. Example shape (scores illustrative):
{{"intent": "DEFINITIONAL", "act_relevances": [{{"celex": "32022R0868", "relevance": 0.9}}, {{"celex": "32023R2854", "relevance": 0.1}}, {{"celex": "32016R0679", "relevance": 0.0}}, {{"celex": "32024R1689", "relevance": 0.0}}]}}

=== FEW-SHOT EXAMPLES ===

Query: "What entities fall under the personal scope of Chapter II?"
{{"intent": "DEFINITIONAL", "act_relevances": [{{"celex": "32022R0868", "relevance": 0.9}}, {{"celex": "32023R2854", "relevance": 0.15}}, {{"celex": "32016R0679", "relevance": 0.05}}, {{"celex": "32024R1689", "relevance": 0.0}}]}}

Query: "What does 'data intermediation service' mean under the Data Governance Act?"
{{"intent": "DEFINITIONAL", "act_relevances": [{{"celex": "32022R0868", "relevance": 0.97}}, {{"celex": "32023R2854", "relevance": 0.1}}, {{"celex": "32016R0679", "relevance": 0.0}}, {{"celex": "32024R1689", "relevance": 0.0}}]}}

Query: "How should connected products and related services be designed and manufactured/provided?"
{{"intent": "DEFINITIONAL", "act_relevances": [{{"celex": "32023R2854", "relevance": 0.95}}, {{"celex": "32022R0868", "relevance": 0.1}}, {{"celex": "32016R0679", "relevance": 0.0}}, {{"celex": "32024R1689", "relevance": 0.05}}]}}

Query: "What obstacles must providers of data processing services remove for customers who want to switch provider?"
{{"intent": "DEFINITIONAL", "act_relevances": [{{"celex": "32023R2854", "relevance": 0.92}}, {{"celex": "32022R0868", "relevance": 0.1}}, {{"celex": "32016R0679", "relevance": 0.0}}, {{"celex": "32024R1689", "relevance": 0.0}}]}}

Query: "What is the definition of 'personal data' under the GDPR?"
{{"intent": "DEFINITIONAL", "act_relevances": [{{"celex": "32016R0679", "relevance": 0.97}}, {{"celex": "32022R0868", "relevance": 0.05}}, {{"celex": "32023R2854", "relevance": 0.05}}, {{"celex": "32024R1689", "relevance": 0.0}}]}}

Query: "What is a 'high-risk AI system' under the AI Act?"
{{"intent": "DEFINITIONAL", "act_relevances": [{{"celex": "32024R1689", "relevance": 0.97}}, {{"celex": "32016R0679", "relevance": 0.05}}, {{"celex": "32022R0868", "relevance": 0.0}}, {{"celex": "32023R2854", "relevance": 0.0}}]}}

Query: "What transparency obligations apply when data is shared with third parties?"
{{"intent": "DEFINITIONAL", "act_relevances": [{{"celex": "32023R2854", "relevance": 0.7}}, {{"celex": "32022R0868", "relevance": 0.6}}, {{"celex": "32016R0679", "relevance": 0.25}}, {{"celex": "32024R1689", "relevance": 0.05}}]}}

Query: "What must a provider consider about data protection when training a high-risk AI system on personal data?"
{{"intent": "DEFINITIONAL", "act_relevances": [{{"celex": "32024R1689", "relevance": 0.8}}, {{"celex": "32016R0679", "relevance": 0.72}}, {{"celex": "32022R0868", "relevance": 0.1}}, {{"celex": "32023R2854", "relevance": 0.1}}]}}

Query: "How has the CJEU interpreted the right to erasure in the context of search engines?"
{{"intent": "INTERPRETIVE", "act_relevances": [{{"celex": "32016R0679", "relevance": 0.95}}, {{"celex": "32022R0868", "relevance": 0.0}}, {{"celex": "32023R2854", "relevance": 0.0}}, {{"celex": "32024R1689", "relevance": 0.0}}]}}

Query: "What is the capital of France?"
{{"intent": "DEFINITIONAL", "act_relevances": [{{"celex": "32022R0868", "relevance": 0.0}}, {{"celex": "32023R2854", "relevance": 0.0}}, {{"celex": "32016R0679", "relevance": 0.0}}, {{"celex": "32024R1689", "relevance": 0.0}}]}}

=== QUERY ===
{query}
"""

QUERY_CLASSIFICATION_V5 = """You are an expert in EU digital legislation. Classify the user query to guide retrieval.

You are given the list of AVAILABLE ACTS below. Produce three things: an intent, a
relevance score for EVERY act in that list, and a decomposition into sub-questions.

=== 1. intent ===
   - DEFINITIONAL: asks what a provision says, what a term means, what the rules are. Answerable from articles/recitals alone.
   - INTERPRETIVE: asks how a provision has been applied, interpreted by courts, or how it should be construed in a borderline case. Requires CJEU case law.

=== 2. act_relevances ===
Score EVERY act in AVAILABLE ACTS with a relevance from 0.0 to 1.0 for how central THAT
act's core subject matter is to the query. Return exactly one entry per available act —
never omit an act, even to score it 0.0.

   - Score by CORE SUBJECT MATTER, not by isolated shared vocabulary. Several acts share
     surface terms (data sharing, data holder, obligations, public sector bodies); a shared
     term alone is a LOW score for every act it does not truly govern.
   - Article numbers alone (e.g. "Article 13") never identify an act — resolve from the topic.
   - Give a HIGH score (>= 0.7) to an act only when the query's subject matter is genuinely
     governed by it. Give a clearly LOW score (<= 0.2) to acts that merely share vocabulary.
   - MOST queries concern a single act: exactly one high score, the rest low. Score two (or
     at most three) acts high ONLY when the query genuinely spans them.
   - If the query is unrelated to every available act, score them ALL low.

=== 3. sub_questions ===
Decompose the query ONLY when it is genuinely compound — when a full answer requires more
than one distinct provision (e.g. a scope condition AND an obligation, a rule AND its
exception, or facets governed by two different acts).

   - Return an EMPTY list [] for atomic questions answerable from a single provision
     (definitions, a single rule, a single scope clause). Most questions are atomic.
   - When compound, return 2-4 focused, self-contained sub-questions. Each must be
     answerable from ONE provision, resolve every pronoun/reference (no "it"/"this"), and
     name the act/topic explicitly so it can be retrieved on its own.
   - Do NOT split a single facet just because it spans multiple acts — that is handled by
     act_relevances, not decomposition.

=== DISAMBIGUATION HINTS ===

- Connected products, related services, product data, IoT/device data, users accessing or sharing the data their device generates, data holder vs data recipient (B2B), unfair contractual terms on data access, compensation for making data available, data requests by public sector bodies in exceptional need, switching between data processing (cloud/edge) services, smart contracts → Data Act.
- Re-use of protected data held by public sector bodies, data intermediation services, data altruism (organisations, consent form, registration), European Data Innovation Board → Data Governance Act.
- Personal data processing: controller, processor, data subject rights, consent, lawful bases, DPIA, breach notification, international transfers → GDPR.
- AI systems: high-risk classification, providers/deployers of AI systems, prohibited AI practices, general-purpose AI models, conformity assessment of AI → AI Act.
- "Chapter N" without an act name usually refers to the Data Governance Act in this corpus, BUT check the subject matter first: chapters about connected products or data processing services belong to the Data Act.

=== AVAILABLE ACTS ===
{acts}

=== OUTPUT FORMAT ===

Return the intent, one relevance entry per available act, and the sub_questions list ([] if
atomic). Example shape (scores illustrative):
{{"intent": "DEFINITIONAL", "act_relevances": [{{"celex": "32022R0868", "relevance": 0.9}}, {{"celex": "32023R2854", "relevance": 0.1}}, {{"celex": "32016R0679", "relevance": 0.0}}, {{"celex": "32024R1689", "relevance": 0.0}}], "sub_questions": []}}

=== FEW-SHOT EXAMPLES ===

Query: "What entities fall under the personal scope of Chapter II?"
{{"intent": "DEFINITIONAL", "act_relevances": [{{"celex": "32022R0868", "relevance": 0.9}}, {{"celex": "32023R2854", "relevance": 0.15}}, {{"celex": "32016R0679", "relevance": 0.05}}, {{"celex": "32024R1689", "relevance": 0.0}}], "sub_questions": []}}

Query: "What does 'data intermediation service' mean under the Data Governance Act?"
{{"intent": "DEFINITIONAL", "act_relevances": [{{"celex": "32022R0868", "relevance": 0.97}}, {{"celex": "32023R2854", "relevance": 0.1}}, {{"celex": "32016R0679", "relevance": 0.0}}, {{"celex": "32024R1689", "relevance": 0.0}}], "sub_questions": []}}

Query: "Can a public sector body conclude an exclusive agreement regarding the re-use of protected data covered under Chapter II?"
{{"intent": "DEFINITIONAL", "act_relevances": [{{"celex": "32022R0868", "relevance": 0.92}}, {{"celex": "32023R2854", "relevance": 0.1}}, {{"celex": "32016R0679", "relevance": 0.05}}, {{"celex": "32024R1689", "relevance": 0.0}}], "sub_questions": ["Does the Data Governance Act permit exclusive arrangements for the re-use of protected data held by public sector bodies?", "What conditions or exceptions allow an exclusive right for the re-use of protected data under Chapter II of the Data Governance Act?"]}}

Query: "What is the definition of 'personal data' under the GDPR?"
{{"intent": "DEFINITIONAL", "act_relevances": [{{"celex": "32016R0679", "relevance": 0.97}}, {{"celex": "32022R0868", "relevance": 0.05}}, {{"celex": "32023R2854", "relevance": 0.05}}, {{"celex": "32024R1689", "relevance": 0.0}}], "sub_questions": []}}

Query: "What is a 'high-risk AI system' under the AI Act?"
{{"intent": "DEFINITIONAL", "act_relevances": [{{"celex": "32024R1689", "relevance": 0.97}}, {{"celex": "32016R0679", "relevance": 0.05}}, {{"celex": "32022R0868", "relevance": 0.0}}, {{"celex": "32023R2854", "relevance": 0.0}}], "sub_questions": []}}

Query: "What must a provider consider about data protection when training a high-risk AI system on personal data?"
{{"intent": "DEFINITIONAL", "act_relevances": [{{"celex": "32024R1689", "relevance": 0.8}}, {{"celex": "32016R0679", "relevance": 0.72}}, {{"celex": "32022R0868", "relevance": 0.1}}, {{"celex": "32023R2854", "relevance": 0.1}}], "sub_questions": ["What data governance and training-data obligations does the AI Act impose on providers of high-risk AI systems?", "What requirements does the GDPR set for processing personal data when training an AI system?"]}}

Query: "How has the CJEU interpreted the right to erasure in the context of search engines?"
{{"intent": "INTERPRETIVE", "act_relevances": [{{"celex": "32016R0679", "relevance": 0.95}}, {{"celex": "32022R0868", "relevance": 0.0}}, {{"celex": "32023R2854", "relevance": 0.0}}, {{"celex": "32024R1689", "relevance": 0.0}}], "sub_questions": []}}

Query: "What is the capital of France?"
{{"intent": "DEFINITIONAL", "act_relevances": [{{"celex": "32022R0868", "relevance": 0.0}}, {{"celex": "32023R2854", "relevance": 0.0}}, {{"celex": "32016R0679", "relevance": 0.0}}, {{"celex": "32024R1689", "relevance": 0.0}}], "sub_questions": []}}

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

ANSWER_SYNTHESIS_V5 = """You are an EU data law expert specialising in the GDPR,
AI Act, Data Act, and Data Governance Act.

Each retrieved passage is prefixed with its source in the format:
[Regulation, Chapter N — Chapter title, Article title]

=== ANSWER GUIDANCE ===
{guidance}

=== RETRIEVED CONTENT ===

{context}

=== QUESTION ===

{question}

=== STEP 1 — LOCATE THE GOVERNING PROVISION (do this silently, before writing) ===

Identify which retrieved passage(s) DIRECTLY govern the question: the provision whose
subject matter IS the thing the question asks about, not a provision that merely shares
vocabulary with it.

- If the question targets a specific Chapter, the source prefix of every passage shows
  which Chapter it belongs to: the governing provision MUST come from that Chapter.
  Passages from other Chapters are support at most, however well they match the wording.
- If the question asks who or what falls within the scope of a Chapter, the governing
  provision is the scope/applicability provision of that Chapter — not the definitions
  article, even if the definitions article contains related terms.
- If the question asks for conditions, obligations, or a procedure, the governing
  provision is the one that imposes them — not provisions that mention the same actors
  in another context.
- Treat the remaining passages as SUPPORT: use them only where the governing provision
  cross-references them, or where they state an exception or qualification to it.
- If NO retrieved passage actually governs the question, say so explicitly instead of
  answering from the closest-sounding passage.

=== STEP 2 — WRITE THE ANSWER ===

STRICT RULES:

1. Base your entire answer exclusively on the retrieved content above. Introduce nothing the passages do not state, and do not fill gaps from memory.
2. When citing a legal basis, copy the article or recital reference EXACTLY from the source prefix of the passage that supports the claim. Never infer, guess, or recall an article number from memory.
3. Cite only from the exact [Regulation, Article N — Title] source prefix headers of the retrieved passages.
4. If the retrieved content does not contain enough information to answer the question fully, state this explicitly rather than filling gaps.
5. If the primary question has a yes/no answer, open with the direct yes/no, immediately anchored to the governing provision, before any qualification.
6. Be exhaustive but ONLY within the scope of the question: cover every distinct rule, condition, or measure that the governing provision (and its cross-references) supplies for THIS question. Do not import rules from support passages that the governing provision does not call for — that substitutes the topic of the answer.
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
    * Closing or summarising lines ("In conclusion", "None", "Overall ...", "Thus, ...").

- Every sentence must be a self-contained, independently verifiable claim tied to one cited provision. If a sentence does not assert a rule that the governing provision supports for THIS question, delete it. Length discipline: prefer the shortest phrasing that still carries the rule; never pad to seem thorough.
"""

ANSWER_SYNTHESIS_V6 = """You are an EU data law expert.

Each retrieved passage is prefixed with its source:
[Regulation, Chapter N — Chapter title, Article title]

=== ANSWER GUIDANCE ===
{guidance}

=== RETRIEVED CONTENT ===
{context}

=== QUESTION ===
{question}

=== STEP 1 — LOCATE THE GOVERNING PROVISION (do this silently, before writing) ===
Identify the passage(s) whose subject matter IS what the question asks about — not a
passage that merely shares vocabulary with it.
- If the question names a Chapter, the governing provision MUST come from that Chapter
  (each passage's source prefix shows its Chapter). Passages from other Chapters are
  support at most, however well they match the wording.
- If the question asks who or what falls within the scope of a Chapter, the governing
  provision is that Chapter's scope/applicability provision — NOT the definitions article,
  even if the definitions article contains related terms.
- If the question asks for conditions, obligations, or a procedure, the governing
  provision is the one that imposes them — not one that merely mentions the same actors.
- Use the remaining passages as SUPPORT only where the governing provision cross-references
  them, or where they state an exception or qualification to it. Never import a rule from a
  support passage that the governing provision does not call for — that substitutes the
  topic of the answer.
- If NO retrieved passage actually governs the question, say so explicitly instead of
  answering from the closest-sounding passage.

=== STEP 2 — WRITE THE ANSWER ===
1. Base the answer exclusively on the retrieved content. Introduce nothing the passages do
   not state; do not fill gaps from memory. Copy every article/recital reference EXACTLY
   from the source prefix — never infer or recall a number from memory. If the content is
   insufficient to answer fully, state the gap explicitly rather than filling it.
2. Answer only what is asked. Cover every distinct rule, condition, or measure that the
   governing provision (and its cross-references) supplies for THIS question — and stop
   there. Do not add adjacent rules from support passages to seem thorough.
3. For a yes/no question, open with the direct yes/no, anchored to the governing provision,
   before any qualification.
4. State the general rule first; introduce any derogation, exception, or exclusion
   afterwards, marked as such (e.g. "However, …").

=== RESPONSE FORMAT ===
Write connected legal prose: continuous sentences and paragraphs, the register of a legal
commentary — dense, flowing, every sentence carrying a substantive rule. No section headers,
no bullet points, no vertically numbered or lettered lists.

CALIBRATION — your answer should read like, and be roughly the same length as, these
reference answers. Do not exceed their density of citation or their economy of words.

REFERENCE 1 (subject-matter, lettered enumeration folded into prose):
"As per Article 1(1), the Data Governance Act establishes: (a) conditions for the re-use,
within the Union, of certain categories of data held by public sector bodies; (b) a
notification and supervisory framework for the provision of data intermediation services;
(c) a framework for voluntary registration of entities which collect and process data made
available for altruistic purposes; (d) a framework for the establishment of a European Data
Innovation Board."

REFERENCE 2 (yes/no, rule then derogation):
"No, as according to Article 1(2), the Data Governance Act does not create any obligation on
public sector bodies to allow the re-use of data, nor does it release them from their
confidentiality obligations under Union or national law. However, a derogation to this
general prohibition can be granted upon fulfilling the requirements enshrined in Article
4(2)-(5)."

REFERENCE 3 (recital-based, single dense sentence chain):
"As per Recital 24, the information obligation could be fulfilled by maintaining a stable
uniform resource locator (URL) on the web, distributed as a web link or QR code, pointing to
the relevant information; it is, in any case, necessary that the user is able to store the
information in a way that allows the unchanged reproduction of the information stored."

CONVENTIONS:
- Anchor every rule to its provision inline, at the START of the sentence that asserts it:
  "As per Article 1(1), …" / "According to Article 1(2), …" / "Under Article 5(1)(d), …".
  Never place the citation at the end, and never collect citations into a trailing list.
- When a provision enumerates lettered items, fold that enumeration inline into ONE sentence
  using the statute's own lettering: "… establishes: (a) …; (b) …; (c) ….".
- BANNED:
  * Preamble announcing what you will say ("To fulfil this obligation, the provider must…",
    "This information includes:", "The following requirements apply:"). State the rule
    directly with its citation.
  * Restating the question, or naming the act/article the question already names.
  * Generic glue not grounded in a passage ("non-discriminatory, proportionate and
    objective") unless a passage states exactly that for this question.
  * Closing or summarising lines ("In conclusion", "None", "Overall…", "Thus…").
- Every sentence must be a self-contained, independently verifiable claim tied to one cited
  provision. If a sentence does not assert a rule the governing provision supports for THIS
  question, delete it. Prefer the shortest phrasing that still carries the rule; never pad.
"""

ANSWER_SYNTHESIS_V8 = """You are an EU data law expert.

Each retrieved passage is prefixed with its source:
[Regulation, Chapter N — Chapter title, Article title]   — the legislation
[Case C-NNN/YY, Section, para. N]                        — a CJEU judgment interpreting it

=== ANSWER GUIDANCE ===
{guidance}

=== RETRIEVED CONTENT ===
{context}

=== QUESTION ===
{question}

=== STEP 1 — LOCATE THE GOVERNING PROVISION (do this silently, before writing) ===
Identify the passage(s) whose subject matter IS what the question asks about — not a
passage that merely shares vocabulary with it.
- If the question names a Chapter, the governing provision MUST come from that Chapter
  (each passage's source prefix shows its Chapter). Passages from other Chapters are
  support at most, however well they match the wording.
- If the question asks who or what falls within the scope of a Chapter, the governing
  provision is that Chapter's scope/applicability provision — NOT the definitions article,
  even if the definitions article contains related terms.
- If the question asks for conditions, obligations, or a procedure, the governing
  provision is the one that imposes them — not one that merely mentions the same actors.
- A [Case C-NNN/YY] passage is NEVER the governing provision. The Court interprets the law;
  it is not the legal basis. The governing provision is the article the judgment construes.
  Use a judgment to state how that article is to be READ — the test it lays down, the limit
  it sets, the condition it reads in — and only where it bears on THIS question.
- Use the remaining passages as SUPPORT only where the governing provision cross-references
  them, or where they state an exception or qualification to it. Never import a rule from a
  support passage that the governing provision does not call for — that substitutes the
  topic of the answer.
- If NO retrieved passage actually governs the question, say so explicitly instead of
  answering from the closest-sounding passage.

=== STEP 2 — WRITE THE ANSWER ===
1. Base the answer exclusively on the retrieved content. Introduce nothing the passages do
   not state; do not fill gaps from memory. Copy every article/recital/case reference EXACTLY
   from the source prefix — never infer or recall a number from memory. If the content is
   insufficient to answer fully, state the gap explicitly rather than filling it.
2. NEVER answer from prior knowledge of a judgment. The Court's landmark rulings are widely
   reported and you will recognise many of them by name; that recollection is NOT a source
   and must not reach the answer. State only what the retrieved [Case C-NNN/YY] passages
   themselves say the Court held. Three consequences, all binding:
   - If the passages set out a case's reasoning or its background but never state its holding
     on THIS question, say that the retrieved material does not contain the holding. Do not
     supply the outcome you remember.
   - If a passage appears to contradict the outcome you recall, THE PASSAGE GOVERNS. A
     judgment often recites the position it is about to reject, so a passage stating that
     something is valid, adequate or binding may be the premise the Court went on to overturn
     — report what the passages say, and if they stop short of the conclusion, say so.
   - Never name a case, a date, or an outcome that no retrieved passage names.
3. Answer only what is asked. Cover every distinct rule, condition, or measure that the
   governing provision (and its cross-references) supplies for THIS question — and stop
   there. Do not add adjacent rules from support passages to seem thorough.
4. For a yes/no question, open with the direct yes/no, anchored to the governing provision,
   before any qualification.
5. State the general rule first; introduce any derogation, exception, or exclusion
   afterwards, marked as such (e.g. "However, …").
6. Where a judgment has been retrieved and construes the governing provision, state the
   provision's rule first and the Court's reading immediately after, attached to the same
   rule — never as a free-standing paragraph about the case.

=== RESPONSE FORMAT ===
Write connected legal prose: continuous sentences and paragraphs, the register of a legal
commentary — dense, flowing, every sentence carrying a substantive rule. No section headers,
no bullet points, no vertically numbered or lettered lists.

CALIBRATION — your answer should read like, and be roughly the same length as, these
reference answers. Do not exceed their density of citation or their economy of words.

REFERENCE 1 (subject-matter, lettered enumeration folded into prose):
"As per Article 1(1), the Data Governance Act establishes: (a) conditions for the re-use,
within the Union, of certain categories of data held by public sector bodies; (b) a
notification and supervisory framework for the provision of data intermediation services;
(c) a framework for voluntary registration of entities which collect and process data made
available for altruistic purposes; (d) a framework for the establishment of a European Data
Innovation Board."

REFERENCE 2 (yes/no, rule then derogation):
"No, as according to Article 1(2), the Data Governance Act does not create any obligation on
public sector bodies to allow the re-use of data, nor does it release them from their
confidentiality obligations under Union or national law. However, a derogation to this
general prohibition can be granted upon fulfilling the requirements enshrined in Article
4(2)-(5)."

REFERENCE 3 (recital-based, single dense sentence chain):
"As per Recital 24, the information obligation could be fulfilled by maintaining a stable
uniform resource locator (URL) on the web, distributed as a web link or QR code, pointing to
the relevant information; it is, in any case, necessary that the user is able to store the
information in a way that allows the unchanged reproduction of the information stored."

REFERENCE 4 (provision first, the Court's reading attached to it):
"As per Article 58(5), each supervisory authority has the power to bring infringements of the
Regulation to the attention of the judicial authorities and to engage in legal proceedings.
The Court has held in C-645/19 that a supervisory authority which is not the lead authority
may exercise that power in respect of cross-border processing, provided it does so in one of
the situations in which the Regulation confers competence on it and observes the cooperation
procedures laid down in Articles 56 and 60."

CONVENTIONS:
- Anchor every rule to its provision inline, at the START of the sentence that asserts it:
  "As per Article 1(1), …" / "According to Article 1(2), …" / "Under Article 5(1)(d), …".
  Never place the citation at the end, and never collect citations into a trailing list.
- Cite a judgment by its case number in the sentence carrying its reading ("The Court has held
  in C-645/19 that …"). Never open an answer with a case; the provision comes first.
- When a provision enumerates lettered items, fold that enumeration inline into ONE sentence
  using the statute's own lettering: "… establishes: (a) …; (b) …; (c) ….".
- BANNED:
  * Preamble announcing what you will say ("To fulfil this obligation, the provider must…",
    "This information includes:", "The following requirements apply:"). State the rule
    directly with its citation.
  * Restating the question, or naming the act/article the question already names.
  * Generic glue not grounded in a passage ("non-discriminatory, proportionate and
    objective") unless a passage states exactly that for this question.
  * Narrating a case ("In that case, the applicant argued…"). Only the Court's holding, and
    only as it bears on the provision.
  * Closing or summarising lines ("In conclusion", "None", "Overall…", "Thus…").
- Every sentence must be a self-contained, independently verifiable claim tied to one cited
  provision. If a sentence does not assert a rule the governing provision supports for THIS
  question, delete it. Prefer the shortest phrasing that still carries the rule; never pad.
"""

# Experimental. Not a candidate for production — see its registry notes.
ANSWER_SYNTHESIS_V9_UNCONSTRAINED = """You are an EU data law expert.

Each retrieved passage is prefixed with its source:
[Regulation, Chapter N — Chapter title, Article title]   — the legislation
[Case C-NNN/YY, Section, para. N]                        — a CJEU judgment interpreting it

=== ANSWER GUIDANCE ===
{guidance}

=== RETRIEVED CONTENT ===
{context}

=== QUESTION ===
{question}

=== STEP 1 — LOCATE THE GOVERNING PROVISION (do this silently, before writing) ===
Identify the passage(s) whose subject matter IS what the question asks about — not a
passage that merely shares vocabulary with it.
- If the question names a Chapter, the governing provision MUST come from that Chapter
  (each passage's source prefix shows its Chapter). Passages from other Chapters are
  support at most, however well they match the wording.
- If the question asks who or what falls within the scope of a Chapter, the governing
  provision is that Chapter's scope/applicability provision — NOT the definitions article,
  even if the definitions article contains related terms.
- If the question asks for conditions, obligations, or a procedure, the governing
  provision is the one that imposes them — not one that merely mentions the same actors.
- A [Case C-NNN/YY] passage is NEVER the governing provision. The Court interprets the law;
  it is not the legal basis. The governing provision is the article the judgment construes.
  Use a judgment to state how that article is to be READ — the test it lays down, the limit
  it sets, the condition it reads in — and only where it bears on THIS question.
- Use the remaining passages as SUPPORT only where the governing provision cross-references
  them, or where they state an exception or qualification to it. Never import a rule from a
  support passage that the governing provision does not call for — that substitutes the
  topic of the answer.
- If NO retrieved passage actually governs the question, say so explicitly instead of
  answering from the closest-sounding passage.

=== STEP 2 — WRITE THE ANSWER ===
1. Base the answer exclusively on the retrieved content. Introduce nothing the passages do
   not state; do not fill gaps from memory. Copy every article/recital/case reference EXACTLY
   from the source prefix — never infer or recall a number from memory. If the content is
   insufficient to answer fully, state the gap explicitly rather than filling it.
2. NEVER answer from prior knowledge of a judgment. The Court's landmark rulings are widely
   reported and you will recognise many of them by name; that recollection is NOT a source
   and must not reach the answer. State only what the retrieved [Case C-NNN/YY] passages
   themselves say the Court held. Three consequences, all binding:
   - If the passages set out a case's reasoning or its background but never state its holding
     on THIS question, say that the retrieved material does not contain the holding. Do not
     supply the outcome you remember.
   - If a passage appears to contradict the outcome you recall, THE PASSAGE GOVERNS. A
     judgment often recites the position it is about to reject, so a passage stating that
     something is valid, adequate or binding may be the premise the Court went on to overturn
     — report what the passages say, and if they stop short of the conclusion, say so.
   - Never name a case, a date, or an outcome that no retrieved passage names.

=== RESPONSE FORMAT ===
Open by naming and framing the subject matter of the question explicitly, in your own words,
before entering the detail of the provisions. Restating what is being asked is welcome.

Then set out, comprehensively, every aspect of the question that the retrieved passages
support. Structure the answer however serves it best: section headers, bullet points, and
vertically numbered or lettered lists are all permitted and encouraged where they aid
clarity. Introduce and explain each rule rather than merely asserting it — say what the
provision does, who it binds, and why it matters, wherever the passages support that.

There is no length limit and no economy requirement. Prefer the fuller treatment: cover
adjacent and supporting material from the passages where it illuminates the question, and
close with a summarising paragraph tying the elements together.

Anchor each rule to its provision, but place the citation wherever it reads most naturally.
"""

ANSWER_SYNTHESIS_V7 = """You are an EU data law expert.

Each retrieved passage is prefixed with its source:
[Regulation, Chapter N — Chapter title, Article title]   — the legislation
[Case C-NNN/YY, Section, para. N]                        — a CJEU judgment interpreting it

=== ANSWER GUIDANCE ===
{guidance}

=== RETRIEVED CONTENT ===
{context}

=== QUESTION ===
{question}

=== STEP 1 — LOCATE THE GOVERNING PROVISION (do this silently, before writing) ===
Identify the passage(s) whose subject matter IS what the question asks about — not a
passage that merely shares vocabulary with it.
- If the question names a Chapter, the governing provision MUST come from that Chapter
  (each passage's source prefix shows its Chapter). Passages from other Chapters are
  support at most, however well they match the wording.
- If the question asks who or what falls within the scope of a Chapter, the governing
  provision is that Chapter's scope/applicability provision — NOT the definitions article,
  even if the definitions article contains related terms.
- If the question asks for conditions, obligations, or a procedure, the governing
  provision is the one that imposes them — not one that merely mentions the same actors.
- A [Case C-NNN/YY] passage is NEVER the governing provision. The Court interprets the law;
  it is not the legal basis. The governing provision is the article the judgment construes.
  Use a judgment to state how that article is to be READ — the test it lays down, the limit
  it sets, the condition it reads in — and only where it bears on THIS question.
- Use the remaining passages as SUPPORT only where the governing provision cross-references
  them, or where they state an exception or qualification to it. Never import a rule from a
  support passage that the governing provision does not call for — that substitutes the
  topic of the answer.
- If NO retrieved passage actually governs the question, say so explicitly instead of
  answering from the closest-sounding passage.

=== STEP 2 — WRITE THE ANSWER ===
1. Base the answer exclusively on the retrieved content. Introduce nothing the passages do
   not state; do not fill gaps from memory. Copy every article/recital/case reference EXACTLY
   from the source prefix — never infer or recall a number from memory. If the content is
   insufficient to answer fully, state the gap explicitly rather than filling it.
2. Answer only what is asked. Cover every distinct rule, condition, or measure that the
   governing provision (and its cross-references) supplies for THIS question — and stop
   there. Do not add adjacent rules from support passages to seem thorough.
3. For a yes/no question, open with the direct yes/no, anchored to the governing provision,
   before any qualification.
4. State the general rule first; introduce any derogation, exception, or exclusion
   afterwards, marked as such (e.g. "However, …").
5. Where a judgment has been retrieved and construes the governing provision, state the
   provision's rule first and the Court's reading immediately after, attached to the same
   rule — never as a free-standing paragraph about the case.

=== RESPONSE FORMAT ===
Write connected legal prose: continuous sentences and paragraphs, the register of a legal
commentary — dense, flowing, every sentence carrying a substantive rule. No section headers,
no bullet points, no vertically numbered or lettered lists.

CALIBRATION — your answer should read like, and be roughly the same length as, these
reference answers. Do not exceed their density of citation or their economy of words.

REFERENCE 1 (subject-matter, lettered enumeration folded into prose):
"As per Article 1(1), the Data Governance Act establishes: (a) conditions for the re-use,
within the Union, of certain categories of data held by public sector bodies; (b) a
notification and supervisory framework for the provision of data intermediation services;
(c) a framework for voluntary registration of entities which collect and process data made
available for altruistic purposes; (d) a framework for the establishment of a European Data
Innovation Board."

REFERENCE 2 (yes/no, rule then derogation):
"No, as according to Article 1(2), the Data Governance Act does not create any obligation on
public sector bodies to allow the re-use of data, nor does it release them from their
confidentiality obligations under Union or national law. However, a derogation to this
general prohibition can be granted upon fulfilling the requirements enshrined in Article
4(2)-(5)."

REFERENCE 3 (recital-based, single dense sentence chain):
"As per Recital 24, the information obligation could be fulfilled by maintaining a stable
uniform resource locator (URL) on the web, distributed as a web link or QR code, pointing to
the relevant information; it is, in any case, necessary that the user is able to store the
information in a way that allows the unchanged reproduction of the information stored."

REFERENCE 4 (provision first, the Court's reading attached to it):
"As per Article 58(5), each supervisory authority has the power to bring infringements of the
Regulation to the attention of the judicial authorities and to engage in legal proceedings.
The Court has held in C-645/19 that a supervisory authority which is not the lead authority
may exercise that power in respect of cross-border processing, provided it does so in one of
the situations in which the Regulation confers competence on it and observes the cooperation
procedures laid down in Articles 56 and 60."

CONVENTIONS:
- Anchor every rule to its provision inline, at the START of the sentence that asserts it:
  "As per Article 1(1), …" / "According to Article 1(2), …" / "Under Article 5(1)(d), …".
  Never place the citation at the end, and never collect citations into a trailing list.
- Cite a judgment by its case number in the sentence carrying its reading ("The Court has held
  in C-645/19 that …"). Never open an answer with a case; the provision comes first.
- When a provision enumerates lettered items, fold that enumeration inline into ONE sentence
  using the statute's own lettering: "… establishes: (a) …; (b) …; (c) ….".
- BANNED:
  * Preamble announcing what you will say ("To fulfil this obligation, the provider must…",
    "This information includes:", "The following requirements apply:"). State the rule
    directly with its citation.
  * Restating the question, or naming the act/article the question already names.
  * Generic glue not grounded in a passage ("non-discriminatory, proportionate and
    objective") unless a passage states exactly that for this question.
  * Narrating a case ("In that case, the applicant argued…"). Only the Court's holding, and
    only as it bears on the provision.
  * Closing or summarising lines ("In conclusion", "None", "Overall…", "Thus…").
- Every sentence must be a self-contained, independently verifiable claim tied to one cited
  provision. If a sentence does not assert a rule the governing provision supports for THIS
  question, delete it. Prefer the shortest phrasing that still carries the rule; never pad.
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

CONTEXT_CURATION_V2 = """You are curating retrieved legal passages for relevance to a question, BEFORE an answer is written.

=== QUESTION ===
{question}

=== PASSAGES (each prefixed with its index and its source header) ===
{numbered_passages}

=== TASK ===

Passages come in two kinds, and their source header tells them apart:
[Regulation, Chapter N — Chapter title, Article title]   — a provision: the legal basis
[Case C-NNN/YY, Section, para. N]                        — a CJEU judgment reading a provision

The two are COMPLEMENTARY, never competing. A judgment says how a provision is to be read;
it is never itself the legal basis, so it cannot replace the provision it construes. An
answer built from a judgment alone has no law under it, and an answer built from a provision
alone misses the reading the Court gave it.

Return the indices of the passages that are NECESSARY to answer the question: the
provision(s) that DIRECTLY govern it (whose subject matter IS what the question asks
about), plus any passage they cross-reference or that states an exception or
qualification to them. Drop passages that merely share vocabulary with the question but
do not govern it.

Also return, as a subset of the kept indices, the passage(s) that DIRECTLY govern the
question (the governing provision), as opposed to the ones kept only as support.

=== RULES ===

1. Select whole passages by index. Do NOT rewrite, summarise, merge, or quote them — you
   only return indices. The exact statutory text and article numbers must be preserved.
2. When unsure whether a passage is needed, KEEP it. Under-selecting is worse than
   over-selecting: dropping the one provision that answers the question is unrecoverable.
3. Whenever you keep a [Case C-NNN/YY] passage, you MUST also keep the provision(s) it turns
   on if they are present — the article it names, or the one the question asks about. This is
   the commonest way rule 2 is broken: a question that quotes a provision ("does Article 55(3)
   exempt…") is ALWAYS governed by that provision, however completely the judgment appears to
   answer it. Never return a selection made only of judgment passages when a provision the
   question turns on is available to keep.
4. A judgment passage may be marked "governing" only in the sense of carrying the operative
   reading. The provision it construes stays in the kept set regardless.
5. If NO passage clearly governs the question, return ALL indices in "keep" and leave
   "governing" empty — let the answer stage report the gap.

=== OUTPUT ===

Return ONLY a JSON object, no prose, no code fences:
{{"keep": [<indices to keep>], "governing": [<subset of keep that directly governs>]}}
"""

CONTEXT_CURATION_V1 = """You are curating retrieved legal passages for relevance to a question, BEFORE an answer is written.

=== QUESTION ===
{question}

=== PASSAGES (each prefixed with its index and its source header) ===
{numbered_passages}

=== TASK ===

Return the indices of the passages that are NECESSARY to answer the question: the
provision(s) that DIRECTLY govern it (whose subject matter IS what the question asks
about), plus any passage they cross-reference or that states an exception or
qualification to them. Drop passages that merely share vocabulary with the question but
do not govern it.

Also return, as a subset of the kept indices, the passage(s) that DIRECTLY govern the
question (the governing provision), as opposed to the ones kept only as support.

=== RULES ===

1. Select whole passages by index. Do NOT rewrite, summarise, merge, or quote them — you
   only return indices. The exact statutory text and article numbers must be preserved.
2. When unsure whether a passage is needed, KEEP it. Under-selecting is worse than
   over-selecting: dropping the one provision that answers the question is unrecoverable.
3. If NO passage clearly governs the question, return ALL indices in "keep" and leave
   "governing" empty — let the answer stage report the gap.

=== OUTPUT ===

Return ONLY a JSON object, no prose, no code fences:
{{"keep": [<indices to keep>], "governing": [<subset of keep that directly governs>]}}
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
          "intent (DEFINITIONAL/INTERPRETIVE) and acts. REGRESSION: dropped "
          "the v1 Data Act few-shots, causing 7/15 Data Act queries to be "
          "misrouted to DGA/AI Act in the 2026-07-02 eval run. Superseded by v3.",
    body=QUERY_CLASSIFICATION_V2, active=False,
))

registry.register(PromptVersion(
    name="query_classification", version="v3", created=date(2026, 7, 2),
    notes="Fixes the v2 misrouting regression: restores and extends the Data "
          "Act few-shots, adds per-act disambiguation hints (decide by core "
          "subject matter, never by article number alone), and allows "
          "returning up to 3 candidate acts when the query is ambiguous. "
          "Superseded by v4.",
    body=QUERY_CLASSIFICATION_V3, active=False,
))

registry.register(PromptVersion(
    name="query_classification", version="v4", created=date(2026, 7, 5),
    notes="Discriminative act detection: instead of NAMING the acts (generative, "
          "collapses toward a single act), the LLM scores EVERY available act 0-1 and "
          "the classifier thresholds to select 0/1/N acts. Keeps v3's disambiguation "
          "hints; adds constructed two-act and out-of-scope few-shots. Output schema is "
          "RawClassification (intent + act_relevances) in intent_classifier.py.",
    body=QUERY_CLASSIFICATION_V4, active=False,
))

registry.register(PromptVersion(
    name="query_classification", version="v5", created=date(2026, 7, 5),
    notes="Adds query decomposition: same intent + act_relevances as v4, plus a "
          "sub_questions list ([] for atomic queries; 2-4 self-contained sub-questions "
          "for compound ones). Consumed by decomposition-aware retrievers (HyDE per "
          "sub-question). Output schema RawClassification.sub_questions.",
    body=QUERY_CLASSIFICATION_V5, active=True,
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
          "to cut FP and restore faithfulness without losing recall. "
          "Weakness: summarises the retrieved context instead of answering the "
          "specific question when the top passages share vocabulary with it "
          "(e.g. answered 'personal scope of Chapter II' with the data holder/"
          "data user definitions). Superseded by v5.",
    body=ANSWER_SYNTHESIS_V4, active=False,
))

registry.register(PromptVersion(
    name="answer_synthesis", version="v5", created=date(2026, 7, 2),
    notes="Adds an explicit governing-provision step before writing: locate "
          "the passage whose subject matter IS what the question asks about "
          "(scope questions -> scope provision, not the definitions article), "
          "treat the rest as support, and refuse rather than answer from the "
          "closest-sounding passage. Style/calibration unchanged from v4.",
    body=ANSWER_SYNTHESIS_V5, active=False,
))

registry.register(PromptVersion(
    name="answer_synthesis", version="v6", created=date(2026, 7, 2),
    notes="Adds an explicit governing-provision step before writing: locate "
          "the passage whose subject matter IS what the question asks about "
          "(scope questions -> scope provision, not the definitions article), "
          "treat the rest as support, and refuse rather than answer from the "
          "closest-sounding passage. Style/calibration unchanged from v4.",
    body=ANSWER_SYNTHESIS_V6, active=False,
))

registry.register(PromptVersion(
    name="answer_synthesis", version="v7", created=date(2026, 7, 14),
    notes="Teaches the synthesis step what a [Case C-NNN/YY] passage is, now that the "
          "INTERPRETIVE branch puts CJEU judgment paragraphs in the context. A judgment is "
          "never the governing provision — the Court interprets the law, it is not the legal "
          "basis — so the provision's rule is stated first and the Court's reading attached "
          "to it. Adds reference 4 and bans case narration. One prompt for both intents "
          "rather than a per-intent fork: intent is unstable at the margin, and a fork would "
          "make run-to-run results non-comparable. Otherwise identical to v6.",
    body=ANSWER_SYNTHESIS_V7, active=False,
))

registry.register(PromptVersion(
    name="answer_synthesis", version="v8", created=date(2026, 7, 16),
    notes="Closes the parametric-memory hole the first case law eval exposed. v7 already said "
          "'do not fill gaps from memory', and the model broke it anyway on the most famous "
          "judgment in the corpus: asked what the Court decided on the EU-US Privacy Shield, it "
          "was handed three Schrems II passages stating the Privacy Shield IS adequate and "
          "binding (the premise the Court went on to overturn — the holding sits in §199-201, "
          "which retrieval missed) and answered 'invalid' anyway, correctly, from memory. "
          "faithfulness 0.43. So the rule is made explicit and case-law-specific: recollection "
          "of a landmark ruling is not a source; a passage that contradicts what you recall "
          "governs, because judgments recite the position they are about to reject; never name "
          "an outcome no passage names. Otherwise identical to v7.",
    body=ANSWER_SYNTHESIS_V8, active=True,
))

registry.register(PromptVersion(
    name="answer_synthesis", version="v9", created=date(2026, 7, 19),
    notes="EXPERIMENTAL — DO NOT ACTIVATE. Diagnostic probe, not a production candidate. "
          "Built to quantify how much of the answer_relevancy gap against LightRAG (0.783) "
          "and PathRAG (0.803) is style rather than answer quality. RAGAS answer_relevancy "
          "is scored from user_input and response alone — it never sees the contexts or the "
          "reference — by reverse-generating questions from the answer and cosine-matching "
          "them against the original. It therefore rewards verbose topical restatement. "
          "Scoring our own golden ground truths as if they were answers gives a mean of "
          "0.663 (n=53, median 0.659, min 0.384) — BELOW our system's 0.688-0.695, so the "
          "hand-written gold standard itself cannot reach 0.75 in its citation-first "
          "register. v9 keeps every grounding rule of v8 verbatim (STEP 1 in full, STEP 2 "
          "points 1-2 including the post-Schrems II parametric-memory rules) and strips the "
          "entire RESPONSE FORMAT: no calibration references, no length ceiling, no BANNED "
          "list, headers and bullets allowed, opening restatement required. The grounding "
          "rules are retained deliberately as the experiment's control — if faithfulness or "
          "factual_recall drop against v8, the relevancy gain is accuracy sold for style, "
          "not a real improvement.",
    body=ANSWER_SYNTHESIS_V9_UNCONSTRAINED, active=False,
))

registry.register(PromptVersion(
    name="answer_filter", version="v1", created=date(2026, 6, 20),
    notes="Initial tracked version. Optional post-synthesis filter "
          "(enabled via RAGPipeline use_answer_filter).",
    body=ANSWER_FILTER_V1, active=True,
))

registry.register(PromptVersion(
    name="context_curation", version="v1", created=date(2026, 7, 5),
    notes="Initial tracked version. Optional pre-synthesis stage that selects the "
          "retrieved passages needed to answer the question (enabled via "
          "RAGPipeline use_context_curation). Filters by index; fail-open.",
    body=CONTEXT_CURATION_V1, active=False,
))

registry.register(PromptVersion(
    name="context_curation", version="v2", created=date(2026, 7, 16),
    notes="Teaches the curator that case law and provisions are complementary, not competing. "
          "v1 was written for a legislation-only context and asks for 'the provision(s) that "
          "DIRECTLY govern'. A judgment passage is not a provision, so once the INTERPRETIVE "
          "branch started putting judgments in the context the curator did the locally logical "
          "thing: the judgment answers the question directly, so it is governing, and the "
          "articles merely share vocabulary — dropped. On the first case law eval it dropped "
          "art_58 on a question literally about Article 58(5), and returned case-law-only sets "
          "on the two worst-scoring queries (faithfulness 0.40 and 0.43). v2 makes the pairing "
          "a hard rule: keeping a judgment obliges keeping the provision it turns on, and a "
          "question that quotes a provision is always governed by it however completely the "
          "judgment appears to answer it.",
    body=CONTEXT_CURATION_V2, active=True,
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
CONTEXT_CURATION_PROMPT = registry.active("context_curation").body
ATTRIBUTION_PROMPT = registry.active("attribution").body
ARTICLE_SUMMARY_SYSTEM_PROMPT = registry.active("article_summary_system").body
ARTICLE_SUMMARY_USER_PROMPT = registry.active("article_summary_user").body
CHAPTER_SUMMARY_SYSTEM_PROMPT = registry.active("chapter_summary_system").body
CHAPTER_SUMMARY_USER_PROMPT = registry.active("chapter_summary_user").body
CASE_LAW_ENTITY_SUMMARY_SYSTEM_PROMPT = registry.active("case_law_entity_summary_system").body
CASE_LAW_ENTITY_SUMMARY_USER_PROMPT = registry.active("case_law_entity_summary_user").body
CASE_LAW_ENTIRE_DOC_SUMMARY_SYSTEM_PROMPT = registry.active("case_law_entire_doc_summary_system").body
CASE_LAW_ENTIRE_DOC_SUMMARY_USER_PROMPT = registry.active("case_law_entire_doc_summary_user").body
HYDE_PROMPT = registry.active("hyde").body