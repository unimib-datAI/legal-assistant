"""Prompts for the retrieval stage: query classification, topic selection, HyDE."""
from datetime import date

from legal_assistant.rag.prompts.registry import PromptVersion, registry


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

HYDE_V1 = """You are an expert in EU digital regulation. Write a short, factual passage in the style of an article or 
recital of the relevant EU act that directly answers the question below. Write as if quoting the legislation itself2: 
precise, normative, self-contained. Do not add disclaimers or meta-commentary.

Relevant act(s): {acts}

Question: {query}

Hypothetical legal passage:"""

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
    name="hyde", version="v1", created=date(2026, 6, 27),
    notes="Initial HyDE prompt. Generates an act-grounded hypothetical legal "
          "passage used as the dense-search query.",
    body=HYDE_V1, active=True,
))
