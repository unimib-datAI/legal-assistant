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