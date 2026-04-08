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

SOCRATIC_KG_QA_GENERATION_PROMPT = """
## ROLE
You are a **Comprehensive Knowledge Archivist** who converts the [Full Document] into detailed,
document-grounded QA pairs.

## OBJECTIVE
Extract as many meaningful Question-Answer pairs as possible from the document.
Use the 5W1H perspectives (Who, What, When, Where, Why, How) **as analytical lenses** to help you
identify and expand potential questions, but do NOT restrict yourself to producing only
5W1H-type questions.
Your goal is to maximize informational coverage, capturing every explicit fact, relation, event,
definition, rationale, and process described in the document.

## INPUT
Full Document: "{document_text}"

## CONSTRAINTS
1. **Context-Independent**
    - Each QA must be self-contained and understandable without referencing the original text.
    - Replace pronouns with explicit entities.

2. **No Hallucination**
    - Use only facts explicitly stated in the document.
    
3. **Expansion-Oriented Thinking**
- For each sentence or factual unit, consider the 5W1H perspectives as prompts to explore:
    - WHO is involved?
    - WHAT happened or is described?
    - WHEN did it occur?
    - WHERE did it occur?
    - WHY did it occur?
    - HOW was it carried out?
    - These perspectives are **guides** to inspire multiple possible QA pairs, even if they are implicit or only partially expressed.
    
4. **Coverage**
    - Extract all possible QA pairs that can be reasonably derived from the document.
    
## OUTPUT FORMAT
Return a JSON list of QA objects:
    [
        {{"question": "...", "answer": "..."}},
        ...
    ]
"""

SOCRATIC_KG_QA_PROCEDURAL_STEPS_PROMPT = """
## ROLE
You are a **Document-Grounded QA Extractor**.

## OBJECTIVE
Convert the full document into high-coverage, explicit-fact QA pairs.

## PROCEDURE
1. Read the document end-to-end.
2. Segment into atomic factual units.
3. For each unit:
    - Generate QAs that capture all explicit information it contains.
    - When forming questions, view the unit through the 5W1H angles (Who, What, When, Where, Why, How) so that different aspects of the same fact can be covered.
4. Merge duplicates and keep the most precise wording.

## INPUT
Full Document: "{document_text}"

## CONSTRAINTS
- Context-Independent QAs only.
- No Hallucination.
- Prefer concise but complete answers.

## OUTPUT FORMAT
Return a JSON list:
[
    {{"question": "...", "answer": "..."}},
...
]
"""

SOCRATIC_TRIPLE_EXTRACTION_PROMPT = """
## ROLE
You are a Semantic Knowledge Graph Builder.
Extract every structured triples (entity1, relation, entity2) from the Q&A pair, following the rules below.

## GOAL
From the question-answer pair, extract only useful, knowledge-ready triples that can serve as entries in a semantic knowledge graph.

## RULES
Extract clean (subject, relation, object) triples following the rules:

1. Split every stated or clearly implied fact into minimal triples; integrate question and answer context when needed.

2. Entities (entity1, entity2) must be short, concrete noun phrases.
    - No pronouns (this, that, it, its, these, those, etc.).
    - Entities must not be unresolved or reference-based pronouns (\eg those, they, someone, anyone, whoever); if such 
    a pronoun appears, rewrite it into a specific, explicit noun phrase or skip the triple.
    - No clauses or relative clauses (no "who/that/which/what/as it ..." inside an entity).
    - No long gerund or sentence-like phrases. If a phrase contains a verb or clause marker, rewrite it into a concise noun concept or skip the triple.
    
3. Relations must be short, canonical verbs or verb phrases.
    - Express a single semantic link between the two entities (\eg causes, leads to, supports, believes, opposes).
    - Must be a compact predicate, not a sentence fragment.
    - No pronouns or clause markers inside the relation (no "its", "that", "as it", "what", etc.).
    - If the source uses an idiomatic or long expression, rewrite it into a simple canonical relation without pronouns or embedded clauses, or skip the triple.
    
4. Include a fact if it can be clearly rewritten into a concise, explicit triple that fits the rules above; otherwise skip it.

5. Output only concise, interpretable, knowledge-ready triples.

## INPUT
Q: {question}
A: {answer}

## OUTPUT FORMAT (JSON List)
- Return a list of JSON objects.
- Return [] if no valid triples exist.

[
    {{"entity1": "Specific_Noun", "relation": "precise_verb_phrase", "entity2": "Specific_Noun"}}
]

"""

GRAPH_RAG_MICROSOFT_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
Identify all entities. For each identified entity, extract the following information:

entity_name: Name of the entity, capitalized
entity_type: One of the following types: [{entity_types}]
entity_description: Comprehensive description of the entity's attributes and activities

Format each entity with the following JSON structure:
{   
    "entity": "entity_name",
    "type": "entity_type",
    "description": "entity_description"
}
    
From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are "clearly related" to each other.
For each pair of related entities, extract the following information:

source_entity: name of the source entity, as identified in step 1
target_entity: name of the target entity, as identified in step 1
relationship_description: explanation as to why you think the source entity and the target entity are related to each other
relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity

Format each relationship with the following JSON structure:
{
    "source": "source_entity",
    "target": "target_entity",
    "description": "relationship_description",
    "strength": relationship_strength
}

Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Output structure must
respect the following JSON format:

{
  "chunk_id": "62021CJ0252_chunk_042",
  "chunk_text": "The request has been made in proceedings between Meta Platforms Inc. and the Bundeskartellamt...",
  "entities": [
    {
      "name": "META PLATFORMS INC.",
      "type": "APPLICANT",
      "description": "Meta Platforms Inc. is the applicant challenging a prohibition decision by the Bundeskartellamt regarding personal data processing"
    },
    {
      "name": "BUNDESKARTELLAMT",
      "type": "NATIONAL_AUTHORITY",
      "description": "The Federal Cartel Office of Germany that prohibited Meta from processing personal data under its general terms of use"
    }
  ],
  "relationships": [
    {
      "source": "META PLATFORMS INC.",
      "target": "BUNDESKARTELLAMT",
      "description": "Meta challenges the Bundeskartellamt prohibition decision as defendant in national proceedings",
      "strength": 9
    }
  ]
}

It is important to note that the final JSON file will be a list of objects whose format is the following:

[
  { "chunk_id": "62021CJ0252_000", "chunk_text": "...", "entities": [], "relationships": [] },
  { "chunk_id": "62021CJ0252_001", "chunk_text": "...", "entities": [], "relationships": [] }
]

---Examples---
Entity types: ORGANIZATION, PERSON

Input:
The Fed is scheduled to meet on Tuesday and Wednesday, with the central bank 
planning to release its latest policy decision on Wednesday at 2:00 p.m. ET, followed 
by a press conference where Fed Chair Jerome Powell will take questions. Investors expect 
the Federal Open Market Committee to hold its benchmark interest rate steady in a range 
of 5.25%-5.5%.

Output:
{
  "chunk_id": "62019CJ0645_chunk_000",
  "chunk_text": "The Fed is scheduled to meet on Tuesday and Wednesday, with the central bank \nplanning to release its latest policy decision on Wednesday at 2:00 p.m. ET, followed \nby a press conference where Fed Chair Jerome Powell will take questions. Investors expect \nthe Federal Open Market Committee to hold its benchmark interest rate steady in a range \nof 5.25%-5.5%.",
  "entities": [
    {
      "name": "FED",
      "type": "ORGANIZATION",
      "description": "The Fed is the Federal Reserve, which is setting interest rates on Tuesday and Wednesday"
    },
    {
      "name": "JEROME POWELL",
      "type": "PERSON",
      "description": "Jerome Powell is the chair of the Federal Reserve"
    },
    {
      "name": "FEDERAL OPEN MARKET COMMITTEE",
      "type": "ORGANIZATION",
      "description": "The Federal Reserve committee makes key decisions about interest rates and the growth of the United States money supply"
    }
  ],
  "relationships": [
    {
      "source": "JEROME POWELL",
      "target": "FED",
      "description": "Jerome Powell is the Chair of the Federal Reserve and will answer questions at a press conference",
      "strength": 9
    }
  ]
}

...More examples...
---Real Data---

Entity types: {entity_types}
Input:
{input_text}
Output:

"""

GRAPH_RAG_MICROSOFT_PROMPT_v2 = """
-Goal-
Given a text document from a Court of Justice of the European Union (CJEU) case law judgment,
identify all entities of the specified types and all relationships among them.

-Steps-

1. Identify all entities. For each entity extract:
   - name: capitalized, always SPECIFIC (never generic names like "REGULATION", "COURT", 
     "SUPERVISORY AUTHORITY" — always use the full specific name e.g. 
     "REGULATION (EU) 2016/679", "COURT OF JUSTICE (GRAND CHAMBER)", 
     "GEGEVENSBESCHERMINGSAUTORITEIT")
   - type: one of the provided entity types
   - description: comprehensive description of the entity's role in this specific case

2. Pay special attention to the document section you are processing:
   - "Legal context" section → focus on REGULATION, DIRECTIVE, ARTICLE, RECITAL, TREATY_PROVISION
   - "Dispute in main proceedings" section → focus on APPLICANT, DEFENDANT, REFERRING_COURT,
     FACTUAL_CIRCUMSTANCE, LEGAL_ISSUE, ADMINISTRATIVE_DECISION
   - "Consideration of questions referred" section → focus on PRELIMINARY_QUESTION, HOLDING,
     LEGAL_PRINCIPLE. Every "By its Nth question the referring court seeks to ascertain whether..."
     is a PRELIMINARY_QUESTION. Every "the answer to the Nth question referred is that Article X 
     must be interpreted as meaning..." is a HOLDING.
   - "On those grounds, the Court hereby rules" section → extract one OPERATIVE_PART entity
     per numbered ruling point and one HOLDING entity summarizing the legal interpretation.
   - "Costs" section → extract nothing, skip entirely.

3. From the entities in step 1, identify all clearly related pairs and for each extract:
   - source, target: entity names from step 1
   - description: why they are related in this specific legal context
   - strength: 1-10 (reserve 9-10 only for direct structural relationships like 
     ARTICLE → REGULATION it belongs to, or HOLDING → PRELIMINARY_QUESTION it answers)

-Critical Rules-
- NEVER use generic entity names: always use the full specific identifier
- NEVER invent entity types outside the provided list
- NEVER extract Official Journal references (e.g. "OJ 2016 L 119") as entities
- NEVER extract paragraph numbers (e.g. "- 37", "2.", "3.") as entities
- ALWAYS extract INTERVENER for Member State governments submitting observations
- ALWAYS extract PRELIMINARY_QUESTION and HOLDING in the reasoning sections
- ALWAYS extract OPERATIVE_PART for each numbered ruling in the dispositif

-Output Format-
Return ONLY a valid JSON object with this exact structure, no preamble, no markdown:

{{
  "chunk_id": "{chunk_id}",
  "chunk_text": "{chunk_text}",
  "entities": [
    {{
      "name": "<SPECIFIC ENTITY NAME IN CAPS>",
      "type": "<ENTITY_TYPE>",
      "description": "<description of role in this specific case>"
    }}
  ],
  "relationships": [
    {{
      "source": "<source entity name>",
      "target": "<target entity name>",
      "description": "<why they are related>",
      "strength": <1-10>
    }}
  ]
}}

---Examples---
Entity types: ORGANIZATION, PERSON

Input:
The Fed is scheduled to meet on Tuesday and Wednesday, with the central bank
planning to release its latest policy decision on Wednesday at 2:00 p.m. ET, followed
by a press conference where Fed Chair Jerome Powell will take questions. Investors expect
the Federal Open Market Committee to hold its benchmark interest rate steady in a range
of 5.25%-5.5%.

Output:
{{
  "chunk_id": "example_chunk_000",
  "chunk_text": "The Fed is scheduled to meet on Tuesday and Wednesday...",
  "entities": [
    {{
      "name": "FEDERAL RESERVE",
      "type": "ORGANIZATION",
      "description": "The Federal Reserve (Fed) is the central bank of the United States, setting interest rates"
    }},
    {{
      "name": "JEROME POWELL",
      "type": "PERSON",
      "description": "Jerome Powell is the Chair of the Federal Reserve"
    }},
    {{
      "name": "FEDERAL OPEN MARKET COMMITTEE",
      "type": "ORGANIZATION",
      "description": "The Federal Reserve committee that makes key decisions about interest rates"
    }}
  ],
  "relationships": [
    {{
      "source": "JEROME POWELL",
      "target": "FEDERAL RESERVE",
      "description": "Jerome Powell is the Chair of the Federal Reserve and will answer press conference questions",
      "strength": 9
    }}
  ]
}}

---Legal Domain Example---
Entity types: COURT, REFERRING_COURT, NATIONAL_AUTHORITY, APPLICANT, DEFENDANT, INTERVENER,
REGULATION, ARTICLE, LEGAL_ISSUE, FACTUAL_CIRCUMSTANCE, PRELIMINARY_QUESTION, HOLDING, OPERATIVE_PART

Input:
The request has been made in proceedings between Facebook Ireland Ltd, Facebook Inc. and
Facebook Belgium BVBA, on the one hand, and the Gegevensbeschermingsautoriteit (the Belgian
Data Protection Authority), as the successor of the Commissie ter bescherming van de
Persoonlijke Levenssfeer (the Belgian Privacy Commission), on the other, concerning injunction
proceedings brought by the President of the Privacy Commission seeking to bring to an end the
processing of personal data of internet users within Belgium by the Facebook online social
network, using cookies, social plug-ins and pixels.
The Belgian Government, the Czech Government and the Italian Government submitted observations.
By its first question, the referring court seeks to ascertain whether Article 58(5) of
Regulation 2016/679 must be interpreted as meaning that a non-lead supervisory authority
may bring proceedings before national courts for cross-border data processing.
The answer to the first question referred is that Article 58(5) of Regulation 2016/679
must be interpreted as meaning that a supervisory authority may exercise that power in
one of the situations where that regulation exceptionally confers competence on it.

Output:
{{
  "chunk_id": "legal_example_chunk_000",
  "chunk_text": "The request has been made in proceedings...",
  "entities": [
    {{
      "name": "FACEBOOK IRELAND LTD",
      "type": "APPLICANT",
      "description": "Facebook Ireland Ltd is an applicant challenging the injunction proceedings regarding personal data processing in Belgium"
    }},
    {{
      "name": "FACEBOOK INC.",
      "type": "APPLICANT",
      "description": "Facebook Inc. is the US parent company and applicant in the proceedings against the Belgian DPA"
    }},
    {{
      "name": "FACEBOOK BELGIUM BVBA",
      "type": "APPLICANT",
      "description": "Facebook Belgium BVBA is the local Belgian entity and applicant in the injunction proceedings"
    }},
    {{
      "name": "GEGEVENSBESCHERMINGSAUTORITEIT",
      "type": "NATIONAL_AUTHORITY",
      "description": "The Belgian Data Protection Authority, successor to the Privacy Commission, respondent in the proceedings"
    }},
    {{
      "name": "COMMISSIE TER BESCHERMING VAN DE PERSOONLIJKE LEVENSSFEER",
      "type": "NATIONAL_AUTHORITY",
      "description": "The former Belgian Privacy Commission, predecessor to the Gegevensbeschermingsautoriteit, which initiated the original injunction proceedings"
    }},
    {{
      "name": "BELGIAN GOVERNMENT",
      "type": "INTERVENER",
      "description": "The Belgian Government submitted observations in the proceedings before the Court"
    }},
    {{
      "name": "CZECH GOVERNMENT",
      "type": "INTERVENER",
      "description": "The Czech Government submitted observations in the proceedings before the Court"
    }},
    {{
      "name": "ITALIAN GOVERNMENT",
      "type": "INTERVENER",
      "description": "The Italian Government submitted observations in the proceedings before the Court"
    }},
    {{
      "name": "PROCESSING OF PERSONAL DATA VIA COOKIES AND SOCIAL PLUG-INS",
      "type": "LEGAL_ISSUE",
      "description": "Facebook's collection of data from Belgian internet users through cookies, social plug-ins and pixels without adequate legal basis"
    }},
    {{
      "name": "BELGIAN INTERNET USERS",
      "type": "FACTUAL_CIRCUMSTANCE",
      "description": "Internet users located within Belgium whose personal data was being processed by Facebook"
    }},
    {{
      "name": "ARTICLE 58(5) OF REGULATION 2016/679",
      "type": "ARTICLE",
      "description": "Provision granting supervisory authorities the power to bring infringements to national courts and initiate legal proceedings"
    }},
    {{
      "name": "Q1 - NON-LEAD SA COMPETENCE FOR CROSS-BORDER PROCEEDINGS",
      "type": "PRELIMINARY_QUESTION",
      "description": "Whether Article 58(5) GDPR must be interpreted as meaning a non-lead supervisory authority may bring proceedings before national courts for cross-border data processing"
    }},
    {{
      "name": "HOLDING Q1 - NON-LEAD SA MAY BRING CROSS-BORDER PROCEEDINGS",
      "type": "HOLDING",
      "description": "Article 58(5) GDPR must be interpreted as meaning a supervisory authority may exercise that power in situations where the regulation exceptionally confers competence on it, provided cooperation procedures are respected"
    }}
  ],
  "relationships": [
    {{
      "source": "FACEBOOK IRELAND LTD",
      "target": "GEGEVENSBESCHERMINGSAUTORITEIT",
      "description": "Facebook Ireland Ltd is subject to injunction proceedings initiated by the Belgian DPA",
      "strength": 9
    }},
    {{
      "source": "GEGEVENSBESCHERMINGSAUTORITEIT",
      "target": "COMMISSIE TER BESCHERMING VAN DE PERSOONLIJKE LEVENSSFEER",
      "description": "The Gegevensbeschermingsautoriteit is the legal successor to the former Belgian Privacy Commission",
      "strength": 10
    }},
    {{
      "source": "BELGIAN GOVERNMENT",
      "target": "GEGEVENSBESCHERMINGSAUTORITEIT",
      "description": "The Belgian Government submitted observations supporting the authority's position",
      "strength": 7
    }},
    {{
      "source": "PROCESSING OF PERSONAL DATA VIA COOKIES AND SOCIAL PLUG-INS",
      "target": "BELGIAN INTERNET USERS",
      "description": "The illegal data processing directly affects Belgian internet users whose data is collected without legal basis",
      "strength": 8
    }},
    {{
      "source": "HOLDING Q1 - NON-LEAD SA MAY BRING CROSS-BORDER PROCEEDINGS",
      "target": "Q1 - NON-LEAD SA COMPETENCE FOR CROSS-BORDER PROCEEDINGS",
      "description": "This holding directly answers the first preliminary question referred by the national court",
      "strength": 10
    }},
    {{
      "source": "HOLDING Q1 - NON-LEAD SA MAY BRING CROSS-BORDER PROCEEDINGS",
      "target": "ARTICLE 58(5) OF REGULATION 2016/679",
      "description": "The holding interprets the scope and conditions of Article 58(5) GDPR",
      "strength": 9
    }}
  ]
}}

---Real Data---
Entity types: {entity_types}
chunk_id: {chunk_id}

Input:
{input_text}

Output:
"""

GRAPH_RAG_MICROSOFT_PROMPT_v3 = """
-Goal-
Given a text document from a Court of Justice of the European Union (CJEU) case law judgment,
identify all entities of the specified types and all relationships among them.

-Steps-

1. Identify all entities. For each entity extract:
   - name: capitalized, always SPECIFIC (never generic names like "REGULATION", "COURT", 
     "SUPERVISORY AUTHORITY" — always use the full specific name e.g. 
     "REGULATION (EU) 2016/679", "COURT OF JUSTICE (GRAND CHAMBER)", 
     "GEGEVENSBESCHERMINGSAUTORITEIT")
   - type: one of the provided entity types
   - description: comprehensive description of the entity's role in this specific case

2. Pay special attention to the document section you are processing:

   - "Legal context" section → focus on REGULATION, DIRECTIVE, ARTICLE, RECITAL, TREATY_PROVISION.

   - "Dispute in main proceedings" section → focus on APPLICANT, DEFENDANT, REFERRING_COURT,
     FACTUAL_CIRCUMSTANCE, LEGAL_ISSUE, ADMINISTRATIVE_DECISION.

   - "Consideration of questions referred" section → this is the most important section.
     Extract the following entity types with priority:

     a) PRELIMINARY_QUESTION — extract one per referred question using this STRICT naming convention:
        "Q{N} - {BRIEF DESCRIPTION OF LEGAL ISSUE}"
        Example: "Q1 - NON-LEAD SA COMPETENCE FOR CROSS-BORDER PROCEEDINGS"
        This name must be IDENTICAL across all chunks that reference the same question.
        NEVER use alternative names like "FIRST QUESTION REFERRED" or "PRELIMINARY QUESTION 1".

     b) HOLDING — extract one per answered question using this STRICT naming convention:
        "HOLDING Q{N} - {BRIEF DESCRIPTION OF COURT ANSWER}"
        Example: "HOLDING Q1 - NON-LEAD SA MAY BRING CROSS-BORDER PROCEEDINGS"
        This name must be IDENTICAL across all chunks that reference the same holding.
        The trigger phrase is: "the answer to the Nth question referred is that..."

     c) LEGAL_PRINCIPLE — extract named legal doctrines and principles established or applied
        in the reasoning. These are concepts broader than a single article. Examples:
        "ONE-STOP SHOP MECHANISM" — the GDPR principle allocating competence to a lead SA
        "SINCERE AND EFFECTIVE COOPERATION" — the duty of non-lead SAs toward the lead SA
        "DIRECT EFFECT" — the principle that an EU provision is directly applicable in national law
        "TERRITORIAL SCOPE OF REGULATION (EU) 2016/679" — the geographic application rule

   - "On those grounds, the Court hereby rules" section → extract one OPERATIVE_PART entity
     per numbered ruling point using this naming convention:
     "OPERATIVE PART - RULING {N}"
     and link it to the corresponding HOLDING Q{N} entity.

   - "Costs" section → extract nothing, skip entirely.

3. From the entities in step 1, identify all clearly related pairs and for each extract:
   - source, target: entity names from step 1
   - description: why they are related in this specific legal context
   - strength: 1-10

   Strength calibration guide:
   - 10: direct structural containment (ARTICLE belongs to REGULATION, HOLDING answers QUESTION)
   - 9:  direct interpretation (HOLDING interprets ARTICLE, QUESTION concerns ARTICLE)
   - 8:  strong thematic link (LEGAL_PRINCIPLE applied in HOLDING, LEGAL_ISSUE triggers QUESTION)
   - 7:  supporting/procedural link (INTERVENER submits observations on, RECITAL supports ARTICLE)
   - 6 or below: indirect or contextual links

-Critical Rules-
- NEVER use generic entity names: always use the full specific identifier
- NEVER invent entity types outside the provided list
- NEVER extract Official Journal references (e.g. "OJ 2016 L 119") as entities
- NEVER extract paragraph numbers (e.g. "- 37", "2.", "3.") as entities
- ALWAYS extract INTERVENER for Member State governments submitting observations
- ALWAYS extract PRELIMINARY_QUESTION and HOLDING in the reasoning sections
- ALWAYS extract LEGAL_PRINCIPLE when a named doctrine is applied in the reasoning
- ALWAYS use the canonical Q{N} naming convention for PRELIMINARY_QUESTION and HOLDING
- RELATIONSHIP DIRECTION: a HOLDING interprets an ARTICLE, so the edge goes
  HOLDING → ARTICLE (not ARTICLE → HOLDING). A HOLDING answers a PRELIMINARY_QUESTION,
  so the edge goes HOLDING → PRELIMINARY_QUESTION.

-Output Format-
Return ONLY a valid JSON object with this exact structure, no preamble, no markdown:

{{
  "chunk_id": "{chunk_id}",
  "chunk_text": "{chunk_text}",
  "entities": [
    {{
      "name": "<SPECIFIC ENTITY NAME IN CAPS>",
      "type": "<ENTITY_TYPE>",
      "description": "<description of role in this specific case>"
    }}
  ],
  "relationships": [
    {{
      "source": "<source entity name>",
      "target": "<target entity name>",
      "description": "<why they are related>",
      "strength": <1-10>
    }}
  ]
}}

---Examples---
Entity types: ORGANIZATION, PERSON

Input:
The Fed is scheduled to meet on Tuesday and Wednesday, with the central bank
planning to release its latest policy decision on Wednesday at 2:00 p.m. ET, followed
by a press conference where Fed Chair Jerome Powell will take questions. Investors expect
the Federal Open Market Committee to hold its benchmark interest rate steady in a range
of 5.25%-5.5%.

Output:
{{
  "chunk_id": "example_chunk_000",
  "chunk_text": "The Fed is scheduled to meet on Tuesday and Wednesday...",
  "entities": [
    {{
      "name": "FEDERAL RESERVE",
      "type": "ORGANIZATION",
      "description": "The Federal Reserve (Fed) is the central bank of the United States, setting interest rates"
    }},
    {{
      "name": "JEROME POWELL",
      "type": "PERSON",
      "description": "Jerome Powell is the Chair of the Federal Reserve"
    }},
    {{
      "name": "FEDERAL OPEN MARKET COMMITTEE",
      "type": "ORGANIZATION",
      "description": "The Federal Reserve committee that makes key decisions about interest rates"
    }}
  ],
  "relationships": [
    {{
      "source": "JEROME POWELL",
      "target": "FEDERAL RESERVE",
      "description": "Jerome Powell is the Chair of the Federal Reserve and will answer press conference questions",
      "strength": 9
    }}
  ]
}}

---Legal Domain Example---
Entity types: COURT, REFERRING_COURT, NATIONAL_AUTHORITY, APPLICANT, DEFENDANT, INTERVENER,
REGULATION, ARTICLE, LEGAL_ISSUE, FACTUAL_CIRCUMSTANCE, LEGAL_PRINCIPLE,
PRELIMINARY_QUESTION, HOLDING, OPERATIVE_PART

Input:
The request has been made in proceedings between Facebook Ireland Ltd, Facebook Inc. and
Facebook Belgium BVBA, on the one hand, and the Gegevensbeschermingsautoriteit (the Belgian
Data Protection Authority), as the successor of the Commissie ter bescherming van de
Persoonlijke Levenssfeer (the Belgian Privacy Commission), on the other, concerning injunction
proceedings brought by the President of the Privacy Commission seeking to bring to an end the
processing of personal data of internet users within Belgium by the Facebook online social
network, using cookies, social plug-ins and pixels.
The Belgian Government, the Czech Government and the Italian Government submitted observations.

In that regard, without prejudice to Article 55(1) of Regulation 2016/679, Article 56(1)
establishes the one-stop shop mechanism, based on an allocation of competences between a
lead supervisory authority and the other supervisory authorities concerned.
The lead supervisory authority cannot eschew sincere and effective cooperation with the other
supervisory authorities concerned.

By its first question, the referring court seeks to ascertain whether Article 55(1), Articles
56 to 58 and Articles 60 to 66 of Regulation 2016/679 must be interpreted as meaning that a
supervisory authority of a Member State which is not the lead supervisory authority may
exercise the power in Article 58(5) in relation to cross-border data processing.

The answer to the first question referred is that Article 58(5) of Regulation 2016/679
must be interpreted as meaning that a supervisory authority may exercise that power in
one of the situations where that regulation exceptionally confers competence on it,
provided that the cooperation and consistency procedures are respected.

On those grounds, the Court (Grand Chamber) hereby rules:
1. Article 55(1), Articles 56 to 58 and Articles 60 to 66 of Regulation (EU) 2016/679,
read together with Articles 7, 8 and 47 of the Charter, must be interpreted as meaning
that a non-lead supervisory authority may exercise the power in Article 58(5) in cross-border
cases, provided that power is exercised in one of the situations where that regulation
confers competence on it and cooperation procedures are respected.

Output:
{{
  "chunk_id": "legal_example_chunk_000",
  "chunk_text": "The request has been made in proceedings...",
  "entities": [
    {{
      "name": "FACEBOOK IRELAND LTD",
      "type": "APPLICANT",
      "description": "Facebook Ireland Ltd is an applicant challenging the injunction proceedings regarding personal data processing in Belgium"
    }},
    {{
      "name": "FACEBOOK INC.",
      "type": "APPLICANT",
      "description": "Facebook Inc. is the US parent company and applicant in the proceedings against the Belgian DPA"
    }},
    {{
      "name": "FACEBOOK BELGIUM BVBA",
      "type": "APPLICANT",
      "description": "Facebook Belgium BVBA is the local Belgian entity and applicant in the injunction proceedings"
    }},
    {{
      "name": "GEGEVENSBESCHERMINGSAUTORITEIT",
      "type": "NATIONAL_AUTHORITY",
      "description": "The Belgian Data Protection Authority, successor to the Privacy Commission, respondent in the proceedings"
    }},
    {{
      "name": "COMMISSIE TER BESCHERMING VAN DE PERSOONLIJKE LEVENSSFEER",
      "type": "NATIONAL_AUTHORITY",
      "description": "The former Belgian Privacy Commission, predecessor to the Gegevensbeschermingsautoriteit"
    }},
    {{
      "name": "BELGIAN GOVERNMENT",
      "type": "INTERVENER",
      "description": "The Belgian Government submitted observations in the proceedings before the Court"
    }},
    {{
      "name": "CZECH GOVERNMENT",
      "type": "INTERVENER",
      "description": "The Czech Government submitted observations in the proceedings before the Court"
    }},
    {{
      "name": "ITALIAN GOVERNMENT",
      "type": "INTERVENER",
      "description": "The Italian Government submitted observations in the proceedings before the Court"
    }},
    {{
      "name": "PROCESSING OF PERSONAL DATA VIA COOKIES AND SOCIAL PLUG-INS",
      "type": "LEGAL_ISSUE",
      "description": "Facebook's collection of data from Belgian internet users through cookies, social plug-ins and pixels without adequate legal basis"
    }},
    {{
      "name": "ARTICLE 55(1) OF REGULATION (EU) 2016/679",
      "type": "ARTICLE",
      "description": "General competence rule — each supervisory authority is competent on the territory of its own Member State"
    }},
    {{
      "name": "ARTICLE 56(1) OF REGULATION (EU) 2016/679",
      "type": "ARTICLE",
      "description": "Establishes the lead supervisory authority for cross-border processing under the one-stop shop mechanism"
    }},
    {{
      "name": "ARTICLE 58(5) OF REGULATION (EU) 2016/679",
      "type": "ARTICLE",
      "description": "Grants supervisory authorities the power to bring infringements to national courts and initiate legal proceedings"
    }},
    {{
      "name": "ONE-STOP SHOP MECHANISM",
      "type": "LEGAL_PRINCIPLE",
      "description": "GDPR principle established in Article 56(1) allocating competence for cross-border processing to a single lead supervisory authority, minimising fragmentation of enforcement"
    }},
    {{
      "name": "SINCERE AND EFFECTIVE COOPERATION",
      "type": "LEGAL_PRINCIPLE",
      "description": "Duty of non-lead supervisory authorities to cooperate with the lead supervisory authority when exercising exceptional competence under Article 58(5)"
    }},
    {{
      "name": "Q1 - NON-LEAD SA COMPETENCE FOR CROSS-BORDER PROCEEDINGS",
      "type": "PRELIMINARY_QUESTION",
      "description": "Whether Articles 55(1), 56-58 and 60-66 of GDPR must be interpreted as meaning a non-lead supervisory authority may exercise the power in Article 58(5) for cross-border processing"
    }},
    {{
      "name": "HOLDING Q1 - NON-LEAD SA MAY BRING CROSS-BORDER PROCEEDINGS",
      "type": "HOLDING",
      "description": "Article 58(5) GDPR must be interpreted as meaning a supervisory authority may exercise that power in situations where the regulation exceptionally confers competence on it, provided cooperation procedures are respected"
    }},
    {{
      "name": "OPERATIVE PART - RULING 1",
      "type": "OPERATIVE_PART",
      "description": "The Court rules that a non-lead supervisory authority may exercise the Article 58(5) power in cross-border cases when the regulation exceptionally confers competence and cooperation procedures are respected"
    }}
  ],
  "relationships": [
    {{
      "source": "FACEBOOK IRELAND LTD",
      "target": "GEGEVENSBESCHERMINGSAUTORITEIT",
      "description": "Facebook Ireland Ltd is subject to injunction proceedings initiated by the Belgian DPA",
      "strength": 9
    }},
    {{
      "source": "GEGEVENSBESCHERMINGSAUTORITEIT",
      "target": "COMMISSIE TER BESCHERMING VAN DE PERSOONLIJKE LEVENSSFEER",
      "description": "The Gegevensbeschermingsautoriteit is the legal successor to the former Belgian Privacy Commission",
      "strength": 10
    }},
    {{
      "source": "BELGIAN GOVERNMENT",
      "target": "GEGEVENSBESCHERMINGSAUTORITEIT",
      "description": "The Belgian Government submitted observations supporting the authority's position",
      "strength": 7
    }},
    {{
      "source": "ONE-STOP SHOP MECHANISM",
      "target": "ARTICLE 56(1) OF REGULATION (EU) 2016/679",
      "description": "The one-stop shop mechanism is established and governed by Article 56(1)",
      "strength": 10
    }},
    {{
      "source": "SINCERE AND EFFECTIVE COOPERATION",
      "target": "ONE-STOP SHOP MECHANISM",
      "description": "Sincere and effective cooperation is a condition for the proper functioning of the one-stop shop mechanism",
      "strength": 8
    }},
    {{
      "source": "Q1 - NON-LEAD SA COMPETENCE FOR CROSS-BORDER PROCEEDINGS",
      "target": "ARTICLE 58(5) OF REGULATION (EU) 2016/679",
      "description": "Q1 concerns the interpretation of the power conferred by Article 58(5)",
      "strength": 9
    }},
    {{
      "source": "Q1 - NON-LEAD SA COMPETENCE FOR CROSS-BORDER PROCEEDINGS",
      "target": "ONE-STOP SHOP MECHANISM",
      "description": "Q1 arises directly from the interplay between Article 58(5) and the one-stop shop mechanism",
      "strength": 8
    }},
    {{
      "source": "HOLDING Q1 - NON-LEAD SA MAY BRING CROSS-BORDER PROCEEDINGS",
      "target": "Q1 - NON-LEAD SA COMPETENCE FOR CROSS-BORDER PROCEEDINGS",
      "description": "This holding directly answers Q1 referred by the national court",
      "strength": 10
    }},
    {{
      "source": "HOLDING Q1 - NON-LEAD SA MAY BRING CROSS-BORDER PROCEEDINGS",
      "target": "ARTICLE 58(5) OF REGULATION (EU) 2016/679",
      "description": "The holding interprets the scope and conditions of Article 58(5) GDPR",
      "strength": 9
    }},
    {{
      "source": "HOLDING Q1 - NON-LEAD SA MAY BRING CROSS-BORDER PROCEEDINGS",
      "target": "SINCERE AND EFFECTIVE COOPERATION",
      "description": "The holding conditions the exercise of Article 58(5) power on compliance with the sincere cooperation principle",
      "strength": 8
    }},
    {{
      "source": "OPERATIVE PART - RULING 1",
      "target": "HOLDING Q1 - NON-LEAD SA MAY BRING CROSS-BORDER PROCEEDINGS",
      "description": "The operative ruling 1 gives binding legal force to the holding on Q1",
      "strength": 10
    }}
  ]
}}

---Real Data---
Entity types: {entity_types}
chunk_id: {chunk_id}

Input:
{input_text}

Output:
"""

GRAPH_RAG_MICROSOFT_PROMPT_v4 = """
-Goal-
Given a text document from a Court of Justice of the European Union (CJEU) case law judgment,
identify all entities of the specified types and all relationships among them.

-Steps-

1. Identify all entities. For each entity extract:
   - name: capitalized, always SPECIFIC (never generic names like "REGULATION", "COURT", 
     "SUPERVISORY AUTHORITY" — always use the full specific name e.g. 
     "REGULATION (EU) 2016/679", "COURT OF JUSTICE (GRAND CHAMBER)", 
     "GEGEVENSBESCHERMINGSAUTORITEIT")
   - type: one of the provided entity types
   - description: comprehensive description of the entity's role in this specific case

2. Pay special attention to the document section you are processing:

   - "Legal context" section → focus on REGULATION, DIRECTIVE, ARTICLE, RECITAL, TREATY_PROVISION.

   - "Dispute in main proceedings" section → focus on APPLICANT, DEFENDANT, REFERRING_COURT,
     FACTUAL_CIRCUMSTANCE, LEGAL_ISSUE, ADMINISTRATIVE_DECISION.

   - "Consideration of questions referred" section → this is the most important section.
     Extract the following entity types with priority:

     a) PRELIMINARY_QUESTION — extract one per referred question using this STRICT naming convention:
        "Q{{N}} - {{BRIEF DESCRIPTION OF LEGAL ISSUE}}"
        Example: "Q1 - NON-LEAD SA COMPETENCE FOR CROSS-BORDER PROCEEDINGS"
        This name must be IDENTICAL across all chunks that reference the same question.
        NEVER use alternative names like "FIRST QUESTION REFERRED" or "PRELIMINARY QUESTION 1".

     b) HOLDING — extract one per answered question using this STRICT naming convention:
        "HOLDING Q{{N}} - {{BRIEF DESCRIPTION OF COURT ANSWER}}"
        Example: "HOLDING Q1 - NON-LEAD SA MAY BRING CROSS-BORDER PROCEEDINGS"
        This name must be IDENTICAL across all chunks that reference the same holding.
        The trigger phrase is: "the answer to the Nth question referred is that..."

     c) LEGAL_PRINCIPLE — extract named legal doctrines and principles established or applied
        in the reasoning. These are concepts broader than a single article. Examples:
        "ONE-STOP SHOP MECHANISM" — the GDPR principle allocating competence to a lead SA
        "SINCERE AND EFFECTIVE COOPERATION" — the duty of non-lead SAs toward the lead SA
        "DIRECT EFFECT" — the principle that an EU provision is directly applicable in national law
        "TERRITORIAL SCOPE OF REGULATION (EU) 2016/679" — the geographic application rule

   - "On those grounds, the Court hereby rules" section → extract one OPERATIVE_PART entity
     per numbered ruling point using this naming convention:
     "OPERATIVE PART - RULING {{N}}"
     and link it to the corresponding HOLDING Q{{N}} entity.

   - "Costs" section → extract the following:
     a) ADMINISTRATIVE_DECISION — one entity representing the cost allocation decision:
        naming convention: "COSTS DECISION - {{CASE_ID}}"
        description: who bears costs and on what legal basis
     b) REFERRING_COURT — if the cost decision is referred back to the national court,
        extract it as the entity responsible for the final costs ruling
     c) INTERVENER — any party explicitly mentioned as bearing or not bearing costs
        (e.g. Member State governments whose observation costs are not recoverable)
     d) Relationships:
        - COSTS DECISION → REFERRING_COURT (if costs are remitted to national court, strength 8)
        - COSTS DECISION → INTERVENER (for each party whose cost status is determined, strength 7)

3. From the entities in step 1, identify all clearly related pairs and for each extract:
   - source, target: entity names from step 1
   - description: why they are related in this specific legal context
   - strength: 1-10

   Strength calibration guide:
   - 10: direct structural containment (ARTICLE belongs to REGULATION, HOLDING answers QUESTION,
         OPERATIVE PART gives binding force to HOLDING)
   - 9:  direct interpretation (HOLDING interprets ARTICLE, QUESTION concerns ARTICLE)
   - 8:  strong thematic link (LEGAL_PRINCIPLE applied in HOLDING, LEGAL_ISSUE triggers QUESTION,
         COSTS DECISION remitted to REFERRING_COURT)
   - 7:  supporting/procedural link (INTERVENER submits observations on, RECITAL supports ARTICLE,
         COSTS DECISION affects INTERVENER cost status)
   - 6 or below: indirect or contextual links

-Critical Rules-
- NEVER use generic entity names: always use the full specific identifier
- NEVER invent entity types outside the provided list
- NEVER extract Official Journal references (e.g. "OJ 2016 L 119") as entities
- NEVER extract paragraph numbers (e.g. "- 37", "2.", "3.") as entities
- ALWAYS extract INTERVENER for Member State governments submitting observations
- ALWAYS extract PRELIMINARY_QUESTION and HOLDING in the reasoning sections
- ALWAYS extract LEGAL_PRINCIPLE when a named doctrine is applied in the reasoning
- ALWAYS use the canonical Q{{N}} naming convention for PRELIMINARY_QUESTION and HOLDING
- ALWAYS extract COSTS DECISION and related entities in the Costs section
- RELATIONSHIP DIRECTION: a HOLDING interprets an ARTICLE, so the edge goes
  HOLDING → ARTICLE (not ARTICLE → HOLDING). A HOLDING answers a PRELIMINARY_QUESTION,
  so the edge goes HOLDING → PRELIMINARY_QUESTION. An OPERATIVE PART gives binding
  force to a HOLDING, so the edge goes OPERATIVE PART → HOLDING.

-Output Format-
Return ONLY a valid JSON object with this exact structure, no preamble, no markdown:

{{
  "chunk_id": "{chunk_id}",
  "chunk_text": "{chunk_text}",
  "entities": [
    {{
      "name": "<SPECIFIC ENTITY NAME IN CAPS>",
      "type": "<ENTITY_TYPE>",
      "description": "<description of role in this specific case>"
    }}
  ],
  "relationships": [
    {{
      "source": "<source entity name>",
      "target": "<target entity name>",
      "description": "<why they are related>",
      "strength": <1-10>
    }}
  ]
}}

---Examples---
Entity types: ORGANIZATION, PERSON

Input:
The Fed is scheduled to meet on Tuesday and Wednesday, with the central bank
planning to release its latest policy decision on Wednesday at 2:00 p.m. ET, followed
by a press conference where Fed Chair Jerome Powell will take questions. Investors expect
the Federal Open Market Committee to hold its benchmark interest rate steady in a range
of 5.25%-5.5%.

Output:
{{
  "chunk_id": "example_chunk_000",
  "chunk_text": "The Fed is scheduled to meet on Tuesday and Wednesday...",
  "entities": [
    {{
      "name": "FEDERAL RESERVE",
      "type": "ORGANIZATION",
      "description": "The Federal Reserve (Fed) is the central bank of the United States, setting interest rates"
    }},
    {{
      "name": "JEROME POWELL",
      "type": "PERSON",
      "description": "Jerome Powell is the Chair of the Federal Reserve"
    }},
    {{
      "name": "FEDERAL OPEN MARKET COMMITTEE",
      "type": "ORGANIZATION",
      "description": "The Federal Reserve committee that makes key decisions about interest rates"
    }}
  ],
  "relationships": [
    {{
      "source": "JEROME POWELL",
      "target": "FEDERAL RESERVE",
      "description": "Jerome Powell is the Chair of the Federal Reserve and will answer press conference questions",
      "strength": 9
    }}
  ]
}}

---Legal Domain Example---
Entity types: COURT, REFERRING_COURT, NATIONAL_AUTHORITY, APPLICANT, DEFENDANT, INTERVENER,
REGULATION, ARTICLE, LEGAL_ISSUE, FACTUAL_CIRCUMSTANCE, LEGAL_PRINCIPLE,
PRELIMINARY_QUESTION, HOLDING, OPERATIVE_PART, ADMINISTRATIVE_DECISION

Input:
The request has been made in proceedings between Facebook Ireland Ltd, Facebook Inc. and
Facebook Belgium BVBA, on the one hand, and the Gegevensbeschermingsautoriteit (the Belgian
Data Protection Authority), as the successor of the Commissie ter bescherming van de
Persoonlijke Levenssfeer (the Belgian Privacy Commission), on the other, concerning injunction
proceedings brought by the President of the Privacy Commission seeking to bring to an end the
processing of personal data of internet users within Belgium by the Facebook online social
network, using cookies, social plug-ins and pixels.
The Belgian Government, the Czech Government and the Italian Government submitted observations.

In that regard, without prejudice to Article 55(1) of Regulation 2016/679, Article 56(1)
establishes the one-stop shop mechanism, based on an allocation of competences between a
lead supervisory authority and the other supervisory authorities concerned.
The lead supervisory authority cannot eschew sincere and effective cooperation with the other
supervisory authorities concerned.

By its first question, the referring court seeks to ascertain whether Article 55(1), Articles
56 to 58 and Articles 60 to 66 of Regulation 2016/679 must be interpreted as meaning that a
supervisory authority of a Member State which is not the lead supervisory authority may
exercise the power in Article 58(5) in relation to cross-border data processing.

The answer to the first question referred is that Article 58(5) of Regulation 2016/679
must be interpreted as meaning that a supervisory authority may exercise that power in
one of the situations where that regulation exceptionally confers competence on it,
provided that the cooperation and consistency procedures are respected.

On those grounds, the Court (Grand Chamber) hereby rules:
1. Article 55(1), Articles 56 to 58 and Articles 60 to 66 of Regulation (EU) 2016/679,
read together with Articles 7, 8 and 47 of the Charter, must be interpreted as meaning
that a non-lead supervisory authority may exercise the power in Article 58(5) in cross-border
cases, provided that power is exercised in one of the situations where that regulation
confers competence on it and cooperation procedures are respected.

Costs:
Since these proceedings are, for the parties to the main proceedings, a step in the action
pending before the referring court, the decision on costs is a matter for that court.
Costs incurred in submitting observations to the Court, other than the costs of those
parties, are not recoverable.

Output:
{{
  "chunk_id": "legal_example_chunk_000",
  "chunk_text": "The request has been made in proceedings...",
  "entities": [
    {{
      "name": "FACEBOOK IRELAND LTD",
      "type": "APPLICANT",
      "description": "Facebook Ireland Ltd is an applicant challenging the injunction proceedings regarding personal data processing in Belgium"
    }},
    {{
      "name": "FACEBOOK INC.",
      "type": "APPLICANT",
      "description": "Facebook Inc. is the US parent company and applicant in the proceedings against the Belgian DPA"
    }},
    {{
      "name": "FACEBOOK BELGIUM BVBA",
      "type": "APPLICANT",
      "description": "Facebook Belgium BVBA is the local Belgian entity and applicant in the injunction proceedings"
    }},
    {{
      "name": "GEGEVENSBESCHERMINGSAUTORITEIT",
      "type": "NATIONAL_AUTHORITY",
      "description": "The Belgian Data Protection Authority, successor to the Privacy Commission, respondent in the proceedings"
    }},
    {{
      "name": "COMMISSIE TER BESCHERMING VAN DE PERSOONLIJKE LEVENSSFEER",
      "type": "NATIONAL_AUTHORITY",
      "description": "The former Belgian Privacy Commission, predecessor to the Gegevensbeschermingsautoriteit"
    }},
    {{
      "name": "BELGIAN GOVERNMENT",
      "type": "INTERVENER",
      "description": "The Belgian Government submitted observations in the proceedings; its costs are not recoverable"
    }},
    {{
      "name": "CZECH GOVERNMENT",
      "type": "INTERVENER",
      "description": "The Czech Government submitted observations in the proceedings; its costs are not recoverable"
    }},
    {{
      "name": "ITALIAN GOVERNMENT",
      "type": "INTERVENER",
      "description": "The Italian Government submitted observations in the proceedings; its costs are not recoverable"
    }},
    {{
      "name": "PROCESSING OF PERSONAL DATA VIA COOKIES AND SOCIAL PLUG-INS",
      "type": "LEGAL_ISSUE",
      "description": "Facebook's collection of data from Belgian internet users through cookies, social plug-ins and pixels without adequate legal basis"
    }},
    {{
      "name": "ARTICLE 55(1) OF REGULATION (EU) 2016/679",
      "type": "ARTICLE",
      "description": "General competence rule — each supervisory authority is competent on the territory of its own Member State"
    }},
    {{
      "name": "ARTICLE 56(1) OF REGULATION (EU) 2016/679",
      "type": "ARTICLE",
      "description": "Establishes the lead supervisory authority for cross-border processing under the one-stop shop mechanism"
    }},
    {{
      "name": "ARTICLE 58(5) OF REGULATION (EU) 2016/679",
      "type": "ARTICLE",
      "description": "Grants supervisory authorities the power to bring infringements to national courts and initiate legal proceedings"
    }},
    {{
      "name": "ONE-STOP SHOP MECHANISM",
      "type": "LEGAL_PRINCIPLE",
      "description": "GDPR principle established in Article 56(1) allocating competence for cross-border processing to a single lead supervisory authority"
    }},
    {{
      "name": "SINCERE AND EFFECTIVE COOPERATION",
      "type": "LEGAL_PRINCIPLE",
      "description": "Duty of non-lead supervisory authorities to cooperate with the lead supervisory authority when exercising exceptional competence under Article 58(5)"
    }},
    {{
      "name": "Q1 - NON-LEAD SA COMPETENCE FOR CROSS-BORDER PROCEEDINGS",
      "type": "PRELIMINARY_QUESTION",
      "description": "Whether Articles 55(1), 56-58 and 60-66 of GDPR must be interpreted as meaning a non-lead supervisory authority may exercise the power in Article 58(5) for cross-border processing"
    }},
    {{
      "name": "HOLDING Q1 - NON-LEAD SA MAY BRING CROSS-BORDER PROCEEDINGS",
      "type": "HOLDING",
      "description": "Article 58(5) GDPR must be interpreted as meaning a supervisory authority may exercise that power in situations where the regulation exceptionally confers competence on it, provided cooperation procedures are respected"
    }},
    {{
      "name": "OPERATIVE PART - RULING 1",
      "type": "OPERATIVE_PART",
      "description": "The Court rules that a non-lead supervisory authority may exercise the Article 58(5) power in cross-border cases when the regulation exceptionally confers competence and cooperation procedures are respected"
    }},
    {{
      "name": "COSTS DECISION - C-645/19",
      "type": "ADMINISTRATIVE_DECISION",
      "description": "The CJEU remits the costs decision to the referring court for the main parties; costs of intervening Member State governments submitting observations are declared non-recoverable"
    }},
    {{
      "name": "HOF VAN BEROEP TE BRUSSEL",
      "type": "REFERRING_COURT",
      "description": "The Court of Appeal Brussels, which referred the preliminary ruling; responsible for deciding on costs for the main parties since the CJEU proceedings are a step in the national action"
    }}
  ],
  "relationships": [
    {{
      "source": "FACEBOOK IRELAND LTD",
      "target": "GEGEVENSBESCHERMINGSAUTORITEIT",
      "description": "Facebook Ireland Ltd is subject to injunction proceedings initiated by the Belgian DPA",
      "strength": 9
    }},
    {{
      "source": "GEGEVENSBESCHERMINGSAUTORITEIT",
      "target": "COMMISSIE TER BESCHERMING VAN DE PERSOONLIJKE LEVENSSFEER",
      "description": "The Gegevensbeschermingsautoriteit is the legal successor to the former Belgian Privacy Commission",
      "strength": 10
    }},
    {{
      "source": "BELGIAN GOVERNMENT",
      "target": "GEGEVENSBESCHERMINGSAUTORITEIT",
      "description": "The Belgian Government submitted observations supporting the authority's position",
      "strength": 7
    }},
    {{
      "source": "ONE-STOP SHOP MECHANISM",
      "target": "ARTICLE 56(1) OF REGULATION (EU) 2016/679",
      "description": "The one-stop shop mechanism is established and governed by Article 56(1)",
      "strength": 10
    }},
    {{
      "source": "SINCERE AND EFFECTIVE COOPERATION",
      "target": "ONE-STOP SHOP MECHANISM",
      "description": "Sincere and effective cooperation is a condition for the proper functioning of the one-stop shop mechanism",
      "strength": 8
    }},
    {{
      "source": "Q1 - NON-LEAD SA COMPETENCE FOR CROSS-BORDER PROCEEDINGS",
      "target": "ARTICLE 58(5) OF REGULATION (EU) 2016/679",
      "description": "Q1 concerns the interpretation of the power conferred by Article 58(5)",
      "strength": 9
    }},
    {{
      "source": "Q1 - NON-LEAD SA COMPETENCE FOR CROSS-BORDER PROCEEDINGS",
      "target": "ONE-STOP SHOP MECHANISM",
      "description": "Q1 arises directly from the interplay between Article 58(5) and the one-stop shop mechanism",
      "strength": 8
    }},
    {{
      "source": "HOLDING Q1 - NON-LEAD SA MAY BRING CROSS-BORDER PROCEEDINGS",
      "target": "Q1 - NON-LEAD SA COMPETENCE FOR CROSS-BORDER PROCEEDINGS",
      "description": "This holding directly answers Q1 referred by the national court",
      "strength": 10
    }},
    {{
      "source": "HOLDING Q1 - NON-LEAD SA MAY BRING CROSS-BORDER PROCEEDINGS",
      "target": "ARTICLE 58(5) OF REGULATION (EU) 2016/679",
      "description": "The holding interprets the scope and conditions of Article 58(5) GDPR",
      "strength": 9
    }},
    {{
      "source": "HOLDING Q1 - NON-LEAD SA MAY BRING CROSS-BORDER PROCEEDINGS",
      "target": "SINCERE AND EFFECTIVE COOPERATION",
      "description": "The holding conditions the exercise of Article 58(5) power on compliance with the sincere cooperation principle",
      "strength": 8
    }},
    {{
      "source": "OPERATIVE PART - RULING 1",
      "target": "HOLDING Q1 - NON-LEAD SA MAY BRING CROSS-BORDER PROCEEDINGS",
      "description": "The operative ruling 1 gives binding legal force to the holding on Q1",
      "strength": 10
    }},
    {{
      "source": "COSTS DECISION - C-645/19",
      "target": "HOF VAN BEROEP TE BRUSSEL",
      "description": "The CJEU remits the costs decision to the referring court for determination as to the main parties",
      "strength": 8
    }},
    {{
      "source": "COSTS DECISION - C-645/19",
      "target": "BELGIAN GOVERNMENT",
      "description": "The Belgian Government's costs as intervener are declared non-recoverable by the CJEU",
      "strength": 7
    }},
    {{
      "source": "COSTS DECISION - C-645/19",
      "target": "CZECH GOVERNMENT",
      "description": "The Czech Government's costs as intervener are declared non-recoverable by the CJEU",
      "strength": 7
    }},
    {{
      "source": "COSTS DECISION - C-645/19",
      "target": "ITALIAN GOVERNMENT",
      "description": "The Italian Government's costs as intervener are declared non-recoverable by the CJEU",
      "strength": 7
    }}
  ]
}}

---Real Data---
Entity types: {entity_types}
chunk_id: {chunk_id}

Input:
{input_text}

Output:
"""