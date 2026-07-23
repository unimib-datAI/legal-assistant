# Knowledge Graph

The Neo4j schema: what each node is and which fields it carries.

> Field names, types and optionality were read from the live database with
> `db.schema.nodeTypeProperties()`, not from the loader source.

## Document assumptions

The schema mirrors the structure EUR-Lex publishes:

- An **act** is composed of one or more **chapters**, and separately of a flat list of
  **recitals**.
- Each **chapter** is composed of one or more **articles**.
  - In some cases the articles are grouped into **sections** rather than hanging directly
    off the chapter (`Chapter → Section → Article`). Both shapes occur in the corpus.
- An **article** is typically composed of a series of **paragraphs**.
- **Articles** may cite other **articles** or **paragraphs**, within and across acts.
- **Case law** interprets **articles** and **paragraphs**; one judgment may interpret
  several provisions, and a provision may be interpreted by several judgments.

![Knowledge graph](../img/KG.png)

## Building it

```bash
legal-assistant graph build          # acts, chapters, sections, articles, paragraphs, recitals
legal-assistant ingest case-law      # fills the CaseLaw stubs with judgment text
legal-assistant graph aske           # optional: Concept nodes for the `topics` retriever
legal-assistant summarize articles   # optional: adds `summary` to Article nodes
```

## Node identity

Articles and chapters keep the same number across different documents, so a plain number
would collide. Every id therefore combines the act's CELEX (the unique identifier EUR-Lex
assigns) with the element's own identifier. A node keeps its identity across reloads and
across acts, which is what makes cross-document citations expressible at all.

Ids are not opaque retrieval parses them:

- **Paragraph ids are zero-padded and dotted**: `32016R0679_012.001` is Article 12(1). A
  `.000` suffix points at the article as a whole.
- **Judgment paragraph ids are sequential integers**, so a passage's neighbours are
  reachable by arithmetic instead of a query  `rag/documents.py::neighbour_ids` uses this
  to recover a holding that follows its reasoning.

---

## Act nodes

### Act

The root of one regulation.

```
id            String       CELEX identifier, e.g. "32016R0679"
title          String      Official title as published
eurolex_url   String       Source page the document was fetched from
```

### Chapter

```
id            String       "<celex>cpt_<roman>"     e.g. "32016R0679cpt_III"
number        String       Roman numeral            e.g. "III"
title         String       e.g. "Rights of the data subject"
summary       String       optional  written by `legal-assistant summarize chapters`
```

### Section

Present only in acts that group articles below the chapter level.

```
id            String       "<celex>cpt_<roman>.sct_<n>"   e.g. "32016R0679cpt_III.sct_1"
title         String       e.g. "Transparency and modalities"
```

### Article

```
id              String     "<celex>art_<n>"    e.g. "32016R0679art_12"
title           String     e.g. "Transparent information, communication and modalities …"
text            String     Full article text
textEmbedding   Float[]    Embedding of `text`, dimension EMBEDDING_DIM (1536)
summary         String     optional  written by `legal-assistant summarize articles`
```

### Paragraph

One numbered paragraph of an article.

```
id              String     "<celex>_<article:3>.<paragraph:3>"   e.g. "32016R0679_012.001"
text            String     Paragraph text
textEmbedding   Float[]    Embedding of `text`
```

### Recital

The numbered preamble items. They hang off the `Act`, not off a chapter.

```
id              String     "<celex>rct_<n>"    e.g. "32016R0679rct_1"
number          String     Display number      e.g. "1"
text            String     Recital text
textEmbedding   Float[]    Embedding of `text`
```

---

## Case law nodes

A `CaseLaw` node is created in two stages. `graph build` writes a **stub**  id, CELEX,
case number — for every judgment listed as "Interpreted by" in an act's EUR-Lex metadata,
together with its `INTERPRETS` edge. `ingest case-law` later fetches the judgment and fills
in the sections, paragraphs and topics below it. A stub with no sections is a judgment not
yet ingested; judgments older than roughly 2012 have no XHTML manifestation in Cellar and
stay stubs.

### CaseLaw

```
id             String      CELEX of the judgment, e.g. "62018CJ0311"
celex          String      Same value, optional  absent on some stubs
case_number    String      Court-style number, e.g. "C-311/18"     optional
summary        String      optional  whole-document LLM summary
```

### CaseLawSection

The judgment's own heading hierarchy, read from the published XHTML rather than inferred.
Sections nest.

```
id            String       "<celex>_sec_<n>"     e.g. "62018CJ0311_sec_3"
celex         String       Owning judgment
heading       String       e.g. "Judgment", "General Information"
depth         Integer      Nesting level, 0 for a top-level section
path          String       Position in the tree, e.g. "3"
summary       String       Per-section LLM summary
```

### CaseLawParagraph

The retrievable unit of a judgment.

```
id                String   "<celex>_par_<n>", or "<celex>_op_<n>" when operative
celex             String   Owning judgment
number            Integer  Paragraph number within the judgment
text              String   Paragraph text
textEmbedding     Float[]  Embedding of `text`
is_operative      Boolean  True for the operative part  the binding holding
section_heading   String   Heading of the section it sits in, denormalised for retrieval
```

`is_operative` matters at query time: the operative part states what the Court actually
held, while the surrounding paragraphs state the reasoning. Retrieval can guarantee it a
context slot of its own (`guarantee_operative`).

### CaseLawTopic

The subject-matter keywords EUR-Lex publishes for a judgment.

```
id       String            "case_law_topic:<celex>:<label>"
celex    String            Owning judgment
label    String            e.g. "Reference for a preliminary ruling"
```

---

## Annex nodes

Only the AI Act has annexes. They hang off the act, in the same structural position as
recitals, because the published markup places them outside the chapter tree.

### Annex

```
id       String   "<celex>anx_<roman>"    e.g. "32024R1689anx_III"
number   String   Roman numeral            e.g. "III"
title    String   e.g. "High-risk AI systems referred to in Article 6(2)"
```

### AnnexPoint

The retrievable unit of an annex. Its id is positional, not parsed from the prose numbering;
the citation a lawyer writes is carried separately in `point_label`.

```
id                String   "<celex>anx_<roman>.p_<nnn>"   e.g. "32024R1689anx_III.p_007"
celex             String   Owning act
text              String   Point text
textEmbedding     Float[]  Embedding of `text`
section_heading   String   Nearest preceding internal heading, denormalised   optional
point_label       String   Derived citation label, e.g. "III, point 1(a)"     optional
```

---

## Obligation nodes

Deontic obligations extracted from the acts, loaded by `ingest obligations`. An obligation
hangs off the passage it was extracted from, and is addressed to actors from a curated,
generated vocabulary. Fields below were read from a live Neo4j with
`db.schema.nodeTypeProperties()`.

### Obligation

```
id                    String   "<source_id>#ob_<n>"   e.g. "32016R0679_012.001#ob_1"
celex                 String   Owning act
modality              String   OBLIGATION | PROHIBITION
obligation_type       String   ACTION | BEING
predicate_text        String   the duty's core, e.g. "shall inform"
predicate_voice       String   active | passive
target                String   what the predicate acts on                       optional
specification         String   standard, method or time to fulfil it            optional
precondition          String   circumstances that trigger it                    optional
beneficiary_text      String   who benefits, as extracted                       optional
<element>_method      String   STATED | CONTEXT | CITATION | BACKGROUND | NONE, one per element
                               (addressee, predicate, target, specification, precondition, beneficiary)
weakest_method        String   weakest populated method, the obligation's trust
```

`weakest_method` is the trust floor: an obligation whose addressee was inferred by
BACKGROUND is weaker evidence than one where every element is STATED, and retrieval and the
checklist both surface it.

### Actor

A subject the legislation defines or addresses. The curated core is generated from each
act's Definitions article (`actors.yaml`); institutional actors and qualified forms the
definitions do not carry are promoted from the extraction's addressee strings during ingest.

```
id           String   slug, e.g. "controller", "provider_of_high_risk_ai_system"
label        String   human-readable form
celex        String   act that defines it; null for cross-cutting subjects   optional
defined_in   String   id of the defining paragraph, when there is one         optional
```

---

## Relationships

The act hierarchy:

```
Act ─CONTAINS→ Chapter ─CONTAINS→ [Section ─CONTAINS→] Article ─CONTAINS→ Paragraph
 ├──CONTAINS→ Recital
 └──CONTAINS→ Annex ─CONTAINS→ AnnexPoint
```

The obligation subgraph, which points back into the acts and into the actor vocabulary:

```
Paragraph | AnnexPoint ─STATES→ Obligation ─ADDRESSED_TO→ Actor
                                     └───────BENEFITS→ Actor
Actor ─IS_A→ Actor
```

`STATES` hangs an obligation off its source passage; `ADDRESSED_TO` and `BENEFITS` link it to
actors; `IS_A` is the actor hierarchy a role filter walks (`(:Actor)-[:IS_A*0..]->(:Actor)`),
so filtering on "provider" reaches "provider of a high-risk AI system".

The judgment hierarchy, which hangs off its own root and points back into the acts:

```
CaseLaw ─HAS_SECTION→ CaseLawSection ─HAS_PARAGRAPH→ CaseLawParagraph
   │                        └──CONTAINS→ CaseLawSection (nested)
   ├──HAS_TOPIC→ CaseLawTopic
   └──INTERPRETS→ Article | Paragraph | Chapter
```

`CONTAINS` is the single containment edge in both hierarchies, so the article side is
reached with a variable-length `(:Act)-[:CONTAINS*]->(:Article)` — chapters may or may not
interpose a section. `INTERPRETS` may point at an article, a paragraph **or** a chapter,
depending on how precise the EUR-Lex metadata is, so a query walking it must handle all
three target labels.

**There is no citation edge.** Article-to-article citations used to be materialised as
`CITES`, derived by regex over the article text ("Articles 12 to 15" expanded to a whole
range, including articles of other acts). Nothing read them, most pointed at ids that did
not exist, and because `CREATE_RELATIONSHIP` does `MATCH … MATCH … MERGE` those edges were
dropped in silence. Cross-references are resolved at query time from the passage text
instead — see [`rag/citations.py`](../src/legal_assistant/rag/citations.py). A graph built
before this change still carries the old edges until it is rebuilt.


## Vector indexes

Four COSINE indexes over `textEmbedding`, sized by `EMBEDDING_DIM`. The act-side three are
created by `graph build`, the judgment one by `ingest case-law`.

| Index | Label | Used by |
|---|---|---|
| `Article` | `Article` | `hybrid` dense article search |
| `Paragraph` | `Paragraph` | `topics` paragraph search |
| `Recital` | `Recital` | recital retrieval |
| `AnnexPoint` | `AnnexPoint` | annex retrieval (AI Act only) |
| `CaseLawParagraph` | `CaseLawParagraph` | `hybrid` INTERPRETIVE branch |

Index name and node label are always the same string  `RagContext._vector_store` relies on
that.

## Where the Cypher lives

Every statement the application runs is in
[`src/legal_assistant/graph/queries.py`](../src/legal_assistant/graph/queries.py), grouped into
`GeneralQueries`, `NodeQueries`, `RelationQueries` and `CaseLawQueries`. Nothing
string-formats Cypher elsewhere; all parameters are bound.
