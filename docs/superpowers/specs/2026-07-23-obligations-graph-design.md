# Obligations graph: design

**Date:** 2026-07-23
**Status:** approved, ready for planning

## 1. Context

The graph today is *documentary*: it models what the corpus says and where. An obligation
graph is *normative*: it models what the corpus requires, of whom, under which conditions.

The two answer different questions. Vector search over article text answers "what does the
law say about X". It does not answer "I am a deployer of a high-risk AI system, what must I
do", because the addressee of a provision is often implicit, or stated in a paragraph other
than the one carrying the duty. That question needs an exact, role-typed filter, which is a
retrieval axis the current pipeline does not have.

The extraction is not designed here. It comes from the workflow published with Galli et al.,
*Approaching the AI Act with AI* (CLSR 60, 2026), whose code is at
`thiagordp/obligation_extraction_for_compliance`. Its two LLM stages are reused as they are;
its detection stage is replaced, for reasons set out in section 3. This document specifies
that integration and everything downstream of it.

## 2. Scope

**In scope**

- Annex nodes for the AI Act: parser, schema, loader, vector index, retrieval branch.
- The obligation data model and the curated addressee vocabulary.
- An ingestion pipeline with validation.
- An obligations branch inside `HybridRetriever`.
- A deterministic compliance checklist pipeline.
- Tests and an eval measuring role recall.

**Out of scope**

- The deontic filtering and analysis stages, reused unchanged from the source repository.
  Their detection stage is replaced; see section 3.
- Conflict, overlap and redundancy detection between obligations. The source paper lists
  it as future work; on realistic pairs, such as accuracy under Article 15(1) against
  transparency under Article 13(3), it is unreliable enough that a wrong verdict is worse
  than no feature.
- `Predicate` as a node. It serves quantitative analysis, not retrieval.
- The per-act reading frame, that is the applicability scaffolding and the question-type to
  slot contract. It is a separate design that will build on this one.

## 3. How the source pipeline is integrated

The source repository has three stages: `obligation_detection`, `obligation_filtering` and
`obligation_analysis`. **Stages 2 and 3 are reused unchanged**, prompts included: they are
the validated part, carrying the paper's 93% filtering precision and above 99% accuracy on
obligation type, addressees and predicate. Their prompts live in `data/raw/prompts/` as text
files rather than in code. **Stage 1 is replaced by a graph-driven detector.**

### 3.0 What the repository actually publishes

Read before assuming a module can be reused. The repository covers detection, filtering,
analysis and the paper's statistics. **It does not publish the paper's fourth module**, the
knowledge-graph representation of section 4.4:

| File | What it does |
|---|---|
| `post_processing.py` | groups sentences by classification label into text files. No normalisation, no graph |
| `visualization.py` | matplotlib frequency plots reproducing Figure 2 and Tables 3 to 6 |
| `notebooks/1.1_..._generate_list_of_addressees_beneficiaries.ipynb` | the only trace of addressee normalisation |

There is no Entity/Person graph, no `HAS_CHILD` or `has_obligation` edge, no Sentence
Transformers clustering and no GraphML export anywhere in `src/`. The paper says as much of
its own design: the ontology "is only partially developed", "remains less mature than the
other modules of the workflow", and its formalisation is "left to future work".

The filtering and analysis **prompts do exist** in the repository, under `data/raw/prompts/`
(committed despite a `.gitignore` entry): a 652-word filtering system prompt and a 2624-word
analysis system prompt, plus their short user templates. The repository's `setup_evaluation/`
also holds real per-sentence LLM outputs, which is what pins the input contract in phase 3.1
with no LLM call.

**What is reused, and how.** Only the four prompts are reused, and they are reused as text:
their verbatim wording is registered as `PromptVersion`s in `rag/prompts/obligations.py`, in
the same versioned form as every other prompt in the project, credited to the paper in the
module docstring. Everything else is this project's own code written against its graph and LLM
client: the two LLM stages (`obligations/filtering.py`, `analysis.py`), a graph-driven
detector (`detection.py`), citation resolution against the graph (`references.py`, adapting
their `solve_references`), and tolerant JSON extraction (`json_io.py`, adapting their
`extract_dict`). The paper's Python modules are not carried in the tree.

So section 4 of this document does not reproduce that module, because there is nothing
published to reproduce. It keeps the paper's two-tier idea, a generalised concept above the
surface forms, and changes both where the concepts come from and what the graph is for: the
paper's are selected by legal experts and exported as a standalone GraphML artefact for
visualisation, ours are generated from the acts' own Definitions articles and written into a
Neo4j graph that already holds `Act`, `Paragraph` and `CaseLaw`, and that has to serve
retrieval. That is why our nodes carry `celex`, anchor to a passage id, and inherit case law
through edges that already exist.

Two discrepancies between the paper and the code, both worth knowing before quoting either.
The paper reports temperature 0.7; `constants.py` sets 0.6. And `PARAGRAPHS_TO_ANALYSE = 2`,
so the published configuration is a smoke test, not the full run behind the reported 729
candidates.

Their detector segments on `div id` matching `^\d+\.\d+`, which is the same numbered
paragraph div this project's `_get_paragraphs` reads. That makes `par_id` map onto our own
ids as `f"{celex}_{par_id}"`. It also makes the detector inherit three defects that this
project has already solved on the parser side:

| | Their detector | Graph-driven detector |
|---|---|---|
| AI Act passages | 509 | 596 paragraphs plus 221 annex points |
| GDPR passages | 372 | 415 |
| Articles never seen | 14 (AI Act), 17 (GDPR) | 0 |
| Colliding ids | 6 | 0 |
| Annexes | invisible | included |
| Source | refetches EUR-Lex | versioned `corpus/*.html` |

The AI Act gap of 87 passages is three things: 19 synthetic `.0` paragraphs for articles with
no numbered subdivision or with a lead-in sentence, and 68 further paragraphs from Article 3,
whose definitions `_get_paragraphs` splits one per entry so that a definition is separately
retrievable and citable. GDPR's gap of 43 is the same two causes, 17 and 26.

The 14 unseen AI Act articles include **Article 16, "Obligations of providers of high-risk AI
systems"**, which is laid out as a lettered list and therefore carries no numbered paragraph
div. For a compliance tool addressed to providers, losing it is disqualifying. GDPR loses
Article 19, "Notification obligation regarding rectification or erasure", the same way.

The collisions come from amending articles, which restate paragraphs of other provisions
under those provisions' own div ids. Six ids appear twice in the AI Act markup: `005.004`,
`011.003`, `017.003`, `043.004`, `047.003` and `058.003`. `_get_paragraphs` re-derives them
precisely because, as its docstring says, "reusing it would silently overwrite that
provision". Their detector takes them at face value, so a naive join would attach Article
108's obligations to Article 17(3).

The graph-driven detector keeps their sentence splitting and marker matching, and replaces
only the fetching and segmentation, drawing passages from `Paragraph` and `AnnexPoint` nodes
instead. Their `standardize_citations` is replaced by `rag/citations.py`, which already
resolves cross-references from passage text.

**Recitals are excluded**, and this holds structurally rather than by configuration: recital
divs are `rct_N` and never match `^\d+\.\d+`, so no recital is among the 509. The
graph-driven detector must preserve that exclusion when selecting nodes. The reason is legal.
Recitals do not bind: a recital cannot be relied on to derogate from the enacting terms, nor
to found an obligation the enacting terms do not impose, which is why their register is
*should* rather than *shall*. A compliance checklist listing non-binding duties is a defect,
not an approximation.

### 3.1 Models

All the obligation stages, vocabulary generation included, run on `EXTRACTION_LLM_MODEL`,
which is `gpt-5-mini` and is kept apart from `RAG_LLM_MODEL`. Retrieval stays on the cheaper
`gpt-4o-mini`; only these offline batch jobs pay for the stronger model.

The split was decided on evidence, not assumption. Regenerating the GDPR vocabulary with each
model, against the same 26 definitions of Article 4, gpt-5-mini caught a definitional `IS_A`
that gpt-4o-mini missed ("supervisory authority concerned" *means a supervisory authority
which* ...) and recovered two entity subjects, `group_of_undertakings` and
`international_organisation`, that gpt-4o-mini dropped, with no false positives added.

**Analysis** is the stage that most needs it: it extracts six elements plus a per-element
extraction method from a system prompt of roughly 4000 words into nested JSON, and a small
model degrades there by dropping elements, mislabelling `extraction_method`, and flattening
nesting. Since `extraction_method` is what `weakest_method` is built on, that degradation
would be silent and load-bearing. **Filtering** is lighter, classifying one sentence into six
categories, but runs on the same model for one setting rather than two.

Note that the paper's own weakest figures, Targets at 0.89 and Specifications at 0.94, are
already the weakest **with a 70B model**. Its footnote about stronger proprietary models
performing at least as well refers to *larger* models and does not extend to small ones.

One mitigation the paper does not use is available here: the source parses free-form JSON,
whereas this project already uses `with_structured_output`. Constraining the schema removes
exactly the failure class small models are worst at.

The record shape stays the source paper's analysis output: obligation type, addressees,
predicate with voice, targets, specifications, pre-conditions, beneficiaries, each element
paired with its extraction method (Stated, Context, Citation, Background-Knowledge, None),
plus the id of the passage it came from, which is the graph's own id.

## 4. Data model

### 4.1 Annexes

Only the AI Act has annexes. GDPR, DGA and the Data Act have none, so this part of the
schema stays empty for three of the four acts.

Annexes hang off the act, in the same structural position as recitals:

```
Act ─CONTAINS→ Chapter ─CONTAINS→ [Section ─CONTAINS→] Article ─CONTAINS→ Paragraph
 ├──CONTAINS→ Recital
 └──CONTAINS→ Annex ─CONTAINS→ AnnexPoint
```

They are binding. Annex III determines whether Chapter III applies at all, and "is my system
high-risk" is among the most common questions the assistant will receive; today it cannot be
answered from the graph. Annex IV opens with *"shall contain at least the following
information"*, an obligation whose addressee resolves through the cross-reference in
Article 11(1).

```
(:Annex)
  id       "<celex>anx_<roman>"        e.g. "32024R1689anx_III"
  number   Roman numeral               e.g. "III"
  title    e.g. "High-risk AI systems referred to in Article 6(2)"

(:AnnexPoint)
  id               "<celex>anx_<roman>.p_<nnn>"   e.g. "32024R1689anx_III.p_007"
  celex            Owning act
  text             Block text
  textEmbedding    Embedding of `text`
  section_heading  Nearest preceding internal heading, denormalised for retrieval
  point_label      Derived citation label, e.g. "III, point 5(b)"; optional
```

**Identity is positional, citation is derived.** An annex `div` has no child `div`s, so
there is no nested element carrying a point's number the way `<article>.<paragraph>` div ids
do for articles. Node identity is therefore positional and always valid, never parsed out of
prose, which is the failure mode the removed `CITES` edges demonstrated.

`point_label` is nonetheless **reliably derivable**, because inspection of the markup shows
the point number is always its own element rather than prose:

- **Table shape**, used by every annex except VI: each numbered point is a `td` pair, label
  cell then text cell, with sub-points as nested tables. Annex III's first top-level table
  yields `1.` / "Biometrics, in so far as ..." followed by `(a)`, `(b)`, `(c)` pairs.
  Two variants occur within it, both confirmed against the corpus: rows are sometimes
  indented by a **blank leading cell**, so Annex VII reads `['', '3.1.', 'The application
  ...']` and the marker is the first non-empty cell rather than cell zero; and a text cell
  wraps its prose in `<p>` in some annexes but in `<span>` in others, so a cell's own text is
  read by walking strings, excluding those belonging to a nested table.
- **Bare paragraph shape**, used by Annex VI: unclassed `<p>` elements strictly alternating
  number then prose (`1.`, "The conformity assessment procedure ...", `2.`, ...).

Text outside any table, such as the sentence opening Annex III that governs every point
under it, is kept as a point with a null label. Internal group headings (`oj-ti-grseq-1`) are
**not** points: they are denormalised onto the points below them as `section_heading`, the
same treatment `CaseLawParagraph` already gets, and they are excluded from the conservation
inventory for the same reason `oj-ti-art` is excluded on the article side, being navigation
rather than substance.

A point's full label composes from its nesting depth, so `"III, point 1(a)"` is built, not
guessed. When neither shape matches, `point_label` is null and the annex is cited whole.

**Annex VI carries no `oj-normal` class at all.** Its body is unclassed `<p>` elements. Any
inventory or splitter keyed on `oj-normal`, which is what `validation/act_source.py` uses
today for articles and recitals, would find zero fragments there and **the validation gate
would pass while the annex is entirely absent from the graph**. Annex VI is not marginal: it
is the internal-control conformity assessment procedure, and its points read "The provider
verifies ...", "The provider examines ...", which are obligations addressed to the provider.
The annex source inventory therefore covers all `<p>` text and all `td` text inside `anx_*`,
excluding `oj-doc-ti` headings, and a test pins this.

`section_heading` follows the precedent already set by `CaseLawParagraph`, which
denormalises its own heading for the same reason.

A fifth vector index, `AnnexPoint`, is created by `graph build`, keeping the invariant that
index name equals node label.

### 4.2 Obligations

```
(:Obligation)
  id                 "<source_id>#ob_<n>"    e.g. "32024R1689_023.003#ob_1"
  celex              Owning act
  modality           OBLIGATION | PROHIBITION
  obligation_type    ACTION | BEING
  predicate_text     e.g. "shall indicate their name, registered trade name ..."
  predicate_voice    active | passive
  target             String, optional
  specification      String, optional
  precondition       String, optional
  beneficiary_text   String, optional
  <element>_method   STATED | CONTEXT | CITATION | BACKGROUND | NONE, one per element
                     for addressee, predicate, target, specification, precondition,
                     beneficiary
  weakest_method     Weakest method among the populated elements, on the fixed ordering
                     STATED > CONTEXT > CITATION > BACKGROUND. NONE marks an absent
                     element and is excluded from the comparison.
```

Two load-bearing choices.

**The obligation hangs off the passage, not the article.** `(:Paragraph)-[:STATES]->` or
`(:AnnexPoint)-[:STATES]->`. The passage id is already the citation unit, so attribution and
cross-referencing work unchanged. The verification that made this safe: `_get_paragraphs` in
`scraper/eurlex_exporter.py` guarantees every article yields at least one `Paragraph`,
including unnumbered single-clause articles, which get a `.0` paragraph. Anchoring at the
paragraph therefore loses no part of the enacting terms.

**The obligation id embeds its source id.** Splitting on `#` recovers the passage without a
round trip to the graph, which is what makes the RRF boost in section 7 cheap.

`weakest_method` makes provenance operational rather than decorative. An obligation whose
addressee was inferred by Background-Knowledge is weaker evidence than one where it was
stated, and that must be visible both in ranking and in the rendered answer.

### 4.3 Actors

```
(:Actor)
  id           "provider", "provider_of_high_risk_ai_system", "controller"
  label        Human-readable form
  celex        Act that defines it; null for cross-cutting subjects
  defined_in   Id of the defining paragraph, when one exists

(:Obligation)-[:ADDRESSED_TO]->(:Actor)
(:Obligation)-[:BENEFITS]->(:Actor)
(:Actor)-[:IS_A]->(:Actor)
```

`IS_A` is what makes role filtering correct. "Provider of a high-risk AI system" is not a
different subject from "provider": it is a provider under a qualification. Kept as unrelated
nodes, a user who says "I am a provider" loses every high-risk duty and vice versa. The
filter therefore walks `IS_A*0..` before matching.

## 5. The addressee vocabulary

A curated file, `src/legal_assistant/obligations/actors.yaml`, of roughly 40 to 60 entries:

```yaml
- id: provider
  label: Provider
  celex: "32024R1689"
  defined_in: "32024R1689_003.003"
- id: provider_of_high_risk_ai_system
  label: Provider of a high-risk AI system
  celex: "32024R1689"
  is_a: [provider]
```

### 5.1 What the vocabulary is for

The vocabulary is what turns a role from a phrase into a stable identifier. It is to
addressees what the CELEX is to acts, and three steps of the pipeline require it.

**The classifier needs a list to score.** `_format_acts` already builds a block of
`- <celex>: <title>` lines and asks the LLM for a 0 to 1 score per line. The addressee
dimension needs the same block. The raw output of the extraction cannot supply it: it
contains hundreds of surface variants of the same subject, so the prompt would carry
hundreds of lines, many of them the same role written differently.

**The graph filter needs an exact id.** The Cypher match is
`(:Obligation)-[:ADDRESSED_TO]->(:Actor)<-[:IS_A*0..]-(:Actor {id: $actor})`. An obligation
labelled `"The provider or its legal representative"` carries a sentence, not an id, and
nothing matches on it.

**The answer must span the hierarchy.** A provider of a high-risk AI system bears both the
duties labelled *Provider* and those labelled *Provider of High-Risk AI System*. Only `IS_A`
joins the two, so without it the user loses one half depending on how the question was
phrased.

**The hierarchy has two tiers, and they come from different places.** This was got wrong
first time and is worth stating plainly, because the second tier is the one that carries the
value.

*Aggregate relations* are stated by the acts themselves and are read off the Definitions
article. AI Act Article 3(8) reads "'operator' means a provider, product manufacturer,
deployer, authorised representative, importer or distributor", which makes each of those an
operator.

*Qualified forms* are not defined anywhere. Article 3 defines 67 terms and not one of them is
a qualified form: it defines `provider`, never "provider of a high-risk AI system". That
phrase comes from Article 16's heading and, in the data, from the addressee strings the
extraction produces, which the source paper counts separately at 23 for high-risk providers
and 9 for general-purpose AI model providers. So qualified actors cannot be generated from
the definitions at all. They are promoted during anchoring, in section 6: an addressee string
that no actor matches but that contains a known actor's label becomes a qualified child of
that actor rather than an unmapped leftover.

### 5.2 How it is produced

Generated by a script, never written by hand, then committed.

The script reads `NodeQueries.GET_DEFINITIONS_BY_ACTS`, which already returns the paragraphs
of each act's Definitions article, and an LLM pass extracts the defined subjects and the
`IS_A` relations between them. These are not invented labels: they are the subjects the
legislature itself defined. The AI Act defines provider, deployer, authorised
representative, importer, distributor and operator in Article 3; GDPR defines controller,
processor, recipient, third party and data subject in Article 4; the Data Act defines data
holder, data recipient and user; the DGA defines data intermediation services provider and
data altruism organisation.

The generated file is versioned in the repository and read by the loader, the same way
`corpus/*.html` is the source the article nodes are built from. The `Actor` nodes live in
the graph; the file is their source, not a structure parallel to it. Committing it means a
regeneration that silently changes the roles shows up as a diff, and a wrong `IS_A` is fixed
by a commit rather than by rerunning a generation. Human review is therefore available but
not a prerequisite.

A subject two acts both define, a Member State or the Commission, is merged into a single
cross-cutting actor with `celex` and `defined_in` null. Keeping one per act would split that
subject's duties in two, and a filter on either half would answer incompletely without saying
so. Pointing at one act's definition would misrepresent the other's.

Two known weaknesses, both mitigated by the file being committed rather than derived.

An `IS_A` derived by an LLM from the definitions is reliable on textual cases, "provider of a
high-risk AI system" resolving to "provider", and weaker on cases needing interpretation, such
as an importer answering for the duties of a provider established outside the Union.

The cross-cutting merge flattens a genuine divergence, two acts using one word for different
roles, into one actor. Nothing detects that automatically. Both errors appear in a diff and
are fixed by a commit instead of being buried inside a cluster.

### 5.3 Anchoring the extracted strings

Extracted addressee strings are free text: the source paper's own output contains values
such as `"The provider or its legal representative"`. Each string is first split on
conjunctions, then anchored by embedding similarity against the actor's label plus its
definition text, above a threshold. One string may anchor to several actors. This is the
source paper's own clustering step, retargeted: instead of discovering clusters, it matches
against fixed targets that already exist.

Anchoring has four outcomes, and only the last discards anything.

1. **The string matches an actor** above threshold, and is anchored to it.
2. **It matches none but contains one.** "providers of high-risk AI systems" contains
   "provider", so it becomes `provider_of_high_risk_ai_systems` with `is_a: [provider]`. This
   supplies the qualified tier of the hierarchy, which the Definitions articles do not carry.
3. **It matches none and contains none, but recurs** at or above a frequency floor. It
   becomes a new actor with an empty `is_a`.
4. **Anything else** goes to the `unmapped` report with occurrence counts.

Outcome 3 is what makes the vocabulary complete, and the GDPR run showed why it is needed. A
Definitions article is not the whole vocabulary: the GDPR imposes duties on the data
protection officer (Articles 37 to 39), certification bodies (Article 43), monitoring bodies
(Article 41), the Board (Article 68) and the Commission (Article 45), none of which Article 4
defines, and on Member States, which nothing defines anywhere. Outcome 2 cannot reach them
either, since "Member State" contains no known label.

The extraction's addressee strings are the ground truth for who actually bears a duty, so
letting them extend the vocabulary makes it complete with respect to the obligations rather
than to one article. Promotions are deterministic and land in the committed file, so a wrong
one appears in a diff before it can affect an answer.

## 6. Ingestion

A new `obligations/` package beside `case_law/`, and a pipeline
`pipelines/obligations_ingest.py` exposed as `legal-assistant ingest obligations`. Annex
parsing extends the existing `graph build`.

Both pass through the gate in `validation/`, with checks of their own:

- every obligation resolves to an existing `Paragraph` or `AnnexPoint`;
- every `IS_A` chain terminates, with no cycles;
- every `ADDRESSED_TO` and `BENEFITS` target exists in the vocabulary;
- the share of unmapped addressee strings stays under a configured ceiling.

A failing check stops the load before anything is written, which is the existing contract of
the gate.

## 7. The obligations branch in HybridRetriever

`HybridRetriever` repeats one pattern three times, and the obligations branch honours it
rather than working around it: **every branch is additive**. It never takes guaranteed slots
from articles, it contributes to the RRF fusion as one more ranked list, and it carries its
own threshold. The docstring states the invariant plainly: the act branch is never made to
depend on the case law branch, so there is no "no match" fallback path.

### 7.1 Classification

`RawClassification` gains an addressee dimension, treated exactly as acts are: the LLM
scores every available actor 0 to 1, and those clearing `addressee_score_threshold` are
kept. `QueryClassification` carries `addressees` and `addressee_scores`.

**The addressee is orthogonal to the intent, not a third value of it.** `intent` is a
`Literal["DEFINITIONAL", "INTERPRETIVE"]` whose values are mutually exclusive, but "what has
the Court said about the duty to inform the deployer" is interpretive and prescriptive at
once. A third value would force a wrong choice. The branch therefore activates whenever the
addressee set is non-empty, whatever the intent.

**The inferred role is surfaced, not hidden.** Deciding whether one is a provider or a
deployer is itself an act of legal qualification, not a lookup. The classified role appears
in the answer and the user can correct it.

### 7.2 Selection

In `_select_obligations`, mirroring `_select_case_law`:

1. **Filter on the graph rather than search.** Obligations in the target acts whose
   `ADDRESSED_TO` walks up `IS_A*0..` to one of the classified addressees. The result is in
   the tens.
2. **Rerank the whole candidate set** with the cross-encoder, against a rendering of the
   obligation: predicate, target, specification. No new vector index and no new BM25 index,
   because the graph filter has already made the set small enough to score exhaustively.
3. Keep what clears `obligation_score_threshold`, capped at `top_k_obligations`.
4. **Contribute twice**, as the case law branch does. Surviving obligations enter the
   context in slots of their own through a new `decorate_obligation`, rendered with their
   structure and their extraction methods. Their source passages become one more ranked list
   in the article RRF, ordered by the score of the obligation that called for them.

### 7.3 The annex branch

Annex points are retrieved as a branch of their own, on the recital pattern: BM25 over the
act-scoped corpus, reranked whole, kept above `annex_score_threshold`, capped at
`top_k_annexes`. They never compete with articles for the guaranteed article slots, so
enabling them cannot regress existing behaviour.

**The branch is built but deliberately left unwired.** It carries no `ParamSpec`, and
`pipeline.py` builds its config solely from `param_specs()`, so nothing can switch it on.
Annex retrieval is not what the obligations work needs: that needs annex *nodes*, so
obligations can be extracted from annex points and hang off them. An obligation carries its
own predicate, target and specification into its own context slot and cites itself through
`point_label`, so its source passage is quotable without ever being retrieved. Annex
retrieval answers a different question, "is my system high-risk" straight from Annex III, and
is a capability to switch on deliberately rather than a parameter to sweep in an eval. Annex
cross-reference resolution in `rag/citations.py` is deferred with it.

### 7.4 New parameters

`use_obligations`, `top_k_obligations`, `obligation_score_threshold`,
`top_k_obligation_bridge`, `addressee_score_threshold`, `use_annexes`, `top_k_annexes`,
`annex_score_threshold`. All declared as `ParamSpec`s so the Streamlit app and the eval
harnesses pick them up without further work.

## 8. The compliance checklist

`pipelines/obligation_checklist.py`, exposed as
`legal-assistant checklist --act <celex> --actor <id>`.

A deterministic Cypher query over (act, actor) returning **every** obligation borne by that
actor, hierarchy included, ordered by article and paragraph. No top-k, no threshold, no
semantic retrieval: completeness holds by construction. The LLM only renders, through a
versioned prompt in `rag/prompts/`.

This is deliberately not routed through the retriever. "Answer my question" and "give me
everything I must do" are different modes, and forcing the second through top-k retrieval
would silently truncate the answer precisely where completeness is the whole point.

## 9. Degradation

The retriever invariant is preserved: **the act branch never depends on the obligations
branch**.

- No addressee above threshold: the branch is skipped and behaviour is identical to today.
- A misclassified addressee: the branch yields irrelevant obligations, which the reranker
  drops at the threshold. The RRF contribution being purely additive, it cannot push out an
  article that would otherwise have been retrieved.
- Obligations not yet ingested: the branch finds nothing and skips, exactly as an
  un-ingested judgment stub does today.

## 10. Tests and evaluation

Unit tests in `tests/obligations/` for conjunction splitting, anchoring, `IS_A` traversal
and the validation checks; in `tests/scraper/` for annex parsing, including a flat annex
with no internal headings and one where `point_label` cannot be derived; in `tests/rag/` for
branch activation, empty degradation and bridge construction. Neo4j and OpenAI are mocked
throughout.

A new eval, **role recall**: a set of questions with a known role and a known expected set of
obligations, scoring the share of expected obligations present in the assembled context. The
dataset is generated from the graph itself, so it needs no annotation.

A non-regression check accompanies it: the existing evals, run with `use_obligations=false`
and `use_annexes=false`, must return unchanged results.

## 11. Implementation order

The scope spans two subsystems, so the order matters: annexes must land before the
extraction runs, or the extraction has to be run twice.

1. **Annexes into the graph.** Parser, `Annex` and `AnnexPoint` nodes, loader, vector index,
   annex retrieval branch. Independently useful: it lets the assistant answer "is my system
   high-risk" from Annex III before a single obligation exists.
2. **Vocabulary and schema.** The generator script, the committed `actors.yaml`, actor
   nodes, obligation schema, validation checks. No retrieval change yet.
3. **Ingestion.** Run the paper's extraction over paragraphs and annex points, anchor the
   addressee strings, load, inspect the unmapped report.
4. **Retrieval branch.** Classifier addressee dimension, `_select_obligations`, the RRF
   contribution, `decorate_obligation`, the new parameters.
5. **Checklist pipeline and evals.**

Steps 1 and 2 are independent of each other and can proceed in parallel.

## 12. Documents to update

Per the ownership table in `CLAUDE.md`:

- `docs/knowledge-graph.md`: Annex, AnnexPoint, Obligation and Actor nodes, the new
  relationships, the fifth vector index. Fields re-derived from a live Neo4j with
  `db.schema.nodeTypeProperties()`, not written from the loader source.
- `docs/architecture.md` and the Layout table in `CLAUDE.md`: the new `obligations/` package.
- `README.md`: `ingest obligations`, `checklist`, and the new retrieval flags.

## 13. Decisions taken, and what was rejected

| Decision | Rejected alternative | Why |
|---|---|---|
| Additive branch with a double contribution | Hard pre-filter of articles by role | A wrongly inferred role would silently delete the right provisions while the answer still looked complete |
| Addressee orthogonal to intent | A third intent value | Interpretive and prescriptive are not mutually exclusive |
| Generated, committed vocabulary | Clustering alone, as in the source paper | Emergent clusters are unstable across rebuilds, carry no normative labels, cannot be enumerated in a prompt, and cannot express `IS_A` |
| Vocabulary as a versioned file | Actors written straight into Neo4j | A regeneration that changes the roles shows up as a diff, and a wrong `IS_A` is fixed by a commit |
| `Actor` / `IS_A` in the existing Neo4j graph | The paper's `Entity` / `Person` / `HAS_CHILD` with a GraphML export | That module is not published, and the paper calls its own ontology "only partially developed". Ours has to serve retrieval inside a graph that already holds acts, passages and case law, not visualisation as a standalone artefact |
| Vocabulary generated from Definitions articles | Generalised concepts selected by legal experts, as in the paper | Adding an act costs nothing, and the subjects are the ones the legislature itself defined rather than a curator's choice |
| Obligation hangs off the passage | Hangs off the article | The passage id is already the citation unit, and every article yields at least one passage |
| Positional annex ids, derived citation labels | Ids parsed from the prose numbering | Repeats the failure mode of the removed `CITES` edges |
| Checklist outside the retriever | Checklist as a retrieval mode | Top-k truncation defeats the purpose of a checklist |

## 14. Known limits

- Obligation elements the source paper measures as weakest, targets at 0.89 accuracy and
  specifications at 0.94, stay the weakest here. `weakest_method` exposes the uncertainty
  rather than resolving it.
- `point_label` composes from the markup's own numbering and covers both observed shapes.
  A third shape appearing in a future act would yield null labels, and the annex would be
  cited whole rather than by point.
- Temporal constraints and logical connectors between obligation elements are not modelled.
  The source paper excludes them too, and their absence means deadlines carried in
  specifications are text, not queryable data.
- Obligations inherit case law for free through the `INTERPRETS` edges already attached to
  their source passage, but nothing yet ranks an obligation by whether the Court has
  construed it. That is a natural follow-up, not part of this design.
