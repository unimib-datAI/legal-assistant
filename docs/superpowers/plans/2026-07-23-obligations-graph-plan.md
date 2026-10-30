# Obligations graph: implementation plan

**Date:** 2026-07-23
**Spec:** [`../specs/2026-07-23-obligations-graph-design.md`](../specs/2026-07-23-obligations-graph-design.md)

Five phases. Phases 1 and 2 are independent and can run in parallel; 3 depends on both, 4 on
3, 5 on 4.

Every phase is written test first, per the project's `test-driven-development` skill: write
the failing test, watch it fail for the right reason, then implement. Neo4j and OpenAI are
mocked throughout, using the factories in `resources.py` as the seam.

---

## Phase 1: Annexes into the graph

Only the AI Act has annexes. GDPR, DGA and the Data Act have none, so three of the four acts
must come through this phase completely unchanged, and that is itself a test.

### 1.1 The annex splitter

**Files:** `scraper/eurlex_exporter.py`, `tests/scraper/test_annex_parser.py`

The unit is a `(label, text)` pair. Two markup shapes occur and both must be handled:

| Shape | Annexes | Structure |
|---|---|---|
| Table | all but VI | `td` pairs, label cell then text cell, sub-points as nested tables |
| Bare paragraph | VI | unclassed `<p>` alternating number then prose |

Traversal note: `find_all("table")` double-counts, because a top-level table in Annex III
contains the nested tables of its own sub-points. Walk top-level tables, then descend.

Two variants inside the table shape, both found while implementing and neither visible from
the paper or from a casual read of the markup:

- Rows may be **indented by a blank leading cell**, so Annex VII reads
  `['', '3.1.', 'The application ...']`. Reading the marker off cell zero gives an empty
  label and pushes the real marker into the text.
- A text cell wraps its prose in `<p>` in some annexes and in `<span>` in others. Annex VII's
  text cells hold a `<span>` followed by the nested tables of their sub-points, so a splitter
  keyed on `<p>` returns empty text for every point there.

Tests to write first:

- Annex III first point yields label `1.` and the Biometrics text, with `(a)`, `(b)`, `(c)`
  as its sub-points.
- **Annex VI yields four points.** This is the regression guard for the shape that carries
  no `oj-normal` class; a splitter keyed on that class returns zero here and must fail.
- Annex VI point 2 text begins "The provider verifies", confirming label and prose are not
  concatenated into one node.
- `section_heading` on a point inside Annex VII resolves to its nearest preceding
  `p.oj-ti-grseq-1`.
- `point_label` composes across nesting: sub-point `(a)` of Annex III point `1.` gives
  `"III, point 1(a)"`.
- An annex matching neither shape yields points with `point_label` null and non-empty text,
  rather than yielding nothing.
- The three annex-less acts return an empty annex list.

Then implement `_get_annexes`, mirroring the docstring style of `_get_paragraphs`, which
already documents its three shapes the same way.

### 1.2 The conservation inventory

**Files:** `validation/act_source.py`, `tests/graph_validation/test_annex_conservation.py`

This is the highest-risk step in the phase and the reason it precedes the loader.

`html_fragments` currently collects `p.oj-normal` inside `art_*` and `rct_*`;
`reconstructed_fragments` collects the text of `Paragraph` and `Recital` nodes. If annexes
reach the graph but not both sides of this check, a dropped annex is invisible. Worse, if
they are added to both sides but keyed on `oj-normal`, **Annex VI contributes zero fragments
to each side and the gate passes while the annex is entirely missing**, which is the silent
failure shape the removed `CITES` edges had.

The annex inventory therefore covers all `<p>` text and all `td` text inside `anx_*`,
excluding both heading classes, `oj-doc-ti` and `oj-ti-grseq-1`, and `reconstructed_fragments`
gains `AnnexPoint`. Excluding group headings is not a shortcut: they are carried as
`section_heading` metadata rather than as point text, which mirrors the article side, where
`_TEXT_CLASS` is `oj-normal` alone and `oj-ti-art` never enters the inventory.

Tests to write first:

- The annex inventory is non-empty for **each of the 13 annexes**, asserted per annex, never
  in aggregate.
- Removing one annex point from a plan makes `conservation` report a violation.
- The three annex-less acts produce a byte-identical inventory to the current implementation.

### 1.3 Schema, loader, index

**Files:** `graph/loader.py`, `graph/queries.py`, `pipelines/graph_build.py`,
`tests/graph/test_annex_loader.py`

`_load_annexes` beside `_load_recitals`, called from `_emit`. Nodes `Annex` and `AnnexPoint`,
`CONTAINS` edges from the act and from the annex. A fifth vector index named `AnnexPoint`,
preserving the invariant that index name equals node label, which `RagContext._vector_store`
relies on.

New queries in `NodeQueries`: `GET_ANNEX_POINTS_BY_ACTS` and `GET_ANNEXES_BY_ACTS`, with
bound parameters, never formatted.

Tests: ids match `<celex>anx_<roman>` and `<celex>anx_<roman>.p_<nnn>`; containment is a tree,
which the existing `structural` check asserts once annexes are in the plan; two builds over
the same input agree, which `build_validated` already checks for free.

### 1.4 The annex retrieval branch (built, then parked)

**Status: implemented and tested, deliberately not wired up.**

The branch exists and passes its tests, but carries no `ParamSpec`, and `pipeline.py` builds
its config solely from `param_specs()` and rejects any override outside it. So the branch is
unreachable from the application until something gives it a switch.

That is on purpose. The obligations work needs annex **nodes**, which phases 1.1 to 1.3
deliver, not annex **retrieval**: an obligation extracted from an annex point carries its own
predicate, target and specification into its own context slot, and cites itself through
`point_label`, so the source passage never has to be retrieved to be quoted. Annex retrieval
buys something different, answering "is my system high-risk" straight from Annex III, and
that is a capability to turn on when it is wanted, not a knob to sweep in an eval.

Not done, and deferred with it: annex cross-reference resolution in `rag/citations.py`.

To activate: add the three `ParamSpec`s, or read the flag from `config.py` as a deployment
setting. Then re-run the non-regression eval, since enabling it changes assembled context.

The design as built:

**Files:** `rag/retrievers/hybrid.py`, `rag/methods/hybrid.py`, `rag/documents.py`,
`rag/citations.py`, `tests/rag/test_annex_branch.py`

On the recital pattern, the closest precedent: BM25 over the act-scoped corpus, reranked
whole, kept above `annex_score_threshold`, capped at `top_k_annexes`, never competing for the
guaranteed article slots. `decorate_annex` renders the citation from `point_label`, falling
back to the annex as a whole.

`rag/citations.py` gains annex cross-reference resolution, so a retrieved article referring
to Annex III can bridge to it.

Tests: with `use_annexes=false` the returned document list is identical to today's, which is
the non-regression guarantee; a high-risk classification query retrieves Annex III points; an
act with no annexes exercises the branch and returns empty without error.

**Exit criteria:** `graph build` loads all four acts through the gate with no new violations;
existing evals unchanged with `use_annexes=false`; "is my system high-risk" retrieves
Annex III.

---

## Phase 2: Vocabulary and schema

Independent of phase 1. Nothing here changes retrieval.

**As built, three deviations from the plan as written.** The models came before the
generator, since the generator produces them. `validation/obligation_source.py` became
`validation/obligation_checks.py`, because the `_source` suffix in that package means a
source inventory, and these are checks. The extraction-record model is deferred to 3.1,
where the source pipeline's real output is pinned as a fixture first, rather than modelled
from the paper's listings.

### 2.1 The vocabulary generator

**Files:** `obligations/vocabulary.py`, `obligations/actors.yaml`, `rag/prompts/vocabulary.py`,
`tests/obligations/test_vocabulary.py`

A script, never a hand-written file. It reads `NodeQueries.GET_DEFINITIONS_BY_ACTS`, which
already returns the paragraphs of each act's Definitions article, and an LLM pass with
structured output extracts the defined subjects and the `IS_A` relations between them. The
generated YAML is committed and read by the loader, the same way `corpus/*.html` is the
source the article nodes are built from.

The LLM comes from `make_chat_llm` with an explicit temperature; the prompt is a versioned
`PromptVersion` in `rag/prompts/`, never inline.

The prompt states the obligation-bearing test as a **criterion**, never as a list of expected
roles. An earlier draft enumerated "providers, deployers, controllers, processors ...", which
invites the model to report another act's vocabulary: run against the DGA it would happily
return a "controller" the DGA never defines.

`IS_A` here covers only what a Definitions article actually states, which is aggregate
definitions such as AI Act Article 3(8), "'operator' means a provider, product manufacturer,
deployer, authorised representative, importer or distributor". Qualified forms are **not**
obtainable at this stage: Article 3 defines 67 terms and none of them is one. They are
promoted in 3.4.

Tests, LLM mocked: a fixture of AI Act Article 3 yields `provider` and `deployer`; an
aggregate definition makes each member `is_a` the aggregate; a subject two acts both define
becomes cross-cutting with a null `celex`; a `defined_in` the query never returned is
dropped; ids are slugs; regenerating over the same fixture produces byte-identical YAML, so a
diff always means a real change.

### 2.2 Actor and Obligation schema

**Files:** `obligations/models.py`, `graph/queries.py`, `tests/obligations/test_models.py`

Pydantic models for the extraction record and the graph nodes, per spec §4.2 and §4.3.
`weakest_method` is computed rather than supplied, on the ordering
STATED > CONTEXT > CITATION > BACKGROUND, with NONE excluded as marking an absent element.

Tests: `weakest_method` over mixed elements; an obligation with every element NONE except the
predicate, the only mandatory element in the source framework; id composition
`<source_id>#ob_<n>` and its inverse, recovering the source id by splitting on `#`, which is
what phase 4 depends on.

### 2.3 Validation checks

**Files:** `validation/obligation_source.py`, `tests/obligations/test_validation.py`

Four checks through the existing gate: every obligation resolves to an existing `Paragraph`
or `AnnexPoint`; every `IS_A` chain terminates; every `ADDRESSED_TO` and `BENEFITS` target
exists in the vocabulary; the unmapped addressee share stays under a configured ceiling.

Tests: each check fails on a crafted bad plan and passes on a good one. A cycle
`a IS_A b IS_A a` must be caught rather than hang the traversal.

**Exit criteria:** `actors.yaml` generated and committed; schema loads into a mocked graph;
every check has a failing case under test.

---

## Phase 3: Extraction and ingestion

Depends on 1 and 2. This phase absorbed the findings from reading the source repository, so
it differs most from a naive reading of the paper.

### 3.1 Pin the source contract

**Files:** `tests/obligations/fixtures/`, `tests/obligations/test_extraction_io.py`

**Done.** No LLM call was needed: the source repository's `setup_evaluation/` holds real
per-sentence analysis outputs, and one is committed verbatim at
`tests/obligations/fixtures/analysis_output_aia_23_3.json`. The reader in
`obligations/extraction_io.py` parses it into `AnalysedObligation` records, and the models in
`obligations/models.py` gained the record shape the real output revealed: each element
(addressees, targets, specifications, pre-conditions, beneficiaries) is a **list** of
`{extraction_method, value}`, the predicate is a single object with a `verb`, and one passage
can yield several obligations. This is the extraction-record model deferred from phase 2, now
built against reality rather than the paper's Listing 2. Mapping a record onto graph nodes and
edges stays in 3.4.

### 3.2 The graph-driven detector

**Files:** `obligations/detection.py`, `tests/obligations/test_detection.py`

Replaces the source repository's `obligation_detection`. Their version fetches EUR-Lex by
URL and segments on `div id` matching `^\d+\.\d+`; ours draws passages from the graph. Their
sentence splitting and marker matching are kept as they are.

What this recovers, verified against `corpus/32024R1689.html`:

- **14 AI Act articles and 17 GDPR articles their detector never sees**, because they carry
  no numbered paragraph div. Among them AI Act Article 16, "Obligations of providers of
  high-risk AI systems", and GDPR Article 19. The graph has them as synthetic `.0`
  paragraphs.
- **6 colliding ids.** `005.004`, `011.003`, `017.003`, `043.004`, `047.003` and `058.003`
  each appear twice in the markup, because amending articles restate other provisions under
  those provisions' own div ids. `_get_paragraphs` re-derives them; their detector does not,
  so a naive `f"{celex}_{par_id}"` join would attach Article 108's obligations to Article
  17(3).
- **Annex points**, invisible to a `^\d+\.\d+` regex.

Recitals stay excluded. This holds structurally in their pipeline, since `rct_N` never
matches the regex, and must be preserved deliberately in ours, where recitals are one node
label away.

Tests: the AI Act yields 596 paragraph passages, not 509, plus 221 annex points; Article 16
is present; the passage carrying `art_108`'s restatement of Article 17(3) has an id distinct
from real Article 17(3); no passage id starts with a recital prefix; annex points appear as
passages.

### 3.3 Filtering and analysis

**Done.** `obligations/filtering.py` sorts a candidate sentence into the prompt's seven
categories and advances only obligation and prohibition; `obligations/analysis.py` drives the
analysis prompt and hands its JSON to `parse_analysis`. Both load the vendored prompts through
`obligations/prompts.py` and parse with `obligations/json_io.py`; neither imports the
reference-only vendored modules. A malformed response yields nothing rather than raising, so
one bad sentence cannot stop a batch.

An end-to-end run on the live graph with gpt-5-mini validated the whole chain on the case that
motivated it: **Article 16, invisible to the paper's detector, yielded all 13 of its lettered
obligations**, each addressed to "Providers of high-risk AI systems", a qualified form that
3.4 promotion will resolve to `provider`.

**Files:** `obligations/filtering.py`, `obligations/analysis.py`, `obligations/prompts.py`,
`obligations/json_io.py`, `config.py`

Adapters that drive the paper's prompts with this project's own LLM client. The prompts are
registered verbatim as `PromptVersion`s in `rag/prompts/obligations.py`; the paper's Python
modules are not kept in the tree. The adapters reproduce their prompt templating (`@Sentence`,
`@Context`, `@Citations`) and JSON parsing in this codebase's idiom. `references.py` adapts
their `solve_references`, resolving the same-act citation ids detection records to article
text from the graph, so the analysis stage receives cited provisions rather than bare ids.

Both stages run on `EXTRACTION_LLM_MODEL` (`gpt-5-mini`), kept apart from the RAG pipeline's
`gpt-4o-mini`. The split earned its keep on the vocabulary: on the GDPR Article 4 definitions
gpt-5-mini caught a definitional `IS_A` and two entity subjects gpt-4o-mini missed. Analysis
is the stage that most needs the stronger model, extracting six elements plus a per-element
extraction method into nested JSON, where a small model drops elements and mislabels
`extraction_method`, exactly what `weakest_method` rests on. Both go through
`with_structured_output`, which the source does not use and which removes the failure class
small models are worst at. Note gpt-5-mini requires its default temperature; a non-default
value is rejected.

Before the full run, replay the paper's 60-sentence validation protocol on the chosen models
and record the result. It is cheap and it turns the model choice from an assumption into a
measurement.

### 3.4 Anchoring and ingestion

**Files:** `obligations/anchoring.py`, `pipelines/obligations_ingest.py`, `cli/main.py`,
`tests/obligations/test_anchoring.py`

Split each addressee string on conjunctions, then match by embedding similarity against actor
label plus definition text, above a threshold, allowing one string to anchor to several
actors.

**Promotion is what completes the vocabulary**, in two forms, both writing back into the
committed file.

A string matching no actor but containing one, "providers of high-risk AI systems"
containing "provider", becomes a new actor with `is_a` pointing at what it qualifies. This is
the qualified tier, which Definitions articles cannot supply; without it `IS_A` stays almost
empty and the role filter loses the split it exists to bridge.

A string matching and containing nothing, but recurring at or above a frequency floor,
becomes a new actor with an empty `is_a`. The GDPR run proved this necessary: Article 4 never
defines the data protection officer, certification bodies, the Board or the Commission, and
nothing anywhere defines Member States, yet all bear duties. Containment cannot reach them,
since "Member State" contains no known label.

Whatever neither matches, contains nor recurs goes to the unmapped report, as a diagnostic
rather than a queue of manual work.

Tests: a recurring unknown string is promoted while a one-off is not; a promoted actor is
written back and survives a reload; promotion is idempotent, so a second ingest over the same
extraction leaves the vocabulary byte-identical.

`legal-assistant ingest obligations` beside `ingest case-law`, following the existing
subparser shape in `cli/main.py`.

**Parallelism.** The passage is the unit of work, and passages run concurrently through
`obligations/concurrency.py`: bounded to `EXTRACTION_MAX_WORKERS` (8) so the request burst
stays under the provider's rate limit, with exponential-backoff retry on a 429 as the second
defence. Order is preserved, so per-passage ordinal numbering is unaffected. Measured on the
live GDPR graph: 3.3s per passage at 8 workers against ~25s sequential, ~7.6x, no 429s, which
puts the full GDPR run near 18 minutes rather than 2 to 3 hours.

Tests, embeddings mocked: `"The provider or its legal representative"` anchors to both
`provider` and `authorised_representative`; a string below threshold lands in `unmapped` and
is not silently dropped; `"providers of high-risk AI systems"` anchors to the qualified actor
rather than the bare one.

**The vocabulary is written into the graph by the build**, not only held as `actors.yaml`.
A first real GDPR write surfaced this: the graph had never held `Actor` nodes, so every
`ADDRESSED_TO` edge pointed at a node that did not exist and Neo4j's `MERGE` dropped it in
silence, the exact failure the removed `CITES` edges had. `build_obligations` now upserts the
whole vocabulary (idempotently), and the ingest validation gained `dangling_edges` and
`conflicting_upserts` so a dropped edge or a duplicate id fails loudly. `STATES` is treated
as cross-document, since its passage endpoint lives in the act plan. A second bounded write
confirmed the edges land: obligations addressed to `controller` resolved to 12 `ADDRESSED_TO`
edges.

**Exit criteria:** obligations loaded for all four acts through the gate; the unmapped report
inspected and under the ceiling; Article 16 obligations present in the graph.

---

## Phase 4: The retrieval branch (built, inert like the annex branch)

**Status: the retriever machinery is built and tested; activation via the shared classifier
is deferred.** The obligations branch in `HybridRetriever` filters obligations by the
classified addressee role (`GET_OBLIGATIONS_FOR_ACTORS`, walking `IS_A`), reranks them, keeps
the survivors above `obligation_score_threshold`, and adds them in their own slots via
`decorate_obligation`, whose header carries the source provision and the `weakest_method`
trust. It is additive: articles keep their slots, and with `use_obligations=False` (the
default) or no classified addressee the returned documents are unchanged.

`QueryClassification` gained `addressees`/`addressee_scores` (default empty), so the branch
reads them where the classifier provides them. The active v5 classification prompt does not
score addressees, so the branch stays inert. Activation is a v6 prompt that adds the addressee
axis plus the classifier fetching the actor list, held back deliberately: it changes the
shared classifier every RAG query uses, so it must land with the non-regression eval as a
guard. Following the annex-branch decision, the obligations flags carry no `ParamSpec`.

### Original design

**Files:** `rag/intent_classifier.py`, `rag/prompts/retrieval.py`, `rag/retrievers/hybrid.py`,
`rag/methods/hybrid.py`, `rag/documents.py`, `tests/rag/test_obligations_branch.py`

### 4.1 Classifier

`RawClassification` gains an addressee dimension scored 0 to 1 per actor, thresholded exactly
as `_select_acts` thresholds acts. `QueryClassification` carries `addressees` and
`addressee_scores`. The actor block is formatted and cached like `_format_acts`.

`intent` stays `Literal["DEFINITIONAL", "INTERPRETIVE"]`. The addressee is orthogonal to it,
not a third value, because "what has the Court said about the duty to inform the deployer" is
interpretive and prescriptive at once and a third value would force a wrong choice.

### 4.2 Selection

`_select_obligations`, mirroring `_select_case_law`:

1. filter on the graph, walking `IS_A*0..` from the classified addressees;
2. rerank the whole candidate set, which the graph filter has made small enough to score
   exhaustively, against a rendering of predicate, target and specification;
3. keep above `obligation_score_threshold`, cap at `top_k_obligations`;
4. contribute twice: own context slots via `decorate_obligation`, and source passages as one
   more ranked list in the article RRF, ordered by the score of the obligation that called for
   them. The source passage comes from splitting the obligation id on `#`, with no round trip
   to the graph.

New `ParamSpec`s: `use_obligations`, `top_k_obligations`, `obligation_score_threshold`,
`top_k_obligation_bridge`, `addressee_score_threshold`.

Tests: no addressee above threshold leaves the returned documents identical to phase 3
behaviour; a misclassified addressee yields obligations the reranker drops, with article slots
unchanged; the bridge orders source passages by obligation score; obligations not yet ingested
make the branch skip, as an un-ingested judgment stub does today.

**Exit criteria:** existing evals with `use_obligations=false` return unchanged results.

---

## Phase 5: Checklist and evaluation

**Files:** `pipelines/obligation_checklist.py`, `rag/prompts/checklist.py`, `cli/main.py`,
`evals/`, `tests/pipelines/test_checklist.py`

**Checklist done.** `legal-assistant checklist --act <celex> --actor <id>`. A deterministic
Cypher query over (act, actor) returning every obligation borne by that actor, hierarchy
included via `IS_A*0..`, ordered by provision. No top-k, no threshold, no semantic retrieval.
The LLM only renders, through the versioned `obligation_checklist` prompt. Validated on the
live graph: the checklist for `controller` returns 103 obligations, the 90 addressed directly
plus 13 from qualified children (`each_controller`, `joint_controllers`, `data_controllers`).

**Role-recall eval: remaining.** It measures the phase-4 retrieval branch, so it follows the
classifier activation. The deterministic checklist recalls the expected set by construction,
which is the completeness guarantee the eval would otherwise establish for the branch.

Original design:

Tests: an actor with a child actor returns the union of both sets; ordering is by article then
paragraph; an actor with no obligations returns an empty checklist rather than an error.

**Role recall eval:** questions with a known role and a known expected obligation set, scoring
the share of expected obligations present in the assembled context. The dataset is generated
from the graph, so it needs no annotation. Accompanied by a non-regression run of the existing
evals with both new branches off.

---

## Documents to update

Per the ownership table in `CLAUDE.md`, at the end of the phase that changes each:

- `docs/knowledge-graph.md` after phases 1 and 3, with fields **re-derived from a live Neo4j**
  using `db.schema.nodeTypeProperties()`, never written from the loader source.
- `docs/architecture.md` and the Layout table in `CLAUDE.md` after phase 2, for the new
  `obligations/` package.
- `README.md` after phases 3, 4 and 5, for `ingest obligations`, `checklist` and the new
  retrieval flags.

## Risks

| Risk | Mitigation |
|---|---|
| An annex shape neither splitter handles is added silently | The per-annex conservation assertion in 1.2 fails rather than passing empty |
| The source pipeline's real output differs from the paper's listings | Phase 3.1 pins the real output as a fixture before any integration code is written |
| `gpt-4o-mini` degrades on the analysis stage, silently corrupting `extraction_method` | A stronger model for analysis, `with_structured_output`, and the 60-sentence protocol replayed before the full run |
| LLM-derived `IS_A` is wrong on interpretive cases, such as an importer answering for a provider established outside the Union | The vocabulary is a committed file, so the error appears in a diff and is fixed by a commit |
| The obligations branch regresses article retrieval | Every branch is additive; the non-regression eval with the flags off is an exit criterion of phases 1 and 4 |
