# Extending the system

The recipes: adding a retrieval strategy, an act, a pipeline, a summarisation task.

For how the system is laid out see [architecture.md](architecture.md); for the Neo4j schema
see [knowledge-graph.md](knowledge-graph.md).

## Reusable parts

These are the pieces to build on. Extending them is a small, local change; copying them is
the mistake they exist to prevent.

| Module | What it gives you |
|---|---|
| `resources.py` | Configured Neo4j / OpenAI clients. Never construct your own. |
| `rag/methods/base.py` | `RagMethod` + `ParamSpec` — the retrieval-strategy contract |
| `rag/methods/registry.py` | The list every consumer reads to discover strategies |
| `rag/context.py` | `RagContext` — expensive shared resources, built once, stores lazy |
| `rag/documents.py` | Passage ids and source headers, shared by every retriever |
| `rag/acts.py` | CELEX ↔ display name, act detection in free text |
| `rag/citations.py` | Provisions cited in a passage's own text |
| `rag/prompts/` | Versioned prompts, so a prompt change is reviewable and reversible |
| `graph/queries.py` | Every Cypher statement the application runs |
| `pipelines/summaries.py` | `SummaryTask` — declare a new node kind to summarise in ~10 lines |
| `logging_setup.py` | `configure_logging()` and `quiet()` |

### `RagContext` — shared resources

Constructing the graph connection, the vector stores, the cross-encoders and the LLM
clients is expensive, so one `RagContext` is built per process and handed to every method.
The two node-level vector stores are `cached_property`, so a method that never touches case
law never pays to open that store. Reach for what you need; do not build your own.

```python
ctx.graph                     # LangChain Neo4jGraph
ctx.embeddings                # the one embedding client
ctx.article_vector_store      # eager
ctx.paragraph_vector_store    # lazy
ctx.case_law_vector_store     # lazy
ctx.classifier                # QueryClassifier
ctx.synthesis_llm / filter_llm / curator_llm
ctx.make_hyde_generator(iterations)
```

### `rag/documents.py` — passage conventions

Every retriever's output flows into the same fusion, reranking, synthesis and attribution
stages. Those stages identify a passage by its id and read it by its bracketed header, so
both conventions live here rather than in each retriever:

`doc_id` · `copy_doc` · `neighbour_ids` · `recital_header` · `decorate_article` ·
`decorate_recital` · `decorate_case_law`

`copy_doc` matters more than it looks: BM25 hands back the very objects held in a
retriever's corpus cache, so enriching in place would corrupt the shared corpus. The
`decorate_*` functions all return copies for the same reason.

### `rag/prompts/` — versioned prompts

A prompt is a `PromptVersion` (text, version, date, changelog note, `active` flag)
registered into one shared registry, grouped by stage: `retrieval.py`, `synthesis.py`,
`summaries.py`, `case_law.py`. Consumers import a plain string:

```python
from legal_assistant.rag.prompts import ANSWER_SYNTHESIS_PROMPT
```

To ship a new version: add `<NAME>_V<n>`, register it with `active=True`, flip the previous
one to `active=False`. Rollback is the reverse flip. To A/B without touching flags, pin the
version — `RAGPipeline(synthesis_prompt_version="v9")`, or
`--prompt-version v9` / `--synthesis-prompt v9`. The version in use is logged on every run
and written into eval output.

Exactly one version per name must be active; a violation fails fast at import.

---

## Adding a new RAG methodology

The worked example. Three files change, nothing else.

### 1. Write the retriever — `rag/retrievers/<name>.py`

A LangChain `BaseRetriever` whose tunables are pydantic fields. Take the shared resources
as fields rather than constructing them.

```python
from typing import Any, List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from legal_assistant.graph.queries import NodeQueries
from legal_assistant.rag.documents import copy_doc, decorate_article, doc_id


class MyRetriever(BaseRetriever):
    """One sentence on what makes this strategy different."""

    graph: Any
    vector_store: Any
    classifier: Any = None

    top_k: int = 5              # every tunable is a field...
    use_expansion: bool = True  # ...named exactly as its ParamSpec

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        classification = self.classifier.classify(query)
        if not classification.acts:
            return []                       # no target acts: return empty, don't guess

        rows = self.graph.query(
            NodeQueries.GET_ARTICLES_BY_ACTS, params={"acts": classification.acts}
        )
        ...
        return [decorate_article(copy_doc(d)) for d in ranked[: self.top_k]]
```

Rules that keep it composable:

- **Decorate before returning.** Synthesis and attribution read the bracketed header.
- **Copy before enriching.** Never mutate a document you got from a cache.
- **Respect the classifier's act selection**, so `--override` and the eval harnesses mean
  the same thing across strategies.
- **Return `[]` rather than falling back** when there is nothing to retrieve. Silent
  fallbacks make eval numbers unreadable.

### 2. Declare it — `rag/methods/<name>.py`

`param_specs()` is the single declaration of the knobs. Each `ParamSpec.name` must match a
field on the retriever, because the config dict is splatted into the constructor.

```python
from typing import Any, Dict, List

from langchain_core.retrievers import BaseRetriever

from legal_assistant.rag.context import RagContext
from legal_assistant.rag.methods.base import ParamSpec, RagMethod
from legal_assistant.rag.retrievers.mine import MyRetriever


class MyRagMethod(RagMethod):
    id = "mine"
    name = "My strategy"
    description = "What it does and when it beats the default."

    def param_specs(self) -> List[ParamSpec]:
        return [
            ParamSpec("top_k", "Top-k", "int", 5, min=1, max=20, step=1),
            ParamSpec("use_expansion", "Expand the query", "bool", True,
                      help="Shown as tooltip in the UI and in `rag methods`."),
        ]

    def build_retriever(self, ctx: RagContext, config: Dict[str, Any]) -> BaseRetriever:
        return MyRetriever(
            graph=ctx.graph,
            vector_store=ctx.article_vector_store,
            classifier=ctx.classifier,
            **config,
        )
```

`kind` drives the frontend widget: `bool` → toggle, `int`/`float` → slider. If a param is
consumed while *building* rather than passed through — as `hyde_iterations` is — pop it
from the config before splatting, the way `methods/hybrid.py` does.

### 3. Register it — `rag/methods/registry.py`

```python
_METHODS: List[RagMethod] = [
    HybridRagMethod(),
    TopicsRagMethod(),
    MyRagMethod(),        # ← the whole integration
]
```

The list is explicit on purpose: import-order magic in a decorator registry is harder to
debug than one line.

### What you get for free

```bash
legal-assistant rag methods                        # documented, with defaults
legal-assistant rag query "..." --method mine
legal-assistant rag query "..." --method mine --override top_k=8
legal-assistant eval retrieval --method mine       # deterministic A/B vs. hybrid
legal-assistant eval ragas --method mine
```

Plus auto-generated hyperparameter controls in the frontend. No CLI change, no eval change,
no UI change.

### Verify it

```bash
legal-assistant rag methods | grep -A3 mine        # registered with the right params
legal-assistant rag query "What is personal data?" --method mine
legal-assistant eval retrieval --method mine --repeats 1
```

`eval retrieval` is the honest comparison: no generation and no LLM judge, so a difference
in `top1_anchor` / `recall_at_k` / `mrr` is a real difference in retrieval.

---

## Other extension points

### A new act

Add its CELEX to `CELEX_TO_ACT_NAME` and its surface forms to `ACT_NAME_KEYWORDS` in
`rag/acts.py` — order matters there, more specific keywords first, so "data governance act"
is never read as "data act". Then `legal-assistant graph build --celex <new> --no-clear`.

Nothing else is hardcoded: `celex_instrument_and_numbers` derives "Regulation 2016/679"
style citations from the CELEX itself, so citation extraction works for a new act without
an entry.

### A new node kind to summarise

Declare a `SummaryTask` in `pipelines/summaries.py` (fetch query, update query, prompt
pair, row→params mappers) and add it to `TASKS`. The async driver, concurrency, error
handling and resumability are shared. Keep the fetch query filtered to `summary IS NULL` so
re-runs stay idempotent.

### A new pipeline

Put the logic in `pipelines/<name>.py` as a plain function returning a result object — no
printing, no `sys.exit`, no Streamlit. Add a subparser in `cli/main.py` whose handler is a
few lines. A Streamlit page, if wanted, calls the same function.
