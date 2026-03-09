# Legal KG V1.1
## Document assumption
The following knowledge graph is evaluated based on the Eurolex document structure. In particular, the following assumptions were considered:
* Each **act** is composed of one or more **chapters**
* Each **chapter** is composed of one or more **articles**
    * In some cases, **articles** will be divided into **sections** rather than directly into **chapters** (Chapter -> Section -> Article)
* **Articles** may cite other **articles** or **paragraphs**
* An **article** is typically composed of a series of **paragraphs**
* **Case law** interprets **articles**; one or more **case law** may refer to a specific **article** or **act**

## Knowledge Graph
![image](img/KG.png)

## Nodes
In order to maintain the uniqueness of nodes such as articles or chapters (which remain constant across different documents), it was decided to utilise a string that combines the unique identifier of the act (CELEX) provided by Eurolex and the identifier of the chapter/article, etc.

### Act
```
id: String 
title: String
eurlex_url: String
```

### Recitals
```
id: String         #CELEX+num_recital
number: String     #Recital number -> 1, 2
text: String       #actual content of the recital
```

### Chapter
```
id: String         #CELEX+num_chapter
number: String     #Roman numerals -> I, II, III
title: String
```

### Section
```
id: String         #CELEX+num_section
title: String
```


### Article
```
id: String        #CELEX+num_article
title: String
text: String
```

### Paragraph
```
id: String         #CELEX+num_paragraph
text: String
```

### Case Law (WIP)
```
id: String           #case law identifier 
```

## Document Parsing and Data Retrieval
All the data used to fill the KG is retrieved from EUR-Lex documents. In particular, specific acts are parsed from the English HTML format into a specific structured data object. Furthermore, the document information section is parsed to match the associated case law. 

PS: The parser ignores ongoing case law; we only consider completed case law. (XXX interprets YYY). 

## Architecture

### Tech Stack

| Layer | Technology | Role |
|---|---|---|
| Language | Python | Entire codebase |
| Graph DB | Neo4j (Docker) | Stores nodes and vector embeddings |
| LLM orchestration | LangChain + `langchain-neo4j` | RetrievalQA chain, Neo4j vector store integration |
| LLM / Embeddings | OpenAI API (`langchain-openai`) | Answer generation and paragraph embeddings |
| Semantic re-ranking | `sentence-transformers` (`all-MiniLM-L6-v2`) | Topic similarity filtering in GraphEnrichedRetriever |
| NLP | NLTK | Tokenization and lemmatization in the ASKE pipeline |
| HTML parsing | BeautifulSoup4 | Parsing EUR-Lex legal documents |
| Web scraping | Playwright (Chromium) | Headless browser downloads bypassing AWS WAF |
| Config | `python-dotenv` | Loads secrets from `.env` |

---

### Pipeline Overview

The system operates in three sequential phases, each with its own entry point:

```
graph_init.py          →   aske_pipeline.py         →   rag_pipeline.py
Phase 1: Graph Init        Phase 2: Topic Extraction     Phase 3: RAG Query
```

---

### Step 1: Graph Initialization (`graph_init.py`)

Downloads the four legal HTML documents from EUR-Lex, parses them into a structured knowledge graph, and generates paragraph-level vector embeddings.

1. `BrowserFetcher` launches a headless Chromium browser, navigates each EUR-Lex URL, and saves the rendered HTML to `docs/` — this bypasses the AWS WAF JavaScript challenge that blocks plain HTTP requests
2. `EURLexHTMLParser` (BeautifulSoup) parses each HTML file into a structured hierarchy: Act → Chapter → Section → Article → Paragraph
3. `MetadataParser` fetches the document metadata page for each act and extracts case-law "Interpreted by" relationships
4. `GraphLoader` writes all nodes and edges to Neo4j using parameterized Cypher queries
5. `Neo4jGraph.generate_text_embeddings` encodes every Paragraph node with `all-MiniLM-L6-v2` and stores the vector on the node
6. `Neo4jGraph.create_vector_index` creates a COSINE similarity vector index on `Paragraph.textEmbedding` for fast nearest-neighbour lookup

---

### Step 2: Topic Extraction (`aske_pipeline.py`)

Runs the ASKE (Automatic Semantic Knowledge Extraction) algorithm over all paragraph nodes, iteratively expanding a set of legal concepts and linking the most relevant topics back to each paragraph.

1. All Paragraph nodes are fetched from Neo4j
2. `TextPreprocessor` tokenizes paragraphs into sentences, lemmatizes each word, and produces sentence-level chunks (first sentence skipped as it usually just contains the paragraph number)
3. `ASKETopicExtractor.run_aske_cycle` runs for `N_GENERATIONS` iterations; each generation has four phases:
   - **Chunk Classification** — cosine similarity between chunk and concept embeddings; chunks above threshold `α` are assigned to that concept
   - **Deactivate Unused** — concepts that received zero classifications are marked inactive and excluded from further enrichment
   - **Terminology Enrichment** — for each active concept, candidate terms are extracted from classified chunks using TF-IDF with bigrams; WordNet definitions are fetched and embedded; terms are scored with a discriminative penalty (`sim_to_concept − 0.5 × max_sim_to_others`) to down-rank generic terms; the top `γ` terms above threshold `β` are added to the concept
   - **Concept Derivation** — terms within each concept are clustered with Affinity Propagation; each distinct cluster spawns a new concept labelled by its centroid term
4. `ASKETopicExtractor.aggregate_topics_by_paragraph` selects the top-3 topics per paragraph based on maximum chunk-level similarity score
5. `Neo4jGraph.update_paragraph_topics` writes Topic nodes and `(Paragraph)-[:RELATED_TO]->(Topic)` edges to Neo4j
6. A human-readable report is written to `results/aske_report.txt` listing every active concept with its associated terms

**Tunable parameters** (top of `aske_pipeline.py`):

| Parameter | Default | Meaning |
|---|---|---|
| `N_GENERATIONS` | 20 | ASKE iterations |
| `ALPHA` | 0.3 | Chunk-classification similarity threshold |
| `BETA` | 0.4 | Terminology-enrichment acceptance threshold |
| `GAMMA` | 10 | Max new terms added per concept per generation |

---

### Step 3: RAG Query (`rag_pipeline.py`)

Answers user questions by combining topic-aware filtering with vector search, re-ranking results, and passing the top-k paragraphs as grounded context to an OpenAI LLM.

1. The user query is encoded with `all-MiniLM-L6-v2` and compared against all Topic node embeddings (cosine similarity threshold: 0.35); the top-5 matching topics are selected
2. `GraphEnrichedRetriever` fetches all Paragraph nodes linked to those topics via `(Paragraph)-[:RELATED_TO]->(Topic)`
3. In parallel, a Neo4j COSINE vector search retrieves the nearest Paragraph nodes to the query embedding
4. Both result sets are deduplicated and merged
5. All candidate paragraphs are re-ranked by cosine similarity to the query; the top-5 are returned as context
6. A LangChain `RetrievalQA` chain passes the context with `ANSWER_SYNTHESIS_PROMPT_v2` to an OpenAI LLM (temperature=0); the prompt enforces strict article citation accuracy
7. The answer is written to `results/prova.txt`

---