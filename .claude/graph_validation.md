# Validazione bloccante e deterministica della costruzione del grafo

> **Stato: proposta, non implementata.** Scritta prima del refactor del codebase e
> riverificata contro il codice attuale il **2026-07-22**: l'analisi è risultata intatta,
> path e cablaggio sono stati corretti.

## Context

Oggi sia il caricamento degli atti sia quello delle sentenze **scrivono su Neo4j nodo per nodo mentre camminano l'albero**. Non esiste alcun punto in cui il grafo che sta per essere scritto venga guardato nel suo insieme, quindi non c'è modo di sapere se il grafo finale rappresenti davvero il documento parsato. Le conseguenze sono già misurabili:

- **Le perdite di testo sono silenziose.** Misurato confrontando i `<p class="oj-normal">` dentro i `div[id^=art_]` con il testo effettivamente catturato da `EURLexHTMLParser`:

  | Atto | frammenti sorgente | catturati | persi |
  |---|---|---|---|
  | 32016R0679 (GDPR) | 1171 | 414 | 8 |
  | 32024R1689 (AI Act) | 1535 | 591 | 19 |
  | 32023R2854 (Data Act) | 774 | 277 | 1 |
  | 32022R0868 (DGA) | 499 | 163 | 11 |

  Non è rumore: sono l'incipit degli articoli "Definizioni" (`For the purposes of this Regulation:`), i sotto-punti annidati dentro una singola definizione, e — in `32024R1689` art_108 — i paragrafi di un articolo modificativo, numerati secondo l'atto modificato (`017.003`) e quindi invisibili al regex `^{article_num}\.\d+$` di [eurlex_exporter.py:183](../src/legal_assistant/scraper/eurlex_exporter.py#L183).

- **Gli archi verso nodi inesistenti non falliscono.** `RelationQueries.CREATE_RELATIONSHIP` ([queries.py:260](../src/legal_assistant/graph/queries.py#L260)) fa `MATCH … MATCH … MERGE`: se un estremo non esiste, la query non scrive nulla e **non solleva niente**. `_load_citations` ([loader.py:198](../src/legal_assistant/graph/loader.py#L198)) genera `Article->Article` verso id sintetizzati da regex (`"Articles 12 to 15"` espande un range intero, anche verso articoli di *altri* atti): la maggior parte di quegli archi non esiste, e nessuno se ne accorge.

- **Il precedente è documentato.** [case_law_html_parsing.md](case_law_html_parsing.md) racconta esattamente questo fallimento sul lato sentenze: `create_case_law_kg` costruiva il parentage unicamente dal `depth`, e un parser sbagliato produceva un albero piatto e invertito con 18 sezioni fantasma — scoperto solo a mano, a valle.

L'obiettivo: **niente viene scritto su Neo4j finché il grafo che sta per essere scritto non è stato ricostruito con una DFS e verificato contro il documento sorgente.** Se il check non passa, la scrittura non avviene.

### Fattibilità sui case law: sì, ed è più forte che sugli atti

Verificato eseguendo il parser su tre sentenze reali (62012CJ0293, 62019CJ0645, 62018CJ0511):

- `_linearize()` ([html_parser.py:122](../src/legal_assistant/case_law/html_parser.py#L122)) produce già un **inventario ordinato ed esatto** della sorgente. Confrontandolo con quanto emesso nell'albero, l'unico item "mancante" è il blocco indice (che viene deliberatamente splittato in topic da `_split_topics`) e gli unici "extra" sono le 3 sezioni sintetiche di preambolo più i topic splittati. **Zero perdite non spiegate, su tutte e tre.**
- Il secondo strato (albero → paragrafi, `split_paragraphs` in [kg_builder.py:65](../src/legal_assistant/case_law/kg_builder.py#L65)) conserva anch'esso il 100%: 150/150, 149/149, 347/347 item di body ritrovati nel testo dei `CaseLawParagraph`.
- Il parsing è già deterministico: riparsare gli stessi byte dà un albero identico (`flatten(...) == flatten(...)` → `True` su tutte e tre).

Quindi sui case law il check è una **verifica di partizione stretta** (ogni item della sorgente finisce in esattamente un nodo), non un confronto fuzzy — ed è verde già oggi. Sugli atti serve prima chiudere le perdite sopra.

### Revisione post-refactor — cosa è cambiato

Il **presupposto architetturale regge**: `GraphLoader` e `kg_builder` toccano il grafo **solo** tramite `upsert_graph_node` e `create_relationship`. Il duck-typing del recorder funziona senza rifattorizzare i builder.

| Affermazione originale | Esito della riverifica |
|---|---|
| `CREATE_RELATIONSHIP` no-op silenzioso su estremo mancante | confermata, `graph/queries.py:260-265` |
| `CITES` generato ma mai letto a runtime | confermata: zero lettori in `rag/`, `frontend/`, `evals/` |
| `load_all_documents` ingoia le eccezioni | confermata, `except Exception` + `logger.error` |
| Regex `^{article_num}\.\d+$` esclude gli articoli modificativi | confermata, `eurlex_exporter.py:183` |
| `_linearize` / `split_paragraphs` / `PREAMBLE_SECTIONS` intatti | confermata, **stesse righe** (122 / 65) |
| 39 frammenti persi | **riprodotta esattamente** (8 / 19 / 1 / 11) |

Cambiano invece:

- **Path**: `service/*` → `src/legal_assistant/*`; `graph/query.py` → `graph/queries.py`; `graph/graph_loader.py` → `graph/loader.py`; `docs/*.html` → `corpus/*.html`; `test/` → `tests/`.
- **`case_law_init.py` non esiste più.** Argparse in `cli/main.py::_cmd_ingest_case_law`, logica in `pipelines/case_law_ingest.py::ingest`.
- **Il `clear_database` non è più nella pagina Streamlit** ma in `pipelines/graph_build.py`.
- **Due strade di scrittura per le sentenze** (scoperta nuova): `build_from_tree` (CLI) e `create_case_law_kg` (pagina Streamlit). La prima è un wrapper della seconda ([kg_builder.py:265](../src/legal_assistant/case_law/kg_builder.py#L265)), quindi **il gate va su `create_case_law_kg`** per coprirle entrambe.

## Approccio

Il punto chiave è **non rifattorizzare i builder**. Sia `GraphLoader` sia `create_case_law_kg` parlano con il grafo attraverso due soli metodi: `upsert_graph_node` e `create_relationship`. Basta passargli un oggetto che ha la stessa interfaccia ma registra invece di scrivere; si valida la registrazione; e solo se passa la si riproduce sul grafo vero.

```
builder(recorder)  →  GraphPlan  →  validate()  →  ok?  →  plan.replay(neo4j)
                                          └── no ──→ GraphValidationError, niente scritture
```

## Implementazione

### 1. `src/legal_assistant/graph/recorder.py` (nuovo)

```python
@dataclass(frozen=True) class NodeOp:  label: str; properties: dict
@dataclass(frozen=True) class EdgeOp:  left_label, right_label, left_id, right_id, rel_type: str

class RecordingGraph:
    """Sostituto in-memory di Neo4jGraph: registra invece di scrivere."""
    def upsert_graph_node(self, node_name, node_properties) -> str   # ritorna node_properties["id"]
    def create_relationship(self, left_node_name, right_node_name, left_id, right_id, relationship) -> None
```

Firme copiate da [client.py:45,54](../src/legal_assistant/graph/client.py#L45). `upsert_graph_node` ritorna l'id perché `CREATE_NODE` fa `RETURN n.id as node_id` e i builder usano quel valore. `RecordingGraph` è duck-typed sulla parte di interfaccia che i builder usano davvero — verificato: non chiamano altro.

### 2. `src/legal_assistant/validation/plan.py` (nuovo)

`GraphPlan` avvolge le operazioni registrate e offre:

- `nodes: dict[str, NodeOp]` / `edges: list[EdgeOp]`
- `dfs(root_id, rel_types) -> Iterator[tuple[int, NodeOp]]` — **la ricostruzione DFS in pre-ordine** su cui poggiano tutti i check strutturali e di conservazione.
- `fingerprint() -> str` — sha256 di una serializzazione canonica (nodi ordinati per id con chiavi ordinate, archi ordinati come tuple; le proprietà volatili come `textEmbedding` escluse). Insensibile all'ordine di emissione, sensibile al contenuto.
- `replay(graph: Neo4jGraph) -> None` — esegue le operazioni nell'ordine registrato.

### 3. `src/legal_assistant/validation/checks.py` (nuovo)

Funzioni pure `(...) -> list[Violation]`, con `Violation(kind, node_id, detail)`:

- `dangling_edges` — estremo non presente fra i nodi registrati.
- `conflicting_upserts` — stesso id scritto due volte con valori diversi per una chiave condivisa (oggi `SET n +=` sovrascrive in silenzio).
- `containment_is_tree` — DFS dalla radice sugli archi di contenimento: ogni nodo strutturale raggiunto **esattamente una volta**; nessun ciclo, nessun orfano, nessun doppio genitore.
- `depth_and_labels` — profondità crescente lungo l'albero e transizioni di label ammesse, prese dallo schema reale documentato in [docs/knowledge-graph.md](../docs/knowledge-graph.md): `Act→Chapter→[Section]→Article→Paragraph`, `Act→Recital`, `CaseLaw→CaseLawSection→CaseLawSection*→CaseLawParagraph`, `CaseLaw→CaseLawTopic`.
- `conservation(source, reconstructed, exempt)` — **il check richiesto**: multiset dei frammenti sorgente normalizzati (NFKC, whitespace collassato, non-alfanumerici rimossi) contro quelli ricostruiti dalla DFS. Segnala sia i mancanti sia i **duplicati** (un frammento consumato due volte è un errore quanto uno perso). Riporta gli estratti, non solo i conteggi.

### 4. `src/legal_assistant/validation/gate.py` (nuovo)

```python
class GraphValidationError(RuntimeError): ...   # porta il report

def build_validated(build_fn, source_inventory, root_id, *, strict=True) -> GraphPlan
```

Registra, valida, e **ri-esegue `build_fn` su un secondo recorder confrontando i fingerprint** — è il check di determinismo, gratuito perché non tocca la rete (stessi byte in ingresso). `strict=False` degrada le violazioni a `logger.warning` e restituisce comunque il piano.

Conformità alle invarianti del progetto: nessun client costruito qui (regola `resources.py`), nessun `basicConfig` (regola `logging_setup.py`), solo `logging.getLogger(__name__)`.

### 5. Inventari sorgente

- **Case law** — `validation/case_law_source.py`: riusa `_linearize()` (va promosso a `linearize()`, pubblico) su **lo stesso HTML già scaricato**, non un secondo fetch. Esenzioni dichiarate: il blocco indice e i tre heading sintetici di `PREAMBLE_SECTIONS` ([kg_builder.py:29](../src/legal_assistant/case_law/kg_builder.py#L29)). Secondo strato: ogni item di body deve ritrovarsi nel testo di un `CaseLawParagraph`, dopo aver strippato il numero di paragrafo con `_PARAGRAPH_NUM` / `_OPERATIVE_NUM` ([kg_builder.py:33,39](../src/legal_assistant/case_law/kg_builder.py#L33)).
- **Atti** — `validation/act_source.py`: tutti i `<p class="oj-normal">` dentro `div[id^=art_]` e `div[id^=rct_]`. Attenzione: `Article.full_text` ([eurlex_exporter.py:158](../src/legal_assistant/scraper/eurlex_exporter.py#L158)) contiene già il testo dei suoi paragrafi — va **escluso** dalla concatenazione DFS o ogni paragrafo risulta duplicato.

### 6. Fix necessari perché gli atti passino al 100%

Tutti in `scraper/eurlex_exporter.py::_get_paragraphs` (179-225):

- **riga ~221** — il ramo `i += 1` scarta l'incipit dell'articolo di definizioni: emetterlo come paragrafo `.0` invece di saltarlo.
- **righe 205-223** — la zip delle definizioni consuma solo `text_parts[i+1]`: far consumare **tutti** i parti fino al prossimo `(N)`, così i sotto-punti annidati non si perdono.
- **riga 183** — allargare la ricerca dei div-paragrafo a qualunque discendente `div[id~=^\d{3}\.\d+$]` dell'articolo, non solo quelli che matchano il numero dell'articolo stesso (sblocca gli articoli modificativi tipo `32024R1689` art_108).

### 7. Rimozione di `CITES` (obsoleto)

Nessuna query lo legge; [queries.py:252](../src/legal_assistant/graph/queries.py#L252) documenta già che il ponte delle citazioni è passato a [rag/citations.py](../src/legal_assistant/rag/citations.py) a runtime. Si cancellano:

- `_extract_citations`, `_extract_article_citations`, `_find_article_references` e la chiave `citations` in `extract_data` ([eurlex_exporter.py:19-70](../src/legal_assistant/scraper/eurlex_exporter.py#L19-L70))
- `_load_citations` e la sua chiamata ([loader.py:36,198-211](../src/legal_assistant/graph/loader.py#L198-L211))
- l'arco `Article-[:CITES]->Paragraph` verso il proprio stesso paragrafo ([loader.py:168-174](../src/legal_assistant/graph/loader.py#L168-L174))

Questo elimina anche l'intera sorgente di archi pendenti.

**Conseguenza da non dimenticare**: [docs/knowledge-graph.md](../docs/knowledge-graph.md) documenta `CITES` come parte dello schema (`Article→Article`, `Article→Paragraph`). Va aggiornato ri-derivando da Neo4j — è il caso previsto dalla tabella "Keeping it current" in [CLAUDE.md](CLAUDE.md).

### 8. Cablaggio del blocco

- **Atti** — `GraphLoader.load_document` costruisce nel recorder, valida, replay. `load_all_documents` ([loader.py:41-52](../src/legal_assistant/graph/loader.py#L41-L52)) **smette di ingoiare le eccezioni**: raccoglie i fallimenti, li riporta, e li propaga al chiamante.
- **`pipelines/graph_build.py`** — spostare `clear_database()` **dopo** la validazione di tutti gli atti (oggi è alle righe 52-54, prima del load): altrimenti un fallimento lascia il DB vuoto.
- **Sentenze** — gate dentro `create_case_law_kg` ([kg_builder.py:159](../src/legal_assistant/case_law/kg_builder.py#L159)), **non** dentro `build_from_tree`: così copre sia la CLI sia la pagina Streamlit [case_law_parser.py:141](../frontend/kg/case_law_parser.py#L141).
- **`pipelines/case_law_ingest.py::ingest`** — `GraphValidationError` entra in `IngestTotals.failed` accanto ai fallimenti di fetch già gestiti, nulla viene scritto per quella sentenza, il batch prosegue.
- **`cli/main.py::_cmd_ingest_case_law`** — **ritorna 1 se `totals.failed` non è vuoto** (oggi ritorna sempre 0). Nuovo flag `--allow-invalid` → `strict=False`. Default: bloccante.
- **`frontend/kg/graph_init.py`** — mostra il report di validazione in caso di fallimento (la pagina è già un guscio sottile: solo rendering).

## Verifica

Nuovo `tests/graph_validation/` con `pytest` (già configurato: `testpaths = ["tests"]`):

1. `test_acts.py` — per ognuno dei 4 `corpus/*.html` (già in repo, nessuna rete): il piano valida senza violazioni, **conservazione 100%**, e il fingerprint coincide con quello in `golden_fingerprints.json`.
2. `test_case_law.py` — su fixture XHTML salvate in `tests/graph_validation/fixtures/` (62012CJ0293 legacy, 62019CJ0645 moderno, 62018CJ0511 grande): conservazione 100% su entrambi gli strati, albero ben formato, `depth` massima 2, zero heading `^Article \d`, `Topics` presente esattamente una volta.
3. `test_gate.py` — con un piano corrotto a mano (arco pendente, sezione duplicata, paragrafo rimosso) il gate solleva e **`RecordingGraph` non ha inoltrato nulla** a un `Neo4jGraph` mockato (`unittest.mock`, come da CLAUDE.md).
4. `test_determinism.py` — due build sugli stessi byte danno fingerprint identici.

End-to-end, dopo i test, con Neo4j attivo:

```bash
legal-assistant ingest case-law --celex 62012CJ0293 62019CJ0645 --reset --skip-embeddings
legal-assistant ingest case-law --celex 62012CJ0293 --allow-invalid   # degrado a warning
echo $?                                                              # 1 se qualcosa è fallito
legal-assistant graph build
```

Controllo finale sul grafo: i `Paragraph` devono salire dei **39 frammenti oggi persi** (1445 → ~1484), e gli archi `CITES` devono sparire (oggi 931 `Article→Article` + 1445 `Article→Paragraph`).

```cypher
MATCH (p:Paragraph) RETURN count(p);
MATCH ()-[r:CITES]->() RETURN count(r);   // atteso: 0
```

## Ordine di esecuzione

1. `recorder.py`, `plan.py`, `checks.py`, `gate.py` + test del gate su piani sintetici.
2. Inventario e test case law — devono essere verdi **senza toccare il parser**.
3. Rimozione di `CITES`; inventario atti; i test atti falliscono e **nominano i 39 frammenti persi**.
4. Fix dell'exporter guidati da quel report, fino a 100%.
5. Cablaggio bloccante nei due entry point + fingerprint golden congelati.
6. Aggiornamento di `docs/knowledge-graph.md`.
