# Validazione bloccante e deterministica della costruzione del grafo

## Context

Oggi sia il caricamento degli atti sia quello delle sentenze **scrivono su Neo4j nodo per nodo mentre camminano l'albero**. Non esiste alcun punto in cui il grafo che sta per essere scritto venga guardato nel suo insieme. Le conseguenze sono già misurabili:

- **Le perdite di testo sono silenziose.** Misurato confrontando i `<p class="oj-normal">` dentro i `div[id^=art_]` con il testo effettivamente catturato da `EURLexHTMLParser._get_paragraphs`:

  | Atto | frammenti sorgente | persi |
  |---|---|---|
  | 32016R0679 (GDPR) | 1175 | 8 |
  | 32024R1689 (AI Act) | 1535 | 19 |
  | 32023R2854 (Data Act) | 774 | 1 |
  | 32022R0868 (DGA) | 499 | 11 |

  Non è rumore: sono l'incipit degli articoli "Definizioni" (`For the purposes of this Regulation:`), i sotto-punti annidati dentro una singola definizione, e — in `32024R1689` art_108 — i paragrafi di un articolo modificativo, numerati secondo l'atto modificato (`017.003`) e quindi invisibili al regex `^{article_num}\.\d+$` di [eurlex_exporter.py:183](service/scraper/eurlex_exporter.py#L183).

- **Gli archi verso nodi inesistenti non falliscono.** `RelationQueries.CREATE_RELATIONSHIP` ([query.py:260](service/graph/query.py#L260)) fa `MATCH … MATCH … MERGE`: se un estremo non esiste, la query non scrive nulla e **non solleva niente**. `_load_citations` ([graph_loader.py:198](service/graph/graph_loader.py#L198)) genera `Article->Article` verso id sintetizzati da regex (`"Articles 12 to 15"` espande un range intero, anche verso articoli di *altri* atti): la maggior parte di quegli archi non esiste, e nessuno se ne accorge.

- **Il precedente è documentato.** [case_law_html_parsing.md](.claude/case_law_html_parsing.md) racconta esattamente questo fallimento sul lato sentenze: `create_case_law_kg` costruiva il parentage unicamente dal `depth`, e un parser sbagliato produceva un albero piatto e invertito con 18 sezioni fantasma — scoperto solo a mano, a valle.

L'obiettivo: **niente viene scritto su Neo4j finché il grafo che sta per essere scritto non è stato ricostruito con una DFS e verificato contro il documento sorgente.** Se il check non passa, la scrittura non avviene.

### Fattibilità sui case law: sì, ed è più forte che sugli atti

Verificato eseguendo il parser su tre sentenze reali (62012CJ0293, 62019CJ0645, 62018CJ0511):

- `_linearize()` ([html_parser.py:122](src/legal_assistant/case_law/html_parser.py#L122)) produce già un **inventario ordinato ed esatto** della sorgente. Confrontandolo con quanto emesso nell'albero, l'unico item "mancante" è il blocco indice (che viene deliberatamente splittato in topic da `_split_topics`) e gli unici "extra" sono le 3 sezioni sintetiche di preambolo più i topic splittati. **Zero perdite non spiegate, su tutte e tre.**
- Il secondo strato (albero → paragrafi, `split_paragraphs` in [kg_builder.py:65](src/legal_assistant/case_law/kg_builder.py#L65)) conserva anch'esso il 100%: 150/150, 149/149, 347/347 item di body ritrovati nel testo dei `CaseLawParagraph`.
- Il parsing è già deterministico: riparsare gli stessi byte dà un albero identico (`flatten(...) == flatten(...)` → `True` su tutte e tre).

Quindi sui case law il check è una **verifica di partizione stretta** (ogni item della sorgente finisce in esattamente un nodo), non un confronto fuzzy — ed è verde già oggi. Sugli atti serve prima chiudere le perdite sopra.

## Approccio

Il punto chiave è **non rifattorizzare i builder**. Sia `GraphLoader` sia `create_case_law_kg` parlano con il grafo attraverso due soli metodi: `upsert_graph_node` e `create_relationship`. Basta passargli un oggetto che ha la stessa interfaccia ma registra invece di scrivere; si valida la registrazione; e solo se passa la si riproduce sul grafo vero.

```
builder(recorder)  →  GraphPlan  →  validate()  →  ok?  →  plan.replay(neo4j)
                                          └── no ──→ GraphValidationError, niente scritture
```

## Implementazione

### 1. `service/graph/recorder.py` (nuovo)

```python
@dataclass(frozen=True) class NodeOp:  label: str; properties: dict
@dataclass(frozen=True) class EdgeOp:  left_label, right_label, left_id, right_id, rel_type: str

class RecordingGraph:
    """Sostituto in-memory di Neo4jGraph: registra invece di scrivere."""
    def upsert_graph_node(self, node_name, node_properties) -> str   # ritorna l'id, come l'originale
    def create_relationship(self, left_node_name, right_node_name, left_id, right_id, relationship) -> None
```

Nessun `Neo4jGraph` viene toccato. `RecordingGraph` è duck-typed sulla parte di interfaccia che i builder usano davvero — verificato: `GraphLoader` e `kg_builder` non chiamano altro.

### 2. `service/validation/plan.py` (nuovo)

`GraphPlan` avvolge le operazioni registrate e offre:

- `nodes: dict[str, NodeOp]` / `edges: list[EdgeOp]`
- `dfs(root_id, rel_types) -> Iterator[tuple[int, NodeOp]]` — **la ricostruzione DFS in pre-ordine** su cui poggiano tutti i check strutturali e di conservazione.
- `fingerprint() -> str` — sha256 di una serializzazione canonica (nodi ordinati per id con chiavi ordinate, archi ordinati come tuple; le proprietà volatili come `textEmbedding` escluse). Insensibile all'ordine di emissione, sensibile al contenuto.
- `replay(graph: Neo4jGraph) -> None` — esegue le operazioni nell'ordine registrato.

### 3. `service/validation/checks.py` (nuovo)

Funzioni pure `(...) -> list[Violation]`, con `Violation(kind, node_id, detail)`:

- `dangling_edges` — estremo non presente fra i nodi registrati.
- `conflicting_upserts` — stesso id scritto due volte con valori diversi per una chiave condivisa (oggi `SET n +=` sovrascrive in silenzio).
- `containment_is_tree` — DFS dalla radice sugli archi di contenimento: ogni nodo strutturale raggiunto **esattamente una volta**; nessun ciclo, nessun orfano, nessun doppio genitore.
- `depth_and_labels` — profondità crescente lungo l'albero e transizioni di label ammesse: `Act→Chapter→[Section]→Article→Paragraph`, `Act→Recital`, `CaseLaw→CaseLawSection→CaseLawSection*→CaseLawParagraph`, `CaseLaw→CaseLawTopic`.
- `conservation(source, reconstructed, exempt)` — **il check richiesto**: multiset dei frammenti sorgente normalizzati (NFKC, whitespace collassato, non-alfanumerici rimossi) contro quelli ricostruiti dalla DFS. Segnala sia i mancanti sia i **duplicati** (un frammento consumato due volte è un errore quanto uno perso). Riporta gli estratti, non solo i conteggi.

### 4. `service/validation/gate.py` (nuovo)

```python
class GraphValidationError(RuntimeError): ...   # porta il report

def build_validated(build_fn, source_inventory, root_id, *, strict=True) -> GraphPlan
```

Registra, valida, e **ri-esegue `build_fn` su un secondo recorder confrontando i fingerprint** — è il check di determinismo, gratuito perché non tocca la rete (stessi byte in ingresso). `strict=False` degrada le violazioni a `logger.warning` e restituisce comunque il piano.

### 5. Inventari sorgente

- **Case law** — `service/validation/case_law_source.py`: riusa `_linearize()` (va promosso a `linearize()`, pubblico) su **lo stesso HTML già scaricato**, non un secondo fetch. Esenzioni dichiarate: il blocco indice e i tre heading sintetici di `PREAMBLE_SECTIONS`. Secondo strato: ogni item di body deve ritrovarsi nel testo di un `CaseLawParagraph`, dopo aver strippato il numero di paragrafo con `_PARAGRAPH_NUM` / `_OPERATIVE_NUM`.
- **Atti** — `service/validation/act_source.py`: tutti i `<p class="oj-normal">` dentro `div[id^=art_]` e `div[id^=rct_]`. Attenzione: `Article.full_text` ([eurlex_exporter.py:158](service/scraper/eurlex_exporter.py#L158)) contiene già il testo dei suoi paragrafi — va **escluso** dalla concatenazione DFS o ogni paragrafo risulta duplicato.

### 6. Fix necessari perché gli atti passino al 100%

- [eurlex_exporter.py:206-223](service/scraper/eurlex_exporter.py#L206-L223) — emettere l'incipit dell'articolo di definizioni come paragrafo `.0` invece di scartarlo nel ramo `i += 1`, e far consumare alla zip delle definizioni **tutti** i parti fino al prossimo `(N)`, non solo `text_parts[i+1]`.
- [eurlex_exporter.py:183](service/scraper/eurlex_exporter.py#L183) — allargare la ricerca dei div-paragrafo a qualunque discendente `div[id~=^\d{3}\.\d+$]` dell'articolo, non solo quelli che matchano il numero dell'articolo stesso (sblocca gli articoli modificativi tipo `32024R1689` art_108).
- **Rimozione di `CITES` (obsoleto).** Nessuna query lo legge; [query.py:247-253](service/graph/query.py#L247-L253) documenta già che il ponte delle citazioni è passato a [service/rag/citations.py](service/rag/citations.py) a runtime. Si cancellano: `_extract_citations`, `_extract_article_citations`, `_find_article_references` e la chiave `citations` in `extract_data` ([eurlex_exporter.py:40-89](service/scraper/eurlex_exporter.py#L40-L89)); `_load_citations` e la sua chiamata ([graph_loader.py:36, 198-211](service/graph/graph_loader.py#L198-L211)); e l'arco `Article-[:CITES]->Paragraph` verso il proprio stesso paragrafo ([graph_loader.py:168-174](service/graph/graph_loader.py#L168-L174)). Questo elimina anche l'intera sorgente di archi pendenti.

### 7. Cablaggio del blocco

- `GraphLoader.load_document` — costruisce nel recorder, valida, replay. `load_all_documents` ([graph_loader.py:49-52](service/graph/graph_loader.py#L49-L52)) **smette di ingoiare le eccezioni**: raccoglie i fallimenti, li riporta, e li propaga al chiamante.
- [frontend/kg/graph_init.py](frontend/kg/graph_init.py) — mostra il report di validazione in caso di fallimento; con "Clear existing database" attivo, il clear va spostato **dopo** la validazione di tutti gli atti, altrimenti un fallimento lascia il DB vuoto.
- `case_law_init.ingest` ([case_law_init.py](case_law_init.py)) — per sentenza: `GraphValidationError` entra in `failed` accanto ai fallimenti di fetch già gestiti, nulla viene scritto per quella sentenza, il batch prosegue. **`main()` ritorna 1 se `failed` non è vuoto** (oggi ritorna sempre 0).
- Flag `--allow-invalid` su `case_law_init.py` → `strict=False`. Default: bloccante.

## Verifica

Nuovo `test/graph_validation/` con `pytest`:

1. `test_acts.py` — per ognuno dei 4 `corpus/*.html` (già in repo, nessuna rete): il piano valida senza violazioni, **conservazione 100%**, e il fingerprint coincide con quello in `golden_fingerprints.json`.
2. `test_case_law.py` — su fixture XHTML salvate in `test/graph_validation/fixtures/` (62012CJ0293 legacy, 62019CJ0645 moderno, 62018CJ0511 grande): conservazione 100% su entrambi gli strati, albero ben formato, `depth` massima 2, zero heading `^Article \d`, `Topics` presente esattamente una volta.
3. `test_gate.py` — con un piano corrotto a mano (arco pendente, sezione duplicata, paragrafo rimosso) il gate solleva e **`RecordingGraph` non ha inoltrato nulla** a un `Neo4jGraph` mockato (`unittest.mock`, come da CLAUDE.md).
4. `test_determinism.py` — due build sugli stessi byte danno fingerprint identici.

End-to-end, dopo i test:

```bash
python case_law_init.py --celex 62012CJ0293 62019CJ0645 --reset --skip-embeddings
python case_law_init.py --celex 62012CJ0293 --allow-invalid   # verifica il degrado a warning
```

più una run di [frontend/kg/graph_init.py](frontend/kg/graph_init.py) sui 4 atti, controllando che i conteggi di `Paragraph` in Neo4j salgano dei ~39 frammenti oggi persi.

## Ordine di esecuzione

1. `recorder.py`, `plan.py`, `checks.py`, `gate.py` + test del gate su piani sintetici.
2. Inventario e test case law → devono essere verdi **senza toccare il parser**.
3. Rimozione di `CITES`; inventario atti; i test atti falliscono e **nominano i 39 frammenti persi**.
4. Fix dell'exporter guidati da quel report, fino a 100%.
5. Cablaggio bloccante nei due entry point + fingerprint golden congelati.
