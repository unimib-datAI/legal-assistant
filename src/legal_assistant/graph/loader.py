import logging
import re

from legal_assistant.graph import Neo4jGraph
from legal_assistant.scraper.eurlex_exporter import EURLexHTMLParser
from legal_assistant.validation import act_source
from legal_assistant.validation.gate import GraphValidationError, build_validated
from legal_assistant.validation.plan import GraphPlan

logger = logging.getLogger(__name__)


class GraphLoader:
    """Loads legal document data into Neo4j from HTML, via a validated plan.

    Nothing is written directly. :meth:`plan_document` builds the graph in memory and
    checks it against the source HTML; only :meth:`write` turns a validated plan into
    database writes. That split is what lets the caller validate every act *before*
    clearing the database.
    """

    def __init__(self, neo4j_graph: Neo4jGraph):
        self.graph = neo4j_graph

    def _emit(self, data):
        """Write one parsed document into ``self.graph``, the real client or a recorder."""
        self._load_act(data['act'])
        self._load_chapters(data['act'], data['chapters'])
        self._load_recitals(data['act'], data['recitals'])
        self._load_annexes(data['act'], data.get('annexes', []))
        self._load_case_law(data['act'], data['case_law'])

    def plan_document(self, config, *, strict: bool = True) -> GraphPlan:
        """Parse, build in memory, and validate. Raises unless the graph is sound.

        Parsing happens once: it reaches the network for the case-law metadata, so the
        determinism check re-runs the *builder* over the parsed data rather than the parser
        itself. Parser determinism is covered by ``tests/graph_validation/``.
        """
        celex = config['celex']
        parser = EURLexHTMLParser(
            html_file_path=config['html_file'],
            celex=celex,
            eurolex_url=config['eurolex_url'],
            document_info_url=config['document_info_url'],
        )
        data = parser.extract_data()

        def build(graph):
            GraphLoader(graph)._emit(data)

        return build_validated(
            build,
            root_id=celex,
            label=f"act {celex}",
            source_inventory=act_source.html_fragments(config['html_file']),
            reconstructed=act_source.reconstructed_fragments,
            conservation_kind="act_text",
            strict=strict,
        )

    def write(self, plan: GraphPlan) -> None:
        """Replay a validated plan onto the real graph."""
        plan.replay(self.graph)

    def load_document(self, config, *, strict: bool = True):
        """Validate one document and write it."""
        plan = self.plan_document(config, strict=strict)
        self.write(plan)
        logger.info("Loaded document %s into graph", config['celex'])
        return plan

    def plan_all_documents(self, documents_config, *, strict: bool = True):
        """Validate every document before any of them is written.

        Failures are collected and re-raised together, rather than logged and swallowed:
        a half-loaded graph is worse than a failed load.
        """
        plans, failures = [], []
        for config in documents_config:
            celex = config['celex']
            try:
                plans.append((celex, self.plan_document(config, strict=strict)))
            except (GraphValidationError, OSError, ValueError, KeyError) as exc:
                logger.error("Validation failed for %s: %s", celex, exc)
                failures.append((celex, exc))

        if failures:
            summary = "\n".join(f"  {celex}: {exc}" for celex, exc in failures)
            raise RuntimeError(
                f"{len(failures)}/{len(documents_config)} document(s) failed validation:\n"
                f"{summary}"
            ) from failures[0][1]
        return plans

    def load_all_documents(self, documents_config, *, strict: bool = True):
        """Validate all documents, then write them. Nothing is written if any fails."""
        plans = self.plan_all_documents(documents_config, strict=strict)
        for celex, plan in plans:
            self.write(plan)
            logger.info("Loaded document %s into graph", celex)
        return plans

    def _load_act(self, act):
        """Create Act node."""
        self.graph.upsert_graph_node(
            node_name="Act",
            node_properties={
                'id': act['celex'],
                'title': act['title'],
                'eurolex_url': act['eurolex_url']
            }
        )

    def _load_chapters(self, act, chapters):
        """Create Chapter nodes and their hierarchical structure."""
        for chapter in chapters:
            chapter_id = f"{act['celex']}{chapter['id']}"
            chapter_number = self._extract_number(chapter['id'], 'cpt_')

            self.graph.upsert_graph_node(
                node_name="Chapter",
                node_properties={
                    'id': chapter_id,
                    'number': chapter_number,
                    'title': chapter['title']
                }
            )

            self.graph.create_relationship(
                left_node_name="Act",
                right_node_name="Chapter",
                left_id=act['celex'],
                right_id=chapter_id,
                relationship="CONTAINS"
            )

            self._load_sections(act, chapter, chapter_id)
            self._load_articles(act, chapter, chapter_id, None)

    def _load_sections(self, act, chapter, chapter_id):
        """Create Section nodes."""
        for section in chapter['sections']:
            section_id = f"{act['celex']}{section['id']}"

            self.graph.upsert_graph_node(
                node_name="Section",
                node_properties={
                    'id': section_id,
                    'title': section['title']
                }
            )

            self.graph.create_relationship(
                left_node_name="Chapter",
                right_node_name="Section",
                left_id=chapter_id,
                right_id=section_id,
                relationship="CONTAINS"
            )

            self._load_articles(act, section, chapter_id, section_id)

    def _load_articles(self, act, parent, chapter_id, section_id):
        """Create Article nodes and paragraphs."""
        for article in parent['articles']:
            article_id = f"{act['celex']}{article['id']}"

            self.graph.upsert_graph_node(
                node_name="Article",
                node_properties={
                    'id': article_id,
                    'title': article['title'],
                    'text': article['full_text']
                }
            )

            if section_id:
                self.graph.create_relationship(
                    left_node_name="Section",
                    right_node_name="Article",
                    left_id=section_id,
                    right_id=article_id,
                    relationship="CONTAINS"
                )
            else:
                self.graph.create_relationship(
                    left_node_name="Chapter",
                    right_node_name="Article",
                    left_id=chapter_id,
                    right_id=article_id,
                    relationship="CONTAINS"
                )

            self._load_paragraphs(act, article, article_id)

    def _load_paragraphs(self, act, article, article_id):
        """Create Paragraph nodes."""
        for paragraph in article['paragraphs']:
            paragraph_id = f"{act['celex']}_{paragraph['id']}"

            self.graph.upsert_graph_node(
                node_name="Paragraph",
                node_properties={
                    'id': paragraph_id,
                    'text': paragraph['text']
                }
            )

            self.graph.create_relationship(
                left_node_name="Article",
                right_node_name="Paragraph",
                left_id=article_id,
                right_id=paragraph_id,
                relationship="CONTAINS"
            )

    def _load_recitals(self, act, recitals):
        """Create Recital nodes."""
        for recital in recitals:
            recital_id = f"{act['celex']}{recital['id']}"

            self.graph.upsert_graph_node(
                node_name="Recital",
                node_properties={
                    'id': recital_id,
                    'number': recital['number'],
                    'text': recital['text']
                }
            )

            self.graph.create_relationship(
                left_node_name="Act",
                right_node_name="Recital",
                left_id=act['celex'],
                right_id=recital_id,
                relationship="CONTAINS"
            )

    def _load_annexes(self, act, annexes):
        """Create Annex and AnnexPoint nodes. Only the AI Act has any.

        Annexes hang off the act, in the same structural position as recitals, because the
        published markup places them outside the chapter tree.
        """
        for annex in annexes:
            annex_id = f"{act['celex']}{annex['id']}"

            self.graph.upsert_graph_node(
                node_name="Annex",
                node_properties={
                    'id': annex_id,
                    'number': annex['number'],
                    'title': annex['title']
                }
            )

            self.graph.create_relationship(
                left_node_name="Act",
                right_node_name="Annex",
                left_id=act['celex'],
                right_id=annex_id,
                relationship="CONTAINS"
            )

            self._load_annex_points(act, annex, annex_id)

    def _load_annex_points(self, act, annex, annex_id):
        """Create the AnnexPoint nodes of one annex.

        The id is positional: an annex div carries no element per point, so deriving identity
        from the prose numbering would repeat the mistake the removed CITES edges made, where
        regex-derived ids mostly pointed at nodes that did not exist. ``point_label`` carries
        the citation a lawyer writes, composed from the markup's own numbering, and is null
        for lead-in prose that has no marker of its own.
        """
        for position, point in enumerate(annex['points'], start=1):
            label = point['label']

            self.graph.upsert_graph_node(
                node_name="AnnexPoint",
                node_properties={
                    'id': f"{annex_id}.p_{position:03d}",
                    'celex': act['celex'],
                    'text': point['text'],
                    'section_heading': point['section_heading'],
                    'point_label': f"{annex['number']}, point {label}" if label else None
                }
            )

            self.graph.create_relationship(
                left_node_name="Annex",
                right_node_name="AnnexPoint",
                left_id=annex_id,
                right_id=f"{annex_id}.p_{position:03d}",
                relationship="CONTAINS"
            )

    def _load_case_law(self, act, case_law_list):
        """Create CaseLaw nodes and their interpretation relationships."""
        for case_law in case_law_list:
            case_law_id = case_law['case_law_identifier']

            try:
                self.graph.upsert_graph_node(
                    node_name="CaseLaw",
                    node_properties={'id': case_law_id}
                )
            except Exception as e:
                logger.debug("CaseLaw node '%s' already exists or could not be created: %s", case_law_id, e)

            if case_law.get('paragraph'):
                paragraph_id = f"{act['celex']}_{case_law['paragraph']}"
                self.graph.create_relationship(
                    left_node_name="CaseLaw",
                    right_node_name="Paragraph",
                    left_id=case_law_id,
                    right_id=paragraph_id,
                    relationship="INTERPRETS"
                )
            elif case_law.get('article'):
                article_id = f"{act['celex']}{case_law['article']}"
                self.graph.create_relationship(
                    left_node_name="CaseLaw",
                    right_node_name="Article",
                    left_id=case_law_id,
                    right_id=article_id,
                    relationship="INTERPRETS"
                )
            elif case_law.get('chapter'):
                chapter_id = f"{act['celex']}{case_law['chapter']}"
                self.graph.create_relationship(
                    left_node_name="CaseLaw",
                    right_node_name="Chapter",
                    left_id=case_law_id,
                    right_id=chapter_id,
                    relationship="INTERPRETS"
                )

    def _extract_number(self, id_string, prefix):
        """Extract number from ID string."""
        match = re.search(f'{prefix}(.+)$', id_string)
        return match.group(1) if match else None
