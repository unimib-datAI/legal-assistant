import logging
import re

from service.scraper.eurlex_exporter import EURLexHTMLParser
from service.graph import Neo4jGraph

logger = logging.getLogger(__name__)


class GraphLoader:
    """Loads legal document data directly into Neo4j from HTML."""

    def __init__(self, neo4j_graph: Neo4jGraph):
        self.graph = neo4j_graph

    def load_document(self, config):
        """
        Load a single document into the graph database.

        Args:
            config: Dict with keys: html_file, celex, author, publication_date,
                   date_of_application, eurolex_url, document_info_url
        """
        parser = EURLexHTMLParser(
            html_file_path=config['html_file'],
            celex=config['celex'],
            eurolex_url=config['eurolex_url'],
            document_info_url=config['document_info_url']
        )

        data = parser.extract_data()

        self._load_act(data['act'])
        self._load_chapters(data['act'], data['chapters'])
        self._load_recitals(data['act'], data['recitals'])
        self._load_citations(data['citations'])
        self._load_case_law(data['act'], data['case_law'])

        logger.info("Loaded document %s into graph", config['celex'])

    def load_all_documents(self, documents_config):
        """
        Load multiple documents into the graph database.

        Args:
            documents_config: List of config dicts
        """
        for config in documents_config:
            try:
                self.load_document(config)
            except Exception as e:
                logger.error("Error loading %s: %s", config['celex'], e)

    def _load_act(self, act):
        """Create Act node."""
        self.graph.create_graph_node(
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

            self.graph.create_graph_node(
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

            self.graph.create_graph_node(
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

            self.graph.create_graph_node(
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

            self.graph.create_graph_node(
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

            self.graph.create_relationship(
                left_node_name="Article",
                right_node_name="Paragraph",
                left_id=article_id,
                right_id=paragraph_id,
                relationship="CITES"
            )

    def _load_recitals(self, act, recitals):
        """Create Recital nodes."""
        for recital in recitals:
            recital_id = f"{act['celex']}{recital['id']}"

            self.graph.create_graph_node(
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

    def _load_citations(self, citations):
        """Create citation relationships between articles."""
        for citation in citations:
            try:
                self.graph.create_relationship(
                    left_node_name="Article",
                    right_node_name="Article",
                    left_id=citation['from_article_id'],
                    right_id=citation['to_article_id'],
                    relationship=citation['citation_type']
                )
            except Exception as e:
                logger.warning("Skipped citation %s -> %s: %s",
                               citation['from_article_id'], citation['to_article_id'], e)

    def _load_case_law(self, act, case_law_list):
        """Create CaseLaw nodes and their interpretation relationships."""
        for case_law in case_law_list:
            case_law_id = case_law['case_law_identifier']

            try:
                self.graph.create_graph_node(
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
