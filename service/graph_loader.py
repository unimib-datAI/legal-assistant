
import re
from service.eurlex_exporter import EURLexHTMLParser
from service.graph import Neo4jGraph


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
        # Extract data using the existing parser
        parser = EURLexHTMLParser(
            html_file_path=config['html_file'],
            celex=config['celex'],
            author=config['author'],
            publication_date=config['publication_date'],
            date_of_application=config['date_of_application'],
            eurolex_url=config['eurolex_url'],
            document_info_url=config['document_info_url']
        )

        data = parser.extract_data()

        # Load into graph
        self._load_act(data['act'])
        self._load_chapters(data['act'], data['chapters'])
        self._load_recitals(data['act'], data['recitals'])
        self._load_citations(data['citations'])
        self._load_case_law(data['act'], data['case_law'])

        print(f"✓ Loaded document {config['celex']} into graph")

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
                print(f"✗ Error loading {config['celex']}: {str(e)}")

    def _load_act(self, act):
        """Create Act node."""
        self.graph.create_graph_node(
            node_name="Act",
            node_properties={
                'id': act['celex'],
                'title': act['title'],
                'author': act['author'],
                'publication_date': act['publication_date'],
                'date_of_application': act['date_of_application'],
                'eurolex_url': act['eurolex_url']
            }
        )

    def _load_chapters(self, act, chapters):
        """Create Chapter nodes and their hierarchical structure."""
        for chapter in chapters:
            chapter_id = f"{act['celex']}{chapter['id']}"
            chapter_number = self._extract_number(chapter['id'], 'cpt_')

            # Create Chapter node
            self.graph.create_graph_node(
                node_name="Chapter",
                node_properties={
                    'id': chapter_id,
                    'number': chapter_number,
                    'title': chapter['title']
                }
            )

            # Create Act -> Chapter relationship
            self.graph.create_relationship(
                left_node_name="Act",
                right_node_name="Chapter",
                left_id=act['celex'],
                right_id=chapter_id,
                relationship="CONTAINS"
            )

            # Load sections within chapter
            self._load_sections(act, chapter, chapter_id)

            # Load articles directly in chapter (not in sections)
            self._load_articles(act, chapter, chapter_id, None)

    def _load_sections(self, act, chapter, chapter_id):
        """Create Section nodes."""
        for section in chapter['sections']:
            section_id = f"{act['celex']}{section['id']}"

            # Create Section node
            self.graph.create_graph_node(
                node_name="Section",
                node_properties={
                    'id': section_id,
                    'title': section['title']
                }
            )

            # Create Chapter -> Section relationship
            self.graph.create_relationship(
                left_node_name="Chapter",
                right_node_name="Section",
                left_id=chapter_id,
                right_id=section_id,
                relationship="CONTAINS"
            )

            # Load articles within section
            self._load_articles(act, section, chapter_id, section_id)

    def _load_articles(self, act, parent, chapter_id, section_id):
        """Create Article nodes and paragraphs."""
        for article in parent['articles']:
            article_id = f"{act['celex']}{article['id']}"

            # Create Article node
            self.graph.create_graph_node(
                node_name="Article",
                node_properties={
                    'id': article_id,
                    'title': article['title'],
                    'full_text': article['full_text']
                }
            )

            # Create parent -> Article relationship
            if section_id:
                # Article is in a Section
                self.graph.create_relationship(
                    left_node_name="Section",
                    right_node_name="Article",
                    left_id=section_id,
                    right_id=article_id,
                    relationship="CONTAINS"
                )
            else:
                # Article is directly in a Chapter
                self.graph.create_relationship(
                    left_node_name="Chapter",
                    right_node_name="Article",
                    left_id=chapter_id,
                    right_id=article_id,
                    relationship="CONTAINS"
                )

            # Load paragraphs
            self._load_paragraphs(act, article, article_id)

    def _load_paragraphs(self, act, article, article_id):
        """Create Paragraph nodes."""
        for paragraph in article['paragraphs']:
            paragraph_id = f"{act['celex']}_{paragraph['id']}"

            # Create Paragraph node
            self.graph.create_graph_node(
                node_name="Paragraph",
                node_properties={
                    'id': paragraph_id,
                    'text': paragraph['text']
                }
            )

            # Create Article -> Paragraph relationship
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

            # Create Recital node
            self.graph.create_graph_node(
                node_name="Recital",
                node_properties={
                    'id': recital_id,
                    'number': recital['number'],
                    'full_text': recital['text']
                }
            )

            # Create Act -> Recital relationship
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
                # Citation might reference non-existent article
                print(f"  ⚠ Skipped citation {citation['from_article_id']} -> {citation['to_article_id']}: {e}")

    def _load_case_law(self, act, case_law_list):
        """Create CaseLaw nodes and their interpretation relationships."""
        for case_law in case_law_list:
            case_law_id = case_law['case_law_identifier']

            # Create CaseLaw node (if not already exists)
            try:
                self.graph.create_graph_node(
                    node_name="CaseLaw",
                    node_properties={
                        'id': case_law_id
                    }
                )
            except Exception:
                pass

            # Create interpretation relationships
            if case_law.get('paragraph'):
                # Case law interprets a specific paragraph
                paragraph_id = f"{act['celex']}_{case_law['paragraph']}"
                self.graph.create_relationship(
                    left_node_name="CaseLaw",
                    right_node_name="Paragraph",
                    left_id=case_law_id,
                    right_id=paragraph_id,
                    relationship="INTERPRETS"
                )
            elif case_law.get('article'):
                # Case law interprets an article
                article_id = f"{act['celex']}{case_law['article']}"
                self.graph.create_relationship(
                    left_node_name="CaseLaw",
                    right_node_name="Article",
                    left_id=case_law_id,
                    right_id=article_id,
                    relationship="INTERPRETS"
                )
            elif case_law.get('chapter'):
                # Case law interprets a chapter
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
