import logging
import re

from bs4 import BeautifulSoup

from legal_assistant.scraper.browser_fetcher import BrowserFetcher

logger = logging.getLogger(__name__)

# One provision of one article: "A02P1", "A04PT7", "A06P1L1LF", "A38P3SNT2", "A01".
#
# Anything after the paragraph number addresses a unit finer than the graph stores (a point,
# a subparagraph, a sentence), so it is matched and discarded rather than read as digits. The
# article number must run to the end or to the first subdivision marker, which is what stops
# a range such as "A12-A15" from matching at all.
_ARTICLE_REF_RE = re.compile(
    r'^A(?P<article>\d+)'
    r'(?:PT(?P<point>\d+)'          # point of a definitions article
    r'|P(?P<paragraph>\d+)(?:[A-Z].*)?'   # paragraph, plus any finer subdivision
    r')?$'
)


class MetadataParser:

    def __init__(self, fetcher: BrowserFetcher | None = None):
        self.fetcher = fetcher or BrowserFetcher()

    def parse_eurovoc_descriptors(self, document_info_url: str) -> list:
        try:
            html = self.fetcher.fetch(document_info_url)
        except (TimeoutError, ValueError) as e:
            logger.warning("Failed to fetch metadata from %s: %s", document_info_url, e)
            return []

        soup = BeautifulSoup(html, "html.parser")
        section = self.extract_div_by_specific_id(soup, "PPLinked_Contents")
        return self.extract_relationship_between_documents_section(section)

    def extract_div_by_specific_id(self, soup, id_prefix, multiple=True):
        """ Extract div elements with specific id prefix. """
        if multiple:
            return soup.find_all('div', id=lambda x: x and x.startswith(id_prefix))
        else:
            return soup.find('div', id=id_prefix)

    def extract_relationship_between_documents_section(self, section):
        """
        Extracts "Interpreted by" relationships from the document section.

        Returns:
            list: List of [article_reference, case_celex] pairs
        """
        case_law = []

        for div in section:
            for dt_elem in div.find_all('dt'):
                dd_elem = dt_elem.find_next_sibling('dd')
                if not dd_elem:
                    continue

                for li_item in dd_elem.find_all('li', class_='defaultUnderlined'):
                    content = self.extract_case_law_content(li_item)
                    if content:
                        case_law.append(content)

        return case_law

    def extract_case_law_content(self, li_item):
        """
        Extract interpretation relationship from a list item.

        Format: "A02P1 Interpreted by [link to 62018CJ0311]"
        Returns: ['A02P1', '62018CJ0311'] or None
        """
        full_text = li_item.get_text(strip=True)

        """
        This filter is necessary to identify 'final judgment' case law, even though there may be case law with
        a preliminary ruling that is trying to get a response from the court.
        """
        if 'Interpreted by' not in full_text:
            return None

        link = li_item.find('a')
        if not link or not link.get('data-celex'):
            return None

        article_reference = full_text.split('Interpreted by')[0].strip()

        if not article_reference:
            return None

        chapter, article, paragraph = self.enrich_article_reference(article_reference)

        return {key: value for key, value in {
            'case_law_identifier': link.get('data-celex'),
            'raw_article_reference': article_reference,
            'chapter': chapter,
            'article': article,
            'paragraph': paragraph
        }.items() if value is not None}

    @staticmethod
    def enrich_article_reference(article_reference):
        """Normalise an EUR-Lex reference to the ids the graph loader writes.

        EUR-Lex format → Database format:
        - 'A01' → chapter=None, article='art_1', paragraph=None
        - 'A02P1' → chapter=None, article='art_2', paragraph='002.001'
        - 'A04PT7' → chapter=None, article='art_4', paragraph='004.7'
        - 'CH8' → chapter='CH8', article=None, paragraph=None

        The reference language goes finer than the graph does: EUR-Lex cites a sentence
        inside a subparagraph inside a point ('A06P1L1LF'), while the smallest node is the
        paragraph. Everything below the paragraph is therefore **truncated**. Reading every
        digit out of the suffix instead, as this once did, invents a paragraph number:
        'A06P1L1LF' became paragraph 11 of an article that has four, and the edge was
        silently dropped by the `MATCH ... MERGE` that writes it.

        Points of a definitions article are the one asymmetry. The loader stores those
        unpadded, ``004.7``, because it builds the id from the parsed marker, while numbered
        paragraphs keep the padded id EUR-Lex puts in the markup, ``006.001``. This has to
        follow the loader, or every Article 4 reference misses a node that does exist.
        """
        if article_reference.startswith('CH'):
            return article_reference, None, None

        match = _ARTICLE_REF_RE.match(article_reference)
        if not match:
            # Ranges ('A12-A15'), annex and preamble references, and anything else EUR-Lex
            # publishes that has no single node to point at. Logged rather than dropped in
            # silence: an unresolved reference means a judgment nothing will ever ingest.
            logger.warning(
                "Unhandled case law reference %r: no interpretation edge will be created.",
                article_reference,
            )
            return None, None, None

        article_num = int(match.group('article'))
        article = f'art_{article_num}'

        if point := match.group('point'):
            return None, article, f'{article_num:03d}.{int(point)}'

        if paragraph := match.group('paragraph'):
            return None, article, f'{article_num:03d}.{int(paragraph):03d}'

        return None, article, None
