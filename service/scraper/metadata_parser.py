import logging

from bs4 import BeautifulSoup

from service.scraper.browser_fetcher import BrowserFetcher

logger = logging.getLogger(__name__)


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

    def enrich_article_reference(self, article_reference):
        """
        Enrich and normalize article reference to match existing database format.

        EUR-Lex format → Database format:
        - 'A01' → chapter=None, article='art_1', paragraph=None
        - 'A02P1' → chapter=None, article='art_2', paragraph='002.001'
        - 'A04PT11' → chapter=None, article='art_4', paragraph='004.011'
        - 'CH8' → chapter='CH8', article=None, paragraph=None
        """
        if article_reference.startswith('CH'):
            return article_reference, None, None

        if article_reference.startswith('A'):
            rest = article_reference[1:]
            article_num, paragraph = None, None

            if 'PT' in rest:
                article_num, part_num = map(int, rest.split('PT'))
                paragraph = f'{article_num:03d}.{part_num:03d}'
            elif 'P' in rest:
                article_num, paragraph_part = rest.split('P')
                article_num = int(article_num)

                paragraph_digits = "".join(filter(str.isdigit, paragraph_part))
                paragraph = f'{article_num:03d}.{int(paragraph_digits):03d}'
            else:
                article_num = int(rest)

            article = f'art_{article_num}' if article_num else None
            return None, article, paragraph

        return None, None, None
