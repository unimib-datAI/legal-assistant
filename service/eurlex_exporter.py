import re

from bs4 import BeautifulSoup
from service.metadata_parser import MetadataParser


class EURLexHTMLParser:
    """EURLex HTML parser to extract structured data from legal documents."""

    def __init__(self, html_file_path, celex, author, publication_date, date_of_application, eurolex_url, document_info_url):
        with open(html_file_path, 'r', encoding='utf-8') as f:
            self.soup = BeautifulSoup(f.read(), 'html.parser')

        self.celex = celex
        self.author = author
        self.publication_date = publication_date
        self.date_of_application = date_of_application
        self.eurolex_url = eurolex_url
        self.document_info_url = document_info_url

    def extract_data(self):
        """Extract structured data from the HTML document"""
        title = self._get_title()
        chapters_data = self._get_chapters()
        recitals_data = self._get_recitals()
        citations = self._extract_citations(chapters_data)
        case_law = self._get_case_law()

        return {
            'act': {
                'celex': self.celex,
                'title': title,
                'author': self.author,
                'publication_date': self.publication_date,
                'date_of_application': self.date_of_application,
                'eurolex_url': self.eurolex_url
            },
            'chapters': chapters_data,
            'recitals': recitals_data,
            'citations': citations,
            'case_law': case_law
        }

    def _extract_citations(self, chapters_data):
        """Extract citations from the entire document"""
        citations = []

        for chapter in chapters_data:
            for article in chapter['articles']:
                citations.extend(self._extract_article_citations(article))

            for section in chapter['sections']:
                for article in section['articles']:
                    citations.extend(self._extract_article_citations(article))

        return citations

    def _extract_article_citations(self, article):
        """Extract citations from a single article"""
        citations = []
        source_article_id = f"{self.celex}{article['id']}"

        for paragraph in article['paragraphs']:
            cited_articles = self._find_article_references(paragraph['text'])

            for cited_article_num in cited_articles:
                citations.append({
                    'from_article_id': source_article_id,
                    'to_article_id': f"{self.celex}art_{cited_article_num}",
                    'citation_type': 'CITES',
                    'paragraph_id': f"{self.celex}_{paragraph['id']}"
                })

        return citations

    def _find_article_references(self, text):
        """Find article references in a given text"""
        cited_articles = set()

        # "Article 5", "Article 89"
        for match in re.finditer(r'\bArticle\s+(\d+)', text, re.IGNORECASE):
            cited_articles.add(int(match.group(1)))

        # "Articles 12 to 15"
        for match in re.finditer(r'\bArticles\s+(\d+)\s+to\s+(\d+)', text, re.IGNORECASE):
            start, end = int(match.group(1)), int(match.group(2))
            cited_articles.update(range(start, end + 1))

        # "Art. 98"
        for match in re.finditer(r'\bArt\.\s+(\d+)', text, re.IGNORECASE):
            cited_articles.add(int(match.group(1)))

        return cited_articles

    def _get_title(self):
        """Extract the main title of the document"""
        title_div = self.soup.find('div', {'class': 'eli-main-title'})
        if title_div:
            title_parts = [p.get_text(strip=True) for p in title_div.find_all('p')]
            return ' '.join(title_parts)
        return None

    def _get_chapters(self):
        """Extract chapters from the document"""
        chapters = []
        chapter_divs = self.soup.find_all('div', id=re.compile(r'^cpt_[^.]+$'))

        for chapter_div in chapter_divs:
            chapter_id = chapter_div.get('id')
            chapters.append({
                'id': chapter_id,
                'title': self._get_chapter_title(chapter_div, chapter_id),
                'sections': self._get_sections(chapter_div, chapter_id),
                'articles': self._get_articles(chapter_div)
            })

        return chapters

    def _get_chapter_title(self, chapter_div, chapter_id):
        """Extract the title of a chapter"""
        title_div = chapter_div.find('div', id=re.compile(f'^{chapter_id}\\.tit_'))
        if title_div:
            title_p = title_div.find('p')
            if title_p:
                return title_p.get_text(strip=True)
        return None

    def _get_sections(self, chapter_div, chapter_id):
        """Extract the sections of the document"""
        sections = []
        section_divs = chapter_div.find_all('div', id=re.compile(f'^{chapter_id}\\.sct_'))

        for section_div in section_divs:
            section_id = section_div.get('id')
            if '.tit_' not in section_id:
                sections.append({
                    'id': section_id,
                    'title': self._get_section_title(section_div, section_id),
                    'articles': self._get_articles(section_div)
                })

        return sections

    def _get_section_title(self, section_div, section_id):
        """Extract the section title"""
        title_div = section_div.find('div', id=re.compile(f'^{section_id}\\.tit_'))
        if title_div:
            title_p = title_div.find('p')
            if title_p:
                return title_p.get_text(strip=True)
        return None

    def _get_articles(self, parent_div):
        """Extract articles from the document"""
        articles = []
        article_divs = parent_div.find_all('div', id=re.compile(r'^art_\d+$'), recursive=False)

        for article_div in article_divs:
            article_id = article_div.get('id')
            article_number_p = article_div.find('p', class_='oj-ti-art')

            full_text = ' '.join(article_div.stripped_strings)

            articles.append({
                'id': article_id,
                'number': article_number_p.get_text(strip=True) if article_number_p else None,
                'title': self._get_article_title(article_div, article_id),
                'paragraphs': self._get_paragraphs(article_div, article_id),
                'full_text': full_text
            })

        return articles

    def _get_article_title(self, article_div, article_id):
        """Extract article title"""
        title_div = article_div.find('div', id=f'{article_id}.tit_1')
        if title_div:
            title_p = title_div.find('p', class_='oj-sti-art')
            if title_p:
                return title_p.get_text(strip=True)
        return None

    def _get_paragraphs(self, article_div, article_id):
        """Extract paragraph from the article"""
        paragraphs = []
        article_num = article_id.split('_')[1].zfill(3)
        paragraph_divs = article_div.find_all('div', id=re.compile(f'^{article_num}\\.\\d+$'))

        for para_div in paragraph_divs:
            para_id = para_div.get('id')
            para_paragraphs = para_div.find_all('p', class_='oj-normal')
            text_parts = [p.get_text(separator=' ', strip=True) for p in para_paragraphs]

            paragraphs.append({
                'id': para_id,
                'text': ' '.join(text_parts)
            })

        return paragraphs

    def _get_recitals(self):
        """Extract recitals from the document"""
        recitals = []
        recital_divs = self.soup.find_all('div', id=re.compile(r'^rct_\d+$'))

        for recital_div in recital_divs:
            recital_id = recital_div.get('id')
            recital_num = recital_id.replace('rct_', '')

            text_parts = []
            for p in recital_div.find_all('p', class_='oj-normal'):
                text_parts.append(p.get_text(separator=' ', strip=True))

            recitals.append({
                'id': recital_id,
                'number': recital_num,
                'text': ' '.join(text_parts)
            })

        return recitals

    def _get_case_law(self):
        """Extract case law information from the document"""
        extractor = MetadataParser()
        return extractor.parse_eurovoc_descriptors(self.document_info_url)

