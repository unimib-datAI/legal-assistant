import re

from bs4 import BeautifulSoup

from legal_assistant.scraper.metadata_parser import MetadataParser

# A standalone "(3)" element: the number half of a split definition entry.
_SPLIT_DEF_NUM_RE = re.compile(r'^\((\d+)\)$')

# Any paragraph div, whatever article it belongs to: "<article:3>.<paragraph>".
_PARAGRAPH_DIV_RE = re.compile(r'^\d{3}\.\d+$')

# An annex root div: "anx_III". Title sub-divs carry a dotted suffix and must not match.
_ANNEX_DIV_RE = re.compile(r'^anx_[IVXLCDM]+$')

# A point's numbering marker, standing alone in its own element: "1.", "(a)", "(iii)".
# Deliberately narrow: a prose word such as "The" must never be taken for a label.
_POINT_LABEL_RE = re.compile(r'^(\d{1,3}\.?|\(\w{1,4}\))$')

# Inside an annex: the two-line heading of the annex itself, and its internal group headings.
_ANNEX_TITLE_CLASS = 'oj-doc-ti'
_ANNEX_HEADING_CLASS = 'oj-ti-grseq-1'


class EURLexHTMLParser:
    """EURLex HTML parser to extract structured data from legal documents."""

    def __init__(self, html_file_path, celex, eurolex_url, document_info_url):
        with open(html_file_path, 'r', encoding='utf-8') as f:
            self.soup = BeautifulSoup(f.read(), 'html.parser')

        self.celex = celex
        self.eurolex_url = eurolex_url
        self.document_info_url = document_info_url

    def extract_data(self):
        """Extract structured data from the HTML document"""
        return {
            'act': {
                'celex': self.celex,
                'title': self._get_title(),
                'eurolex_url': self.eurolex_url
            },
            'chapters': self._get_chapters(),
            'recitals': self._get_recitals(),
            'annexes': self._get_annexes(),
            'case_law': self._get_case_law()
        }

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

    def _get_annexes(self):
        """Extract the annexes of the document.

        Only some acts have any: the AI Act has thirteen, the other three in the corpus have
        none. Annexes hang off the act itself, not off a chapter, in the same structural
        position as recitals.
        """
        annexes = []
        for annex_div in self.soup.find_all('div', id=_ANNEX_DIV_RE):
            annex_id = annex_div.get('id')
            annexes.append({
                'id': annex_id,
                'number': annex_id.split('_', 1)[1],
                'title': self._get_annex_title(annex_div),
                'points': self._get_annex_points(annex_div),
            })

        return annexes

    def _get_annex_points(self, annex_div):
        """Split one annex into its numbered points.

        An annex div has no child divs, so unlike an article there is no element per point
        to read. Two shapes occur in the published markup:

        1. **Tables**: each numbered point is a ``td`` pair, the marker cell then the text
           cell, with sub-points as tables nested inside the text cell. Most annexes.
        2. **Bare paragraphs**: unclassed ``<p>`` alternating a marker and its prose. Annex
           VI is the case, and it carries no ``oj-normal`` class at all, so anything keyed on
           that class sees an empty annex rather than failing.

        A point's ``label`` carries its whole path within the annex, "1(a)" and not "(a)",
        because Article 6(2) cites "Annex III, point 1(a)". The annex number is prepended by
        the loader, which is what knows it.
        """
        if annex_div.find('table') is not None:
            return self._table_points(annex_div)
        return self._bare_points(annex_div)

    def _table_points(self, annex_div):
        """Points from the table shape, in document order.

        Prose outside the tables is kept as a point with a null label. Annex III opens with
        the sentence that governs every point under it, and losing unnumbered lead-in text
        would drop published words from the graph.
        """
        points, heading = [], None
        for element in annex_div.find_all(['table', 'p']):
            if element.name == 'table':
                # Nested tables are reached from the point that owns them, not on their own.
                if element.find_parent('table') is None:
                    points.extend(self._points_from_table(element, '', heading))
                continue

            if element.find_parent('table') is not None:
                continue

            classes = element.get('class') or []
            if _ANNEX_TITLE_CLASS in classes:
                continue
            if _ANNEX_HEADING_CLASS in classes:
                heading = element.get_text(separator=' ', strip=True)
                continue
            if text := element.get_text(separator=' ', strip=True):
                points.append({'label': None, 'text': text, 'section_heading': heading})

        return points

    def _points_from_table(self, table, prefix, heading):
        """One table is one point: marker cell, text cell, then its nested sub-points.

        Some annexes indent a point with a blank leading column, so Annex VII's rows read
        ``['', '3.1.', 'The application ...']``. Taking the marker off cell zero would give
        an empty label and push the real marker into the text, hence the skip.
        """
        cells = [td for td in table.find_all('td') if td.find_parent('table') is table]
        while cells and not cells[0].get_text(strip=True):
            cells.pop(0)
        if len(cells) < 2:
            return []

        label = prefix + self._normalise_label(cells[0].get_text(separator=' ', strip=True))
        body = cells[1]
        points = [{
            'label': label,
            'text': self._own_text(body),
            'section_heading': heading,
        }]

        for nested in body.find_all('table'):
            if nested.find_parent('table') is table:
                points.extend(self._points_from_table(nested, label, heading))

        return points

    @staticmethod
    def _own_text(cell):
        """A cell's own prose, excluding anything belonging to a table nested inside it.

        Walks strings rather than ``<p>`` elements: a cell wraps its text in ``<p>`` in some
        annexes and in ``<span>`` in others, and Annex VII's text cells hold a ``<span>``
        followed by the nested tables of their sub-points.
        """
        owner = cell.find_parent('table')
        parts = [
            text
            for node in cell.find_all(string=True)
            if node.find_parent('table') is owner and (text := node.strip())
        ]
        return ' '.join(parts)

    @staticmethod
    def _normalise_label(marker):
        """"1." and "1" are the same point; "(a)" keeps its brackets so paths stay readable."""
        return marker.rstrip('.')

    @staticmethod
    def _bare_points(annex_div):
        """Points from alternating unclassed ``<p>``: a marker, then the prose it numbers."""
        points, label, heading = [], None, None
        for p in annex_div.find_all('p'):
            classes = p.get('class') or []
            if _ANNEX_TITLE_CLASS in classes:
                continue
            if not (text := p.get_text(separator=' ', strip=True)):
                continue
            if _ANNEX_HEADING_CLASS in classes:
                heading = text
                continue
            if _POINT_LABEL_RE.match(text):
                label = text.rstrip('.')
                continue
            points.append({'label': label, 'text': text, 'section_heading': heading})
            label = None

        return points

    @staticmethod
    def _get_annex_title(annex_div):
        """The annex heading, which the markup splits in two.

        Both halves are ``p.oj-doc-ti``: the first restates the number ("ANNEX III"), which
        is already carried by ``number``, and the second is the title proper.
        """
        headings = annex_div.find_all('p', class_=_ANNEX_TITLE_CLASS)
        if len(headings) < 2:
            return None
        return headings[1].get_text(separator=' ', strip=True)

    def _get_article_title(self, article_div, article_id):
        """Extract article title"""
        title_div = article_div.find('div', id=f'{article_id}.tit_1')
        if title_div:
            title_p = title_div.find('p', class_='oj-sti-art')
            if title_p:
                return title_p.get_text(strip=True)
        return None

    @staticmethod
    def _div_text(div):
        """Concatenated oj-normal text of one paragraph div."""
        parts = [p.get_text(separator=' ', strip=True) for p in div.find_all('p', class_='oj-normal')]
        return ' '.join(t for t in parts if t)

    def _get_paragraphs(self, article_div, article_id):
        """Extract the paragraphs of one article.

        Three shapes occur in the published markup:

        1. **Numbered paragraphs**: ``div`` children whose id is ``<article>.<paragraph>``.
        2. **Amending articles**: an article whose body restates paragraphs of *another*
           act, so their div ids carry the amended article's number, not this one's
           (AI Act art_108 holds ``017.003``, ``019.004``, …). Their ids must be re-derived
           from the containing article: ``017.003`` is also the real id of AI Act Article
           17(3), and reusing it would silently overwrite that provision.
        3. **Unnumbered articles**: definitions and single-clause articles, whose text sits
           directly under the article div.
        """
        article_num = article_id.split('_')[1].zfill(3)
        paragraphs = []
        used_ids = set()
        consumed_divs = []

        own_pattern = re.compile(f'^{article_num}\\.\\d+$')
        for para_div in article_div.find_all('div', id=own_pattern):
            para_id = para_div.get('id')
            used_ids.add(para_id)
            consumed_divs.append(para_div)
            paragraphs.append({'id': para_id, 'text': self._div_text(para_div)})

        # Shape 2: paragraph divs belonging to another article.
        foreign = [d for d in article_div.find_all('div', id=_PARAGRAPH_DIV_RE)
                   if d.get('id') not in used_ids]
        seq = 0
        for para_div in foreign:
            seq += 1
            while f'{article_num}.{seq:03d}' in used_ids:
                seq += 1
            para_id = f'{article_num}.{seq:03d}'
            used_ids.add(para_id)
            consumed_divs.append(para_div)
            paragraphs.append({'id': para_id, 'text': self._div_text(para_div)})

        # Text that sits directly under the article, outside every paragraph div.
        consumed = {id(p) for div in consumed_divs for p in div.find_all('p', class_='oj-normal')}
        text_parts = [
            t for p in article_div.find_all('p', class_='oj-normal')
            if id(p) not in consumed and (t := p.get_text(separator=' ', strip=True))
        ]

        if paragraphs:
            # An amending article introduces each restated paragraph with a connective line
            # ("In Article 17, the following paragraph is added:"). Those lines are the
            # article's own words and belong in the graph.
            if text_parts:
                paragraphs.insert(0, {'id': f'{article_num}.0', 'text': ' '.join(text_parts)})
            return paragraphs

        # Shape 3: no numbered subdivisions, i.e. definitions or a single-clause article.
        if not text_parts:
            return paragraphs

        if not any(_SPLIT_DEF_NUM_RE.match(t) for t in text_parts):
            paragraphs.append({'id': f'{article_num}.0', 'text': ' '.join(text_parts)})
            return paragraphs

        # A definitions article splits each entry across several oj-normal elements: a
        # standalone "(N)" followed by the definition, which may itself run across further
        # elements (nested sub-points). Group everything up to the next "(N)".
        groups: list[tuple[str | None, list[str]]] = []
        for part in text_parts:
            match = _SPLIT_DEF_NUM_RE.match(part)
            if match:
                groups.append((match.group(1), []))
            elif groups:
                groups[-1][1].append(part)
            else:
                # The article's opening line ("For the purposes of this Regulation:"),
                # which precedes the first definition.
                groups.append((None, [part]))

        for def_num, body in groups:
            text = ' '.join(body)
            if not text:
                continue
            if def_num is None:
                paragraphs.append({'id': f'{article_num}.0', 'text': text})
            else:
                paragraphs.append({'id': f'{article_num}.{def_num}', 'text': f'({def_num}) {text}'})

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

