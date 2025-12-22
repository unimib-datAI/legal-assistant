import csv
import os
import re
from service.eurlex_exporter import EURLexHTMLParser
from service.iterator import DataIterator


class CsvExporter:
    """Class to export legal document data into CSV files."""

    def __init__(self, documents_config):
        self.documents_config = documents_config

    def process_all(self, output_dir='csv_output'):
        os.makedirs(output_dir, exist_ok=True)
        all_data = []

        for config in self.documents_config:
            try:
                exporter = EURLexHTMLParser(
                    html_file_path=config['html_file'],
                    celex=config['celex'],
                    author=config['author'],
                    publication_date=config['publication_date'],
                    date_of_application=config['date_of_application'],
                    eurolex_url=config['eurolex_url'],
                    document_info_url=config['document_info_url']
                )
                all_data.append(exporter.extract_data())
            except Exception as e:
                print(f"Errore processando {config['celex']}: {str(e)}")

        if all_data:
            self._write_final_csv_files(all_data, output_dir)
            print("CSV files generated successfully.")

        return all_data

    def _write_csv(self, filename, headers, rows, output_dir):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

    def _extract_number(self, id_string, prefix):
        match = re.search(f'{prefix}(.+)$', id_string)
        return match.group(1) if match else None

    def _write_final_csv_files(self, all_data, output_dir):
        self._write_acts(all_data, output_dir)
        self._write_chapters(all_data, output_dir)
        self._write_sections(all_data, output_dir)
        self._write_articles(all_data, output_dir)
        self._write_paragraphs(all_data, output_dir)
        self._write_recitals(all_data, output_dir)
        self._write_act_chapter_relations(all_data, output_dir)
        self._write_chapter_section_relations(all_data, output_dir)
        self._write_chapter_article_relations(all_data, output_dir)
        self._write_section_article_relations(all_data, output_dir)
        self._write_article_paragraph_relations(all_data, output_dir)
        self._write_act_recital_relations(all_data, output_dir)
        self._write_citations(all_data, output_dir)
        self._write_case_law_article_relations(all_data, output_dir)
        self._write_case_law_paragraph_relations(all_data, output_dir)
        self._write_case_law_chapter_relations(all_data, output_dir)

    def _write_acts(self, all_data, output_dir):
        iterator = DataIterator(all_data)
        rows = []
        for act, _ in iterator.iter_acts():
            rows.append([
                act['celex'], act['title'], act['author'],
                act['publication_date'], act['date_of_application'], act['eurolex_url']
            ])
        self._write_csv('acts.csv',
                        ['celex', 'title', 'author', 'publication_date', 'date_of_application', 'eurolex_url'],
                        rows, output_dir)

    def _write_chapters(self, all_data, output_dir):
        iterator = DataIterator(all_data)
        rows = []
        for act, chapter in iterator.iter_chapters():
            chapter_id = f"{act['celex']}{chapter['id']}"
            number = self._extract_number(chapter['id'], 'cpt_')
            rows.append([chapter_id, number, chapter['title']])

        self._write_csv('chapters.csv', ['chapter_id', 'number', 'title'], rows, output_dir)

    def _write_sections(self, all_data, output_dir):
        iterator = DataIterator(all_data)
        rows = []
        for act, chapter, section in iterator.iter_sections():
            section_id = f"{act['celex']}{section['id']}"
            rows.append([section_id, section['title']])

        self._write_csv('sections.csv', ['section_id', 'title'], rows, output_dir)

    def _write_articles(self, all_data, output_dir):
        iterator = DataIterator(all_data)
        rows = []
        for act, chapter, section, article in iterator.iter_articles():
            rows.append([f"{act['celex']}{article['id']}", article['title'], article['full_text']])
        self._write_csv('articles.csv', ['article_id', 'title', 'full_text'], rows, output_dir)

    def _write_paragraphs(self, all_data, output_dir):
        iterator = DataIterator(all_data)
        rows = []
        for act, chapter, section, article, paragraph in iterator.iter_paragraphs():
            rows.append([f"{act['celex']}_{paragraph['id']}", paragraph['text']])
        self._write_csv('paragraphs.csv', ['paragraph_id', 'text'], rows, output_dir)

    def _write_act_chapter_relations(self, all_data, output_dir):
        iterator = DataIterator(all_data)
        rows = []
        for act, chapter in iterator.iter_chapters():
            rows.append([
                act['celex'],
                f"{act['celex']}{chapter['id']}"
            ])
        self._write_csv('act_has_chapter.csv', ['from_celex', 'to_chapter_id'], rows, output_dir)

    def _write_chapter_section_relations(self, all_data, output_dir):
        iterator = DataIterator(all_data)
        rows = []
        for act, chapter, section in iterator.iter_sections():
            rows.append([
                f"{act['celex']}{chapter['id']}",
                f"{act['celex']}{section['id']}"
            ])
        self._write_csv('chapter_has_section.csv', ['from_chapter_id', 'to_section_id'], rows, output_dir)

    def _write_chapter_article_relations(self, all_data, output_dir):
        iterator = DataIterator(all_data)
        rows = []
        for act, chapter, section, article in iterator.iter_articles():
            if section is None:  # Only articles directly inside the chapter
                rows.append([
                    f"{act['celex']}{chapter['id']}",
                    f"{act['celex']}{article['id']}"
                ])
        self._write_csv('chapter_has_article.csv', ['from_chapter_id', 'to_article_id'], rows, output_dir)

    def _write_section_article_relations(self, all_data, output_dir):
        iterator = DataIterator(all_data)
        rows = []
        for act, chapter, section, article in iterator.iter_articles():
            if section is not None:  # Only articles inside sections
                rows.append([
                    f"{act['celex']}{section['id']}",
                    f"{act['celex']}{article['id']}"
                ])
        self._write_csv('section_has_article.csv', ['from_section_id', 'to_article_id'], rows, output_dir)

    def _write_article_paragraph_relations(self, all_data, output_dir):
        iterator = DataIterator(all_data)
        rows = []
        for act, chapter, section, article, paragraph in iterator.iter_paragraphs():
            rows.append([
                f"{act['celex']}{article['id']}",
                f"{act['celex']}_{paragraph['id']}"
            ])
        self._write_csv('article_has_paragraph.csv',
                        ['from_article_id', 'to_paragraph_id'], rows, output_dir)

    def _write_citations(self, all_data, output_dir):
        iterator = DataIterator(all_data)
        rows = []
        for act, data in iterator.iter_acts():
            for citation in data['citations']:
                rows.append([
                    citation['from_article_id'],
                    citation['to_article_id'],
                    citation['citation_type'],
                    citation['paragraph_id']
                ])
        self._write_csv('article_citations.csv',
                        ['from_article_id', 'to_article_id', 'citation_type', 'paragraph_id'],
                        rows, output_dir)

    def _write_recitals(self, all_data, output_dir):
        iterator = DataIterator(all_data)
        rows = []
        for act, recital in iterator.iter_recitals():
            recital_id = f"{act['celex']}{recital['id']}"
            rows.append([recital_id, recital['text']])
        self._write_csv('recitals.csv', ['recital_id', 'full_text'], rows, output_dir)

    def _write_act_recital_relations(self, all_data, output_dir):
        iterator = DataIterator(all_data)
        rows = []
        for act, recital in iterator.iter_recitals():
            rows.append([
                act['celex'],
                f"{act['celex']}{recital['id']}"
            ])
        self._write_csv('act_has_recital.csv', ['from_celex', 'to_recital_id'], rows, output_dir)

    def _write_case_law_paragraph_relations(self, all_data, output_dir):
        iterator = DataIterator(all_data)
        rows = []

        for act, data in iterator.iter_acts():
            for case_law in data.get('case_law', []):
                if case_law.get('paragraph'):
                    rows.append([
                        case_law['case_law_identifier'],
                        f"{act['celex']}_{case_law['paragraph']}"
                    ])

        self._write_csv('case_law_interprets_paragraph.csv',
                        ['from_case_law_id', 'to_paragraph_id'],
                        rows, output_dir)

    def _write_case_law_article_relations(self, all_data, output_dir):
        iterator = DataIterator(all_data)
        rows = []

        for act, data in iterator.iter_acts():
            for case_law in data.get('case_law', []):
                if case_law.get('article') and not case_law.get('paragraph'):
                    rows.append([
                        case_law['case_law_identifier'],
                        f"{act['celex']}{case_law['article']}"
                    ])

        self._write_csv('case_law_interprets_article.csv',
                        ['from_case_law_id', 'to_article_id'],
                        rows, output_dir)

    def _write_case_law_chapter_relations(self, all_data, output_dir):
        iterator = DataIterator(all_data)
        rows = []

        for act, data in iterator.iter_acts():
            for case_law in data.get('case_law', []):
                if case_law.get('chapter') and not case_law.get('paragraph') and not case_law.get('article'):
                    rows.append([
                        case_law['case_law_identifier'],
                        f"{act['celex']}{case_law['chapter']}"
                    ])

        self._write_csv('case_law_interprets_chapter.csv',
                        ['from_case_law_id', 'to_chapter_id'],
                        rows, output_dir)