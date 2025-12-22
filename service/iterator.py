class DataIterator:
    """Iterator class for navigating hierarchical legal document data"""

    def __init__(self, all_data):
        self.all_data = all_data

    def iter_acts(self):
        """Iterate over all acts"""
        for data in self.all_data:
            yield data['act'], data

    def iter_chapters(self):
        """Iterate over all chapters across all acts"""
        for act, data in self.iter_acts():
            for chapter in data['chapters']:
                yield act, chapter

    def iter_sections(self):
        """Iterate over all sections across all chapters"""
        for act, data in self.iter_acts():
            for chapter in data['chapters']:
                for section in chapter['sections']:
                    yield act, chapter, section

    def iter_articles(self):
        """Iterate over all articles, whether in chapters or sections"""
        for act, data in self.iter_acts():
            for chapter in data['chapters']:
                # Articles directly under chapter
                for article in chapter['articles']:
                    yield act, chapter, None, article

                # Articles inside sections
                for section in chapter['sections']:
                    for article in section['articles']:
                        yield act, chapter, section, article

    def iter_paragraphs(self):
        """Iterate over all paragraphs across all articles"""
        for act, chapter, section, article in self.iter_articles():
            for paragraph in article['paragraphs']:
                yield act, chapter, section, article, paragraph

    def iter_recitals(self):
        """Iterate over all recitals across all acts"""
        for act, data in self.iter_acts():
            for recital in data.get('recitals', []):
                yield act, recital