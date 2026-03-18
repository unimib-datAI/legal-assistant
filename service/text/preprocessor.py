import logging
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Stateless NLP preprocessing for legal paragraph text.

    Handles tokenization, lemmatization, and chunking — all pure text
    operations with no graph or embedding model dependency.
    """

    def __init__(self) -> None:
        self._lemmatizer = WordNetLemmatizer()

    def _tokenize_paragraphs(self, paragraphs: list[dict]) -> list[dict]:
        """Split each paragraph's text into sentences, preserving paragraph_id."""
        tokenized = []
        for para in paragraphs:
            tokenized.append({
                "paragraph_id": para["paragraph_id"],
                "text": sent_tokenize(para["text"]),
            })
        logger.debug("Tokenized %d paragraphs into sentences", len(tokenized))
        return tokenized

    def _lemmatize_paragraphs(self, paragraphs: list[dict]) -> list[dict]:
        """Lemmatize each sentence in a tokenized paragraph, preserving paragraph_id.

        Expects tokenized input: {"paragraph_id": ..., "text": ["sentence1", ...]}
        """
        lemmatized = []
        for para in paragraphs:
            sentences = [
                " ".join(self._lemmatizer.lemmatize(w.lower()) for w in word_tokenize(sentence))
                for sentence in para["text"]
            ]
            lemmatized.append({
                "paragraph_id": para["paragraph_id"],
                "text": sentences,
            })
        logger.debug("Lemmatized %d paragraphs", len(lemmatized))
        return lemmatized

    def to_chunks(self, paragraphs: list[dict], skip_first: bool = True) -> list[dict]:
        """Tokenize, lemmatize, and split paragraphs into sentence-level chunks.

        Combines the three preprocessing steps into a single call.

        Args:
            paragraphs: Raw paragraphs from the KG — {"paragraph_id": ..., "text": str}
            skip_first: Skip the first sentence per paragraph (often just a number/label)

        Returns:
            List of dicts with 'paragraph_id', 'chunk_index', and 'text'
        """
        tokenized = self._tokenize_paragraphs(paragraphs)
        lemmatized = self._lemmatize_paragraphs(tokenized)
        return self._to_chunks(lemmatized, skip_first)

    @staticmethod
    def _to_chunks(lemmatized_paragraphs: list[dict], skip_first: bool) -> list[dict]:
        chunks = []
        for para in lemmatized_paragraphs:
            sentences = para["text"]
            start_idx = 1 if skip_first and len(sentences) > 1 else 0
            for chunk_idx, sentence in enumerate(sentences[start_idx:]):
                chunks.append({
                    "paragraph_id": para["paragraph_id"],
                    "chunk_index": chunk_idx,
                    "text": sentence,
                })
        logger.debug("Created %d chunks from %d paragraphs", len(chunks), len(lemmatized_paragraphs))
        return chunks
