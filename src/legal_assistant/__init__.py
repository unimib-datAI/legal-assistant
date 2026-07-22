"""Graph-RAG over EU digital-regulation legislation (AI Act, DGA, Data Act, GDPR).

See the README for the layout. Nothing is imported here on purpose: the subpackages pull
in heavy dependencies (Neo4j drivers, sentence-transformers, OpenAI clients), so importing
``legal_assistant`` stays cheap and callers import exactly the module they need.
"""

__version__ = "0.1.0"
