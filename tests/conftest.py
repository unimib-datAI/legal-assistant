"""Shared pytest fixtures.

Unit tests must not touch Neo4j or the OpenAI API — mock those boundaries with
``unittest.mock`` (see .claude/CLAUDE.md). Test modules mirror the package layout:
``src/legal_assistant/rag/acts.py`` -> ``tests/rag/test_acts.py``.
"""
