"""The CLI surface of ``graph build``: flags in, exit code out.

Two things are worth pinning here rather than leaving to a manual run. The default is no
longer destructive, which is the breaking change in this work, and the chained ingest must
be opt-in, since it turns a four-document job into a several-hundred-fetch one.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from legal_assistant.cli.main import build_parser
from legal_assistant.pipelines.graph_build import GraphBuildResult

GDPR = "32016R0679"


def _parse(*argv: str):
    return build_parser().parse_args(["graph", "build", *argv])


def _result(**kwargs) -> GraphBuildResult:
    defaults = {"celex_ids": [GDPR], "indexed_labels": ["Paragraph"], "skipped": [], "failed": []}
    return GraphBuildResult(**{**defaults, **kwargs})


@pytest.fixture
def run():
    """Invoke the subcommand with the pipeline stubbed, returning (exit code, mocks)."""
    with patch("legal_assistant.pipelines.graph_build.build_graph") as build, \
         patch("legal_assistant.cli.main._ingest_case_law_for") as chain:
        build.return_value = _result()
        yield lambda args: (args.func(args), build, chain)


def test_the_database_is_kept_by_default(run):
    _, build, _ = run(_parse("--celex", GDPR))
    assert build.call_args.kwargs["clear_db"] is False


def test_clear_is_opt_in(run):
    _, build, _ = run(_parse("--celex", GDPR, "--clear"))
    assert build.call_args.kwargs["clear_db"] is True


def test_no_clear_is_gone_rather_than_silently_accepted():
    """It described the default once. Leaving it as a no-op would mislead old scripts."""
    with pytest.raises(SystemExit):
        _parse("--no-clear")


def test_force_is_passed_through(run):
    _, build, _ = run(_parse("--celex", GDPR, "--force"))
    assert build.call_args.kwargs["force"] is True


def test_case_law_is_not_ingested_unless_asked(run):
    _, _, chain = run(_parse("--celex", GDPR))
    chain.assert_not_called()


def test_with_case_law_chains_the_ingest_for_the_acts_built(run):
    _, _, chain = run(_parse("--celex", GDPR, "--with-case-law"))
    chain.assert_called_once_with([GDPR])


def test_a_failed_act_exits_non_zero(run):
    with patch("legal_assistant.pipelines.graph_build.build_graph") as build:
        build.return_value = _result(celex_ids=[], failed=[(GDPR, "parser regression")])
        args = _parse("--celex", GDPR)
        assert args.func(args) == 1


def test_a_fully_skipped_run_is_a_success(run):
    """Resuming a finished build is not an error, it is the expected no-op."""
    with patch("legal_assistant.pipelines.graph_build.build_graph") as build:
        build.return_value = _result(celex_ids=[], skipped=[GDPR])
        args = _parse("--celex", GDPR)
        assert args.func(args) == 0
