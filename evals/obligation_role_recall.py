"""Role-recall for the obligations branch: does a role query surface that role's duties?

The dataset is generated from the graph, so it needs no manual annotation: every actor that
bears obligations is a case, its query is "What obligations does a <role> have under <act>?",
and its expected set is exactly the obligations addressed to that actor and its qualified
children. Two things are measured, both without answer synthesis:

- classified:  the addressee classifier picks the intended role for the query.
- has_obligations: the graph filter returns a non-empty obligation set for the role.

A run needs only the classifier LLM and Neo4j, no embeddings or reranker, so it is cheap.

Example:
    legal-assistant eval roles --act 32016R0679
    legal-assistant eval roles --act 32016R0679 --limit 10
"""
import argparse
import logging
from collections import Counter

from legal_assistant.graph.queries import NodeQueries
from legal_assistant.logging_setup import configure_logging
from legal_assistant.pipelines.obligation_checklist import fetch_checklist
from legal_assistant.rag.acts import CELEX_TO_ACT_NAME
from legal_assistant.rag.intent_classifier import QueryClassifier
from legal_assistant.resources import make_chat_llm, make_langchain_graph

configure_logging()
logger = logging.getLogger(__name__)


def _roles_bearing_obligations(graph, celex: str) -> list[dict]:
    """Actors that are the direct addressee of at least one obligation of the act."""
    rows = graph.query(
        """
        MATCH (act:Act {id:$celex})-[:CONTAINS*]->()-[:STATES]->(:Obligation)-[:ADDRESSED_TO]->(a:Actor)
        RETURN a.id AS id, a.label AS label, count(*) AS n
        ORDER BY n DESC
        """,
        params={"celex": celex},
    )
    return rows


def run(act: str, limit: int | None) -> None:
    graph = make_langchain_graph()
    classifier = QueryClassifier(graph, make_chat_llm(temperature=0.0))
    act_name = CELEX_TO_ACT_NAME.get(act, act)

    roles = _roles_bearing_obligations(graph, act)
    if limit:
        roles = roles[:limit]

    tally: Counter = Counter()
    print(f"\nRole recall for {act_name} ({act}), {len(roles)} role(s)\n")
    print(f"{'role':40} {'clf':>4} {'obls':>5}  query-classified-as")
    for role in roles:
        query = f"What obligations does a {role['label']} have under the {act_name}?"
        addressees, _ = classifier.classify_addressees(query)
        expected = fetch_checklist(graph, act, role["id"])

        classified = role["id"] in addressees
        has_obls = len(expected) > 0
        tally["classified"] += int(classified)
        tally["has_obligations"] += int(has_obls)
        tally["total"] += 1
        print(f"{role['id'][:40]:40} {'Y' if classified else 'n':>4} "
              f"{len(expected):>5}  {addressees[:3]}")

    total = tally["total"] or 1
    print(f"\nclassified   : {tally['classified']}/{total} "
          f"({100*tally['classified']/total:.0f}%)")
    print(f"has_obligations: {tally['has_obligations']}/{total} "
          f"({100*tally['has_obligations']/total:.0f}%)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--act", default="32016R0679", help="CELEX id of the act.")
    parser.add_argument("--limit", type=int, help="Evaluate at most N roles.")
    args = parser.parse_args(argv)
    run(args.act, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
