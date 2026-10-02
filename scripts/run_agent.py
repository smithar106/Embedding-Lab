"""
Run the minimal agent and make the loop visible.

Prints the separation between the LLM's DECISION and the deterministic Python
TOOL EXECUTION, and saves a structured trace to results/agent_runs/.

Usage (from the project root, after ingesting the corpus):

    python scripts/run_agent.py                       # run 3 demo queries
    python scripts/run_agent.py --query "Say hello."
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from agent.agent import run_agent  # noqa: E402
from tools.retrieval_tool import TOOL_SCHEMA, TOOL_DESCRIPTION  # noqa: E402

DEMO_QUERIES = [
    "Which component stores energy in an electric field instead of chemically?",
    "Say hello.",
    "What is the boiling point of tungsten?",
]


def print_tool_schema() -> None:
    print("TOOL")
    print(f"  name       : {TOOL_SCHEMA['function']['name']}")
    print(f"  description: {TOOL_DESCRIPTION}")
    print(f"  parameters : {json.dumps(TOOL_SCHEMA['function']['parameters'])}")
    print()


def print_trace(trace: dict) -> None:
    print("USER")
    print(f"  {trace['user_query']}\n")

    if trace["tool_requested"]:
        print("AGENT DECISION")
        print(f"  Tool requested: {trace['tool_name']}\n")
        print("TOOL ARGUMENTS")
        for k, v in trace["tool_arguments"].items():
            print(f"  {k}: {v}")
        print()
        print("TOOL EXECUTION")
        print("  retrieval_search()")
        print("      ↓")
        print("  embed query → pgvector similarity search")
        print("      ↓")
        print(f"  {len(trace['tool_results'])} chunks returned\n")
        print("TOOL RESULT")
        for r in trace["tool_results"]:
            print(f"  {r['rank']}. {r['chunk_id']} — {r['title']}")
        print()
    else:
        print("AGENT DECISION")
        print("  No tool.\n")

    print("AGENT FINAL ANSWER")
    print(f"  {trace['final_answer']}\n")

    print("TRACE")
    print(f"  citations: {trace['citations']}")
    if trace["citation_validity"] is not None:
        print(f"  citation validity: {trace['citation_validity']['ratio']}")
    print(f"  latency: decision {trace['latency'].get('agent_decision_ms')} ms | "
          f"tool {trace['latency'].get('tool_execution_ms', 0)} ms | "
          f"generation {trace['latency'].get('final_generation_ms')} ms | "
          f"total {trace['latency'].get('total_ms')} ms")
    print()


def save_trace(trace: dict) -> Path:
    d = Path("results/agent_runs")
    d.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = d / f"agent_{ts}.json"
    out.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimal retrieval agent")
    parser.add_argument("--query", default=None)
    args = parser.parse_args()

    print_tool_schema()

    queries = [args.query] if args.query else DEMO_QUERIES
    for q in queries:
        trace = run_agent(q)
        print("=" * 72)
        print_trace(trace)
        saved = save_trace(trace)
        print(f"Saved trace to {saved}")
        print()


if __name__ == "__main__":
    main()
