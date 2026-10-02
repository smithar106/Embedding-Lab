"""
Evaluate the agent against a small golden dataset.

Measures a NEW layer — agent quality:
  - tool-selection accuracy  (does the agent call retrieval when it should, and
                              skip it when it shouldn't?)
  - citation validity        (citations restricted to returned chunks)
  - expected-fact coverage   (for retrieval questions)
  - insufficient-evidence behaviour (for out-of-corpus questions)

Usage (from the project root, after ingesting the corpus):

    python scripts/evaluate_agent.py
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
from evaluation.answer_metrics import citation_validity, fact_coverage  # noqa: E402


def load_golden_agent(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the agent against a golden dataset")
    parser.add_argument("--questions", default="data/evaluation/golden_agent.json")
    args = parser.parse_args()

    items = load_golden_agent(args.questions)

    decisions = []       # (expected_tool_use, actual_tool_use) for retrieval/direct kinds
    coverage = []
    validity = []
    insufficiency = []

    per_question = {}
    for it in items:
        trace = run_agent(it["question"])
        per_question[it["question_id"]] = {
            "kind": it["kind"],
            "question": it["question"],
            "expected_tool_use": it["expected_tool_use"],
            "tool_requested": trace["tool_requested"],
            "final_answer": trace["final_answer"],
            "citations": trace["citations"],
            "citation_validity": trace["citation_validity"],
            "refused": trace["refused"],
            "latency": trace["latency"],
        }

        if it["kind"] in ("retrieval", "direct"):
            decisions.append((it["expected_tool_use"], trace["tool_requested"]))
        if it["kind"] == "retrieval" and it.get("expected_facts"):
            fc = fact_coverage(trace["final_answer"], it["expected_facts"])
            coverage.append(fc["coverage"] if fc["coverage"] is not None else 0.0)
        if trace["tool_requested"] and trace["citation_validity"] is not None:
            validity.append(trace["citation_validity"]["ratio"])
        if it["kind"] == "out_of_corpus":
            insufficiency.append(1.0 if trace["refused"] else 0.0)

    correct = sum(1 for exp, act in decisions if exp == act)
    total = len(decisions)
    accuracy = correct / total if total else 0.0

    print("=" * 72)
    print("AGENT EVALUATION")
    print("=" * 72)
    print(f"Tool-selection accuracy : {correct}/{total} = {accuracy:.3f}")
    print(f"Expected-fact coverage  : {sum(coverage)/len(coverage):.3f}" if coverage else "Expected-fact coverage  : n/a")
    print(f"Citation validity       : {sum(validity)/len(validity):.3f}" if validity else "Citation validity       : n/a")
    print(f"Insufficient-evidence   : {sum(insufficiency)}/{len(insufficiency)} refused correctly" if insufficiency else "Insufficient-evidence   : n/a")
    print()

    for qid, row in per_question.items():
        mark = "✓" if (row["expected_tool_use"] is None or row["expected_tool_use"] == row["tool_requested"]) else "✗"
        print(f"  {qid} [{row['kind']:<12}] tool={row['tool_requested']} (expected {row['expected_tool_use']}) {mark}")
    print()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    payload = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "tool_selection_accuracy": accuracy,
        "expected_fact_coverage": round(sum(coverage) / len(coverage), 3) if coverage else None,
        "citation_validity": round(sum(validity) / len(validity), 3) if validity else None,
        "insufficient_evidence_rate": round(sum(insufficiency) / len(insufficiency), 3) if insufficiency else None,
        "per_question": per_question,
    }
    out = results_dir / f"agent_evaluation_{ts}.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
