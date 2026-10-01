"""
Run the golden evaluation questions through every model and report metrics.

Usage (from the project root, after ingesting the corpus you want to evaluate):

    python scripts/evaluate_models.py
    python scripts/evaluate_models.py --questions data/evaluation/golden_questions_hard.json
    python scripts/evaluate_models.py --top-k 5 --relevance-level document
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion.chunker import CHUNK_OVERLAP, CHUNK_SIZE  # noqa: E402
from models.embedding_models import MODELS  # noqa: E402
from evaluation.dataset import load_golden_questions  # noqa: E402
from evaluation.evaluator import evaluate, failure_analysis  # noqa: E402

_DISPLAY = {"minilm": "MiniLM", "bge": "BGE", "e5": "E5"}


def _fmt(v: float) -> str:
    return f"{v:.3f}"


def print_per_question(per_question: dict, models: list[str]) -> None:
    for qid, qdata in per_question.items():
        print("=" * 72)
        print(f"QUESTION {qid}")
        print(f'"{qdata["question"]}"')
        print()
        print("Ground truth:")
        for rid in qdata["relevant"]:
            print(f"  {rid}")
        print()

        for model in models:
            m = qdata["models"][model]
            print(f"{_DISPLAY.get(model, model).upper()}")
            for r in m["ranks"]:
                mark = "✓" if r["relevant"] else "-"
                print(f"  {r['id']:<28} {mark}")
            rank = m["first_relevant_rank"]
            print(f"  First relevant rank: {rank if rank is not None else 'not in top-k'}")
            print(f"  RR: {m['reciprocal_rank']:.3f}")
            print()
    print()


def print_aggregate(aggregate: dict, models: list[str]) -> None:
    print("=" * 72)
    print("AGGREGATE BENCHMARK")
    print("=" * 72)
    header = f"{'MODEL':<8} {'R@1':>7} {'R@3':>7} {'R@5':>7} {'MRR':>8} {'Hit@5':>7} {'Avg latency':>13}"
    print(header)
    print("-" * 72)
    for model in models:
        a = aggregate[model]
        name = _DISPLAY.get(model, model)
        print(
            f"{name:<8} {_fmt(a['recall_at_1']):>7} {_fmt(a['recall_at_3']):>7} "
            f"{_fmt(a['recall_at_5']):>7} {_fmt(a['mrr']):>8} {_fmt(a['hit_rate_at_5']):>7} "
            f"{a['avg_total_ms']:>7} ms"
        )
    print()
    print("Note: recall/mrr/hit-rate measure retrieval effectiveness against ground")
    print("truth. Latency is warm retrieval (embed + vector search). Raw cosine")
    print("scores are NOT compared across models — each model has its own score")
    print("distribution.")


def print_failures(cases: list[dict], models: list[str]) -> None:
    if not cases:
        print("\nNo notable failure cases — every model retrieved all relevant evidence at rank 1.")
        return
    print("\n" + "=" * 72)
    print("FAILURE ANALYSIS")
    print("=" * 72)
    for c in cases:
        print(f"\n{c['question_id']}: {c['question']}")
        for model in models:
            rank = c["first_relevant_rank"][model]
            label = rank if rank is not None else "not in top-k"
            print(f"  {_DISPLAY.get(model, model).upper():<8} first relevant rank = {label}")
        if c["big_gaps"]:
            for g in c["big_gaps"]:
                print(f"  -> {_DISPLAY.get(g['earlier'], g['earlier'])} retrieved evidence substantially earlier than {_DISPLAY.get(g['later'], g['later'])}")


def save_results(payload: dict) -> Path:
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = results_dir / f"evaluation_{ts}.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval models against golden questions")
    parser.add_argument("--questions", default="data/evaluation/golden_questions.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--relevance-level", default="document", choices=["document", "chunk"])
    args = parser.parse_args()

    questions = load_golden_questions(args.questions)
    models = list(MODELS)

    result = evaluate(questions, models=models, top_k=args.top_k, relevance_level=args.relevance_level)

    print_per_question(result["per_question"], models)
    print_aggregate(result["aggregate"], models)
    cases = failure_analysis(result["per_question"], models, top_k=args.top_k)
    print_failures(cases, models)

    payload = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k": args.top_k,
            "similarity": "cosine",
            "relevance_level": args.relevance_level,
        },
        "models": {name: MODELS[name].hf_id for name in models},
        "num_questions": len(questions),
        "aggregate": result["aggregate"],
        "per_question": result["per_question"],
    }
    out = save_results(payload)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
